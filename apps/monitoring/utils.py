import os
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import torch
import segmentation_models_pytorch as smp
from .models import Images
import logging
from typing import Optional, Tuple, Dict, Union

# Configuración del dispositivo
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
INVERT_MASK = False  # Ajustar según el dataset (agua=blanco, fondo=negro)
CHECKPOINT_PATH = os.path.join(os.path.dirname(__file__), '..', 'checkpoints', 'unet_resnet34_best.pth')

# Configuración del logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def preprocess_image_from_array(img_array: np.ndarray, target_size: Tuple[int, int] = (256, 256)) -> Optional[torch.Tensor]:
    """
    Preprocesa una imagen desde un array de numpy para el modelo UNet.
    
    Args:
        img_array: Array de imagen en formato BGR de OpenCV
        target_size: Tamaño objetivo (ancho, alto)
    
    Returns:
        Tensor de PyTorch listo para el modelo o None si hay error
    """
    try:
        if img_array is None:
            raise ValueError("Array de imagen es None")
        
        # Convertir BGR a RGB
        img = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
        
        # Redimensionar
        img = cv2.resize(img, target_size)
        
        # Normalizar
        img = img / 255.0
        
        # Cambiar a formato (C, H, W)
        img = np.transpose(img, (2, 0, 1))
        
        # Convertir a tensor y añadir dimensión de batch
        img_tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        
        return img_tensor
        
    except Exception as e:
        logger.error(f"Error en preprocess_image_from_array: {e}")
        return None

def preprocess_image(image_path: str, target_size: Tuple[int, int] = (256, 256)) -> Optional[torch.Tensor]:
    """
    Preprocesa una imagen desde archivo para el modelo UNet.
    
    Args:
        image_path: Ruta al archivo de imagen
        target_size: Tamaño objetivo (ancho, alto)
    
    Returns:
        Tensor de PyTorch listo para el modelo o None si hay error
    """
    try:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Imagen no encontrada: {image_path}")
        
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"No se pudo cargar la imagen: {image_path}")
        
        return preprocess_image_from_array(img, target_size)
        
    except FileNotFoundError as e:
        logger.error(f"Archivo no encontrado: {e}")
        return None
    except Exception as e:
        logger.error(f"Error en preprocess_image: {e}")
        return None

def load_unet_model() -> Optional[torch.nn.Module]:
    """
    Carga el modelo UNet con pesos preentrenados.
    
    Returns:
        Modelo UNet cargado o None si hay error
    """
    try:
        # Cargar modelo UNet con backbone ResNet34
        model = smp.Unet(
            encoder_name="resnet34",
            encoder_weights=None,  # Pesos cargados desde checkpoint
            in_channels=3,
            classes=1,
            activation="sigmoid"
        ).to(DEVICE)
        
        # Cargar checkpoint
        if os.path.exists(CHECKPOINT_PATH):
            checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=True)
            model.load_state_dict(checkpoint)
            model.eval()
            logger.info("✅ Modelo UNet cargado desde checkpoint")
        else:
            logger.warning("⚠️ Checkpoint no encontrado, inicializando modelo sin pesos preentrenados")
        
        return model
        
    except torch.cuda.OutOfMemoryError:
        logger.error("Error: Memoria GPU insuficiente para cargar el modelo")
        return None
    except Exception as e:
        logger.error(f"Error en load_unet_model: {e}")
        return None

def validate_inputs(image_path: str, resolution_m_per_px: float) -> None:
    """
    Valida los parámetros de entrada.
    
    Args:
        image_path: Ruta a la imagen
        resolution_m_per_px: Resolución en metros por píxel
        
    Raises:
        ValueError: Si los parámetros no son válidos
        FileNotFoundError: Si la imagen no existe
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Imagen no encontrada: {image_path}")
    
    if resolution_m_per_px <= 0:
        raise ValueError("La resolución debe ser un valor positivo")

def detect_water_and_vegetation(
    image_path: str,
    model: Optional[torch.nn.Module] = None,
    resolution_m_per_px: float = 0.1,
    water_threshold: float = 0.5,
    green_range: Tuple[Tuple[int, int, int], Tuple[int, int, int]] = ((40, 50, 50), (80, 255, 255)),
    invert_water_mask: bool = False,
    target_size: Tuple[int, int] = (256, 256)
) -> Dict[str, float]:
    """
    Detecta y cuantifica áreas de agua y vegetación en una imagen.
    
    Args:
        image_path: Ruta al archivo de imagen
        model: Modelo UNet preentrenado (opcional)
        resolution_m_per_px: Resolución en metros por píxel
        water_threshold: Umbral para la detección de agua (0-1)
        green_range: Rango de colores HSV para vegetación ((H_min, S_min, V_min), (H_max, S_max, V_max))
        invert_water_mask: Si invertir la máscara de agua
        target_size: Tamaño objetivo para el procesamiento del modelo
        
    Returns:
        Diccionario con porcentajes y áreas en m² de agua y vegetación
        
    Raises:
        ValueError: Si los parámetros no son válidos
        FileNotFoundError: Si la imagen no existe
    """
    try:
        # Validar entradas
        validate_inputs(image_path, resolution_m_per_px)
        
        if not (0 <= water_threshold <= 1):
            raise ValueError("water_threshold debe estar entre 0 y 1")
        
        # Cargar imagen una sola vez
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"No se pudo cargar la imagen: {image_path}")
        
        original_height, original_width = img.shape[:2]
        total_pixels = original_height * original_width
        
        # === DETECCIÓN DE AGUA CON MODELO UNET ===
        water_percentage = 0.0
        water_area_m2 = 0.0
        
        try:
            # Cargar modelo si no se proporciona
            if model is None:
                model = load_unet_model()
                if model is None:
                    raise RuntimeError("No se pudo cargar el modelo UNet")
            
            # Preprocesar imagen para el modelo
            img_tensor = preprocess_image_from_array(img, target_size)
            if img_tensor is None:
                raise RuntimeError("No se pudo preprocesar la imagen para el modelo")
            
            # Inferencia
            with torch.no_grad():
                output = model(img_tensor)
                # Redimensionar la salida al tamaño original
                water_mask = torch.nn.functional.interpolate(
                    output, 
                    size=(original_height, original_width), 
                    mode='bilinear', 
                    align_corners=False
                )
                
                # Aplicar umbral y convertir a numpy
                water_mask = (water_mask > water_threshold).float().squeeze().cpu().numpy()
                
                if invert_water_mask:
                    water_mask = 1 - water_mask
            
            # Calcular métricas de agua
            water_area_px = np.sum(water_mask > 0)
            water_percentage = (water_area_px / total_pixels) * 100
            water_area_m2 = water_area_px * (resolution_m_per_px ** 2)
            
        except Exception as e:
            logger.error(f"Error en detección de agua: {e}")
            # Continuar con vegetación aunque falle la detección de agua
        
        # === DETECCIÓN DE VEGETACIÓN CON ANÁLISIS HSV ===
        vegetation_percentage = 0.0
        vegetation_area_m2 = 0.0
        
        try:
            # Convertir a HSV
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            
            # Crear máscara de vegetación
            lower_green = np.array(green_range[0])
            upper_green = np.array(green_range[1])
            veg_mask = cv2.inRange(hsv, lower_green, upper_green)
            
            # Calcular métricas de vegetación
            veg_area_px = np.sum(veg_mask > 0)
            vegetation_percentage = (veg_area_px / total_pixels) * 100
            vegetation_area_m2 = veg_area_px * (resolution_m_per_px ** 2)
            
        except Exception as e:
            logger.error(f"Error en detección de vegetación: {e}")
        
        # Resultados
        results = {
            'water_percentage': float(water_percentage),
            'water_area_m2': float(water_area_m2),
            'vegetation_percentage': float(vegetation_percentage),
            'vegetation_area_m2': float(vegetation_area_m2),
            'total_pixels': int(total_pixels),
            'image_dimensions': (original_width, original_height)
        }
        
        logger.info(f"Detección completada - Agua: {water_percentage:.2f}%, Vegetación: {vegetation_percentage:.2f}%")
        return results
        
    except FileNotFoundError as e:
        logger.error(f"Archivo no encontrado: {e}")
        raise
    except ValueError as e:
        logger.error(f"Error de validación: {e}")
        raise
    except torch.cuda.OutOfMemoryError:
        logger.error("Error: Memoria GPU insuficiente")
        return {
            'water_percentage': 0.0, 'water_area_m2': 0.0, 
            'vegetation_percentage': 0.0, 'vegetation_area_m2': 0.0,
            'total_pixels': 0, 'image_dimensions': (0, 0)
        }
    except Exception as e:
        logger.error(f"Error inesperado en detect_water_and_vegetation: {e}")
        return {
            'water_percentage': 0.0, 'water_area_m2': 0.0, 
            'vegetation_percentage': 0.0, 'vegetation_area_m2': 0.0,
            'total_pixels': 0, 'image_dimensions': (0, 0)
        }

# Función de conveniencia para usar con configuración predeterminada
def quick_detect_water_vegetation(image_path: str, resolution_m_per_px: float = 0.1) -> Dict[str, float]:
    """
    Función simplificada para detección rápida de agua y vegetación.
    
    Args:
        image_path: Ruta al archivo de imagen
        resolution_m_per_px: Resolución en metros por píxel
        
    Returns:
        Diccionario con resultados de la detección
    """
    return detect_water_and_vegetation(image_path, resolution_m_per_px=resolution_m_per_px)

def calculate_turbidity(image_path, model=None, water_mask=None):
    try:
        if water_mask is None:
            if model is None:
                model = load_unet_model()
                if model is None:
                    raise ValueError("No se pudo cargar el modelo UNet")
            
            img_tensor = preprocess_image(image_path)
            if img_tensor is None:
                raise ValueError("No se pudo preprocesar la imagen")
            
            with torch.no_grad():
                output = model(img_tensor)
                water_mask = (output > 0.5).float().squeeze().cpu().numpy()
                if INVERT_MASK:
                    water_mask = 1 - water_mask
                water_mask = cv2.resize(water_mask, (cv2.imread(image_path).shape[1], cv2.imread(image_path).shape[0]))
                water_mask = (water_mask > 0).astype(np.uint8) * 255

        img = cv2.imread(image_path)
        if img is None:
            raise ValueError("No se pudo cargar la imagen")
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        water_areas = img[water_mask > 0]
        if len(water_areas) == 0:
            return 0.0
        turbidity = np.std(water_areas[:, 1]) + np.std(water_areas[:, 2])
        turbidity_normalized = min(turbidity / 100, 1.0)
        return float(turbidity_normalized * 100)
    except Exception as e:
        print(f"Error en calculate_turbidity: {e}")
        return 0.0

# def calculate_turbidity(image_path, water_mask=None):
#     try:
#         # Si no se proporciona una máscara, generar una con UNet
#         if water_mask is None:
#             model = load_unet_model()
#             if model is None:
#                 raise ValueError("No se pudo cargar el modelo UNet")
            
#             img_tensor = preprocess_image(image_path)
#             if img_tensor is None:
#                 raise ValueError("No se pudo preprocesar la imagen")
            
#             with torch.no_grad():
#                 output = model(img_tensor)
#                 water_mask = (output > 0.5).float().squeeze().cpu().numpy()
#                 if INVERT_MASK:
#                     water_mask = 1 - water_mask
#                 water_mask = cv2.resize(water_mask, (cv2.imread(image_path).shape[1], cv2.imread(image_path).shape[0]))
#                 water_mask = (water_mask > 0).astype(np.uint8) * 255

#         # Calcular turbidez
#         img = cv2.imread(image_path)
#         if img is None:
#             raise ValueError("No se pudo cargar la imagen")
#         hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
#         water_areas = img[water_mask > 0]
#         if len(water_areas) == 0:
#             return 0.0
#         turbidity = np.std(water_areas[:, 1]) + np.std(water_areas[:, 2])  # S y V en HSV
#         turbidity_normalized = min(turbidity / 100, 1.0)
#         return float(turbidity_normalized * 100)
#     except Exception as e:
#         print(f"Error en calculate_turbidity: {e}")
#         return 0.0

def prepare_dataset(use_external_only=True):
    try:
        images = []
        turbidity_values = []
        vegetation_percentages = []
        water_percentages = []
        
        model = load_unet_model()
        if model is None:
            print("No se pudo cargar el modelo UNet")
            return None, None, None, None
        
        if not use_external_only:
            # Parte original para imágenes de la BD (comentada para usar solo externo)
            # drone_images = Images.objects.filter(water_percentage__isnull=False)
            # print(f"Procesando {len(drone_images)} imágenes de la base de datos")
            # for img_obj in drone_images:
            #     print(f"Procesando imagen: {img_obj.image.path}")
            #     img = Image.open(img_obj.image.path).convert('RGB')
            #     img = img.resize((256, 256))
            #     img_array = np.array(img) / 255.0
            #     images.append(img_array)
            #     
            #     det = detect_water_and_vegetation(img_obj.image.path, model=model, resolution_m_per_px=img_obj.metadata.get('resolution_m_per_px', 0.1))
            #     turbidity = calculate_turbidity(img_obj.image.path, model=model)
            #     turbidity_values.append(turbidity)
            #     vegetation_percentages.append(img_obj.vegetation_percentage or det['vegetation_percentage'])
            #     water_percentages.append(img_obj.water_percentage or det['water_percentage'])
            pass  # Ignorar BD si use_external_only=True
        
        external_images, external_turbidity, external_veg, external_water = load_external_dataset(model=model)
        if external_images is not None and len(external_images) > 0:
            print(f"Agregando {len(external_images)} imágenes del dataset externo")
            images.extend(external_images)
            turbidity_values.extend(external_turbidity)
            vegetation_percentages.extend(external_veg)
            water_percentages.extend(external_water)
        else:
            print("No se cargaron imágenes del dataset externo")
            return None, None, None, None
        
        print(f"Dataset preparado con {len(images)} imágenes (solo externo)")
        return np.array(images), np.array(turbidity_values), np.array(vegetation_percentages), np.array(water_percentages)
    except Exception as e:
        print(f"Error en prepare_dataset: {e}")
        return None, None, None, None

def train_unet_model(dataset_images=None, dataset_turbidity=None, dataset_veg=None, dataset_water=None):
    try:
        if dataset_images is None:
            dataset_images, _, dataset_veg, dataset_water = prepare_dataset(use_external_only=True)
            if dataset_images is None or len(dataset_images) == 0:
                print("No hay datos suficientes para entrenar")
                return None

        checkpoint_dir = os.path.dirname(CHECKPOINT_PATH)
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
            print(f"Directorio de checkpoints creado: {checkpoint_dir}")

        print(f"Entrenando con {len(dataset_images)} imágenes")
        model = smp.Unet(
            encoder_name="resnet34",
            encoder_weights="imagenet",
            in_channels=3,
            classes=1,
            activation="sigmoid"
        ).to(DEVICE)

        X_train, X_val = train_test_split(dataset_images, test_size=0.2, random_state=42)
        print(f"Conjunto de entrenamiento: {len(X_train)} imágenes, validación: {len(X_val)} imágenes")
        
        X_train = torch.tensor(X_train, dtype=torch.float32).permute(0, 3, 1, 2).to(DEVICE)
        X_val = torch.tensor(X_val, dtype=torch.float32).permute(0, 3, 1, 2).to(DEVICE)

        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = smp.losses.DiceLoss(mode='binary')

        epochs = 40
        best_iou = 0
        for epoch in range(epochs):
            model.train()
            print(f"Época {epoch+1}/{epochs}")
            for i, img in enumerate(X_train):
                img = img.unsqueeze(0)
                optimizer.zero_grad()
                output = model(img)
                mask = (img[:, 0, :, :] > 0.5).float()
                loss = criterion(output, mask.unsqueeze(0))
                loss.backward()
                optimizer.step()
                if i % 10 == 0:
                    print(f"  Imagen {i+1}/{len(X_train)}, pérdida: {loss.item():.4f}")
            
            model.eval()
            with torch.no_grad():
                iou = 0
                for img in X_val:
                    img = img.unsqueeze(0)
                    output = model(img)
                    mask = (img[:, 0, :, :] > 0.5).float()
                    pred = (output > 0.5).float()
                    intersection = (pred * mask).sum()
                    union = (pred + mask).sum() - intersection
                    iou += intersection / (union + 1e-10)
                iou = iou / len(X_val)
                print(f"  IoU de validación: {iou:.4f}")
            
            if iou > best_iou:
                best_iou = iou
                torch.save(model.state_dict(), CHECKPOINT_PATH)
                print(f"✅ Mejor modelo guardado en época {epoch+1} con IoU: {iou:.4f}")

        print("✅ Entrenamiento completado")
        return model
    except Exception as e:
        print(f"Error en train_unet_model: {e}")
        return None
def calculate_ndvi(image_path):
    try:
        img = np.array(Image.open(image_path).convert('RGB'))
        nir = img[:, :, 0].astype(float)
        red = img[:, :, 2].astype(float)
        ndvi = (nir - red) / (nir + red + 1e-10)
        return float(np.mean(ndvi))
    except Exception as e:
        print(f"Error en calculate_ndvi: {e}")
        return None

def analyze_biodiversity(image_path):
    try:
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError("No se pudo cargar la imagen")
        data = img.reshape(-1, 3)
        clf = RandomForestClassifier(n_estimators=100)
        labels = np.random.randint(0, 3, size=data.shape[0])  # Simulación
        clf.fit(data, labels)
        prediction = clf.predict(data)
        unique, counts = np.unique(prediction, return_counts=True)
        return {int(k): float(v / len(prediction)) for k, v in zip(unique, counts)}
    except Exception as e:
        print(f"Error en analyze_biodiversity: {e}")
        return {}

def load_external_dataset(dataset_path='datasets/water_bodies', model=None):
    try:
        images = []
        turbidity_values = []
        vegetation_percentages = []
        water_percentages = []
        
        image_dir = os.path.join(dataset_path, 'images')
        mask_dir = os.path.join(dataset_path, 'masks')
        
        for img_name in os.listdir(image_dir):
            img_path = os.path.join(image_dir, img_name)
            mask_path = os.path.join(mask_dir, img_name)
            
            img = Image.open(img_path).convert('RGB')
            img = img.resize((256, 256))
            img_array = np.array(img) / 255.0
            images.append(img_array)
            
            det = detect_water_and_vegetation(img_path, model=model)
            turbidity = calculate_turbidity(img_path, model=model)
            turbidity_values.append(turbidity)
            vegetation_percentages.append(det['vegetation_percentage'])
            water_percentages.append(det['water_percentage'])
        
        return np.array(images), np.array(turbidity_values), np.array(vegetation_percentages), np.array(water_percentages)
    except Exception as e:
        print(f"Error en load_external_dataset: {e}")
        return None, None, None, None

def adjust_image(image_path, brightness=0, contrast=1.0, saturation=1.0, sepia=0.0, hue=0, opacity=1.0, blue_boost=1.0):
    try:
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError("No se pudo cargar la imagen")

        # Ajuste de brillo y contraste
        img = cv2.convertScaleAbs(img, alpha=contrast, beta=brightness)

        # Ajuste de saturación y hue (en HSV)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        s = cv2.multiply(s, saturation)
        s = np.clip(s, 0, 255)
        h = (h + hue) % 180  # Ajuste de hue
        hsv_adjusted = cv2.merge([h, s, v])
        img = cv2.cvtColor(hsv_adjusted, cv2.COLOR_HSV2BGR)

        # Efecto sepia
        if sepia > 0:
            sepia_matrix = np.array([[0.272, 0.543, 0.131],
                                     [0.349, 0.686, 0.168],
                                     [0.393, 0.769, 0.189]])
            img = cv2.transform(img, sepia_matrix * sepia)
            img = np.clip(img, 0, 255)

        # Resaltar azul
        b, g, r = cv2.split(img)
        b = cv2.multiply(b, blue_boost)
        b = np.clip(b, 0, 255)
        img = cv2.merge([b, g, r])

        # Opacidad (mezcla con blanco para simular)
        if opacity < 1.0:
            white = np.full_like(img, 255)
            img = cv2.addWeighted(img, opacity, white, 1 - opacity, 0)

        # Codificar la imagen ajustada a bytes (formato JPEG)
        _, buffer = cv2.imencode('.jpg', img)
        return buffer.tobytes()
    except Exception as e:
        print(f"Error en adjust_image: {e}")
        return None
    
def process_db_images_with_model():
    try:
        # Cargar el modelo entrenado (usa el checkpoint)
        model = load_unet_model()
        if model is None:
            print("No se pudo cargar el modelo UNet")
            return
        
        # Obtener todas las imágenes de la BD (o filtra según necesites, ej: water_percentage__isnull=True)
        drone_images = Images.objects.all()  # O usa .filter() para procesar solo algunas
        print(f"Procesando {len(drone_images)} imágenes de la base de datos con el modelo entrenado")
        
        for img_obj in drone_images:
            print(f"Procesando imagen: {img_obj.image.path}")
            
            # Calcular detección de agua y vegetación con el modelo
            resolution = img_obj.metadata.get('resolution_m_per_px', 0.1)
            det = detect_water_and_vegetation(img_obj.image.path, model=model, resolution_m_per_px=resolution)
            
            # Calcular turbidez con el modelo
            # turbidity = calculate_turbidity(img_obj.image.path, model=model)
            
            # Otros cálculos (ej: NDVI, biodiversidad)
            # ndvi = calculate_ndvi(img_obj.image.path)
            # biodiversity = analyze_biodiversity(img_obj.image.path)
            
            # Actualizar el objeto en la BD
            img_obj.water_percentage = det['water_percentage']
            img_obj.water_area_m2 = det['water_area_m2']
            img_obj.vegetation_percentage = det['vegetation_percentage']
            img_obj.vegetation_area_m2 = det['vegetation_area_m2']
            # img_obj.turbidity = turbidity 
            img_obj.save()
        
        print("✅ Procesamiento completado para todas las imágenes en la BD")
    except Exception as e:
        print(f"Error en process_db_images_with_model: {e}")
