import os
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from .models import Images
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

def detect_water_and_vegetation(image_path, resolution_m_per_px=0.1):
    try:
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError("No se pudo cargar la imagen")
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        # Máscara para agua (azul en HSV)
        lower_blue = np.array([100, 50, 50])
        upper_blue = np.array([130, 255, 255])
        water_mask = cv2.inRange(hsv, lower_blue, upper_blue)
        water_area_px = np.sum(water_mask > 0)
        water_percentage = (water_area_px / (img.shape[0] * img.shape[1])) * 100
        water_area_m2 = water_area_px * (resolution_m_per_px ** 2)
        
        # Máscara para vegetación (verde en HSV)
        lower_green = np.array([40, 50, 50])
        upper_green = np.array([80, 255, 255])
        veg_mask = cv2.inRange(hsv, lower_green, upper_green)
        veg_area_px = np.sum(veg_mask > 0)
        vegetation_percentage = (veg_area_px / (img.shape[0] * img.shape[1])) * 100
        vegetation_area_m2 = veg_area_px * (resolution_m_per_px ** 2)
        
        return {
            'water_percentage': water_percentage,
            'water_area_m2': water_area_m2,
            'vegetation_percentage': vegetation_percentage,
            'vegetation_area_m2': vegetation_area_m2
        }
    except Exception as e:
        print(f"Error en detect_water_and_vegetation: {e}")
        return {'water_percentage': 0, 'water_area_m2': 0, 'vegetation_percentage': 0, 'vegetation_area_m2': 0}

def calculate_turbidity(image_path, water_mask=None):
    try:
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError("No se pudo cargar la imagen")
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        if water_mask is None:
            lower_blue = np.array([100, 50, 50])
            upper_blue = np.array([130, 255, 255])
            water_mask = cv2.inRange(hsv, lower_blue, upper_blue)
        water_areas = img[water_mask > 0]
        if len(water_areas) == 0:
            return 0
        turbidity = np.std(water_areas[:, 1]) + np.std(water_areas[:, 2])  # S y V en HSV
        turbidity_normalized = min(turbidity / 100, 1.0)  # Normalizar a [0, 100]
        return turbidity_normalized * 100
    except Exception as e:
        print(f"Error en calculate_turbidity: {e}")
        return 0

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
        # Convertir claves a int estándar para evitar problemas con JSON
        return {int(k): float(v / len(prediction)) for k, v in zip(unique, counts)}
    except Exception as e:
        print(f"Error en analyze_biodiversity: {e}")
        return {}

def load_external_dataset(dataset_path='datasets/water_bodies'):
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
            
            # Calcular valores de regresión
            det = detect_water_and_vegetation(img_path)
            turbidity = calculate_turbidity(img_path)
            turbidity_values.append(turbidity)
            vegetation_percentages.append(det['vegetation_percentage'])
            water_percentages.append(det['water_percentage'])
        
        return np.array(images), np.array(turbidity_values), np.array(vegetation_percentages), np.array(water_percentages)
    except Exception as e:
        print(f"Error en load_external_dataset: {e}")
        return None, None, None, None

def prepare_dataset():
    try:
        images = []
        turbidity_values = []
        vegetation_percentages = []
        water_percentages = []
        
        drone_images = Images.objects.filter(water_percentage__isnull=False)
        
        for img_obj in drone_images:
            img = Image.open(img_obj.image.path).convert('RGB')
            img = img.resize((256, 256))
            img_array = np.array(img) / 255.0
            images.append(img_array)
            
            # Usar valores almacenados o recalcular
            turbidity = calculate_turbidity(img_obj.image.path)
            det = detect_water_and_vegetation(img_obj.image.path, resolution_m_per_px=img_obj.metadata.get('resolution_m_per_px', 0.1))
            turbidity_values.append(turbidity)
            vegetation_percentages.append(img_obj.vegetation_percentage or det['vegetation_percentage'])
            water_percentages.append(img_obj.water_percentage or det['water_percentage'])
        
        external_images, external_turbidity, external_veg, external_water = load_external_dataset()
        if external_images is not None and len(external_images) > 0:
            images.extend(external_images)
            turbidity_values.extend(external_turbidity)
            vegetation_percentages.extend(external_veg)
            water_percentages.extend(external_water)
        
        return np.array(images), np.array(turbidity_values), np.array(vegetation_percentages), np.array(water_percentages)
    except Exception as e:
        print(f"Error en prepare_dataset: {e}")
        return None, None, None, None

def train_cnn_model(dataset_images=None, dataset_turbidity=None, dataset_veg=None, dataset_water=None):
    try:
        if dataset_images is None:
            dataset_images, dataset_turbidity, dataset_veg, dataset_water = prepare_dataset()
            if dataset_images is None or len(dataset_images) == 0:
                print("No hay datos suficientes para entrenar")
                return None

        # Combinar salidas para multi-output
        y = np.column_stack((dataset_turbidity, dataset_veg, dataset_water))

        # Dividir dataset en train y validation
        X_train, X_val, y_train, y_val = train_test_split(
            dataset_images, y, test_size=0.2, random_state=42
        )

        # Data augmentation
        datagen = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )
        datagen.fit(X_train)

        # Crear modelo
        model = Sequential([
            Conv2D(32, (3,3), activation='relu', input_shape=(256, 256, 3)),
            MaxPooling2D((2,2)),
            Conv2D(64, (3,3), activation='relu'),
            MaxPooling2D((2,2)),
            Conv2D(128, (3,3), activation='relu'),
            MaxPooling2D((2,2)),
            Flatten(),
            Dense(256, activation='relu'),
            Dense(3, activation='linear')  # 3 salidas: turbidez, veg %, agua %
        ])

        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )

        # Generadores train y val
        train_gen = datagen.flow(X_train, y_train, batch_size=32)
        val_gen = datagen.flow(X_val, y_val, batch_size=32)

        # Entrenar
        model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=10
        )

        # Guardar modelo
        model_dir = os.path.join(os.path.dirname(__file__), 'models')
        os.makedirs(model_dir, exist_ok=True)
        model.save(os.path.join(model_dir, 'cnn_model.keras'))

        print("✅ Modelo CNN entrenado y guardado en 'models/cnn_model.h5'")
        return model

    except Exception as e:
        print(f"Error en train_cnn_model: {e}")
        return None
    
def adjust_image(image_path, brightness=0, contrast=1.0, saturation=1.0, sepia=0.0, hue=0, opacity=1.0, blue_boost=1.0):
    """
    Ajusta varios parámetros en la imagen.
    - brightness: -255 a 255
    - contrast: 0.0 a 3.0
    - saturation: 0.0 a 3.0
    - sepia: 0.0 a 1.0 (intensidad del efecto sepia)
    - hue: -180 a 180 (ajuste de color)
    - opacity: 0.0 a 1.0
    - blue_boost: 1.0 a 2.0 (resaltar azul para agua)
    """
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

        # Guardar temporal
        temp_path = image_path + '_adjusted.jpg'
        cv2.imwrite(temp_path, img)
        return temp_path
    except Exception as e:
        print(f"Error en adjust_image: {e}")
        return None