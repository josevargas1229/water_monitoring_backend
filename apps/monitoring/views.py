from rest_framework import viewsets, status
from rest_framework.decorators import action
from rest_framework.response import Response
from .models import Ecosystem, Images
from .serializers import EcosystemSerializer, ImagesSerializer
from .utils import calculate_ndvi, detect_water_and_vegetation, calculate_turbidity, load_unet_model, analyze_biodiversity, train_unet_model, adjust_image, process_db_images_with_model
from apps.alerts.models import Alert
from rest_framework import status
from django.contrib.gis.geos import Polygon
from django.core.files.base import ContentFile
from django.db.models import DateField
from django.db.models.functions import TruncDate
import json
class EcosystemViewSet(viewsets.ModelViewSet):
    queryset = Ecosystem.objects.all()
    serializer_class = EcosystemSerializer

class ImagesViewSet(viewsets.ModelViewSet):
    queryset = Images.objects.all()
    serializer_class = ImagesSerializer

    def perform_create(self, serializer):
        image = serializer.save()
        resolution = image.metadata.get('resolution_m_per_px', 0.1)
        det = detect_water_and_vegetation(image.image.path, resolution)
        # ndvi = calculate_ndvi(image.image.path)
        # turbidity = calculate_turbidity(image.image.path)
        bio_analysis = analyze_biodiversity(image.image.path)
        # image.ndvi_score = ndvi
        # image.turbidity = turbidity
        image.vegetation_percentage = det['vegetation_percentage']
        image.vegetation_area_m2 = det['vegetation_area_m2']
        image.water_percentage = det['water_percentage']
        image.water_area_m2 = det['water_area_m2']
        image.biodiversity_analysis = bio_analysis
        image.save()
        # if image.water_percentage < 50 or (ndvi is not None and ndvi < 0.3):
        #     Alert.objects.create(
        #         ecosystem=image.ecosystem,
        #         type='sequia',
        #         details={'ndvi': ndvi, 'water_percentage': image.water_percentage},
        #         confidence=0.85
        #     )
    
    @action(detail=False, methods=['get'], url_path='by-ecosystem/(?P<ecosystem_id>\d+)')
    def by_ecosystem(self, request, ecosystem_id=None):
        """
        Recupera imágenes asociadas a un ecosistema específico.

        Args:
            request: Objeto de solicitud HTTP que contiene parámetros de consulta.
            ecosystem_id (int, opcional): ID del ecosistema para filtrar imágenes.

        Query Parameters:
            year (int, opcional): Filtra imágenes por año de captura (ej. ?year=2020).
            capture_date (str, opcional): Filtra imágenes por fecha exacta (formato YYYY-MM-DD, ej. ?capture_date=2020-01-01).
            include_adjusted (bool, opcional): Si es 'true', incluye imágenes ajustadas; de lo contrario, solo originales (predeterminado: 'false').

        Returns:
            Response: Respuesta HTTP con los datos serializados de las imágenes o un mensaje de error.

        Raises:
            ValueError: Si el parámetro 'year' no es un número entero válido.
            Exception: Para errores inesperados durante la ejecución.
        """
        try:
            queryset = Images.objects.filter(ecosystem_id=ecosystem_id)
            
            # Filtrar por año si se proporciona ?year=2020
            year = request.query_params.get('year')
            if year:
                try:
                    year = int(year)
                    queryset = queryset.filter(capture_date__year=year)
                except ValueError:
                    return Response({'error': 'El parámetro "year" debe ser un número entero válido'}, status=status.HTTP_400_BAD_REQUEST)
            
            # Alternativamente, filtrar por fecha exacta si se proporciona ?capture_date=2020-01-01
            capture_date = request.query_params.get('capture_date')
            if capture_date:
                queryset = queryset.filter(capture_date__date=capture_date)  # Filtra por día exacto
            
            # Por defecto, solo originales; si ?include_adjusted=true, incluir ajustadas (pero las filtramos primero)
            include_adjusted = request.query_params.get('include_adjusted', 'false').lower() == 'true'
            if not include_adjusted:
                queryset = queryset.filter(is_adjusted=False)  # Solo originales por defecto
            
            if not queryset.exists():
                return Response({'message': 'No se encontraron imágenes para este ecosistema con los filtros aplicados'}, status=status.HTTP_404_NOT_FOUND)
            
            # Ordenar por capture_date descendente (más reciente primero)
            queryset = queryset.order_by('-capture_date')
            
            serializer = ImagesSerializer(queryset, many=True, context={'include_adjusted': include_adjusted})
            return Response(serializer.data, status=status.HTTP_200_OK)
        except Exception as e:
            return Response({'error': f'Error al recuperar imágenes: {str(e)}'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    
    @action(detail=False, methods=['get'], url_path='capture-dates/(?P<ecosystem_id>\d+)')
    def capture_dates(self, request, ecosystem_id=None):
        """
        Recupera las fechas distintas de captura de imágenes asociadas a un ecosistema específico.

        Args:
            request: Objeto de solicitud HTTP.
            ecosystem_id (int, opcional): ID del ecosistema para filtrar imágenes.

        Query Parameters:
            include_adjusted (bool, opcional): Si es 'true', incluye imágenes ajustadas; de lo contrario, solo originales (predeterminado: 'false').

        Returns:
            Response: Respuesta HTTP con una lista de fechas distintas en formato YYYY-MM-DD.

        Raises:
            Exception: Para errores inesperados durante la ejecución.
        """
        try:
            queryset = Images.objects.filter(ecosystem_id=ecosystem_id)
            
            # Por defecto, solo originales; si ?include_adjusted=true, incluir ajustadas
            include_adjusted = request.query_params.get('include_adjusted', 'false').lower() == 'true'
            if not include_adjusted:
                queryset = queryset.filter(is_adjusted=False)
            
            if not queryset.exists():
                return Response({'message': 'No se encontraron imágenes para este ecosistema con los filtros aplicados'}, status=status.HTTP_404_NOT_FOUND)
            
            # Obtener fechas distintas truncando capture_date a nivel de día
            dates = queryset.annotate(date=TruncDate('capture_date', output_field=DateField())).values('date').distinct()
            
            # Convertir las fechas a formato de cadena YYYY-MM-DD y eliminar duplicados
            date_list = sorted({date['date'].strftime('%Y-%m-%d') for date in dates}, reverse=True)
            
            return Response({'capture_dates': date_list}, status=status.HTTP_200_OK)
        except Exception as e:
            return Response({'error': f'Error al recuperar fechas de captura: {str(e)}'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        
    @action(detail=False, methods=['post'], url_path='adjust-and-analyze/(?P<image_id>\d+)')
    def adjust_and_analyze(self, request, image_id):
        """
        Ajusta una imagen original aplicando parámetros de edición (brillo, contraste, etc.). 
        Crea un nuevo registro para la imagen ajustada.

        Args:
            request: Objeto de solicitud HTTP que contiene los parámetros de ajuste.
            image_id (int): ID de la imagen original a ajustar.

        Form Data:
            brightness (float, opcional): Factor de ajuste de brillo (predeterminado: 0).
            contrast (float, opcional): Factor de ajuste de contraste (predeterminado: 1.0).
            saturation (float, opcional): Factor de ajuste de saturación (predeterminado: 1.0).
            sepia (float, opcional): Factor de ajuste de efecto sepia (predeterminado: 0.0).
            hue (float, opcional): Factor de ajuste de tono (predeterminado: 0).
            opacity (float, opcional): Factor de ajuste de opacidad (predeterminado: 1.0).
            blue_boost (float, opcional): Factor de refuerzo de color azul (predeterminado: 1.0).
            description (str, opcional): Descripción de la imagen ajustada (predeterminado: descripción de la imagen original).

        Returns:
            Response: Respuesta HTTP con los detalles de la imagen ajustada, su URL y métricas calculadas, o un mensaje de error.

        Raises:
            Images.DoesNotExist: Si el image_id proporcionado no existe.
            Exception: Para errores inesperados durante el ajuste o análisis de la imagen.
        """
        try:
            original_image = Images.objects.get(id=image_id)
            if original_image.is_adjusted:
                return Response({'error': 'Solo se pueden ajustar imágenes originales'}, status=status.HTTP_400_BAD_REQUEST)
        except Images.DoesNotExist:
            return Response({'error': 'Imagen no encontrada'}, status=status.HTTP_404_NOT_FOUND)

        # Parámetros de ajuste
        brightness = float(request.data.get('brightness', 0))
        contrast = float(request.data.get('contrast', 1.0))
        saturation = float(request.data.get('saturation', 1.0))
        sepia = float(request.data.get('sepia', 0.0))
        hue = float(request.data.get('hue', 0))
        opacity = float(request.data.get('opacity', 1.0))
        blue_boost = float(request.data.get('blue_boost', 1.0))
        description = request.data.get('description', original_image.description)

        # Aplicar ajustes y obtener bytes de la imagen ajustada
        adjusted_bytes = adjust_image(original_image.image.path, brightness, contrast, saturation, sepia, hue, opacity, blue_boost)
        if not adjusted_bytes:
            return Response({'error': 'Error al ajustar la imagen'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        # Cargar el modelo UNet
        model = load_unet_model()
        if model is None:
            return Response(
                {'error': 'No se pudo cargar el modelo UNet'},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

        # Crear nuevo registro para la imagen ajustada
        adjusted_image = Images(
            ecosystem=original_image.ecosystem,
            parent_image=original_image,
            is_adjusted=True,
            description=description,
            metadata=original_image.metadata.copy(),
            capture_date=original_image.capture_date
        )

        # Asignar la imagen ajustada usando ContentFile
        original_filename = original_image.image.name.split('/')[-1]
        adjusted_filename = f'adjusted_{original_filename}'
        adjusted_image.image.save(adjusted_filename, ContentFile(adjusted_bytes))

        # Recalcular métricas
        resolution = adjusted_image.metadata.get('resolution_m_per_px', 0.1)
        det = detect_water_and_vegetation(adjusted_image.image.path, model=model, resolution_m_per_px=resolution)
        # ndvi = calculate_ndvi(adjusted_image.image.path)
        # turbidity = calculate_turbidity(adjusted_image.image.path)
        # bio_analysis = analyze_biodiversity(adjusted_image.image.path)
        # adjusted_image.ndvi_score = ndvi
        # adjusted_image.turbidity = turbidity
        adjusted_image.vegetation_percentage = det['vegetation_percentage']
        adjusted_image.vegetation_area_m2 = det['vegetation_area_m2']
        adjusted_image.water_percentage = det['water_percentage']
        adjusted_image.water_area_m2 = det['water_area_m2']
        # adjusted_image.biodiversity_analysis = bio_analysis
        adjusted_image.save()

        return Response({
            'message': 'Imagen ajustada y analizada exitosamente',
            'adjusted_image_id': adjusted_image.id,
            'adjusted_image_url': adjusted_image.image.url,
            'adjusted_metrics': {
                # 'ndvi_score': float(ndvi) if ndvi is not None else None,
                # 'turbidity': float(turbidity),
                'vegetation_percentage': float(det['vegetation_percentage']),
                'vegetation_area_m2': float(det['vegetation_area_m2']),
                'water_percentage': float(det['water_percentage']),
                'water_area_m2': float(det['water_area_m2']),
                # 'biodiversity_analysis': bio_analysis
            }
        }, status=status.HTTP_201_CREATED)
        
    @action(detail=False, methods=['post'], url_path='upload-multiple')
    def upload_multiple(self, request):
        """
        Permite la carga de múltiples imágenes asociadas a un ecosistema.

        Args:
            request: Objeto de solicitud HTTP que contiene los datos y archivos enviados.

        Form Data:
            ecosystem_id (int, opcional): ID del ecosistema al que se asociarán las imágenes.
            ecosystem_name (str, opcional): Nombre del ecosistema si se crea uno nuevo (predeterminado: 'Cuerpo de agua nuevo').
            coordinates (list, opcional): Coordenadas para crear un polígono de ubicación del ecosistema.
            images (list of files): Lista de imágenes a cargar.
            descriptions (list of str, opcional): Lista de descripciones para las imágenes.
            capture_dates (list, opcional): Lista de fechas de captura para cada imagen (formato YYYY-MM-DD).

        Returns:
            Response: Respuesta HTTP con los detalles de las imágenes procesadas y el ecosistema, o un mensaje de error.

        Raises:
            Ecosystem.DoesNotExist: Si el ecosystem_id proporcionado no existe.
            ValueError, TypeError: Si las coordenadas no son válidas para crear un polígono.
            Exception: Para errores inesperados durante el procesamiento de las imágenes.
        """
        ecosystem_id = request.data.get('ecosystem_id') 
        ecosystem_name = request.data.get('ecosystem_name', 'Cuerpo de agua nuevo')
        images = request.FILES.getlist('images')
        descriptions = request.data.get('descriptions', [])
        coordinates = request.data.get('coordinates')
        capture_dates = request.data.get('capture_dates', [])

        if not images:
            return Response({'error': 'Se requieren imágenes'}, status=status.HTTP_400_BAD_REQUEST)
        
        if ecosystem_id:
            try:
                ecosystem = Ecosystem.objects.get(id=ecosystem_id)
            except Ecosystem.DoesNotExist:
                return Response({'error': 'Ecosistema no encontrado'}, status=status.HTTP_404_NOT_FOUND)
        else:
            # Crear un nuevo ecosistema si no se proporciona ID
            location = None
            if coordinates:
                try:
                    coordinates = json.loads(coordinates)
                    # Convertir las coordenadas en un objeto Polygon
                    location = Polygon(coordinates)
                except (ValueError, TypeError):
                    return Response({'error': 'Coordenadas inválidas para el polígono'}, status=status.HTTP_400_BAD_REQUEST)
            
            ecosystem = Ecosystem.objects.create(
                name=ecosystem_name,
                location=location
            )
        
        model = load_unet_model()
        if model is None:
            return Response(
                {'error': 'No se pudo cargar el modelo UNet'},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
        
        responses = []
        for i, image in enumerate(images):
            description = descriptions[i] if i < len(descriptions) else ''
            capture_date = capture_dates[i] if i < len(capture_dates) else None
            
            if not capture_date:
                return Response({'error': 'Se requiere una fecha de captura para cada imagen'}, status=status.HTTP_400_BAD_REQUEST)
            
            drone_image = Images(
                ecosystem=ecosystem,
                image=image,
                description=description,
                metadata={'resolution_m_per_px': 0.1},
                capture_date=capture_date,
                is_adjusted=False,  # Valor por defecto
                parent_image=None  # Valor por defecto
            )
            drone_image.save()
            
            resolution = drone_image.metadata.get('resolution_m_per_px', 0.1)
            det = detect_water_and_vegetation(drone_image.image.path, model=model, resolution_m_per_px=resolution)
            # ndvi = calculate_ndvi(drone_image.image.path)
            # turbidity = calculate_turbidity(drone_image.image.path)
            # bio_analysis = analyze_biodiversity(drone_image.image.path)
            # drone_image.ndvi_score = ndvi
            # drone_image.turbidity = turbidity
            drone_image.vegetation_percentage = det['vegetation_percentage']
            drone_image.vegetation_area_m2 = det['vegetation_area_m2']
            drone_image.water_percentage = det['water_percentage']
            drone_image.water_area_m2 = det['water_area_m2']
            # drone_image.biodiversity_analysis = bio_analysis
            drone_image.save()
            # if drone_image.water_percentage < 50 or (ndvi is not None and ndvi < 0.3):
            #     Alert.objects.create(
            #         ecosystem=drone_image.ecosystem,
            #         type='sequia',
            #         details={ 'water_percentage': drone_image.water_percentage},
            #         confidence=0.85
            #     )
            responses.append({
                'id': drone_image.id,
                'image': drone_image.image.url,
                'description': description,
                'water_percentage': drone_image.water_percentage,
                'water_area_m2': drone_image.water_area_m2,
                'vegetation_percentage': drone_image.vegetation_percentage,
                'vegetation_area_m2': drone_image.vegetation_area_m2,
                # 'turbidity': drone_image.turbidity
            })

        return Response({
            'message': 'Imágenes subidas exitosamente',
            'ecosystem_id': ecosystem.id,
            'ecosystem_name': ecosystem.name,
            'images': responses
        }, status=status.HTTP_201_CREATED)

    @action(detail=False, methods=['post'], url_path='train-model')
    def train_model(self, request):
        model = train_unet_model()
        if model:
            return Response({'message': 'Modelo entrenado exitosamente'}, status=status.HTTP_200_OK)
        return Response({'error': 'Fallo al entrenar el modelo'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    @action(detail=False, methods=['post'], url_path='train-and-process-db')
    def train_and_process_db(self, request):
        # Luego, procesar imágenes en BD con el modelo entrenado
        process_db_images_with_model()
        return Response({'message': 'Modelo entrenado y base de datos procesada exitosamente'}, status=status.HTTP_200_OK)

    def process_db_images_with_model(model=None):
        try:
            # Usar el modelo proporcionado o cargar uno si no se pasa
            if model is None:
                model = load_unet_model()
                if model is None:
                    print("No se pudo cargar el modelo UNet")
                    return
            
            # Obtener todas las imágenes de la BD
            drone_images = Images.objects.all()
            print(f"Procesando {len(drone_images)} imágenes de la base de datos con el modelo entrenado")
            
            for img_obj in drone_images:
                print(f"Procesando imagen: {img_obj.image.path}")
                
                # Calcular detección de agua y vegetación
                resolution = img_obj.metadata.get('resolution_m_per_px', 0.1)
                det = detect_water_and_vegetation(img_obj.image.path, model=model, resolution_m_per_px=resolution)
                
                # Calcular turbidez
                # turbidity = calculate_turbidity(img_obj.image.path, model=model)
                
                # Calcular NDVI y biodiversidad
                # ndvi = calculate_ndvi(img_obj.image.path)
                # biodiversity = analyze_biodiversity(img_obj.image.path)
                
                # Actualizar el objeto en la BD
                img_obj.water_percentage = det['water_percentage']
                img_obj.water_area_m2 = det['water_area_m2']
                img_obj.vegetation_percentage = det['vegetation_percentage']
                img_obj.vegetation_area_m2 = det['vegetation_area_m2']
                # img_obj.turbidity = turbidity
                # if ndvi is not None:
                #     img_obj.ndvi = ndvi
                # if biodiversity:
                #     img_obj.biodiversity = biodiversity
                img_obj.save()
            
            print("✅ Procesamiento completado para todas las imágenes en la BD")
        except Exception as e:
            print(f"Error en process_db_images_with_model: {e}")