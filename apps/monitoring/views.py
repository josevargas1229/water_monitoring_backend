from rest_framework import viewsets
from rest_framework.decorators import action
from rest_framework.response import Response
from .models import Ecosystem, Images
from .serializers import EcosystemSerializer, ImagesSerializer
from .utils import calculate_ndvi, detect_water_and_vegetation, calculate_turbidity, analyze_biodiversity, train_cnn_model, adjust_image
from apps.alerts.models import Alert
from rest_framework import status


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
        ndvi = calculate_ndvi(image.image.path)
        turbidity = calculate_turbidity(image.image.path)
        bio_analysis = analyze_biodiversity(image.image.path)
        image.ndvi_score = ndvi
        image.turbidity = turbidity
        image.vegetation_percentage = det['vegetation_percentage']
        image.vegetation_area_m2 = det['vegetation_area_m2']
        image.water_percentage = det['water_percentage']
        image.water_area_m2 = det['water_area_m2']
        image.biodiversity_analysis = bio_analysis
        image.save()
        if image.water_percentage < 50 or (ndvi is not None and ndvi < 0.3):
            Alert.objects.create(
                ecosystem=image.ecosystem,
                type='sequia',
                details={'ndvi': ndvi, 'water_percentage': image.water_percentage},
                confidence=0.85
            )
    
    @action(detail=False, methods=['get'], url_path='by-ecosystem/(?P<ecosystem_id>\d+)')
    def by_ecosystem(self, request, ecosystem_id=None):
        try:
            images = Images.objects.filter(ecosystem_id=ecosystem_id)
            if not images.exists():
                return Response({'message': 'No se encontraron imágenes para este ecosistema'}, status=status.HTTP_404_NOT_FOUND)
            
            serializer = ImagesSerializer(images, many=True)
            return Response(serializer.data, status=status.HTTP_200_OK)
        except Exception as e:
            return Response({'error': f'Error al recuperar imágenes: {str(e)}'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        
    @action(detail=False, methods=['post'], url_path='adjust-and-analyze/(?P<image_id>\d+)')
    def adjust_and_analyze(self, request, image_id=None):
        try:
            image = Images.objects.get(id=image_id)
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

        # Aplicar ajustes
        adjusted_path = adjust_image(image.image.path, brightness, contrast, saturation, sepia, hue, opacity, blue_boost)
        if not adjusted_path:
            return Response({'error': 'Error al ajustar la imagen'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        # Recalcular métricas
        resolution = image.metadata.get('resolution_m_per_px', 0.1)
        det = detect_water_and_vegetation(adjusted_path, resolution)
        ndvi = calculate_ndvi(adjusted_path)
        turbidity = calculate_turbidity(adjusted_path)
        bio_analysis = analyze_biodiversity(adjusted_path)


        return Response({
            'message': 'Análisis recalculado con ajustes',
            'adjusted_metrics': {
                'ndvi_score': float(ndvi) if ndvi is not None else None,
                'turbidity': float(turbidity),
                'vegetation_percentage': float(det['vegetation_percentage']),
                'vegetation_area_m2': float(det['vegetation_area_m2']),
                'water_percentage': float(det['water_percentage']),
                'water_area_m2': float(det['water_area_m2']),
                'biodiversity_analysis': bio_analysis
            }
        }, status=status.HTTP_200_OK)
        
    @action(detail=False, methods=['post'], url_path='upload-multiple')
    def upload_multiple(self, request):
        ecosystem_id = request.data.get('ecosystem_id') 
        ecosystem_name = request.data.get('ecosystem_name', 'Cuerpo de agua nuevo')
        images = request.FILES.getlist('images')
        descriptions = request.data.get('descriptions', [])

        if not images:
            return Response({'error': 'Se requieren imágenes'}, status=status.HTTP_400_BAD_REQUEST)
        
        if ecosystem_id:
            try:
                ecosystem = Ecosystem.objects.get(id=ecosystem_id)
            except Ecosystem.DoesNotExist:
                return Response({'error': 'Ecosistema no encontrado'}, status=status.HTTP_404_NOT_FOUND)
        else:
            # Crear un nuevo ecosistema si no se proporciona ID
            ecosystem = Ecosystem.objects.create(
                name=ecosystem_name,
                location=None  # agregar logica para ubicación posteriormente
            )

        responses = []
        for i, image in enumerate(images):
            description = descriptions[i] if i < len(descriptions) else ''
            drone_image = Images(
                ecosystem=ecosystem,
                image=image,
                description=description,
                metadata={'resolution_m_per_px': 0.1}  # Ajustar según el drone
            )
            drone_image.save()
            resolution = drone_image.metadata.get('resolution_m_per_px', 0.1)
            det = detect_water_and_vegetation(drone_image.image.path, resolution)
            ndvi = calculate_ndvi(drone_image.image.path)
            turbidity = calculate_turbidity(drone_image.image.path)
            bio_analysis = analyze_biodiversity(drone_image.image.path)
            drone_image.ndvi_score = ndvi
            drone_image.turbidity = turbidity
            drone_image.vegetation_percentage = det['vegetation_percentage']
            drone_image.vegetation_area_m2 = det['vegetation_area_m2']
            drone_image.water_percentage = det['water_percentage']
            drone_image.water_area_m2 = det['water_area_m2']
            drone_image.biodiversity_analysis = bio_analysis
            drone_image.save()
            if drone_image.water_percentage < 50 or (ndvi is not None and ndvi < 0.3):
                Alert.objects.create(
                    ecosystem=drone_image.ecosystem,
                    type='sequia',
                    details={'ndvi': ndvi, 'water_percentage': drone_image.water_percentage},
                    confidence=0.85
                )
            responses.append({
                'id': drone_image.id,
                'image': drone_image.image.url,
                'description': description,
                'water_percentage': drone_image.water_percentage,
                'water_area_m2': drone_image.water_area_m2,
                'vegetation_percentage': drone_image.vegetation_percentage,
                'vegetation_area_m2': drone_image.vegetation_area_m2,
                'turbidity': drone_image.turbidity
            })

        return Response({
            'message': 'Imágenes subidas exitosamente',
            'ecosystem_id': ecosystem.id,  # Devuelve el ID del ecosistema usado/creado para referencia
            'ecosystem_name': ecosystem.name,
            'images': responses
        }, status=status.HTTP_201_CREATED)

    @action(detail=False, methods=['post'], url_path='train-model')
    def train_model(self, request):
        model = train_cnn_model()
        if model:
            return Response({'message': 'Modelo entrenado exitosamente'}, status=status.HTTP_200_OK)
        return Response({'error': 'Fallo al entrenar el modelo'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
