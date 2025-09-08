from celery import shared_task
from .utils import calculate_ndvi, detect_dry_water_bodies, analyze_biodiversity, train_cnn_model
from .models import Images
from apps.alerts.models import Alert

@shared_task
def process_drone_image(image_id):
    try:
        image = Images.objects.get(id=image_id)
        ndvi = calculate_ndvi(image.image.path)
        water_det = detect_dry_water_bodies(image.image.path)
        bio_analysis = analyze_biodiversity(image.image.path)
        image.ndvi_score = ndvi
        image.water_detection = water_det
        image.biodiversity_analysis = bio_analysis
        image.save()
        if water_det.get('water_percentage', 0) < 50 or (ndvi is not None and ndvi < 0.3):
            Alert.objects.create(
                ecosystem=image.ecosystem,
                type='sequia',
                details={'ndvi': ndvi, 'water_percentage': water_det.get('water_percentage', 0)},
                confidence=0.85
            )
    except Exception as e:
        print(f"Error en process_drone_image: {e}")

@shared_task
def train_ml_models(dataset_images, dataset_labels):
    try:
        model = train_cnn_model(dataset_images, dataset_labels)
        if model:
            print("Modelo CNN entrenado exitosamente")
        else:
            print("Fallo al entrenar el modelo CNN")
    except Exception as e:
        print(f"Error en train_ml_models: {e}")