from rest_framework import serializers
from .models import Ecosystem, Images

class EcosystemSerializer(serializers.ModelSerializer):
    class Meta:
        model = Ecosystem
        fields = '__all__'

class ImagesSerializer(serializers.ModelSerializer):
    description = serializers.CharField(max_length=500, required=False, allow_blank=True)

    class Meta:
        model = Images
        fields = ['id', 'ecosystem', 'image', 'description', 'metadata', 'capture_date', 'ndvi_score', 'water_detection', 'biodiversity_analysis']