from rest_framework import serializers
from .models import Ecosystem, Images

class EcosystemSerializer(serializers.ModelSerializer):
    class Meta:
        model = Ecosystem
        fields = '__all__'

class ImagesSerializer(serializers.ModelSerializer):
    description = serializers.CharField(max_length=500, required=False, allow_blank=True)
    adjusted_images = serializers.SerializerMethodField()

    class Meta:
        model = Images
        fields = [
            'id',
            'ecosystem',
            'image',
            'description',
            'metadata',
            'capture_date',
            'vegetation_percentage',
            'vegetation_area_m2',
            'water_percentage', 
            'water_area_m2',
            'is_adjusted',
            'parent_image',
            'adjusted_images',
        ]

    def get_adjusted_images(self, obj):
        # Solo incluir adjusted_images si include_adjusted=True y si la imagen es original
        if self.context.get('include_adjusted', False) and not obj.is_adjusted:
            adjusted = obj.adjusted_images.all()  # Usa el related_name del modelo
            return ImagesSerializer(adjusted, many=True, context=self.context).data
        return []