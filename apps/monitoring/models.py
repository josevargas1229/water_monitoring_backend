from django.contrib.gis.db import models

class Ecosystem(models.Model):
    name = models.CharField(max_length=100)
    location = models.PolygonField(null=True)
    created_at = models.DateTimeField(auto_now_add=True)

class Images(models.Model):
    ecosystem = models.ForeignKey(Ecosystem, on_delete=models.CASCADE, related_name='images')
    parent_image = models.ForeignKey('self', on_delete=models.SET_NULL, null=True, blank=True, related_name='adjusted_images')
    is_adjusted = models.BooleanField(default=False)
    image = models.ImageField(upload_to='drones/')
    description = models.CharField(max_length=500, blank=True, null=True)
    metadata = models.JSONField(default=dict)
    capture_date = models.DateTimeField()
    created_at = models.DateTimeField(auto_now_add=True)
    ndvi_score = models.FloatField(null=True)
    water_detection = models.JSONField(null=True)
    biodiversity_analysis = models.JSONField(null=True)
    turbidity = models.FloatField(null=True)
    vegetation_percentage = models.FloatField(null=True)
    water_percentage = models.FloatField(null=True)
    water_area_m2 = models.FloatField(null=True)
    vegetation_area_m2 = models.FloatField(null=True)

    class Meta:
        ordering = ['-capture_date']

    def __str__(self):
        status = "Adjusted" if self.is_adjusted else "Original"
        return f"{status} for {self.ecosystem.name} - {self.capture_date}"