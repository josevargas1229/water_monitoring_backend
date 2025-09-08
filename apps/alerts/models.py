from django.db import models
from apps.monitoring.models import Ecosystem

class Alert(models.Model):
    ecosystem = models.ForeignKey(Ecosystem, on_delete=models.CASCADE)
    alert_date = models.DateTimeField(auto_now_add=True)
    type = models.CharField(max_length=50)
    details = models.JSONField()
    confidence = models.FloatField()