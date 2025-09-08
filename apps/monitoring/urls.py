from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import EcosystemViewSet, ImagesViewSet

router = DefaultRouter()
router.register(r'ecosystems', EcosystemViewSet)
router.register(r'images', ImagesViewSet)

urlpatterns = [
    path('', include(router.urls)),
]