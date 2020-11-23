from django.urls import path
from .views import image_process_ajax, process_img

urlpatterns = [
    path('', image_process_ajax, name='imageupload'),
    path('image_asiba', process_img, name="process_img")
]
