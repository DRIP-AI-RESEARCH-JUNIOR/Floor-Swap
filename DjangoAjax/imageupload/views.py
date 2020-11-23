from django.shortcuts import render
from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
from PIL import Image
from base64 import decodestring
import io, os
import base64
import numpy as np
import cv2
from imageupload.indoor import tiles_generator, pred_evaluator, swapperTile
from django.contrib.staticfiles.storage import staticfiles_storage
# Create your views here.
def image_process_ajax(request):
    return render(request, 'imageupload/image.html')

def url_Generator(final):
    out = Image.fromarray(final)
    b = io.BytesIO()
    out.save(b, format='PNG')
    b = b.getvalue()
    b64_im = base64.b64encode(b)
    image_url = u'data:img/jpeg;base64,' + b64_im.decode('utf-8')
    return image_url

@csrf_exempt
def process_img(request):
    global predicted_image, original_image
    if (request.POST.get('icon')):
        file_path = staticfiles_storage.path("imageupload/img/Tile_1.jpg")
        img_data = bytes(request.POST['icon'], 'ascii')
        z = img_data[img_data.find(b'/9'):]
        im = Image.open(io.BytesIO(base64.b64decode(z))).convert('RGB')
        pred, img_original = pred_evaluator(im)
        predicted_image = pred
        original_image = img_original
        filled_tiles = tiles_generator(file_path, pred)
        final = swapperTile(pred, img_original, filled_tiles)
        image_url = url_Generator(final)
        return JsonResponse({'status':'success', 'image':image_url})
    if (request.POST.get('option')):
        option = int(request.POST['option'])
        file_path = staticfiles_storage.path("imageupload/img/Tile_"+str(option)+".jpg")
        filled_tiles = tiles_generator(file_path, predicted_image)
        final = swapperTile(predicted_image, original_image, filled_tiles)
        image_url = url_Generator(final)
        return JsonResponse({'status': 'success', 'image': image_url})