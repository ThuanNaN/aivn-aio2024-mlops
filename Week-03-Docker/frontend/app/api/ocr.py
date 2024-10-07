import os
import requests
import io
from PIL import Image
from utils import plot_bbox

BACKEND_URL = os.getenv("BACKEND_URL", "http://0.0.0.0:8000")

def ocr_api(image_path: str):
    image = Image.open(image_path)
    img_name = image_path.split("/")[-1]
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='JPEG')
    img_byte_arr = img_byte_arr.getvalue()

    url = f"{BACKEND_URL}/ocr/predict"
    files = {'file_upload': (img_name, img_byte_arr, 'image/jpeg')}
    headers = {'accept': 'application/json'}

    response = requests.post(url, headers=headers, files=files)

    if response.status_code == 200:
        json_results = response.json()['data']
        image_with_bbox = plot_bbox(json_results, image)[0]
        return "Suscess", image_with_bbox
    else:
        return "Error: API request failed.", None
