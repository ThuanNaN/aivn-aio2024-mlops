import os
from PIL import Image

def log_image(image_path, image_name, log_dir: str = "./app/logs/images"):
    img = Image.open(image_path).convert("RGB")
    save_name = os.path.basename(image_name)
    save_path = os.path.join(log_dir, save_name)
    img.save(save_path)