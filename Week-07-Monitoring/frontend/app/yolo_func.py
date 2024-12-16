import gradio as gr
import requests
from PIL import Image
import numpy as np
import os
from utils import draw_bounding_boxes
from dotenv import load_dotenv
load_dotenv()

API_URL = os.getenv("API_URL")

def detect_objects(image):
    """
    Send image to YOLO detection API and process results
    
    :param image: Input image (numpy array or PIL Image)
    :return: Annotated image, detection results text
    """
    # Convert numpy array to PIL Image if needed
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    
    # Prepare image for upload
    image_bytes = None
    with gr.Blocks():
        with gr.Column():
            # Create a temporary file to upload
            temp_filename = "temp_upload.jpg"
            image.save(temp_filename)
            
            # Prepare file for upload
            files = {'file': open(temp_filename, 'rb')}
            
            try:
                # Send request to API
                url = f"{API_URL}/v1/yolo/detect"
                response = requests.post(url, files=files)
                response.raise_for_status()  # Raise an exception for bad responses
                
                # Parse response
                result = response.json()
                
                pil_image = Image.open(temp_filename).convert("RGB")
                annotated_image = draw_bounding_boxes(pil_image, result.get('detections', []))

                # Prepare detection results text
                results_text = "Detected Objects:\n"
                for detection in result.get('detections', []):
                    results_text += (
                        f"- {detection['class']} "
                        f"(Confidence: {detection['confidence']:.2f})\n"
                    )
                
                # Clean up temporary file
                os.remove(temp_filename)
                
                return annotated_image, results_text
            
            except requests.RequestException as e:
                return image, f"Error detecting objects: {str(e)}"
            finally:
                # Ensure file is closed
                if 'files' in locals():
                    files['file'].close()