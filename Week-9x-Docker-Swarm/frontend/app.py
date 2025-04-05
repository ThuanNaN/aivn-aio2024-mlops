import os
import io
import requests
import gradio as gr
from PIL import Image, ImageDraw
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get backend URL from environment variable or use default
BACKEND_URL = os.getenv("BACKEND_URL", "http://yolov8-api:8000")

def process_image(image):
    """
    Sends the image to the YOLOv8 API and visualizes the results.
    """
    # Save image to bytes
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)
    
    # Send to API
    try:
        files = {'file': ('image.png', img_byte_arr, 'image/png')}
        response = requests.post(f"{BACKEND_URL}/detect/", files=files)
        
        if response.status_code == 200:
            res = response.json()
            results = res.get('results', [])
            
            # Create a copy of the original image for drawing
            draw_image = image.copy()
            draw = ImageDraw.Draw(draw_image)
            
            # Draw boxes on the image
            for detection in results:
                # Extract coordinates in the new format
                x_min = detection['x_min']
                y_min = detection['y_min']
                x_max = detection['x_max']
                y_max = detection['y_max']
                
                # Create label with class name and confidence
                label = f"{detection['class_name']} {detection['confidence']:.2f}"
                
                # Draw bounding box
                draw.rectangle([x_min, y_min, x_max, y_max], outline="red", width=2)
                
                # Calculate text size using a simpler approach
                font_size = 12
                text_width = len(label) * font_size * 0.6  # Approximate width
                text_height = font_size + 4
                
                # Draw label background
                draw.rectangle([x_min, y_min, x_min + text_width, y_min + text_height], fill="red")
                
                # Draw label
                draw.text((x_min, y_min), label, fill="white")
            
            # Summary message
            num_detections = len(results)
            summary = f"Detected {num_detections} objects"
            
            return draw_image, summary, json.dumps(results, indent=2)
        else:
            return image, f"Error: {response.status_code}", response.text
    except Exception as e:
        return image, f"Error: {str(e)}", ""

# Create Gradio interface
with gr.Blocks(title="YOLOv8 Object Detection") as app:
    gr.Markdown("# YOLOv8 Object Detection")
    gr.Markdown("Upload an image to detect objects using YOLOv8")
    
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(type="pil", label="Input Image")
            submit_btn = gr.Button("Detect Objects")
        
        with gr.Column():
            output_image = gr.Image(type="pil", label="Detection Results")
            output_text = gr.Textbox(label="Summary")
            output_json = gr.JSON(label="Detailed Results")
    
    submit_btn.click(
        fn=process_image,
        inputs=[input_image],
        outputs=[output_image, output_text, output_json]
    )
    
    gr.Markdown("## How to use")
    gr.Markdown("1. Upload an image using the input panel")
    gr.Markdown("2. Click 'Detect Objects'")
    gr.Markdown("3. View the results with bounding boxes on the right")


# Launch the app
if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=7860)
