import io
import base64
import os
from typing import List, Optional
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from pydantic import BaseModel
import uvicorn
from ultralytics import YOLO
import cv2

# Initialize FastAPI app
app = FastAPI(
    title="YOLOv8 Object Detection API",
    description="API for detecting objects in images using YOLOv8",
    version="1.0.0"
)

# Load YOLOv8 model from environment variable or use default
model_path = os.getenv("MODEL_PATH", "yolov8n.pt")
model = YOLO(model_path)

class DetectionResult(BaseModel):
    class_id: int
    class_name: str
    confidence: float
    x_min: float
    y_min: float
    x_max: float
    y_max: float

class DetectionResponse(BaseModel):
    results: List[DetectionResult]
    base64_img: Optional[str] = None

@app.get("/")
def read_root():
    return {"message": "Welcome to YOLOv8 Object Detection API"}


@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.post("/detect/", response_model=DetectionResponse)
async def detect_objects(
    file: UploadFile = File(...),
    conf_threshold: float = Form(0.25),
    iou_threshold: float = Form(0.45),
    show_boxes: bool = Form(True),
    return_image: bool = Form(False)
):
    # Validate file type
    if file.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
        raise HTTPException(status_code=400, detail="Only JPEG and PNG images are supported")
    
    # Read the image file
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))
    
    # Convert PIL Image to NumPy array (YOLOv8 expects RGB format)
    image_np = np.array(image)
    
    # Run YOLOv8 inference
    results = model(image_np, conf=conf_threshold, iou=iou_threshold)[0]
    
    # Process results
    detection_results = []
    for box in results.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        class_id = int(box.cls[0].item())
        confidence = float(box.conf[0].item())
        class_name = results.names[class_id]
        
        detection_results.append(
            DetectionResult(
                class_id=class_id,
                class_name=class_name,
                confidence=confidence,
                x_min=x1,
                y_min=y1,
                x_max=x2,
                y_max=y2
            )
        )
    
    response = {"results": detection_results}
    
    # Optionally render boxes on the image and return it
    if return_image:
        if show_boxes:
            # Use the plotted image with boxes from YOLOv8
            result_image = results.plot()
        else:
            # Just use the original image
            result_image = image_np
            
        # Convert numpy array to base64 string
        _, buffer = cv2.imencode('.jpg', result_image)
        base64_img = base64.b64encode(buffer).decode('utf-8')
        response["base64_img"] = base64_img

    return response

@app.get("/models")
def get_available_models():
    return {
        "available_models": [
            {"name": "YOLOv8n", "size": "small", "path": "yolov8n.pt"},
            {"name": "YOLOv8s", "size": "medium", "path": "yolov8s.pt"},
            {"name": "YOLOv8m", "size": "large", "path": "yolov8m.pt"},
            {"name": "YOLOv8l", "size": "xlarge", "path": "yolov8l.pt"},
            {"name": "YOLOv8x", "size": "xxlarge", "path": "yolov8x.pt"}
        ]
    }

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
