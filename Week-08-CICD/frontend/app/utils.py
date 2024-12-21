import pandas as pd
import numpy as np
import cv2

def clean_data(df: pd.DataFrame):
    df.rename(columns={'Change %': 'Change'}, inplace=True)
    df.rename(columns={'Vol.': 'Volume'}, inplace=True)

    df.sort_values('Date', inplace=True)
    df['Volume'] = (
        df['Volume']
        .str.replace('K', 'e3')
        .str.replace('M', 'e6')
        .str.replace('B', 'e9')
        .astype(float)
    )
    columns_to_clean = ['Price', 'Open', 'High', 'Low']
    for col in columns_to_clean:
        df[col] = df[col].str.replace(',', '').astype(float)

    df['Change'] = df['Change'].str.replace('%', '').astype(float) / 100
    return df

def draw_bounding_boxes(image, detections):
    image = np.array(image)
    for detection in detections:
        # Extract bounding box coordinates and class
        x1 = int(detection["bounding_box"]["x1"])
        y1 = int(detection["bounding_box"]["y1"])
        x2 = int(detection["bounding_box"]["x2"])
        y2 = int(detection["bounding_box"]["y2"])
        label = detection["class"]
        confidence = detection["confidence"]
        
        # Draw rectangle
        color = (0, 255, 0) if confidence > 0.5 else (0, 0, 255) 
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        
        # Add label and confidence
        text = f"{label} ({confidence:.2f})"
        cv2.putText(image, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return image