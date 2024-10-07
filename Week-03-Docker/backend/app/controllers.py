import easyocr
import numpy as np
from PIL import Image
from app.schemas import OCR_Output

def build_extract_ocr(languages):
    EASY_OCR = easyocr.Reader(languages, gpu=False)
    return EASY_OCR

def extract_ocr(image_path, languages: list = ["en"]) -> list[OCR_Output]:
    try:
        EASY_OCR = build_extract_ocr(languages)
    except:
        raise Exception("Error during OCR initialization")
    
    try:
        pil_img = Image.open(image_path).convert("RGB")
        array_img = np.asarray(pil_img)
        ocr_result = EASY_OCR.readtext(array_img, slope_ths=0.5,
                                    height_ths=1.0, width_ths=1.5)
        
        return_data = []
        for line in ocr_result:
            bbox = [[int(x) for x in sublist] for sublist in line[0]]
            text = str(line[1])
            score = float(line[2])
            return_data.append(
                OCR_Output(
                    bbox=bbox,
                    text=text,
                    score=score
                )
            )
        return return_data
    except:
        raise Exception("Error during OCR processing")
