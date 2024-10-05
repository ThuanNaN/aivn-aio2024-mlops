import numpy as np
import gradio as gr
from PIL import Image, ImageDraw, ImageFont
import easyocr

def plot_bbox(easy_ocr_result, image):
    boxes = [line[0] for line in easy_ocr_result]
    texts = [line[1] for line in easy_ocr_result]
    scores = [line[2] for line in easy_ocr_result]
    image = Image.fromarray(image)

    draw = ImageDraw.Draw(image)
    rectangle_width = 2
    font_size = 20
    font = ImageFont.truetype("./JetBrainsMono-Medium.ttf", size=font_size)  

    for box, text in zip(boxes, texts):
        top_left     = (int(box[0][0]), int(box[0][1]))
        bottom_right = (int(box[2][0]), int(box[2][1]))

        # Check if x1 < x0
        if top_left[0] > bottom_right[0]:
            top_left, bottom_right = (bottom_right[0], top_left[1]), (top_left[0], bottom_right[1])
        
        # Check if y1 < y0
        if top_left[1] > bottom_right[1]:
            top_left, bottom_right = (top_left[0], bottom_right[1]), (bottom_right[0], top_left[1])
        
        text_pos = (top_left[0], top_left[1]-font_size-rectangle_width)
        draw.rectangle([top_left, bottom_right], outline="green", width=2)
        draw.text(text_pos, text, fill="blue", font=font)

    return image, boxes, texts, scores


def create_reader(languages: list):
    return easyocr.Reader(languages, gpu=False)

def extract_ocr(input_image, languages):
    EASY_OCR = create_reader(languages)
    ocr_result = EASY_OCR.readtext(input_image, slope_ths=0.5,
                                   height_ths=1.0, width_ths=1.5)
    image = plot_bbox(ocr_result, input_image)[0]
    return np.array(image)


choices = ["vi", "en", "ko", "ch_sim"]

demo = gr.Interface(
    fn=extract_ocr,
    inputs=["image", 
            gr.CheckboxGroup(choices=choices, label="Languages")],
    outputs=["image"],
)
demo.launch()