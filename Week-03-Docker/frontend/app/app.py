import os
import gradio as gr
from api.ocr import ocr_api

demo = gr.Interface(
    fn = ocr_api,
    inputs = gr.Image(type="filepath", label="Input Image"),
    outputs=[gr.Text(label="Status"), gr.Image(label="Output Image")],
    title = "OCR",
    description = "This is a OCR application",
)

demo.launch(
    server_name="0.0.0.0",
    server_port=3000,
    share=False
)
