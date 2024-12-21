import gradio as gr
import pandas as pd
from yolo_func import detect_objects
from btc_func import plot_csv


with gr.Blocks() as demo:
    gr.Markdown("# 📈 AIVN 2024 MLOPS")
    
    with gr.Tab("BTC Price"):
        with gr.Row():
            csv_input = gr.File(label="Upload your CSV file", file_types=[".csv"])
        with gr.Row():
            with gr.Column():
                input_date_from = gr.Number(0, label="Input From Date")
                input_date_to = gr.Number(14, label="Input To Date")
                next_days = gr.Number(7, label="Next Days")
            with gr.Column():
                upload_button = gr.Button("Plot")
                show_v1_pred = gr.Checkbox(label="Show V1 Model Predictions")
                show_v2_pred = gr.Checkbox(label="Show V2 Model Predictions")
        with gr.Row():
            visualize_date = gr.Slider(label="Visualize Date", step=1)
        with gr.Row():
            line_plot = gr.Plot()
        with gr.Row():
            output_text = gr.Textbox(label="Output Message", lines=2)

        def update_slider(file):
            if file is None:
                return gr.update(maximum=0)
            df = pd.read_csv(file.name)
            return gr.update(maximum=len(df))

        csv_input.change(
            fn=update_slider,
            inputs=[csv_input],
            outputs=[visualize_date]
        )
        upload_button.click(
            fn=plot_csv,
            inputs=[csv_input, visualize_date, input_date_from, input_date_to, next_days, show_v1_pred, show_v2_pred],
            outputs=[line_plot, output_text]
        )
        visualize_date.change(
            fn=plot_csv,
            inputs=[csv_input, visualize_date, input_date_from, input_date_to, next_days, show_v1_pred, show_v2_pred],
            outputs=[line_plot, output_text]
        )
        show_v1_pred.change(
            fn=plot_csv,
            inputs=[csv_input, visualize_date, input_date_from, input_date_to, next_days, show_v1_pred, show_v2_pred],
            outputs=[line_plot, output_text]
        )
        show_v2_pred.change(
            fn=plot_csv,
            inputs=[csv_input, visualize_date, input_date_from, input_date_to, next_days, show_v1_pred, show_v2_pred],
            outputs=[line_plot, output_text]
        )

    with gr.Tab("Detections"):
        gr.Markdown("Object detection using YOLO11")
        with gr.Row():
            input_image = gr.Image(type="pil", label="Upload Image")
        with gr.Row():
            detect_btn = gr.Button("Detect Objects")

        with gr.Row():
            annotated_image = gr.Image(label="Annotated Image")
        with gr.Row():
            detection_results = gr.Textbox(label="Detection Results")
    
        # Bind detection function
        detect_btn.click(
            fn=detect_objects, 
            inputs=input_image, 
            outputs=[annotated_image, detection_results]
        )

demo.launch(server_name="0.0.0.0", server_port=7860)
