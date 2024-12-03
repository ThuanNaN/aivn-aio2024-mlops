import gradio as gr
import pandas as pd
import plotly.graph_objects as go
from utils import clean_data
from api import predict_btc_df

def plot_csv(file, 
             visualize_date: int = 7, 
             input_date_from: int = 0,
             input_date_to: int = 7,
             next_days: int = 1,
             show_v1_pred=False,
             show_v2_pred=False):
    if file is None:
        return None, "Please upload a CSV file.", 0
    try:
        df = pd.read_csv(file.name)
        input_df = df.copy()
        visualize_df = df.copy()

        if show_v1_pred:
            input_df = input_df.iloc[input_date_from:input_date_to]
            v1_predictions = predict_btc_df(input_df, next_days, "v1")
        
        if show_v2_pred:
            input_df = input_df.iloc[input_date_from:input_date_to]
            v2_predictions = predict_btc_df(input_df, next_days, "v2")

        visualize_df = visualize_df.iloc[0:visualize_date]
        df_cleaned = clean_data(visualize_df)
        
        error = ""
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df_cleaned['Date'],
            y=df_cleaned['Price'],
            mode='lines+markers',
            name='BTC Price'
        ))
        if show_v1_pred:
            if v1_predictions.get("error"):
                error += v1_predictions["error"] + "\n"
                v1_predictions = {"predictions": []}
            fig.add_trace(go.Scatter(
                x=df_cleaned['Date'][input_date_to:input_date_to+next_days],
                y=v1_predictions["predictions"],
                mode='lines+markers',
                name='V1 Predictions'
            ))
        if show_v2_pred:
            if v2_predictions.get("error"):
                error += v2_predictions["error"] + "\n"
                v2_predictions = {"predictions": []}
            fig.add_trace(go.Scatter(
                x=df_cleaned['Date'][input_date_to:input_date_to+next_days],
                y=v2_predictions["predictions"],
                mode='lines+markers',
                name='V2 Predictions'
            ))
        fig.update_layout(
            title="BTC Price Over Time",
            xaxis_title="Time",
            yaxis_title="Price (USD)",
            template="plotly_white",
            height=700,
            width=1450,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            )
        )
        if error != "":
            return fig, error
        return fig, "Plot created successfully!"
    
    except Exception as e:
        return None, f"Error processing the file: {str(e)}"

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

    with gr.Tab("Gold Price"):
        gr.Markdown("Coming soon...")

demo.launch(server_name="0.0.0.0", server_port=3000, debug=True)
