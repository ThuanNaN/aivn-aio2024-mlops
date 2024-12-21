import os
import pandas as pd
import plotly.graph_objects as go
from utils import clean_data
import requests
from dotenv import load_dotenv
load_dotenv()

API_URL = os.getenv("API_URL")
HEADERS = {'accept': 'application/json'}

def predict_btc_df(input_df: pd.DataFrame, next_days: int=1, model_version: str="v1")-> dict:
    url = f"{API_URL}/v1/btc/predict"

    json_pretty = input_df.to_json(orient='records')
    json_data = {
        "json_str": json_pretty,
        "next_days": next_days,
        "model_version": model_version
    }
    response = requests.post(url, headers=HEADERS, json=json_data)
    if response.status_code == 200:
        response_json = response.json() 
        return response_json
    else:
        return {
            "error": f"Error: {response.status_code}, {response.text}"
        }

def plot_csv(file, 
             visualize_date: int = 7, 
             input_date_from: int = 0,
             input_date_to: int = 7,
             next_days: int = 1,
             show_v1_pred=False,
             show_v2_pred=False):
    if file is None:
        return None, "Please upload a CSV file."
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
            height=720,
            width=1080,
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