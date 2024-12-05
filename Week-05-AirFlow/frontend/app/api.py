import requests
import pandas as pd
import os
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

