import pandas as pd
from io import StringIO
import pickle
import torch
from contextlib import asynccontextmanager
from fastapi import FastAPI, APIRouter
from app.core.config import BTC_Config
from app.schema.btc import BTCModel
from app.api.v1.controller.predict_btc import predict_futures, RNN_Model

def load_scaler(path: str):
    with open(path, 'rb') as f:
        scaler = pickle.load(f)
    return scaler

def load_model(config: BTC_Config):
    model = RNN_Model(
        input_size=config.input_size,
        hidden_size=config.hidden_size,
        output_size=config.output_size,
        num_layers=config.num_layers
    )
    model_state_dict = torch.load(config.model_path, weights_only=True)
    model.load_state_dict(model_state_dict)
    return model

btc_models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    config = BTC_Config()
    btc_models["config"] = config
    btc_models["features_scaler"] = load_scaler(config.features_scaler_path)
    btc_models["target_scaler"] = load_scaler(config.target_scaler_path)
    btc_models["model"] = load_model(config)
    yield
    btc_models.clear()

router = APIRouter(lifespan=lifespan)


@router.post("/predict",
             tags=["BTC"], 
             description="Predict the future price of Bitcoin")
async def predict_btc(input_data: BTCModel):
    df_data = pd.read_json(StringIO(input_data.json_str), orient='records')
    next_days = input_data.next_days
    predictions = await predict_futures(btc_models, df_data, next_days=next_days)

    if isinstance(predictions, Exception):
        return {"error": predictions}
    
    return {"predictions": predictions}

