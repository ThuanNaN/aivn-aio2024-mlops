import pandas as pd
from io import StringIO
import pickle
import torch
from contextlib import asynccontextmanager
from fastapi import FastAPI, APIRouter, HTTPException, status
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
    model_state_dict = torch.load(config.model_path, weights_only=True, map_location='cpu')
    model.load_state_dict(model_state_dict)
    return model

def load_config(version: str):
    print(f"Loading model artifacts for version {version}")
    artifacts = {}
    try:
        config = BTC_Config(version=version)
        artifacts["config"] = config
        artifacts["features_scaler"] = load_scaler(config.features_scaler_path)
        artifacts["target_scaler"] = load_scaler(config.target_scaler_path)
        artifacts["model"] = load_model(config)
    except Exception as e:
        print(f"Error loading model artifacts for version {version}")
        print(e)
    return artifacts

model_artifacts = {}
@asynccontextmanager
async def lifespan(app: FastAPI):
    model_artifacts["v1"] = load_config("v0.1")
    model_artifacts["v2"] = load_config("v0.2")
    yield
    model_artifacts.clear()

router = APIRouter(lifespan=lifespan)


@router.post("/predict",
             description="Predict the future price of Bitcoin")
async def predict_btc(input_data: BTCModel):
    next_days = input_data.next_days
    model_version = input_data.model_version
    try:
        df_data = pd.read_json(StringIO(input_data.json_str), orient='records')
    except:
        return HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid JSON data"
        )
    btc_models = model_artifacts.get(model_version, None)
    if btc_models is None:
        return HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid model version"
        )
    predictions = await predict_futures(btc_models, df_data, next_days=next_days)
    if isinstance(predictions, Exception):
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(predictions)
        )
    return {"predictions": predictions}

