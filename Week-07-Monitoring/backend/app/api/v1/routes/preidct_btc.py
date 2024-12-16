import os
import pandas as pd
from io import StringIO
import pickle
import yaml
import torch
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI, APIRouter, HTTPException, status
from app.core.config import BTC_Config
from app.schema.btc import BTCModel
from app.api.v1.controller.predict_btc import predict_futures, RNN_Model
import mlflow
from mlflow import MlflowClient
from dotenv import load_dotenv
load_dotenv()

LOCAL_ARTIFACTS = Path("/DATA/artifacts")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")

def load_scaler(path: str):
    with open(f"./scalers/{path}", 'rb') as f:
        scaler = pickle.load(f)
    return scaler


def load_model(config: dict, data_version: str, run_id: str):
    try:
        model = RNN_Model(
            input_size=5,
            hidden_size=config["hidden_size"],
            output_size=1,
            num_layers=config["num_layers"]
        )
        model_path = f"{LOCAL_ARTIFACTS}/{data_version}/{run_id}/model.pth"
        model_state_dict = torch.load(model_path, weights_only=True, map_location='cpu')
        model.load_state_dict(model_state_dict)
        return model
    except Exception as e:
        return e


def load_config(data_version: str):
    print(f"Loading model and artifacts")
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    artifacts = {}
    try:
        deploy_config = BTC_Config()

        client = MlflowClient()
        alias_mv = client.get_model_version_by_alias(deploy_config.registered_name, 
                                                     deploy_config.model_alias)
        
        print("Downloading model artifacts with run_id:", alias_mv.run_id)

        scalers_artifact_uri = f"runs:/{alias_mv.run_id}/scalers"
        mlflow.artifacts.download_artifacts(scalers_artifact_uri, dst_path=".")

        config_artifact_uri = f"runs:/{alias_mv.run_id}/config"
        mlflow.artifacts.download_artifacts(config_artifact_uri, dst_path=".")

        with open("./config/btc_config.yaml", 'rb') as f:
            config = yaml.safe_load(f)

        artifacts["config"] = config
        artifacts["deploy_config"] = deploy_config
        artifacts["features_scaler"] = load_scaler(deploy_config.features_scaler_path)
        artifacts["target_scaler"] = load_scaler(deploy_config.target_scaler_path)

        model_uri = f"models:/{deploy_config.registered_name}@production"
        # TODO: Load the model from the model_uri
        # artifacts["model"] = mlflow.pyfunc.load_model(model_uri)

        artifacts["model"] = load_model(config["model_config"], data_version, alias_mv.run_id)


    except Exception as e:
        print(f"Error loading model model artifacts")
        print(e)
    return artifacts

model_artifacts = {}
@asynccontextmanager
async def lifespan(app: FastAPI):
    model_artifacts["v1"] = load_config("v0.1")
    # model_artifacts["v2"] = load_config("v0.2")
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

