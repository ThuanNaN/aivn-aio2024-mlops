from contextlib import asynccontextmanager
from fastapi import FastAPI, APIRouter


gold_models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    config = None
    yield
    gold_models.clear()

router = APIRouter(lifespan=lifespan)

@router.post("/predict",
             description="Predict the future price of Bitcoin")
async def predict_btc(input_data: str):
    
    return {"predictions": "Not implemented yet"}   

