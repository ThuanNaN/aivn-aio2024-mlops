from fastapi import FastAPI
from app.api.v1 import router as v1_router


app = FastAPI()
app.include_router(v1_router, prefix="/v1")

