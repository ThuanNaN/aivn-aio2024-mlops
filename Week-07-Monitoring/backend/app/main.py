from fastapi import FastAPI
from app.api.v1 import router as v1_router

from prometheus_fastapi_instrumentator import Instrumentator

app = FastAPI()
app.include_router(v1_router, prefix="/v1")

instrumentator = Instrumentator().instrument(app).expose(app)
