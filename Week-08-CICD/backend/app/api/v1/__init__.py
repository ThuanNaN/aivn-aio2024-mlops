from fastapi import APIRouter
from app.api.v1.routes.preidct_btc import router as btc_router
from app.api.v1.routes.yolo import router as yolo_router

router = APIRouter()

# Health Check
@router.get("/health")
async def health_check():
    return {"status": "ok"}

# Include the v1 router
router.include_router(btc_router, prefix="/btc", tags=["BTC"])
router.include_router(yolo_router, prefix="/yolo", tags=["Detect"])
