from fastapi import APIRouter, File, UploadFile, status, HTTPException
from app.controllers import extract_ocr
from app.schemas import OCR_Response
from app.utils import log_image


router = APIRouter()

@router.post("/ocr/predict")
async def predict_ocr(file_upload: UploadFile = File(...)):
    file_path = file_upload.file
    file_name = file_upload.filename
    log_image(file_path, file_name)
    ocr_result = extract_ocr(file_path)
    return OCR_Response(
        data=ocr_result,
        status_code=status.HTTP_200_OK
    )