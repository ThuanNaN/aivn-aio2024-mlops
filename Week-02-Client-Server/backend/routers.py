from fastapi import APIRouter, File, UploadFile, status, HTTPException
from controllers import extract_ocr
from schemas import OCR_Response


router = APIRouter()

@router.post("/ocr/predict")
async def predict_ocr(file_upload: UploadFile = File(...)):
    ocr_result = extract_ocr(file_upload.file)
    return OCR_Response(
        data=ocr_result,
        status_code=status.HTTP_200_OK
    )