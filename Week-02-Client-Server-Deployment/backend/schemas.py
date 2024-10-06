from pydantic import BaseModel

class OCR_Output(BaseModel):
    bbox: list[list[int]]
    text: str
    score: float

class OCR_Response(BaseModel):
    data: list[OCR_Output]
    status_code: int