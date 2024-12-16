from pydantic import BaseModel


class GoldModel(BaseModel):
    json_str: str
    next_days: int = 1