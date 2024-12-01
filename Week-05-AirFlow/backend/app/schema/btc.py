from pydantic import BaseModel


class BTCModel(BaseModel):
    json_str: str
    next_days: int = 1