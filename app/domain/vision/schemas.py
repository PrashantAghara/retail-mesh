from pydantic import BaseModel


class ShelfQueryResponse(BaseModel):
    image_path: str
    answer: str
