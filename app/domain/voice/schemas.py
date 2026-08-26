from pydantic import BaseModel


class VoiceQueryResponse(BaseModel):
    transcribed_query: str
    category: str | None
    response_text: str
