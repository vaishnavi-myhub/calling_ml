from pydantic import BaseModel, Field


class TtsTestBody(BaseModel):
    prompt: str = Field(default="Say a short welcome message for a voice assistant.", min_length=1)
    language: str = "English"
    voice: str = "cloned"