from pydantic import BaseModel, ConfigDict, Field


class CampaignCreateBody(BaseModel):
    name: str = Field(min_length=1)


class ChatBody(BaseModel):
    message: str = Field(min_length=1)
    system_prompt: str | None = Field(default=None, alias="systemPrompt")
    model_config = ConfigDict(populate_by_name=True)


class ChatHistoryItem(BaseModel):
    role: str
    content: str


class VoiceTurnBody(BaseModel):
    question: str = Field(min_length=1)
    language: str = "auto"
    system_prompt: str | None = Field(default=None, alias="systemPrompt")
    voice: str | None = None
    # Prior turns of this conversation, oldest first -- the typed-question endpoint is
    # stateless server-side, so the client resends its own transcript each turn instead
    # of the server tracking a session (which the live WebSocket pipeline already does).
    history: list[ChatHistoryItem] = Field(default_factory=list)
    model_config = ConfigDict(populate_by_name=True)


class TranscriptionBody(BaseModel):
    audio_base64: str = Field(min_length=1, alias="audioBase64")
    filename: str = "audio.webm"
    language: str | None = None
    model_config = ConfigDict(populate_by_name=True)


class SpeechBody(BaseModel):
    text: str = Field(min_length=1)
    voice: str | None = None
    language: str = "auto"