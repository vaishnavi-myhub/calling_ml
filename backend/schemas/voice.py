from dataclasses import dataclass


@dataclass(frozen=True)
class ChatRequest:
    message: str
    system_prompt: str | None = None
    # Prior turns of this same conversation, oldest first, as {"role": "user"|"assistant",
    # "content": ...} -- sent to the LLM before the current message so it can follow up
    # on earlier questions ("mi restaurant" -> "ela unnav") instead of answering each
    # message in isolation.
    history: tuple[dict[str, str], ...] = ()

    @classmethod
    def from_payload(cls, payload: object) -> "ChatRequest | None":
        if not isinstance(payload, dict):
            return None
        message = payload.get("message")
        system_prompt = payload.get("systemPrompt")
        if not isinstance(message, str) or not message.strip():
            return None
        return cls(message.strip(), system_prompt.strip() if isinstance(system_prompt, str) else None)


@dataclass(frozen=True)
class TranscriptionRequest:
    audio_base64: str
    filename: str = "audio.wav"
    language: str | None = None

    @classmethod
    def from_payload(cls, payload: object) -> "TranscriptionRequest | None":
        if not isinstance(payload, dict):
            return None
        audio = payload.get("audioBase64")
        filename = payload.get("filename", "audio.wav")
        language = payload.get("language")
        if not isinstance(audio, str) or not audio.strip():
            return None
        return cls(audio.strip(), filename if isinstance(filename, str) else "audio.wav", language if isinstance(language, str) else None)


@dataclass(frozen=True)
class SpeechRequest:
    text: str
    voice: str | None = None
    language: str = "auto"

    @classmethod
    def from_payload(cls, payload: object) -> "SpeechRequest | None":
        if not isinstance(payload, dict):
            return None
        text = payload.get("text")
        voice = payload.get("voice")
        language = payload.get("language", "auto")
        if not isinstance(text, str) or not text.strip():
            return None
        return cls(text.strip(), voice if isinstance(voice, str) else None, language if isinstance(language, str) else "auto")