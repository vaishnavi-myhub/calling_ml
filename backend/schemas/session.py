from dataclasses import dataclass


@dataclass(frozen=True)
class VoiceTurnRequest:
    question: str
    language: str = "auto"
    system_prompt: str | None = None
    voice: str | None = None

    @classmethod
    def from_payload(cls, payload: object) -> "VoiceTurnRequest | None":
        if not isinstance(payload, dict):
            return None
        question = payload.get("question")
        if not isinstance(question, str) or not question.strip():
            return None
        system_prompt = payload.get("systemPrompt")
        language = payload.get("language", "auto")
        voice = payload.get("voice")
        return cls(
            question=question.strip(),
            language=language.strip() if isinstance(language, str) and language.strip() else "auto",
            system_prompt=system_prompt.strip() if isinstance(system_prompt, str) else None,
            voice=voice.strip() if isinstance(voice, str) else None,
        )