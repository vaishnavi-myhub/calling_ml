from typing import Any

from ..models import SystemService
from ..repositories import SqlRepository
from .voice import VoiceService


class OverviewService:
    def __init__(self, repository: SqlRepository, voice: VoiceService) -> None:
        self.repository = repository
        self.voice = voice

    def get_overview(self) -> dict[str, Any]:
        metrics = self.repository.call_metrics()
        provider_status = self.voice.provider_status()

        services = [
            SystemService("LLM", provider_status["llm"]["model"] if provider_status["llm"]["configured"] else "Not configured", "Online" if provider_status["llm"]["configured"] else "Degraded"),
            SystemService("STT", provider_status["stt"]["model"] or "Faster-Whisper", "Online" if provider_status["stt"]["configured"] else "Degraded"),
            SystemService("TTS", str(provider_status["tts"]["model"]), "Online" if provider_status["tts"]["configured"] else "Degraded"),
            SystemService("Database", "SQLite", "Online"),
        ]

        return {
            "metrics": {
                "activeCalls": metrics["activeCalls"],
                "callsToday": metrics["callsToday"],
                "successfulCalls": metrics["successfulCalls"],
                "averageDuration": metrics["averageDuration"],
                "averageLatency": metrics["averageLatency"],
                "leadsQualified": metrics["leadsQualified"],
            },
            "activity": [point.to_dict() for point in self.repository.activity_series()],
            "recentCalls": [call.to_dict() for call in self.repository.calls[:10]],
            "services": [service.to_dict() for service in services],
        }
