"""API request and response schemas."""

from .campaigns import CampaignCreate
from .voice import ChatRequest, SpeechRequest, TranscriptionRequest
from .session import VoiceTurnRequest
from .http import CampaignCreateBody, ChatBody, SpeechBody, TranscriptionBody, VoiceTurnBody
from .tts_test import TtsTestBody

__all__ = ["CampaignCreate", "CampaignCreateBody", "ChatBody", "ChatRequest", "SpeechBody", "SpeechRequest", "TranscriptionBody", "TranscriptionRequest", "TtsTestBody", "VoiceTurnBody", "VoiceTurnRequest"]