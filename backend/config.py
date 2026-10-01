import json
from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_DIR = Path(__file__).resolve().parent

# Single source of truth for supported spoken languages. The frontend fetches this
# list from GET /api/voice/languages instead of hardcoding it, and voice.py's
# language-code lookup indexes into the same list, so there is exactly one place
# that knows what languages this deployment supports.
#
# "auto" (the default) lets Whisper detect the spoken language itself per utterance
# from its full ~99-language vocabulary, so this list is not the ceiling on what's
# understood -- it's just the set a caller can force manually if auto-detect ever
# guesses wrong on a short utterance. Every code below is one faster-whisper (and the
# underlying Whisper model) actually recognizes. A few official Indian languages
# (Odia, Konkani, Maithili, Dogri, Bodo, Manipuri, Santali, Kashmiri) are not in
# Whisper's trained language set at all, so they cannot be added honestly here.
SUPPORTED_LANGUAGES: tuple[dict[str, str], ...] = (
    {"value": "auto", "label": "Auto detect (any language / mixed languages)", "code": "auto"},
    {"value": "English", "label": "English", "code": "en"},
    {"value": "Hindi", "label": "Hindi", "code": "hi"},
    {"value": "Telugu", "label": "Telugu", "code": "te"},
    {"value": "Tamil", "label": "Tamil", "code": "ta"},
    {"value": "Kannada", "label": "Kannada", "code": "kn"},
    {"value": "Malayalam", "label": "Malayalam", "code": "ml"},
    {"value": "Bengali", "label": "Bengali", "code": "bn"},
    {"value": "Marathi", "label": "Marathi", "code": "mr"},
    {"value": "Gujarati", "label": "Gujarati", "code": "gu"},
    {"value": "Punjabi", "label": "Punjabi", "code": "pa"},
    {"value": "Urdu", "label": "Urdu", "code": "ur"},
    {"value": "Nepali", "label": "Nepali", "code": "ne"},
    {"value": "Sinhala", "label": "Sinhala", "code": "si"},
    {"value": "Sanskrit", "label": "Sanskrit", "code": "sa"},
    {"value": "Spanish", "label": "Spanish", "code": "es"},
    {"value": "French", "label": "French", "code": "fr"},
    {"value": "German", "label": "German", "code": "de"},
    {"value": "Portuguese", "label": "Portuguese", "code": "pt"},
    {"value": "Arabic", "label": "Arabic", "code": "ar"},
    {"value": "Chinese", "label": "Chinese", "code": "zh"},
    {"value": "Japanese", "label": "Japanese", "code": "ja"},
    {"value": "Korean", "label": "Korean", "code": "ko"},
    {"value": "Russian", "label": "Russian", "code": "ru"},
    {"value": "Indonesian", "label": "Indonesian", "code": "id"},
    {"value": "Turkish", "label": "Turkish", "code": "tr"},
    {"value": "Vietnamese", "label": "Vietnamese", "code": "vi"},
    {"value": "Thai", "label": "Thai", "code": "th"},
)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(BACKEND_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    host: str = "127.0.0.1"
    port: int = 4000
    # Console log verbosity for this app's own logger (see logging_setup.py) -- separate
    # from uvicorn's own access-log lines ("INFO: 127.0.0.1 - GET ... 200 OK"), which
    # always print regardless. INFO shows each pipeline stage (VAD turn boundaries,
    # transcription results + detected language, translation calls, retrieval hits/
    # misses, errors); DEBUG adds finer detail. Set LOG_LEVEL=WARNING to quiet it down.
    log_level: str = "INFO"
    # Off by default: uvicorn's --reload was found to hang or leave orphaned "phantom"
    # server processes still listening on the port after a code edit, on this machine --
    # confirmed by direct reproduction (edited a file with --reload running and watched
    # the old worker never get replaced). Restart the server manually after changes
    # instead. Set RELOAD=true to re-enable if this turns out to be machine-specific.
    reload: bool = False
    frontend_origin: str = "http://localhost:5173"
    frontend_origins: str = "http://localhost:5173,http://localhost:8080,http://127.0.0.1:5173,http://127.0.0.1:8080"

    llm_url: str = "http://127.0.0.1:11434"
    llm_model: str = "qwen3:8b"
    llm_api_key: str = ""
    llm_keep_alive: str = "10m"
    llm_num_predict: int = 96
    llm_num_ctx: int = 4096
    llm_think: bool = False

    stt_url: str = ""
    stt_api_key: str = ""
    stt_mode: str = "local"
    stt_model: str = "large-v3"
    stt_device: str = "cuda"
    stt_compute_type: str = "float16"
    # Used only for live-caption previews while the caller is still speaking (see
    # live_session.py's interim transcription) -- measured stt_model="small" at ~3s
    # even for a warm model on a well under one second clip, too slow to ever complete
    # before the next update or before the caller finishes talking. "base" measured at
    # ~0.8s for the same clip, close enough to real-time for progressively-updating
    # captions. The final, authoritative transcription that actually drives the LLM's
    # answer still uses the full stt_model -- only these in-progress previews are faster
    # and less precise.
    stt_interim_model: str = "base"
    # Comma-separated language codes that auto-detection may choose from (e.g.
    # "en,hi,te"); empty means any of Whisper's ~100 languages. Whisper confuses
    # closely related languages -- measured: real Telugu speech detected as Tamil at
    # 24-43% confidence, then transcribed as Tamil gibberish -- so restricting it to
    # the languages callers actually use turns those near-misses into the right answer.
    stt_languages: str = ""
    # How many CPU-bound model calls (CPU-mode STT, Piper TTS, embeddings, translation,
    # intent classification) can run at once, for handling multiple simultaneous live
    # calls. GPU-mode calls (CUDA STT, XTTS) stay pinned to a single dedicated thread
    # regardless of this setting -- see _GPU_MODEL_EXECUTOR in voice.py for why. Default
    # of 2 is conservative: faster-whisper/ctranslate2 already uses multiple internal
    # CPU threads per single inference call, so a large worker count here would have
    # concurrent calls compete for the same CPU cores rather than genuinely parallelize.
    # Raise this only if you've confirmed the machine has cores to spare.
    model_cpu_workers: int = 2

    tts_url: str = ""
    tts_enabled: bool = False
    tts_api_key: str = ""
    tts_mode: str = "local"
    tts_engine: str = "xtts_v2"
    tts_gpu: bool = True
    tts_model: str = "tts_models/multilingual/multi-dataset/xtts_v2"
    tts_model_path: str = ""
    tts_model_paths: str = ""
    tts_speaker_wav: str = ""
    # JSON map of voice profile name -> reference WAV path, e.g. {"default": "...", "cloned": "..."}.
    # Replaces the single global TTS_SPEAKER_WAV as the primary source of voice profiles;
    # TTS_SPEAKER_WAV still works as a fallback "default" profile for existing setups.
    tts_voice_profiles: str = ""

    database_url: str = f"sqlite:///{(BACKEND_DIR / 'data' / 'aura.db').as_posix()}"
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

    # Local IndicTrans2 ONNX bundle used to translate LLM replies into Indic languages
    # the LLM itself cannot write reliably (see services/translation.py). Empty/missing
    # -> translation is skipped and those languages fall back to direct LLM generation.
    indic_translation_model_path: str = str(BACKEND_DIR / "data" / "models" / "indictrans2-en-indic-dist-200M-ONNX-fp32")
    # Reverse direction: translates an Indic-language question into English before
    # document retrieval and LLM generation, since cross-lingual embedding similarity
    # (Indic query vs. English document) was measured to be far weaker than same-language
    # matching -- without this, a correct answer in the document goes unfound.
    indic_translation_reverse_model_path: str = str(BACKEND_DIR / "data" / "models" / "indictrans2-indic-en-dist-200M-ONNX-int8")

    # Energy-based end-of-speech detection for the live call WebSocket pipeline.
    # 700ms measured too tight: a real speech sample had a 960ms gap between two words
    # (an ordinary inter-word pause, not the caller finishing their turn) that still
    # triggered utterance_end, splitting one sentence into two separate, incomplete
    # questions sent to the LLM. 1100ms gives real pauses more headroom.
    #
    # Raised from 1100 to 1250: this is also the fallback the semantic pause-check
    # (below) races against, and measured across 15 real pause checks, the race was
    # razor-thin on average (200ms onset + ~634ms transcribe + ~146ms classify =
    # ~980ms) and lost outright in slower cases -- a sentence correctly judged
    # "complete" still ended up waiting out the slow standard timer because the
    # verdict arrived at essentially the same moment the timer fired anyway. 150ms
    # of extra worst-case wait (only when the classifier is inconclusive) buys the
    # fast 450ms path a real, reliable margin for the common case instead of a coin
    # flip -- see _PAUSE_ONSET_MS below for the other half of this fix.
    vad_silence_ms: int = 1250
    # Shorter silence wait used only once the live transcript-so-far already *ends
    # like a finished sentence* (terminal punctuation) -- see
    # EndpointDetector.set_completion_hint. The 1100ms default above stays the
    # fallback for anything that doesn't clearly sound finished, which is exactly the
    # case the 700ms regression above hit (a mid-sentence pause has no terminal
    # punctuation yet, so it keeps the full, safe wait).
    vad_silence_ms_fast: int = 450
    # The opposite case: the transcript-so-far was judged cut off mid-sentence ("I
    # would like to know about the..."), so the caller is almost certainly still
    # thinking. Real mid-sentence thinking pauses run 1-2s, longer than the default
    # above -- measured: a ~1.1s pause after "I would like to know the restaurant"
    # split one question into two turns on the default wait alone.
    vad_silence_ms_extended: int = 2000
    vad_min_speech_ms: int = 250
    live_sample_rate: int = 16000
    # Diagnostic only, off by default: when true, every live-call turn's raw mic audio
    # (exactly what gets fed to Whisper) and every synthesized TTS sentence are saved
    # as .wav files under debug_audio_dir. Needed to actually verify what's happening
    # in a real session -- server logs show the *text* Whisper produced and which voice
    # was picked, but not whether the captured audio was genuine clean speech, noise,
    # feedback, or empty, which is exactly the kind of thing that turns "it sounds
    # wrong" into a fixable, verified diagnosis instead of a guess. Turn off again once
    # done debugging -- this has no privacy controls and grows disk usage per turn.
    debug_save_audio: bool = False
    debug_audio_dir: str = str(BACKEND_DIR / "data" / "debug_audio")

    # Telephony bridge (/api/telephony/session): a second, richer live endpoint for a
    # remote call-handling backend (e.g. a Plivo-based platform) to connect to as its
    # own AI "brain", instead of the browser-facing /api/voice/session/live. Separate
    # settings block because its concerns are different -- no document grounding (a
    # remote caller's system prompt is authoritative, not this project's demo RAG
    # corpus), real tool-calling, and GPU concurrency protection, none of which the
    # browser endpoint needs.
    #
    # One shared 8GB laptop GPU (measured via nvidia-smi this session: Ollama's own
    # qwen3:8b already holds ~6.5GB at idle, ~1.4GB free) -- a second concurrent call
    # risks severe latency degradation or outright OOM, not just a slow response. 1 is
    # the only value verified safe; raise only after measuring real concurrent-call
    # headroom, not by assumption.
    telephony_max_concurrent_calls: int = 1
    # Safety net matching the browser pipeline's own established pattern (see
    # vad_silence_ms's comment on measured races) -- a call stuck mid-turn (a hung
    # Ollama request, a stuck tool call) must not hold the one concurrency slot
    # forever and silently block every other call.
    telephony_turn_timeout_s: int = 20
    telephony_tool_call_timeout_s: int = 8
    # Plivo's own configured outbound rate for this deployment (see codecs docstring
    # in the calling platform repo: "Plivo audio/x-l16;rate=16000 in, audio/x-l16
    # @24000 out"). Piper's voices are all 22050Hz (every *.onnx.json under
    # data/models/piper/) -- TTS output is resampled to this rate before being sent
    # over the wire, so the remote bridge never has to know Piper's native rate.
    telephony_output_sample_rate: int = 24000

    @field_validator("port")
    @classmethod
    def _validate_port(cls, value: int) -> int:
        return value if 0 < value < 65536 else 4000

    @field_validator("llm_num_predict", "llm_num_ctx")
    @classmethod
    def _validate_positive(cls, value: int) -> int:
        return value if value > 0 else 96

    @property
    def frontend_origins_list(self) -> tuple[str, ...]:
        return tuple(origin.strip() for origin in self.frontend_origins.split(",") if origin.strip())

    def voice_profiles(self) -> dict[str, str]:
        """Named voice profile -> reference WAV path, for TTS voice cloning."""
        profiles: dict[str, str] = {}
        if self.tts_voice_profiles:
            try:
                parsed = json.loads(self.tts_voice_profiles)
                if isinstance(parsed, dict):
                    profiles.update({str(name): str(path) for name, path in parsed.items()})
            except json.JSONDecodeError:
                pass
        if self.tts_speaker_wav and "default" not in profiles:
            profiles["default"] = self.tts_speaker_wav
        return profiles


settings = Settings()
