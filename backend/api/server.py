from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any

import json
import time

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

from ..config import settings
from ..logging_setup import get_logger
from ..schemas import (
    CampaignCreate,
    CampaignCreateBody,
    ChatBody,
    ChatRequest,
    SpeechBody,
    SpeechRequest,
    TranscriptionBody,
    TranscriptionRequest,
    TtsTestBody,
    VoiceTurnBody,
    VoiceTurnRequest,
)
from ..services.documents import document_rag
from ..services.factory import campaigns, overview, repository, voice
from ..services.live_session import LiveSession
from ..services.telephony_bridge import telephony_session_endpoint
from ..services.voice import _CPU_MODEL_EXECUTOR, _GPU_MODEL_EXECUTOR, ProviderUnavailable

logger = get_logger("server")


@asynccontextmanager
async def lifespan(_: FastAPI):
    logger.info("warming up models (STT, embeddings, translation, LLM)...")
    warmup_started = time.monotonic()
    voice.warmup()
    logger.info("warmup complete (%.1fs) -- ready to accept requests", time.monotonic() - warmup_started)
    yield
    logger.info("shutting down")
    # Without this, the background thread pools that model calls run on (see
    # _GPU_MODEL_EXECUTOR / _CPU_MODEL_EXECUTOR in voice.py) keep this process alive
    # after uvicorn's shutdown signal, since ThreadPoolExecutor threads aren't daemons
    # and Python won't exit while a non-daemon thread is running. Under `--reload`, that
    # stalls StatReload indefinitely: it detects a file change, tries to stop the old
    # worker to start a new one, and simply never proceeds -- confirmed by editing a
    # file with a live --reload server running and watching it hang with no new worker
    # ever starting.
    _GPU_MODEL_EXECUTOR.shutdown(wait=False, cancel_futures=True)
    _CPU_MODEL_EXECUTOR.shutdown(wait=False, cancel_futures=True)


app = FastAPI(
    title="Aura Voice API",
    version="1.0.0",
    description="Local AI voice calling backend with LLM, STT, and TTS integrations.",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=list(settings.frontend_origins_list),
    allow_origin_regex=r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$",
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "X-Filename", "filename"],
)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def provider_call(operation: Any) -> dict[str, object]:
    try:
        return operation()
    except ProviderUnavailable as error:
        raise HTTPException(status_code=503, detail=str(error)) from error


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "aura-voice-backend", "timestamp": now_iso()}


@app.get("/api/overview")
def get_overview() -> dict[str, object]:
    return overview.get_overview()


@app.get("/api/calls")
def get_calls() -> dict[str, object]:
    return {"calls": [call.to_dict() for call in repository.calls]}


@app.get("/api/services")
def get_services() -> dict[str, object]:
    return {"services": [service.to_dict() for service in repository.services]}


@app.get("/api/campaigns")
def get_campaigns() -> dict[str, object]:
    return {"campaigns": [campaign.to_dict() for campaign in repository.campaigns]}


@app.post("/api/campaigns", status_code=201)
def create_campaign(body: CampaignCreateBody) -> dict[str, object]:
    campaign = campaigns.create(CampaignCreate(body.name.strip()))
    return campaign.to_dict()


@app.get("/api/voice/providers")
def get_provider_status() -> dict[str, object]:
    return voice.provider_status()


@app.get("/api/voice/languages")
def get_languages() -> dict[str, object]:
    return {"languages": voice.languages()}


@app.get("/api/voice/voices")
def get_voices() -> dict[str, object]:
    return {"voices": voice.voices()}


@app.get("/api/voice/documents")
def get_documents() -> dict[str, object]:
    return {"documents": [{"filename": document.filename, "chunks": len(document.chunks)} for document in document_rag.documents]}


@app.post("/api/voice/documents/upload")
async def upload_document(request: Request) -> dict[str, object]:
    content_type = request.headers.get("content-type", "")
    payload = b""
    filename = "upload.bin"

    try:
        if "multipart/form-data" in content_type.lower():
            form = await request.form()
            uploaded = form.get("file")
            if uploaded is None:
                raise HTTPException(status_code=400, detail="No file was uploaded.")
            payload = await uploaded.read()
            filename = getattr(uploaded, "filename", None) or filename
        else:
            payload = await request.body()
            filename = request.headers.get("x-filename") or request.headers.get("filename") or filename

        if not payload:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")

        document = document_rag.add_document(filename, payload)
        return {"message": "Document uploaded and added to response context.", "document": document, "totalDocuments": document_rag.count()}
    except HTTPException:
        raise
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except RuntimeError as error:
        raise HTTPException(status_code=415, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"Unable to read document: {error}") from error


@app.post("/api/voice/chat")
def chat(body: ChatBody) -> dict[str, object]:
    request = ChatRequest(body.message.strip(), body.system_prompt.strip() if body.system_prompt else None)
    return provider_call(lambda: voice.chat(request))


@app.post("/api/voice/session/turn")
def voice_session_turn(body: VoiceTurnBody) -> dict[str, object]:
    request = VoiceTurnRequest(body.question.strip(), body.language, body.system_prompt.strip() if body.system_prompt else None, body.voice)
    return provider_call(lambda: voice.answer_with_audio(request))


@app.post("/api/voice/session/stream")
def voice_session_stream(body: VoiceTurnBody) -> StreamingResponse:
    # No audio here to detect language from (unlike the live WS pipeline), so an
    # explicit selection wins, and "auto" falls back to a script-based guess on the
    # typed text itself, defaulting to English for plain Latin-script input.
    target_code = body.language if body.language != "auto" else (voice.detect_script_language(body.question) or "en")
    history = tuple({"role": entry.role, "content": entry.content} for entry in body.history)

    def events():
        try:
            for chunk in voice.stream_answer(body.question.strip(), target_code, history):
                # Typed questions get spoken back too, the same as a live mic call --
                # each sentence is synthesized as soon as it's translated/generated,
                # not just at the very end, so this stays low-latency the same way the
                # live WS session's sentence-by-sentence audio already is. A synthesis
                # failure (e.g. TTS temporarily down) shouldn't break the text reply
                # the caller can already see, so it's swallowed and just skipped.
                if chunk.get("sentence") and settings.tts_enabled:
                    try:
                        speech = voice.synthesize(SpeechRequest(str(chunk["sentence"]), body.voice, target_code))
                        chunk = {**chunk, "audioBase64": speech["audioBase64"], "contentType": speech["contentType"]}
                    except ProviderUnavailable:
                        pass
                yield json.dumps(chunk) + "\n"
        except ProviderUnavailable as error:
            yield json.dumps({"error": str(error), "done": True}) + "\n"

    return StreamingResponse(events(), media_type="application/x-ndjson")


@app.post("/api/voice/transcribe")
def transcribe(body: TranscriptionBody) -> dict[str, object]:
    request = TranscriptionRequest(body.audio_base64, body.filename, body.language)
    return provider_call(lambda: voice.transcribe(request))


@app.post("/api/voice/synthesize")
def synthesize(body: SpeechBody) -> dict[str, object]:
    request = SpeechRequest(body.text.strip(), body.voice, body.language)
    return provider_call(lambda: voice.synthesize(request))


@app.post("/api/voice/test-tts")
def test_tts(body: TtsTestBody) -> dict[str, object]:
    return provider_call(lambda: voice.test_qwen_to_tts(body.prompt.strip(), body.language, body.voice))


@app.websocket("/api/voice/session/live")
async def voice_session_live(websocket: WebSocket) -> None:
    """Full-duplex live call pipeline: client streams mic PCM16 frames in,
    server streams back transcript/assistant-text/audio messages sentence by
    sentence. See services/live_session.py for the protocol and orchestration.
    """
    await websocket.accept()
    logger.info("live call WebSocket connected (%s)", websocket.client)
    session: LiveSession | None = None
    try:
        while True:
            message = await websocket.receive()
            if message["type"] == "websocket.disconnect":
                break
            text = message.get("text")
            if text is not None:
                control = json.loads(text)
                if control.get("type") == "start":
                    session = LiveSession(
                        voice=voice,
                        repository=repository,
                        language=control.get("language", "auto"),
                        voice_name=control.get("voice"),
                        sample_rate=int(control.get("sampleRate") or settings.live_sample_rate),
                    )
                    await websocket.send_json({"type": "ready"})
                elif control.get("type") == "stop":
                    break
                continue
            audio_bytes = message.get("bytes")
            if audio_bytes is not None and session is not None:
                await session.handle_audio_chunk(audio_bytes, websocket.send_json, websocket.send_bytes)
    except WebSocketDisconnect:
        logger.info("live call WebSocket disconnected")
    finally:
        if session is not None:
            await session.finalize()


@app.websocket("/api/telephony/session")
async def telephony_session(websocket: WebSocket) -> None:
    """Richer live endpoint for a remote call-handling platform (e.g. a
    Plivo-based backend) to connect to as its own AI brain. See
    services/telephony_bridge.py for the wire protocol and current stage.
    """
    await telephony_session_endpoint(websocket)
