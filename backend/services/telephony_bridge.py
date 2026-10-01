"""Telephony bridge: a richer live-call endpoint for a remote call-handling
platform (e.g. a Plivo-based backend) to connect to as its own AI "brain",
alongside (not replacing) the browser-facing live endpoint in api/server.py.

Reuses the browser pipeline's proven pieces directly (EndpointDetector, STT,
TTS, garbled/implausible-speech guards, confidence-gated language switching)
but does NOT use stream_answer/_enrich_with_documents -- a remote caller
supplies its own complete system prompt (persona, knowledge, rules), and
grounding that against this project's own demo document store would answer
from the wrong knowledge base entirely, a correctness bug not a style choice.

Scope, deliberately: English, Hindi, Telugu, and Hindi/Telugu-English code-
mixing are fully handled (the project's own proven, tested generation path).
Other languages are not yet wired into the generation-instruction logic here
-- this mirrors the user's own explicit phased priority (these four first,
others added gradually) rather than silently guessing at untested behavior
for languages this project hasn't verified.

WIRE PROTOCOL
-------------
Client -> server, JSON text control messages:
    {"type": "start", "system_instruction": str, "tools": [...] | None,
     "language": str | None, "voice_name": str | None,
     "vad_silence_ms": int | None, "transcript_languages": [str] | None,
     "sample_rate": int | None}
        Setup handshake. Must be the first message. `tools`, if given, is a
        list of either Ollama's native {"type":"function","function":{name,
        description,parameters}} shape, or a bare {name,description,
        parameters} dict (auto-wrapped) -- accepts either so a client built
        against a different provider's declaration shape doesn't need its
        own translation layer.
    {"type": "send_text", "text": str, "end_of_turn": bool}
        Injects a turn starting from text instead of transcribed audio (e.g.
        an opening greeting nudge).
    {"type": "function_result", "id": str, "result": {...}}
        Reply to a "function_call" event (below), correlated by id.
    {"type": "set_language", "language": str}
        Pins the established reply language going forward (e.g. after the
        client's own switch_language tool fires).
    {"type": "stop"}
        Ends the session.
Client -> server, binary frames: raw PCM16 mono audio at `sample_rate`
    (default 16000).

Server -> client, JSON text events:
    {"type": "ready"}
    {"type": "connection_state", "state": "connected" | "busy"}
    {"type": "error", "message": str}
    {"type": "transcript", "speaker": "user" | "agent", "text": str,
     "is_final": true, "language": str | None, "language_confidence": float | None}
        Fired only for FINAL text, never revising interim partials -- the
        local pipeline's own interim captions are confirmed to revise as
        more audio arrives, so forwarding those would corrupt whatever
        stores this transcript on the client side.
    {"type": "function_call", "name": str, "args": {...}, "id": str}
    {"type": "interrupted"}
    {"type": "generation_complete"}
Server -> client, binary frames: raw PCM16 mono audio at
    settings.telephony_output_sample_rate (24000 by default).
"""

import asyncio
import contextlib
import json
import time
import wave
import base64
import io
from uuid import uuid4

from fastapi import WebSocket, WebSocketDisconnect

from ..config import settings
from ..logging_setup import get_logger
from ..schemas import SpeechRequest
from .factory import voice as voice_service
from .live_session import EndpointDetector, _trim_trailing_silence, _LANGUAGE_SWITCH_CONFIDENCE, _MIN_LANGUAGE_CONFIDENCE
from .resample import resample_pcm16
from . import translation
from .voice import (
    ProviderUnavailable,
    _is_code_mixed,
    _stt_uses_gpu,
    _take_speakable,
    _tts_uses_gpu,
    run_on_model_thread,
    speech_rate_implausible,
    transcript_looks_garbled,
)

logger = get_logger("telephony_bridge")

#: One process-wide gate on how many telephony sessions may run concurrently,
#: sized to this machine's actual measured GPU headroom (see
#: telephony_max_concurrent_calls's comment in config.py) -- not per-worker,
#: since there is exactly one GPU regardless of how many requests arrive.
_capacity = asyncio.Semaphore(max(1, settings.telephony_max_concurrent_calls))

#: Generic "didn't catch that" line -- the remote caller's system_instruction
#: owns persona/tone for everything else, but a garbled-transcript recovery
#: happens before the LLM is even involved, so it needs its own fixed text.
_REPEAT_REQUESTS: dict[str, str] = {
    "en": "Sorry, I didn't catch that. Could you say it again?",
    "hi": "माफ़ कीजिए, मैं ठीक से सुन नहीं पाया। क्या आप दोबारा बोल सकते हैं?",
    "te": "క్షమించండి, నాకు సరిగ్గా వినపడలేదు. మళ్ళీ చెప్పగలరా?",
}

#: Safety net matching the browser pipeline's own established pattern -- a
#: turn stuck on a hung Ollama/tool-call request must not hold the one
#: concurrency slot (see _capacity above) forever.
_TURN_TIMEOUT_S = settings.telephony_turn_timeout_s
_TOOL_CALL_TIMEOUT_S = settings.telephony_tool_call_timeout_s


def _wav_bytes_to_pcm(wav_bytes: bytes) -> tuple[bytes, int]:
    """Returns (raw PCM16 mono bytes, sample_rate) from a WAV file's bytes --
    Piper's synthesize() returns a full WAV container (base64-encoded), not
    raw PCM, so this has to be unwrapped before resampling/sending over the
    wire as raw frames."""
    with wave.open(io.BytesIO(wav_bytes), "rb") as wav_file:
        rate = wav_file.getframerate()
        pcm = wav_file.readframes(wav_file.getnframes())
        if wav_file.getnchannels() == 2:
            # Piper voices are mono; defensive only, never expected to trigger.
            import audioop

            pcm = audioop.tomono(pcm, wav_file.getsampwidth(), 0.5, 0.5)
    return pcm, rate


def _normalize_tool(tool: dict) -> dict:
    """Accepts either Ollama's native {"type":"function","function":{...}}
    shape or a bare {name,description,parameters} dict and returns Ollama's
    shape -- so a client built against a different provider's declaration
    format doesn't need its own translation layer."""
    if tool.get("type") == "function" and "function" in tool:
        return tool
    return {
        "type": "function",
        "function": {
            "name": tool.get("name"),
            "description": tool.get("description", ""),
            "parameters": tool.get("parameters", {"type": "object", "properties": {}}),
        },
    }


class TelephonySession:
    """One telephony call's state: VAD/turn-detection, the STT->LLM->TTS turn
    loop, and barge-in. Reuses VoiceService.transcribe_pcm/synthesize/
    is_turn_complete, transcript_looks_garbled, speech_rate_implausible, and
    EndpointDetector exactly as the browser pipeline does -- only the
    generation step (no document grounding, real tool-calling, a remote
    caller's own system prompt) and the wire protocol are new."""

    def __init__(
        self,
        system_instruction: str,
        tools: list[dict] | None = None,
        language: str | None = None,
        voice_name: str | None = None,
        vad_silence_ms: int | None = None,
        transcript_languages: list[str] | None = None,
        sample_rate: int = 16000,
    ) -> None:
        self.voice = voice_service
        self.system_instruction = system_instruction
        self.tools = [_normalize_tool(t) for t in (tools or [])]
        self.language = language or "auto"
        self.voice_name = voice_name
        self.transcript_languages = transcript_languages or []
        self.sample_rate = sample_rate
        self.detector = EndpointDetector(
            sample_rate, vad_silence_ms or settings.vad_silence_ms, settings.vad_min_speech_ms,
            settings.vad_silence_ms_fast, settings.vad_silence_ms_extended,
        )
        self._frame_buffer = bytearray()
        self._utterance_buffer = bytearray()
        #: Same role as LiveSession's _established_language -- the session's
        #: best-known reply language, reused for turns whose own per-utterance
        #: detection is too unreliable to trust alone. See _run_turn for the
        #: asymmetric confidence-bar logic this drives.
        self._established_language: str | None = None
        #: OpenAI/Ollama-shaped running history -- {"role": "user"|"assistant"|"tool", ...}.
        self._history: list[dict[str, object]] = []
        self._active_task: asyncio.Task | None = None
        #: Keyed by the tool-call id sent in a "function_call" event; resolved
        #: when the matching "function_result" control message arrives.
        self._pending_function_results: dict[str, asyncio.Future] = {}
        self._turn_seq = 0
        self._session_id = str(uuid4())[:8]

    def set_language(self, language: str) -> None:
        self._established_language = language

    def provide_function_result(self, call_id: str, result: object) -> None:
        future = self._pending_function_results.get(call_id)
        if future is not None and not future.done():
            future.set_result(result)

    # ------------------------------------------------------------- audio in
    async def handle_audio_chunk(self, chunk: bytes, send_json, send_bytes) -> None:
        self._frame_buffer.extend(chunk)
        frame_size = self.detector.frame_size_bytes()
        while len(self._frame_buffer) >= frame_size:
            frame = bytes(self._frame_buffer[:frame_size])
            del self._frame_buffer[:frame_size]
            event = self.detector.process(frame)
            if event == "speech_start":
                if self._active_task and not self._active_task.done():
                    # Caller started talking again while the agent's own reply
                    # was still generating/being sent -- true barge-in. Unlike
                    # the browser client (which masks TTS echo via a client-
                    # side mic-mute hack), clean-duplex phone audio has no
                    # such leakage to guard against, so this is a genuine
                    # interrupt signal, not noise.
                    self._active_task.cancel()
                    await send_json({"type": "interrupted"})
                preroll = self.detector.pop_preroll()
                self._utterance_buffer.clear()
                if preroll:
                    self._utterance_buffer.extend(preroll)
                self._utterance_buffer.extend(frame)
            elif event == "speech":
                self._utterance_buffer.extend(frame)
            elif event == "utterance_end":
                self._utterance_buffer.extend(frame)
                end_silence_ms = self.detector.last_end_silence_ms
                utterance = bytes(self._utterance_buffer)
                self._utterance_buffer.clear()
                if self._active_task and not self._active_task.done():
                    self._active_task.cancel()
                self._active_task = asyncio.create_task(self._run_turn_guarded(utterance, end_silence_ms, send_json, send_bytes))
            # plain "silence" outside an utterance is pre-speech noise; drop it.

    async def start_text_turn(self, text: str, send_json, send_bytes) -> None:
        """Entry point for an injected text turn (e.g. an opening greeting
        nudge) -- skips STT, goes straight to generation."""
        if self._active_task and not self._active_task.done():
            self._active_task.cancel()
        self._active_task = asyncio.create_task(self._run_text_turn_guarded(text, send_json, send_bytes))

    # ----------------------------------------------------------------- turn
    async def _run_turn_guarded(self, pcm: bytes, end_silence_ms: int, send_json, send_bytes) -> None:
        try:
            await asyncio.wait_for(self._run_turn(pcm, end_silence_ms, send_json, send_bytes), timeout=_TURN_TIMEOUT_S)
        except asyncio.CancelledError:
            raise
        except asyncio.TimeoutError:
            logger.error("telephony bridge[%s]: turn exceeded %ds timeout -- abandoning", self._session_id, _TURN_TIMEOUT_S)
        except Exception:
            logger.exception("telephony bridge[%s]: turn failed", self._session_id)
            with contextlib.suppress(Exception):
                await send_json({"type": "error", "message": "internal error"})

    async def _run_text_turn_guarded(self, text: str, send_json, send_bytes) -> None:
        try:
            await asyncio.wait_for(self._generate_and_speak(text, self._established_language or "en", send_json, send_bytes), timeout=_TURN_TIMEOUT_S)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("telephony bridge[%s]: text turn failed", self._session_id)

    async def _run_turn(self, pcm: bytes, end_silence_ms: int, send_json, send_bytes) -> None:
        self._turn_seq += 1
        turn_started = time.monotonic()
        stt_pcm = _trim_trailing_silence(pcm, self.sample_rate, end_silence_ms) if end_silence_ms else pcm
        transcript = await run_on_model_thread(self.voice.transcribe_pcm, stt_pcm, self.sample_rate, self.language, gpu=_stt_uses_gpu())
        question = str(transcript["text"]).strip()
        if not question:
            return

        duration_ms = len(stt_pcm) / 2 / self.sample_rate * 1000
        if transcript_looks_garbled(question) or transcript.get("low_confidence") or speech_rate_implausible(question, duration_ms):
            reply_code = self.language if self.language != "auto" else (self._established_language or "en")
            logger.info("telephony bridge[%s]: transcript looks garbled/implausible -- asking to repeat (in %r)", self._session_id, reply_code)
            await self._ask_to_repeat(reply_code, send_json, send_bytes)
            return

        detected_language = transcript.get("language")
        confidence = float(transcript.get("language_probability") or 0.0)
        if self.language == "auto":
            # Same asymmetric confidence-gating as the browser pipeline's
            # LiveSession._run_turn: switching AWAY from an established non-
            # English language (to English, or to a different non-English
            # language) needs a much higher bar than establishing one in the
            # first place -- see that module's comment for the measured
            # incidents (English<->Telugu flip-flopping, then Telugu<->Hindi)
            # this protects against.
            established_non_english = self._established_language and self._established_language != "en"
            switching_away = bool(established_non_english and detected_language and detected_language != self._established_language)
            required_confidence = _LANGUAGE_SWITCH_CONFIDENCE if switching_away else _MIN_LANGUAGE_CONFIDENCE
            if detected_language and confidence >= required_confidence:
                self._established_language = str(detected_language)
            elif self._established_language:
                detected_language = self._established_language

        target_code = self.voice.resolve_target_language(self.language, detected_language)
        source_code = detected_language

        logger.info(
            "telephony bridge[%s]: transcribed (%dms): %r [detected=%s, confidence=%.2f] -> replying in %r",
            self._session_id, int((time.monotonic() - turn_started) * 1000), question, detected_language, confidence, target_code,
        )
        await send_json({
            "type": "transcript", "speaker": "user", "text": question, "is_final": True,
            "language": detected_language, "language_confidence": confidence,
        })

        await self._generate_and_speak(question, target_code, send_json, send_bytes, source_code=source_code)

    # ------------------------------------------------------ generate+speak
    async def _generate_and_speak(self, question: str, target_code: str, send_json, send_bytes, source_code: str | None = None) -> None:
        self._history.append({"role": "user", "content": question})

        is_code_mixed = _is_code_mixed(question)
        needs_translation = translation.is_supported(target_code)
        generation_code = "en" if needs_translation else target_code

        if generation_code == "en" and target_code != "en":
            language_note = self._english_generation_instruction(source_code, question)
        elif generation_code != "en":
            language_note = self.voice.language_instruction(self.voice._language_name_for_code(generation_code))
        else:
            language_note = self.voice.language_instruction("English")

        messages: list[dict[str, object]] = [
            {"role": "system", "content": f"{self.system_instruction}\n\n{language_note}"},
            *self._history,
        ]

        assistant_text = ""
        output_text = ""
        spoken_length = 0
        seq = 0
        tool_rounds = 0

        while True:  # bounded below by tool_rounds; a round with no tool call always returns
            raw_text = ""
            pending_tool_calls: list[dict] = []
            async for chunk in self._stream(messages):
                if chunk.get("tool_calls"):
                    pending_tool_calls.extend(chunk["tool_calls"])
                token = chunk.get("token") or ""
                if token:
                    raw_text += token
                    ready, consumed = _take_speakable(raw_text[spoken_length:])
                    spoken_length += consumed
                    for piece in ready:
                        seq += 1
                        spoken = await self._speak(piece, target_code, needs_translation, is_code_mixed, seq, send_json, send_bytes)
                        output_text = f"{output_text} {spoken}".strip()
                if chunk.get("done"):
                    break

            if pending_tool_calls:
                tool_rounds += 1
                self._history.append({"role": "assistant", "content": raw_text, "tool_calls": pending_tool_calls})
                if tool_rounds > max(1, len(self.tools)) + 3:
                    logger.warning("telephony bridge[%s]: too many tool-call rounds (%d) -- stopping", self._session_id, tool_rounds)
                    break
                for call in pending_tool_calls:
                    result = await self._invoke_tool(call, send_json)
                    self._history.append({
                        "role": "tool",
                        "tool_call_id": call.get("id", ""),
                        "content": json.dumps(result),
                    })
                messages = [messages[0], *self._history]
                spoken_length = len(raw_text)  # don't re-speak content from the tool-call round
                continue

            remainder = raw_text[spoken_length:].strip()
            if remainder:
                seq += 1
                spoken = await self._speak(remainder, target_code, needs_translation, is_code_mixed, seq, send_json, send_bytes)
                output_text = f"{output_text} {spoken}".strip()
            assistant_text = output_text
            break

        self._history.append({"role": "assistant", "content": assistant_text})
        await send_json({"type": "transcript", "speaker": "agent", "text": assistant_text, "is_final": True})
        await send_json({"type": "generation_complete"})

    async def _stream(self, messages: list[dict[str, object]]):
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue()

        def _produce() -> None:
            try:
                for item in self.voice.stream_chat_raw(messages, self.tools or None):
                    loop.call_soon_threadsafe(queue.put_nowait, item)
            except ProviderUnavailable as error:
                loop.call_soon_threadsafe(queue.put_nowait, {"error": str(error), "done": True})
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, None)

        producer = asyncio.create_task(asyncio.to_thread(_produce))
        try:
            while True:
                item = await queue.get()
                if item is None:
                    break
                yield item
        finally:
            await producer

    def _english_generation_instruction(self, source_code: str | None, question: str) -> str:
        if source_code and source_code != "en":
            source_name = self.voice._language_name_for_code(source_code)
            if self.voice.has_native_script(question):
                source_hint = f"The user's message is written in {source_name}. "
            else:
                source_hint = f"The user's message is {source_name}, written phonetically using English letters -- read it as {source_name}. "
        else:
            source_hint = "The user's question may be written in a non-English language or script. "
        return (
            f"{source_hint}Regardless of what language the question is written in, you must write your "
            "entire answer in English, using the Latin alphabet only. Never answer in the question's own "
            "language or script, even partially. This is only an internal step: a separate, reliable "
            "translation system converts your English answer into the user's own language afterward, so "
            "never mention English, translation, or any language limitation to the user."
        )

    async def _speak(self, raw_sentence: str, target_code: str, needs_translation: bool, is_code_mixed: bool, seq: int, send_json, send_bytes) -> str:
        output = self.voice._localize_sentence(raw_sentence, target_code, is_code_mixed) if needs_translation else raw_sentence
        await self._synthesize_and_send(output, target_code, seq, send_bytes)
        return output

    async def _synthesize_and_send(self, text: str, target_code: str, seq: int, send_bytes) -> None:
        try:
            speech = await run_on_model_thread(self.voice.synthesize, SpeechRequest(text, self.voice_name, target_code), gpu=_tts_uses_gpu())
        except ProviderUnavailable:
            logger.exception("telephony bridge[%s]: TTS failed for seq=%d", self._session_id, seq)
            return
        wav_bytes = base64.b64decode(speech["audioBase64"])
        pcm, native_rate = _wav_bytes_to_pcm(wav_bytes)
        pcm_out = resample_pcm16(pcm, native_rate, settings.telephony_output_sample_rate)
        await send_bytes(pcm_out)

    async def _invoke_tool(self, call: dict, send_json) -> dict:
        name = call.get("function", {}).get("name") or call.get("name")
        args = call.get("function", {}).get("arguments") or call.get("args") or {}
        call_id = str(call.get("id") or uuid4())
        logger.info("telephony bridge[%s]: tool call %s(%r)", self._session_id, name, args)
        future: asyncio.Future = asyncio.get_running_loop().create_future()
        self._pending_function_results[call_id] = future
        try:
            await send_json({"type": "function_call", "name": name, "args": args, "id": call_id})
            result = await asyncio.wait_for(future, timeout=_TOOL_CALL_TIMEOUT_S)
            return result if isinstance(result, dict) else {"success": True, "result": result}
        except asyncio.TimeoutError:
            logger.warning("telephony bridge[%s]: tool call %s timed out after %ds", self._session_id, name, _TOOL_CALL_TIMEOUT_S)
            return {"success": False, "error": "tool call timed out"}
        finally:
            self._pending_function_results.pop(call_id, None)

    async def _ask_to_repeat(self, target_code: str, send_json, send_bytes) -> None:
        phrase = _REPEAT_REQUESTS.get(target_code, _REPEAT_REQUESTS["en"])
        await send_json({"type": "transcript", "speaker": "agent", "text": phrase, "is_final": True})
        await self._synthesize_and_send(phrase, target_code, 0, send_bytes)
        await send_json({"type": "generation_complete"})


async def _receive_setup(websocket: WebSocket) -> TelephonySession | None:
    message = await websocket.receive()
    if message["type"] == "websocket.disconnect":
        return None
    text = message.get("text")
    if text is None:
        await websocket.send_json({"type": "error", "message": "First message must be a JSON 'start' control message, not binary audio."})
        return None
    try:
        control = json.loads(text)
    except json.JSONDecodeError:
        await websocket.send_json({"type": "error", "message": "First message was not valid JSON."})
        return None
    if control.get("type") != "start":
        await websocket.send_json({"type": "error", "message": f"Expected 'start', got {control.get('type')!r}."})
        return None
    system_instruction = control.get("system_instruction")
    if not system_instruction or not isinstance(system_instruction, str):
        await websocket.send_json({"type": "error", "message": "'start' requires a non-empty 'system_instruction'."})
        return None
    return TelephonySession(
        system_instruction=system_instruction,
        tools=control.get("tools"),
        language=control.get("language"),
        voice_name=control.get("voice_name"),
        vad_silence_ms=control.get("vad_silence_ms"),
        transcript_languages=control.get("transcript_languages"),
        sample_rate=int(control.get("sample_rate") or 16000),
    )


async def telephony_session_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    logger.info("telephony bridge connected (%s)", websocket.client)

    if _capacity.locked():
        logger.warning("telephony bridge: at capacity (%d), refusing new session", settings.telephony_max_concurrent_calls)
        await websocket.send_json({"type": "connection_state", "state": "busy"})
        await websocket.close(code=1013, reason="At capacity")
        return

    async with _capacity:
        session: TelephonySession | None = None
        try:
            session = await _receive_setup(websocket)
            if session is None:
                return
            logger.info(
                "telephony bridge: session started (language=%s, voice=%s, sample_rate=%d, tools=%d)",
                session.language, session.voice_name, session.sample_rate, len(session.tools),
            )
            await websocket.send_json({"type": "ready"})
            await websocket.send_json({"type": "connection_state", "state": "connected"})

            while True:
                message = await websocket.receive()
                if message["type"] == "websocket.disconnect":
                    break
                text = message.get("text")
                if text is not None:
                    try:
                        control = json.loads(text)
                    except json.JSONDecodeError:
                        continue
                    control_type = control.get("type")
                    if control_type == "stop":
                        break
                    elif control_type == "send_text":
                        await session.start_text_turn(str(control.get("text", "")), websocket.send_json, websocket.send_bytes)
                    elif control_type == "function_result":
                        session.provide_function_result(str(control.get("id", "")), control.get("result"))
                    elif control_type == "set_language":
                        session.set_language(str(control.get("language", "")))
                    continue
                audio_bytes = message.get("bytes")
                if audio_bytes is not None:
                    await session.handle_audio_chunk(audio_bytes, websocket.send_json, websocket.send_bytes)
        except WebSocketDisconnect:
            logger.info("telephony bridge: disconnected")
        except Exception:
            logger.exception("telephony bridge: session failed")
            with contextlib.suppress(Exception):
                await websocket.send_json({"type": "error", "message": "internal error"})
        finally:
            if session is not None and session._active_task and not session._active_task.done():
                session._active_task.cancel()
            logger.info("telephony bridge: session ended (language=%s)", session.language if session else None)
