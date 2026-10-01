"""Per-connection orchestration for the Live Call WebSocket pipeline.

Buffers incoming mic PCM, detects end-of-speech with a lightweight energy-based
detector (no compiled dependency — deliberately not webrtcvad, which has no
prebuilt wheel for this Python version and would need a C build toolchain),
transcribes the finished utterance with the local Whisper model, streams the
LLM reply, and synthesizes+emits each finished sentence's audio as soon as
it's ready so playback can start well before the full reply is done.
"""

import asyncio
import base64
import json
import os
import time
import wave
from collections import deque
from collections.abc import Awaitable, Callable
from uuid import uuid4

import numpy as np

from ..config import settings
from ..logging_setup import get_logger
from ..models import CallRecord
from ..repositories import SqlRepository
from ..schemas import SpeechRequest
from . import translation
from .voice import ProviderUnavailable, VoiceService, _stt_uses_gpu, _tts_uses_gpu, run_on_model_thread, speech_rate_implausible, transcript_looks_garbled

logger = get_logger("live_session")

SendJson = Callable[[dict[str, object]], Awaitable[None]]
SendBytes = Callable[[bytes], Awaitable[None]]

_FRAME_MS = 20
_CALIBRATION_FRAMES = 25  # ~500ms of ambient noise sampled before speech can be detected
_MAX_HISTORY_TURNS = 6  # recent question/answer pairs kept as LLM context, bounded by llm_num_ctx
# How often to re-transcribe the in-progress utterance for live captions. Was 700ms
# when STT ran on CPU (3-7s per pass, so faster just piled up skipped passes); on
# GPU a pass takes ~100-300ms, so captions can keep up with speech much more closely.
_INTERIM_INTERVAL_MS = 400
# A frame quieter than this fraction of the caller's typical speech loudness (75th
# percentile of their recent speech frames) counts as silence -- see EndpointDetector.
# Tuned by replaying 12 recorded live turns: 0.20 let room noise start phantom turns,
# 0.25 ended the noisy ones 1-9s sooner without cutting into any voiced speech. The cap
# keeps a loud caller's quieter words audible.
_SPEECH_RELATIVE_THRESHOLD = 0.25
_SPEECH_RELATIVE_CAP = 900.0
_SPEECH_LEVEL_WINDOW_MS = 4000
# How long a silence inside an utterance has to last before it's worth transcribing
# what's been said so far and asking whether it's a finished thought (see
# LiveSession._launch_pause_check). Shorter would fire on ordinary gaps between
# words, spending GPU time on checks that speech resuming immediately throws away.
#
# Lowered from 200: this delay eats directly into the pause-check's race against the
# standard silence timer (see vad_silence_ms in config.py for the measured numbers).
# 130ms trades a modest increase in how often a check fires on an ordinary inter-word
# gap (each one is cheap and self-cancels the moment speech resumes) for real headroom
# in that race.
_PAUSE_ONSET_MS = 130
# Below this, Whisper's own language auto-detection is unreliable enough that its guess
# shouldn't be trusted on its own -- observed a short/unclear utterance misdetected as
# Spanish (producing a garbled transcript and a reply in the wrong language entirely) for
# what was actually spoken in an Indian language. Short utterances give the model the
# least signal to work with, so this matters most on exactly the kind of quick turns a
# live conversation has.
_MIN_LANGUAGE_CONFIDENCE = 0.6
# A "complete" verdict from the pause-check classifier is only trusted enough to grant
# the fast 450ms close if at least this much audio has actually been captured for the
# utterance so far. Found via a real captured turn: a ~250-300ms blip (two brief energy
# bursts, mostly silence around them) closed the whole turn in exactly 1.00s total --
# the classifier judged the resulting one/two-word fragment transcript "complete," the
# fast path fired, and Whisper then hallucinated a fluent, unrelated 6-word sentence
# ("I'm gonna take her down.") to fill in for what was, physically, nowhere near enough
# audio to contain it (impossible at any normal human speaking rate). The classifier
# itself can't be made perfectly reliable on a fragment this short -- the fix is to
# never let a "complete" verdict shorten the wait until there's enough real audio for
# that verdict to mean anything, falling back to the standard/extended wait instead so
# the caller has more real time to actually finish speaking.
_MIN_FAST_CLOSE_MS = 600
# Once a session has established which language the caller is using, actually
# switching away from it mid-conversation needs a much higher bar than establishing
# it in the first place -- see the switching_away check in _run_turn for why.
_LANGUAGE_SWITCH_CONFIDENCE = 0.85

# Short, natural acknowledgment interjections spoken the instant a turn starts, while
# the real reply is still being generated -- masks the dead-air gap between "you
# finished talking" and "the answer is ready" (VAD silence-confirmation alone is
# ~1.1s, plus generation time on top), which real human phone conversation never has:
# a person says "okay" or "mm" almost reflexively before actually answering. Kept
# deliberately short (not "let me look that up for you") so it works for trivial fast
# turns too, not just slow document-lookup ones -- a long filler before a already-fast
# reply would make the exchange feel slower, not more natural. Never repeats back to
# back within a session (see _pick_filler), matching how a real person varies it.
_FILLER_PHRASES: dict[str, list[str]] = {
    "en": ["Okay.", "Alright.", "Sure.", "Got it.", "Mm-hm.", "One moment."],
    # Written natively rather than machine-translated from the English list: a
    # translation model handed "Mm-hm." produces nothing a native speaker would say,
    # and translating at turn time would cost ~900ms on CPU for a one-word phrase.
    "hi": ["ठीक है।", "अच्छा।", "हाँ जी।", "एक सेकंड।", "जी, बताता हूँ।"],
    "te": ["సరే.", "అలాగే.", "సరేనండి.", "ఒక్క నిమిషం.", "అవునండి."],
}
# Other languages get the English list translated once, on first use, and cached.
_TRANSLATABLE_FILLERS = ["Okay.", "Alright.", "Sure.", "One moment."]
# Reply to a turn that is *only* a request to switch language ("can you talk in
# Telugu?"). Such turns are usually short and, measured live, often garbled by STT
# ("तिल्गु में बात कर सकते हैं आपु") -- sent to the LLM they got "I don't have enough
# information". A person just confirms and carries on, so this does too, instantly.
_LANGUAGE_SWITCH_CONFIRMATIONS: dict[str, str] = {
    "en": "Sure, let's continue in English. How can I help you?",
    "hi": "ज़रूर, हम हिंदी में बात करते हैं। बताइए, मैं आपकी क्या मदद कर सकता हूँ?",
    "te": "సరే, తెలుగులో మాట్లాడుకుందాం. చెప్పండి, నేను మీకు ఎలా సహాయం చేయగలను?",
}
# Longer than this, a language request probably carries a real question too ("tell me
# your opening hours in Telugu"), which the LLM still needs to answer.
_MAX_SWITCH_ONLY_WORDS = 8
# Said instead of answering when the transcript is clearly garbled (see
# transcript_looks_garbled) -- what a person on a phone does with audio they didn't
# catch, rather than confidently answering whatever the mishearing happened to say.
_REPEAT_REQUESTS: dict[str, list[str]] = {
    "en": ["Sorry, I didn't catch that. Could you say it again?", "Sorry, could you repeat that?"],
    "hi": ["माफ़ कीजिए, मैं ठीक से सुन नहीं पाया। क्या आप दोबारा बोल सकते हैं?", "माफ़ कीजिए, ज़रा फिर से बोलिए?"],
    "te": ["క్షమించండి, నాకు సరిగ్గా వినపడలేదు. మళ్ళీ చెప్పగలరా?", "క్షమించండి, ఇంకొకసారి చెప్తారా?"],
}
_filler_translations: dict[str, list[str]] = {}
_speech_cache: dict[tuple[str, str | None, str], dict[str, object]] = {}

# Whisper hallucinates fluent-sounding but wrong text when handed several seconds of
# near-silence/background noise instead of genuine speech -- confirmed directly on real
# session audio: a 3.4s low-level tail (room tone in the low hundreds RMS, well under
# the speech threshold but never quite reaching true digital silence) produced a
# confident, fluent, entirely nonsensical Hindi sentence; another tail produced the
# classic Whisper silence-hallucination phrase "I'm gonna be like,". end_silence_ms
# (below) is exactly how long EndpointDetector waited, frame by frame, after the
# caller's last loud frame before ending the turn -- faster-whisper's own vad_filter=True
# apparently doesn't reliably strip this kind of quiet-but-not-silent tail on its own,
# so trimming it here too, short of a small pad (never clipping a trailing word's
# consonant), keeps the STT call to genuine speech instead of the multi-second
# "make sure they're really done" wait tacked on the end.
_TRAILING_SILENCE_PAD_MS = 300


def _trim_trailing_silence(pcm: bytes, sample_rate: int, trailing_ms: int) -> bytes:
    trim_ms = trailing_ms - _TRAILING_SILENCE_PAD_MS
    if trim_ms <= 0:
        return pcm
    trim_bytes = int(trim_ms / 1000 * sample_rate) * 2
    if trim_bytes <= 0 or trim_bytes >= len(pcm):
        return pcm
    return pcm[:-trim_bytes]


class EndpointDetector:
    """Fixed-frame energy VAD: tracks a continuously-adaptive noise floor (a low
    percentile of recent frame energy, not a one-time snapshot), then flags
    speech-start/utterance-end from RMS energy relative to that floor.

    The floor used to be measured once, in the first ~500ms, and frozen for the
    rest of the call -- measured directly (via server-side diagnostic logging)
    that this goes stale: a session's very first turn, before any assistant audio
    had even played back, still ran the detector "speaking" for 12+ seconds
    straight on ordinary short speech, meaning the room/mic's real ambient level
    was already above whatever got captured in that first half-second sample.
    Recomputing the floor continuously from a rolling window (see _recent_rms)
    keeps it anchored to what's actually ambient right now instead of a stale
    guess from session start.

    The min-speech-frames confirmation delay (waiting for several consecutive
    loud frames before trusting that speech has actually started, so a brief
    noise blip doesn't false-trigger) necessarily sees real speech audio
    *before* it can confirm it as such -- measured directly: without buffering
    that audio, an utterance starting right at session start ("Hi there...")
    came out transcribed as "there...", since "Hi" fell inside the confirmation
    window and was discarded rather than kept. pre_roll retains the most recent
    not-yet-confirmed frames so they can be recovered once speech_start
    actually fires, instead of asking every utterance to sacrifice its first
    word.
    """

    # Continuously-adaptive floor window: long enough to genuinely reflect the
    # current ambient level (not one noisy/quiet frame), short enough to adapt
    # within a couple of seconds if the room's actual noise level changes
    # mid-call (e.g. an AC or fan switching on).
    _FLOOR_WINDOW_MS = 4000

    def __init__(
        self, sample_rate: int, silence_ms: int, min_speech_ms: int,
        fast_silence_ms: int | None = None, extended_silence_ms: int | None = None,
    ) -> None:
        self.frame_samples = max(1, sample_rate * _FRAME_MS // 1000)
        self._silence_frames_needed = max(1, silence_ms // _FRAME_MS)
        # Used instead of the above only once the semantic check has judged the current
        # pause: a finished thought ends sooner (mark_turn_complete), a sentence cut off
        # mid-way gets more patience (mark_turn_incomplete). Unjudged pauses keep the
        # default wait.
        self._fast_silence_frames_needed = max(1, min(fast_silence_ms or silence_ms, silence_ms) // _FRAME_MS)
        self._extended_silence_frames_needed = max(1, max(extended_silence_ms or silence_ms, silence_ms) // _FRAME_MS)
        self._pause_onset_frames = max(1, _PAUSE_ONSET_MS // _FRAME_MS)
        # Bumped whenever speech resumes after any silence, so a verdict about one
        # pause can never be applied to a later one (the check runs asynchronously
        # and may finish after the caller has already started talking again).
        self._pause_id = 0
        self._complete_pause_id: int | None = None
        self._incomplete_pause_id: int | None = None
        self.last_end_silence_ms = 0
        self.last_end_mode = "standard"
        self._min_speech_frames = max(1, min_speech_ms // _FRAME_MS)
        # A safety net, not a normal limit: a real single turn in this kind of Q&A
        # conversation essentially never runs this long. Observed sustained background
        # noise keep the detector "speaking" for 20+ seconds straight, which Whisper
        # then hallucinates plausible-sounding nonsense from (random foreign-language
        # text, "cough, cough, cough") rather than failing loudly -- a silent
        # worst-case latency and correctness problem. Forcing utterance_end past this
        # cap bounds the damage even when the adaptive floor below doesn't fully
        # solve the underlying cause (e.g. a genuinely noisy room or mic).
        self._max_speech_frames = max(1, 12_000 // _FRAME_MS)
        self._noise_floor = 150.0
        # Only frames judged background (not loud) -- see process() for why.
        self._recent_rms: deque[float] = deque(maxlen=max(1, self._FLOOR_WINDOW_MS // _FRAME_MS))
        # Every recent frame, used solely to recover when the floor is so far below
        # the room's real level that nothing ever counts as background (see process()).
        self._all_rms: deque[float] = deque(maxlen=max(1, self._FLOOR_WINDOW_MS // _FRAME_MS))
        # Loudness of the caller's own recent speech (see process()): lets steady
        # background sound that is far quieter than their voice count as silence.
        self._speech_rms: deque[float] = deque(maxlen=max(1, _SPEECH_LEVEL_WINDOW_MS // _FRAME_MS))
        self._speech_level = 0.0
        self._speaking = False
        self._speech_frames = 0
        self._silence_frames = 0
        self._speaking_duration_frames = 0
        self._frames_seen = 0
        self._last_log_frame = 0
        preroll_frames = max(self._min_speech_frames, _CALIBRATION_FRAMES) + 2
        self._preroll: deque[bytes] = deque(maxlen=preroll_frames)

    def frame_size_bytes(self) -> int:
        return self.frame_samples * 2  # 16-bit PCM

    def pop_preroll(self) -> bytes:
        """Returns and clears the buffered pre-confirmation audio -- call this
        exactly when a "speech_start" event is returned, and prepend the
        result to the utterance before the frame that triggered speech_start."""
        data = b"".join(self._preroll)
        self._preroll.clear()
        return data

    @property
    def pause_id(self) -> int:
        return self._pause_id

    @property
    def speaking(self) -> bool:
        return self._speaking

    def in_pause(self) -> bool:
        """Mid-utterance and silent long enough to be worth checking whether the
        caller has actually finished (see _PAUSE_ONSET_MS)."""
        return self._speaking and self._silence_frames >= self._pause_onset_frames

    def mark_turn_complete(self, pause_id: int) -> None:
        """Lets the current pause end the turn on the shorter fast threshold.
        Ignored if speech has resumed since pause_id was read."""
        if self._speaking and pause_id == self._pause_id:
            self._complete_pause_id = pause_id

    def mark_turn_incomplete(self, pause_id: int) -> None:
        """Gives the current pause the longer, extended threshold -- the caller
        stopped mid-sentence and is most likely still thinking."""
        if self._speaking and pause_id == self._pause_id:
            self._incomplete_pause_id = pause_id

    def process(self, frame_bytes: bytes) -> str:
        samples = np.frombuffer(frame_bytes, dtype="<i2").astype("float32")
        rms = float(np.sqrt(np.mean(samples**2))) if samples.size else 0.0
        self._frames_seen += 1

        threshold = max(self._noise_floor * 3.0, 250.0)
        if self._speech_level:
            # Relative to the caller's own voice, not just to the quietest background.
            # Measured on a real session: speech at 1,700-5,400 RMS, then 8+ seconds of
            # room sound at 240-830 after the caller stopped -- far above the floor
            # (~30-50), so every "is it still speech?" check said yes and each turn ran
            # into the 12s cap ("Hello. Hello." took 9.8s to end). Capped so a very loud
            # caller can't raise the bar above where their own quieter speech sits.
            threshold = max(threshold, min(self._speech_level * _SPEECH_RELATIVE_THRESHOLD, _SPEECH_RELATIVE_CAP))
        is_loud = rms > threshold
        self._all_rms.append(rms)
        if is_loud and self._speaking:
            self._speech_rms.append(rms)
            if len(self._speech_rms) >= 10 and self._frames_seen % 5 == 0:
                # 75th percentile, not the median: in a noisy room, background frames that
                # clear the floor outnumber the caller's own voiced frames, and a median
                # sat at the background's level instead of theirs (measured: 550 vs. a
                # voice peaking at 4,000-6,000, making the relative threshold useless).
                self._speech_level = float(np.percentile(np.asarray(self._speech_rms), 75))
        # The floor learns only from frames that aren't speech. It used to learn from
        # every frame, and a caller who starts talking the instant the mic opens gave
        # it nothing *but* speech to learn from -- measured: the floor rose to the
        # voice's own level, and the start of every such sentence was dropped as
        # "background" ("people tonight." for "I want to book a table for four people
        # tonight."), or the whole utterance was never detected at all.
        if not is_loud:
            self._recent_rms.append(rms)
            if len(self._recent_rms) >= 10:
                # A low percentile of recent background, not a one-time snapshot from
                # session start, so it keeps adapting if the room's real ambient level
                # rises or falls over a long call.
                self._noise_floor = float(np.percentile(np.asarray(self._recent_rms), 15))

        if self._frames_seen - self._last_log_frame >= 250:  # ~every 5s of audio
            self._last_log_frame = self._frames_seen
            logger.info("vad: rms=%.0f floor=%.0f is_loud=%s speaking=%s", rms, self._noise_floor, is_loud, self._speaking)

        if is_loud:
            if self._speaking and self._silence_frames:
                self._pause_id += 1
                self._complete_pause_id = None
            self._silence_frames = 0
            self._speech_frames += 1
            if not self._speaking and self._speech_frames >= self._min_speech_frames:
                self._speaking = True
                self._speaking_duration_frames = 0
                self._pause_id += 1
                self._complete_pause_id = None
                return "speech_start"
            if not self._speaking:
                self._preroll.append(frame_bytes)
                return "silence"
            self._speaking_duration_frames += 1
            if self._speaking_duration_frames >= self._max_speech_frames:
                # 12s with no background frame at all usually means the floor is far
                # below a genuinely noisy room, not a 12-second sentence. Re-learn it
                # from everything recent so the detector doesn't stay stuck "speaking".
                recovered = float(np.percentile(np.asarray(self._all_rms), 15))
                if recovered > self._noise_floor:
                    self._noise_floor = recovered
                    self._recent_rms.clear()
                    self._recent_rms.extend([recovered] * 10)
                    logger.info("vad: no background audio for %dms -- noise floor re-learned at %.0f", self._speaking_duration_frames * _FRAME_MS, recovered)
                self._speaking = False
                self._silence_frames = 0
                self.last_end_silence_ms = 0
                self.last_end_mode = "max-length"
                return "utterance_end"
            return "speech"

        self._speech_frames = 0
        if self._speaking:
            self._silence_frames += 1
            self._speaking_duration_frames += 1
            if self._complete_pause_id == self._pause_id:
                mode, needed = "fast", self._fast_silence_frames_needed
            elif self._incomplete_pause_id == self._pause_id:
                mode, needed = "extended", self._extended_silence_frames_needed
            else:
                mode, needed = "standard", self._silence_frames_needed
            if self._silence_frames >= needed:
                self.last_end_silence_ms = self._silence_frames * _FRAME_MS
                self.last_end_mode = mode
                self._speaking = False
                self._silence_frames = 0
                return "utterance_end"
            return "speech"
        self._preroll.append(frame_bytes)
        return "silence"


class LiveSession:
    def __init__(self, voice: VoiceService, repository: SqlRepository, language: str, voice_name: str | None, sample_rate: int) -> None:
        self.voice = voice
        self.repository = repository
        self.language = language or "auto"
        self.voice_name = voice_name
        self.sample_rate = sample_rate or settings.live_sample_rate
        self.detector = EndpointDetector(
            self.sample_rate, settings.vad_silence_ms, settings.vad_min_speech_ms,
            settings.vad_silence_ms_fast, settings.vad_silence_ms_extended,
        )

        self._frame_buffer = bytearray()
        self._utterance_buffer = bytearray()
        self._active_task: asyncio.Task | None = None
        self._session_id = str(uuid4())
        self._started_at = time.monotonic()
        self._turns: list[dict[str, str]] = []
        self._latencies_ms: list[int] = []
        self._had_error = False
        # The first confidently-detected (or explicitly requested -- see
        # requested_reply_language in _run_turn) spoken language this session, reused
        # for later turns whose own detection is too unreliable to trust on its own
        # (see _MIN_LANGUAGE_CONFIDENCE) -- once we genuinely know what language the
        # caller is using, a later short/unclear utterance, or a moment of shaky STT
        # confidence, shouldn't have to re-gamble on a fresh guess. A *confident*
        # later detection of a different language still overrides it normally.
        self._established_language: str | None = None
        self._last_filler: str | None = None

        self._interim_task: asyncio.Task | None = None
        self._frames_since_interim = 0
        self._interim_interval_frames = max(1, _INTERIM_INTERVAL_MS // _FRAME_MS)
        self._turn_seq = 0
        # Transcription of the utterance-so-far taken at the start of the current pause
        # (see _launch_pause_check). If the turn then ends without further speech, the
        # audio is the same apart from trailing silence, so this is reused as the final
        # transcript instead of transcribing everything a second time.
        self._pause_stt_task: asyncio.Task | None = None
        self._pause_stt_id: int | None = None
        self._pause_stt_bytes: int = 0
        # Interim (live-caption) transcription used to re-decode the *entire*
        # utterance-so-far every ~400ms -- cost grew with how long the caller had
        # been talking, for no benefit (the same early words got re-transcribed over
        # and over). Each confirmed pause (see _classify_pause) now locks in
        # everything up to that point as stable text; interim passes between pauses
        # only decode the new audio since the last confirmation and append to it, so
        # the per-pass cost stays bounded instead of growing with utterance length.
        self._confirmed_caption = ""
        self._confirmed_bytes = 0

        logger.info("session %s started (requested_language=%s, sample_rate=%d)", self._session_id[:8], self.language, self.sample_rate)
        if settings.debug_save_audio:
            os.makedirs(settings.debug_audio_dir, exist_ok=True)
            logger.info("session %s: debug audio saving is ON -- writing to %s", self._session_id[:8], settings.debug_audio_dir)

    def _save_debug_audio(self, label: str, pcm: bytes, sample_rate: int | None = None) -> None:
        if not settings.debug_save_audio or not pcm:
            return
        path = os.path.join(settings.debug_audio_dir, f"{self._session_id[:8]}_{self._turn_seq:02d}_{label}.wav")
        try:
            with wave.open(path, "wb") as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(sample_rate or self.sample_rate)
                wav_file.writeframes(pcm)
            logger.info("saved debug audio: %s (%.1fs)", path, len(pcm) / 2 / (sample_rate or self.sample_rate))
        except OSError:
            logger.exception("failed to save debug audio to %s", path)

    async def handle_audio_chunk(self, chunk: bytes, send_json: SendJson, send_bytes: SendBytes) -> None:
        self._frame_buffer.extend(chunk)
        frame_size = self.detector.frame_size_bytes()
        while len(self._frame_buffer) >= frame_size:
            frame = bytes(self._frame_buffer[:frame_size])
            del self._frame_buffer[:frame_size]
            event = self.detector.process(frame)
            if event == "speech_start":
                logger.debug("speech started")
                self._frames_since_interim = 0
                self._confirmed_caption = ""
                self._confirmed_bytes = 0
                preroll = self.detector.pop_preroll()
                if preroll:
                    self._utterance_buffer.extend(preroll)
                self._utterance_buffer.extend(frame)
            elif event == "speech":
                self._utterance_buffer.extend(frame)
                if self.detector.in_pause():
                    # One check per pause, and only one in flight at a time (they share
                    # the single GPU worker); a check left over from an earlier pause
                    # just delays this one until it finishes.
                    if self._pause_stt_id != self.detector.pause_id and (
                        self._pause_stt_task is None or self._pause_stt_task.done()
                    ):
                        self._launch_pause_check(bytes(self._utterance_buffer), send_json)
                    continue
                self._frames_since_interim += 1
                # Live captions: re-transcribe what's been said so far, periodically,
                # while the caller is still talking -- not just once at utterance_end.
                # Skipped while a previous interim pass is still running so slow
                # transcription can't pile up a backlog of stale requests.
                if self._frames_since_interim >= self._interim_interval_frames and (
                    self._interim_task is None or self._interim_task.done()
                ):
                    self._frames_since_interim = 0
                    # Only the audio since the last confirmed pause, not the whole
                    # utterance -- see _confirmed_caption above.
                    snapshot = bytes(self._utterance_buffer[self._confirmed_bytes:])
                    self._interim_task = asyncio.create_task(
                        self._send_interim_transcript(snapshot, self._confirmed_caption, send_json)
                    )
            elif event == "utterance_end":
                self._utterance_buffer.extend(frame)
                utterance_ms = len(self._utterance_buffer) / 2 / self.sample_rate * 1000
                cached_stt = None
                if self._pause_stt_task is not None and self._pause_stt_id == self.detector.pause_id:
                    cached_stt = self._pause_stt_task
                elif self._pause_stt_task is not None and not self._pause_stt_task.done():
                    self._pause_stt_task.cancel()
                self._pause_stt_task = None
                self._pause_stt_id = None
                end_silence_ms = self.detector.last_end_silence_ms
                logger.info(
                    "utterance ended (%.0fms of audio, %dms of trailing silence, %s end-of-turn) -- %s",
                    utterance_ms, end_silence_ms, self.detector.last_end_mode,
                    "reusing pause-check transcript" if cached_stt else "starting transcription",
                )
                if self._interim_task and not self._interim_task.done():
                    self._interim_task.cancel()
                utterance = bytes(self._utterance_buffer)
                self._utterance_buffer.clear()
                await self._start_turn(utterance, send_json, send_bytes, cached_stt, end_silence_ms)
            # plain "silence" outside an utterance is pre-speech noise; drop it.

    async def _send_interim_transcript(self, pcm: bytes, prefix: str, send_json: SendJson) -> None:
        try:
            # An empty new-audio slice (e.g. a confirmed pause just fired) still has a
            # prefix worth showing, so don't bail out before the empty-pcm check below.
            transcript = {"text": ""}
            if pcm:
                transcript = await run_on_model_thread(self.voice.transcribe_pcm, pcm, self.sample_rate, self.language, True, gpu=_stt_uses_gpu())
            new_text = str(transcript["text"]).strip()
            combined = f"{prefix} {new_text}".strip() if new_text else prefix
            # A pause check sends its own (more accurate, main-model) caption; an
            # older snapshot landing after it would visibly roll the caption back.
            if combined and not self.detector.in_pause():
                logger.info("interim: decoded %dms new audio (prefix %d chars) -> %r", len(pcm) / 2 / self.sample_rate * 1000, len(prefix), combined)
                # No language field here on purpose: this caption is often prefix + a
                # freshly-decoded new slice, decoded separately and possibly in a
                # different detected language than the prefix's own -- sending a
                # single "language" for the combined text would be misleading. The
                # authoritative "transcript" event (is_final=true) always carries it.
                await send_json({"type": "transcript_interim", "text": combined, "is_final": False})
        except (asyncio.CancelledError, ProviderUnavailable):
            pass  # best-effort live captions; a failed/cancelled pass just skips this update

    def _launch_pause_check(self, pcm: bytes, send_json: SendJson) -> None:
        """At the start of a pause: transcribe what's been said so far with the main
        STT model, then ask whether it's a finished thought. Transcription and
        classification are separate tasks on purpose -- if the turn ends on the
        standard timer first, _run_turn reuses the transcript without also waiting
        on the classification call it no longer needs."""
        pause_id = self.detector.pause_id
        self._pause_stt_id = pause_id
        self._pause_stt_bytes = len(pcm)
        self._pause_stt_task = asyncio.create_task(self._pause_transcribe(pcm))
        asyncio.create_task(self._classify_pause(self._pause_stt_task, pause_id, len(pcm), send_json))

    async def _pause_transcribe(self, pcm: bytes) -> dict[str, object] | None:
        started = time.monotonic()
        try:
            transcript = await run_on_model_thread(self.voice.transcribe_pcm, pcm, self.sample_rate, self.language, gpu=_stt_uses_gpu())
        except ProviderUnavailable:
            return None
        transcript["stt_ms"] = int((time.monotonic() - started) * 1000)
        return transcript

    async def _classify_pause(self, stt_task: asyncio.Task, pause_id: int, audio_bytes: int, send_json: SendJson) -> None:
        try:
            transcript = await stt_task
        except asyncio.CancelledError:
            return
        # Speech resumed, or the turn already ended on the standard timer.
        if not transcript or pause_id != self.detector.pause_id or not self.detector.speaking:
            return
        text = str(transcript.get("text", "")).strip()
        if not text:
            return
        # Main-model transcript of the whole utterance-so-far, not just a fresh slice --
        # its own language detection is as trustworthy as the eventual final one.
        await send_json({
            "type": "transcript_interim", "text": text, "is_final": False,
            "language": transcript.get("language"), "language_confidence": float(transcript.get("language_probability") or 0.0),
        })
        duration_ms = audio_bytes / 2 / self.sample_rate * 1000
        if transcript_looks_garbled(text) or transcript.get("low_confidence") or speech_rate_implausible(text, duration_ms):
            # Not a real sentence, so no verdict: judged "cut off" it would earn the
            # extended 2s wait -- measured making garbled turns feel slower, not better.
            # Also not locked in as a confirmed prefix -- a wrong transcript would
            # otherwise get stuck at the front of every caption for the rest of the turn.
            logger.info("pause check: %r -> garbled/low-confidence/implausible rate (%.0fms for %d words), standard wait", text, duration_ms, len(text.split()))
            return
        # This is the accurate main-model transcript of everything up to this pause --
        # lock it in so later interim passes only decode audio after it (see
        # _confirmed_caption in __init__), and stay accurate even for the part they
        # skip re-decoding, since this came from the same (more accurate) model the
        # final answer will use.
        self._confirmed_caption = text
        self._confirmed_bytes = audio_bytes
        started = time.monotonic()
        complete = await asyncio.to_thread(self.voice.is_turn_complete, text)
        duration_ms = audio_bytes / 2 / self.sample_rate * 1000
        if complete and duration_ms < _MIN_FAST_CLOSE_MS:
            # Too little real audio for a "complete" verdict to be trustworthy -- see
            # _MIN_FAST_CLOSE_MS. Neither mark_turn_complete nor mark_turn_incomplete:
            # just let the standard wait apply, same as an "unknown" verdict.
            logger.info("pause check: %r -> complete but only %.0fms captured, ignoring (standard wait)", text, duration_ms)
        elif complete:
            self.detector.mark_turn_complete(pause_id)
        elif complete is False:
            self.detector.mark_turn_incomplete(pause_id)
        logger.info(
            "pause check: %r -> %s (stt %sms, classify %dms)",
            text, {True: "complete", False: "cut off", None: "unknown"}[complete],
            transcript.get("stt_ms"), int((time.monotonic() - started) * 1000),
        )

    async def _start_turn(
        self, pcm: bytes, send_json: SendJson, send_bytes: SendBytes,
        cached_stt: asyncio.Task | None = None, end_silence_ms: int = 0,
    ) -> None:
        if self._active_task and not self._active_task.done():
            self._active_task.cancel()
            await send_json({"type": "interrupted"})
        self._active_task = asyncio.create_task(self._run_turn(pcm, send_json, send_bytes, cached_stt, end_silence_ms))

    async def _run_turn(
        self, pcm: bytes, send_json: SendJson, send_bytes: SendBytes,
        cached_stt: asyncio.Task | None = None, end_silence_ms: int = 0,
    ) -> None:
        turn_started = time.monotonic()
        end_mode = self.detector.last_end_mode
        self._turn_seq += 1
        filler_task: asyncio.Task | None = None
        # Exactly the audio Whisper actually transcribes -- saved before transcription
        # so a bad transcript can be checked against what was really captured (genuine
        # speech vs. noise/silence/feedback) instead of guessing from the text alone.
        self._save_debug_audio("mic_in", pcm)
        try:
            transcript = None
            if cached_stt is not None:
                try:
                    transcript = await cached_stt
                except asyncio.CancelledError:
                    if asyncio.current_task() and asyncio.current_task().cancelling():
                        raise
                    transcript = None
            reused_stt = transcript is not None
            if reused_stt:
                # The cached transcript came from a pause-time snapshot, not this full
                # (possibly longer) buffer -- its own byte length is what its
                # plausibility has to be checked against, not len(pcm).
                question_audio_bytes = self._pause_stt_bytes
            else:
                stt_pcm = _trim_trailing_silence(pcm, self.sample_rate, end_silence_ms) if end_silence_ms else pcm
                if len(stt_pcm) != len(pcm):
                    logger.info(
                        "trimmed %dms trailing silence before STT (%.0fms end-of-turn wait)",
                        int((len(pcm) - len(stt_pcm)) / 2 / self.sample_rate * 1000), end_silence_ms,
                    )
                transcript = await run_on_model_thread(self.voice.transcribe_pcm, stt_pcm, self.sample_rate, self.language, gpu=_stt_uses_gpu())
                question_audio_bytes = len(stt_pcm)
            question = str(transcript["text"]).strip()
            stt_ms = int((time.monotonic() - turn_started) * 1000)
            if not question:
                logger.info("transcription empty (%dms) -- nothing to answer", stt_ms)
                await send_json({"type": "empty_transcript"})
                return
            logger.info(
                "transcribed (%dms%s): %r [detected=%s, confidence=%.2f, low_confidence=%s]",
                stt_ms, ", reused from pause check" if reused_stt else "", question,
                transcript.get("language"), float(transcript.get("language_probability") or 0.0),
                transcript.get("low_confidence"),
            )
            await send_json({
                "type": "transcript", "role": "user", "text": question, "is_final": True,
                "language": transcript.get("language"), "language_confidence": float(transcript.get("language_probability") or 0.0),
            })

            question_duration_ms = question_audio_bytes / 2 / self.sample_rate * 1000
            if transcript_looks_garbled(question) or transcript.get("low_confidence") or speech_rate_implausible(question, question_duration_ms):
                # Don't let a decoding loop (or audio Whisper itself wasn't confident
                # about -- see low_confidence in transcribe_pcm -- or a transcript with
                # more words than the captured audio could physically contain, see
                # speech_rate_implausible) pick the reply language, reach the LLM, or
                # land in the conversation history (where it skewed later replies too).
                reply_code = self.language if self.language != "auto" else (self._established_language or "en")
                logger.info(
                    "transcript looks garbled/low-confidence/implausible (%.0fms for %d words) -- asking the caller to repeat (in %r) instead of answering",
                    question_duration_ms, len(question.split()), reply_code,
                )
                await self._ask_to_repeat(reply_code, send_json, send_bytes)
                return

            detected_language = transcript.get("language")
            confidence = float(transcript.get("language_probability") or 0.0)
            if self.language == "auto":
                # Whisper is a fundamentally English-trained model, so a short,
                # code-mixed utterance ("restuarant eh time ki open avuthundhi?")
                # specifically tends to get misheard *as English* -- a confident-but-
                # wrong single-language verdict, since it has to pick exactly one label
                # for audio that's genuinely two languages at once. Measured directly:
                # the same caller, same call, got "te" for one turn (correctly) and a
                # >=0.6-confidence "en" for the very next, flipping the whole reply
                # language back to English mid-conversation. The reverse direction isn't
                # a real problem -- Whisper doesn't confidently mishear plain English as
                # Telugu -- so establishing a language from an English (or not-yet-set)
                # session keeps the low bar (confirmed necessary -- a flat higher bar in
                # both directions was tested and blocked the legitimate English->Telugu
                # switch in turn 3 too). But a switch AWAY from an already-established
                # *non-English* language -- to English, or to a second, different
                # non-English language -- needs the higher bar either way: reported live,
                # a caller speaking Telugu throughout had the session flip to Hindi
                # mid-conversation on a single misheard turn, the same "confidently wrong
                # single-utterance guess overriding a whole established session" failure
                # as the earlier English case, just between two non-English languages
                # instead of non-English-vs-English. Whisper confuses related Indic
                # languages often enough (see stt_languages in config.py) that this
                # matters just as much between e.g. Telugu and Hindi as it does for
                # English.
                established_non_english = self._established_language and self._established_language != "en"
                switching_away_from_established = (
                    established_non_english
                    and detected_language
                    and detected_language != self._established_language
                )
                required_confidence = _LANGUAGE_SWITCH_CONFIDENCE if switching_away_from_established else _MIN_LANGUAGE_CONFIDENCE
                if detected_language and confidence >= required_confidence:
                    self._established_language = str(detected_language)
                elif self._established_language:
                    logger.info(
                        "detection (%s, %.2f) not confident enough to switch away from established session language %r -- keeping it"
                        if switching_away_from_established else
                        "detection (%s, %.2f) not confident enough to establish -- using established session language %r instead",
                        detected_language, confidence, self._established_language,
                    )
                    detected_language = self._established_language

            target_code = self.voice.resolve_target_language(self.language, detected_language)
            source_code = detected_language
            if self.language == "auto" and target_code == "en":
                # Whisper tags a whole utterance with one language, and for genuinely
                # code-mixed speech it often picks "en" even when the words themselves
                # are clearly mixed -- the transcript still shows it plainly (e.g.
                # "restaurant ka phone number kya hai"), just mistagged. Without this,
                # a code-mixed question that got tagged "en" this way skipped the
                # mixed-reply logic entirely and came back in plain English, no matter
                # how clearly mixed the actual words were.
                mixed_language = self.voice.detect_mixed_language(question)
                if mixed_language and translation.is_supported(mixed_language):
                    logger.info("question reads as %s-English mixed despite being tagged 'en' -- replying in the mix instead", mixed_language)
                    target_code = mixed_language
                    source_code = mixed_language
            if self.language == "auto":
                requested = await asyncio.to_thread(self.voice.requested_reply_language, question)
                if requested:
                    # Seeds _established_language rather than a separate, stronger
                    # override -- a hard override was measured locking every later
                    # turn into the requested language regardless of what the caller
                    # actually said next ("detection (en, 0.56)... replying in 'te'"),
                    # which is the opposite of matching the caller. Feeding it into the
                    # same established-language state the normal per-turn detection
                    # already uses means a confident later switch (see
                    # _LANGUAGE_SWITCH_CONFIDENCE above) still works normally -- this
                    # only guarantees the request itself is honored immediately and
                    # survives a later *unconfident* guess, same protection every
                    # other established language already gets.
                    if requested != self._established_language:
                        logger.info("caller asked to continue in %r -- replying in it from now on", requested)
                    self._established_language = requested
                    target_code = requested
                    confirmation = _LANGUAGE_SWITCH_CONFIRMATIONS.get(requested)
                    if confirmation and len(question.split()) <= _MAX_SWITCH_ONLY_WORDS:
                        await send_json({"type": "assistant_text", "text": confirmation, "final": False})
                        await self._speak_sentence(confirmation, requested, 1, send_json, send_bytes, cache=True)
                        await send_json({"type": "assistant_text", "text": confirmation, "final": True})
                        self._turns.append({"question": question, "answer": confirmation})
                        logger.info("turn complete (language switch to %r, %dms): %r", requested, int((time.monotonic() - turn_started) * 1000), confirmation)
                        return
            logger.info("replying in %r (source=%r)", target_code, source_code)
            history = tuple(
                message
                for turn in self._turns[-_MAX_HISTORY_TURNS:]
                for message in ({"role": "user", "content": turn["question"]}, {"role": "assistant", "content": turn["answer"]})
            )

            # Fires now, in parallel with the real reply's generation below (not
            # before it) -- see _speak_filler's docstring for why this doesn't add
            # to how soon generation starts, only guarantees what gets heard first.
            filler_task = asyncio.create_task(self._speak_filler(target_code, send_json, send_bytes))

            assistant_text = ""
            seq = 0
            first_audio_latency_ms: int | None = None
            first_sentence = True
            answer_started = time.monotonic()
            first_timing: dict[str, int] = {}
            first_tts_ms = 0
            filler_heard_ms: int | None = None

            async for chunk in self._stream_answer(question, target_code, history, source_code):
                if chunk.get("error"):
                    await send_json({"type": "error", "message": chunk["error"]})
                    self._had_error = True
                    break
                if chunk.get("done"):
                    assistant_text = str(chunk.get("full_text", assistant_text))
                    break
                sentence = str(chunk["sentence"])
                assistant_text = f"{assistant_text} {sentence}".strip()
                await send_json({"type": "assistant_text", "text": assistant_text, "final": False})
                seq += 1
                if first_sentence:
                    # Guarantees send order (filler heard before the real answer)
                    # without having delayed when generation itself began.
                    filler_done_at = await filler_task
                    if filler_done_at is not None:
                        filler_heard_ms = int((filler_done_at - turn_started) * 1000)
                    first_timing = dict(chunk.get("timing") or {})
                    first_sentence = False
                tts_started = time.monotonic()
                await self._speak_sentence(sentence, target_code, seq, send_json, send_bytes)
                if first_audio_latency_ms is None:
                    first_tts_ms = int((time.monotonic() - tts_started) * 1000)
                    first_audio_latency_ms = int((time.monotonic() - turn_started) * 1000)

            await send_json({"type": "assistant_text", "text": assistant_text, "final": True})
            self._turns.append({"question": question, "answer": assistant_text})
            if first_audio_latency_ms is not None:
                self._latencies_ms.append(first_audio_latency_ms)
            total_ms = int((time.monotonic() - turn_started) * 1000)
            logger.info("turn complete (%dms total, first reply at %sms): %r", total_ms, first_audio_latency_ms, assistant_text)
            # Every stage of one turn on one line, measured from the moment the caller
            # stopped talking -- "heard" figures are what the caller actually waits.
            logger.info(
                "turn timing [%s]: end-of-turn wait %dms (%s) | stt %dms%s | lang/setup %dms | "
                "first sentence generated +%sms | translate/rewrite %sms | first tts %dms | "
                "filler heard at %sms | answer heard at %sms",
                target_code, end_silence_ms, end_mode,
                stt_ms, " (reused)" if reused_stt else "",
                int((answer_started - turn_started) * 1000) - stt_ms,
                first_timing.get("generated_ms", "?"), first_timing.get("localize_ms", "?"), first_tts_ms,
                end_silence_ms + filler_heard_ms if filler_heard_ms is not None else "-",
                end_silence_ms + first_audio_latency_ms if first_audio_latency_ms is not None else "-",
            )
        except asyncio.CancelledError:
            logger.info("turn cancelled (interrupted by new speech)")
            raise
        except Exception as error:  # keep the socket alive; surface the failure to the client
            self._had_error = True
            logger.exception("turn failed")
            await send_json({"type": "error", "message": str(error)})
        finally:
            # If the loop above never reached a first real sentence (an error or an
            # empty/interrupted turn), the filler task could otherwise be left
            # pending/unretrieved -- _speak_filler already swallows its own
            # exceptions, so this just makes sure it's actually finished or
            # cancelled rather than orphaned.
            if filler_task is not None and not filler_task.done():
                filler_task.cancel()

    async def _stream_answer(self, question: str, target_code: str, history: tuple[dict[str, str], ...], source_code: str | None = None):
        """Bridges the synchronous voice.stream_answer generator (blocking LLM +
        translation calls) into the event loop via a background thread + queue."""
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue()

        def _produce() -> None:
            try:
                for item in self.voice.stream_answer(question, target_code, history, source_code):
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

    async def _filler_options(self, target_code: str) -> list[str]:
        if target_code in _FILLER_PHRASES:
            return _FILLER_PHRASES[target_code]
        if not translation.is_supported(target_code):
            return _FILLER_PHRASES["en"]
        if target_code not in _filler_translations:
            # Off the event loop: the translation model is a blocking ~1s CPU call.
            _filler_translations[target_code] = await asyncio.to_thread(
                lambda: [translation.translate_from_english(phrase, target_code) for phrase in _TRANSLATABLE_FILLERS]
            )
        return _filler_translations[target_code]

    async def _ask_to_repeat(self, target_code: str, send_json: SendJson, send_bytes: SendBytes) -> None:
        import random

        if target_code in _REPEAT_REQUESTS:
            phrase = random.choice(_REPEAT_REQUESTS[target_code])
        else:
            phrase = random.choice(_REPEAT_REQUESTS["en"])
            if translation.is_supported(target_code):
                phrase = await asyncio.to_thread(translation.translate_from_english, phrase, target_code)
        await send_json({"type": "assistant_text", "text": phrase, "final": False})
        await self._speak_sentence(phrase, target_code, 1, send_json, send_bytes, cache=True)
        await send_json({"type": "assistant_text", "text": phrase, "final": True})

    async def _speak_filler(self, target_code: str, send_json: SendJson, send_bytes: SendBytes) -> float | None:
        """Speaks a short acknowledgment immediately, in parallel with the real
        reply's generation (see _run_turn -- this task and the real generation
        both start at the same time; only the *sending order* is synchronized,
        via awaiting this task right before the first real sentence is spoken,
        so the filler is always heard first without adding to how soon
        generation itself begins). Never lets a filler failure break the turn --
        worst case the caller just doesn't hear an acknowledgment this time.
        Returns when the filler's audio was sent (for the turn timing log)."""
        if not settings.tts_enabled:
            return None
        import random

        try:
            options = await self._filler_options(target_code)
            choices = [phrase for phrase in options if phrase != self._last_filler] or options
            phrase = random.choice(choices)
            self._last_filler = phrase
            await self._speak_sentence(phrase, target_code, 0, send_json, send_bytes, cache=True)
            return time.monotonic()
        except Exception:
            logger.exception("filler synthesis failed -- continuing without it")
            return None

    async def _speak_sentence(
        self, sentence: str, target_code: str, seq: int, send_json: SendJson, send_bytes: SendBytes, cache: bool = False,
    ) -> None:
        if not settings.tts_enabled:
            return
        # Fillers are a small fixed set, so their audio is synthesized once per server
        # lifetime and reused -- the acknowledgment then costs no synthesis at all at
        # the moment the caller stops talking, however busy the CPU is right then.
        cache_key = (target_code, self.voice_name, sentence)
        speech = _speech_cache.get(cache_key) if cache else None
        if speech is None:
            try:
                # target_code, not self.language: self.language is the caller's raw request
                # (often literally "auto"), which isn't a real voice-model key and was
                # silently falling back to the English voice for every single reply
                # regardless of what language was actually being spoken -- target_code is
                # this turn's actually-resolved language ("hi", "te", ...).
                speech = await run_on_model_thread(self.voice.synthesize, SpeechRequest(sentence, self.voice_name, target_code), gpu=_tts_uses_gpu())
            except ProviderUnavailable as error:
                await send_json({"type": "error", "message": str(error)})
                return
            if cache:
                _speech_cache[cache_key] = speech
        audio_bytes = base64.b64decode(speech["audioBase64"])
        if settings.debug_save_audio:
            # Already a complete WAV file (Piper's own sample rate, e.g. 22050Hz) --
            # written as-is rather than through _save_debug_audio, which would wrap it
            # with a second, wrong-sample-rate header on top of its real one.
            path = os.path.join(settings.debug_audio_dir, f"{self._session_id[:8]}_{self._turn_seq:02d}_tts_out_{target_code}_{seq}.wav")
            try:
                with open(path, "wb") as audio_file:
                    audio_file.write(audio_bytes)
                logger.info("saved debug audio: %s (voice=%s, text=%r)", path, target_code, sentence)
            except OSError:
                logger.exception("failed to save debug audio to %s", path)
        await send_json({"type": "audio", "seq": seq, "text": sentence, "contentType": speech["contentType"]})
        await send_bytes(audio_bytes)

    async def finalize(self) -> None:
        if self._active_task and not self._active_task.done():
            self._active_task.cancel()
        elapsed_total = int(time.monotonic() - self._started_at)
        logger.info("session %s ended (%ds, %d turns, error=%s)", self._session_id[:8], elapsed_total, len(self._turns), self._had_error)
        if not self._turns:
            return
        elapsed = int(time.monotonic() - self._started_at)
        record = CallRecord(
            id=self._session_id,
            caller="Live Call Test",
            agent="Local AI Agent",
            direction="Test",
            language=self.language,
            duration=f"{elapsed // 60:02d}:{elapsed % 60:02d}",
            lead="Not analyzed",
            sentiment=None,
            time="",
            status="Failed" if self._had_error else "Completed",
        )
        transcript = json.dumps(self._turns, ensure_ascii=False)
        latency_ms = round(sum(self._latencies_ms) / len(self._latencies_ms)) if self._latencies_ms else None
        await asyncio.to_thread(self.repository.add_call, record, transcript, latency_ms)
