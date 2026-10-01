import asyncio
import unittest

import numpy as np

from backend.services import live_session as live_session_module
from backend.services.live_session import EndpointDetector, LiveSession


class EndpointDetectorTests(unittest.TestCase):
    """The energy-based VAD is what decides when the caller has finished
    speaking — get this wrong and the live pipeline either never answers
    (misses utterance_end) or answers mid-sentence (fires too early)."""

    def _frames(self, amplitude: float, seconds: float, sample_rate: int = 16000):
        detector_frame_ms = 20
        frame_samples = sample_rate * detector_frame_ms // 1000
        total_frames = int(seconds * 1000 / detector_frame_ms)
        rng = np.random.default_rng(0)
        for _ in range(total_frames):
            samples = rng.normal(0, amplitude, frame_samples).astype("<i2")
            yield samples.tobytes()

    def test_silence_only_never_starts_speech(self):
        detector = EndpointDetector(sample_rate=16000, silence_ms=700, min_speech_ms=250)
        events = [detector.process(frame) for frame in self._frames(amplitude=20, seconds=2)]
        self.assertNotIn("speech_start", events)
        self.assertNotIn("utterance_end", events)

    def test_loud_then_silence_produces_one_utterance(self):
        detector = EndpointDetector(sample_rate=16000, silence_ms=700, min_speech_ms=250)
        events = []
        events += [detector.process(frame) for frame in self._frames(amplitude=20, seconds=0.5)]  # calibration
        events += [detector.process(frame) for frame in self._frames(amplitude=4000, seconds=1.0)]  # speech
        events += [detector.process(frame) for frame in self._frames(amplitude=20, seconds=1.0)]  # trailing silence

        self.assertEqual(events.count("speech_start"), 1)
        self.assertEqual(events.count("utterance_end"), 1)
        # utterance_end must come after speech_start
        self.assertLess(events.index("speech_start"), events.index("utterance_end"))

    def test_brief_noise_burst_below_min_speech_is_ignored(self):
        detector = EndpointDetector(sample_rate=16000, silence_ms=700, min_speech_ms=250)
        events = []
        events += [detector.process(frame) for frame in self._frames(amplitude=20, seconds=0.5)]
        events += [detector.process(frame) for frame in self._frames(amplitude=4000, seconds=0.06)]  # too short
        events += [detector.process(frame) for frame in self._frames(amplitude=20, seconds=1.0)]

        self.assertNotIn("speech_start", events)

    def test_speech_from_the_first_frame_is_detected(self):
        # No quiet lead-in to learn the room from: a floor that learned from every
        # frame calibrated to the voice itself and never detected this utterance.
        detector = EndpointDetector(sample_rate=16000, silence_ms=700, min_speech_ms=250)
        events = [detector.process(frame) for frame in self._frames(amplitude=4000, seconds=1.0)]
        events += [detector.process(frame) for frame in self._frames(amplitude=20, seconds=1.0)]
        self.assertEqual(events.count("speech_start"), 1)
        self.assertEqual(events.count("utterance_end"), 1)

    def test_steady_room_noise_after_speech_does_not_keep_the_turn_open(self):
        # Speech at ~4000 RMS, then 5s of room sound at ~500: far above a quiet floor,
        # far below the voice. Measured live, this ran every turn into the 12s cap.
        detector = EndpointDetector(sample_rate=16000, silence_ms=1100, min_speech_ms=250)
        for frame in self._frames(amplitude=30, seconds=0.5):
            detector.process(frame)
        for frame in self._frames(amplitude=4000, seconds=1.5):
            detector.process(frame)
        events = [detector.process(frame) for frame in self._frames(amplitude=500, seconds=5.0)]
        self.assertIn("utterance_end", events)
        self.assertLessEqual(events.index("utterance_end") * 20, 1300)
        self.assertNotIn("speech_start", events)  # and the room noise doesn't start a new turn

    def _utterance_then_pause(self, detector, pause_seconds, verdict=None):
        """Speech, then a pause; returns how long into the pause the turn ended (or None)."""
        for frame in self._frames(amplitude=20, seconds=0.5):
            detector.process(frame)
        for frame in self._frames(amplitude=4000, seconds=1.0):
            detector.process(frame)
        pause_id = None
        for index, frame in enumerate(self._frames(amplitude=20, seconds=pause_seconds)):
            if detector.process(frame) == "utterance_end":
                return (index + 1) * 20
            if verdict and pause_id is None and detector.in_pause():
                pause_id = detector.pause_id
                getattr(detector, verdict)(pause_id)
        return None

    def test_finished_thought_ends_on_fast_threshold(self):
        detector = EndpointDetector(16000, silence_ms=1100, min_speech_ms=250, fast_silence_ms=450, extended_silence_ms=2000)
        ended_at = self._utterance_then_pause(detector, 2.0, verdict="mark_turn_complete")
        self.assertEqual(ended_at, 440)  # 450ms threshold at 20ms frames = 22 frames
        self.assertEqual(detector.last_end_mode, "fast")

    def test_mid_sentence_pause_gets_extended_patience(self):
        # A 1.5s thinking pause: the standard 1100ms wait would split the sentence.
        detector = EndpointDetector(16000, silence_ms=1100, min_speech_ms=250, fast_silence_ms=450, extended_silence_ms=2000)
        self.assertIsNone(self._utterance_then_pause(detector, 1.5, verdict="mark_turn_incomplete"))

    def test_unjudged_pause_keeps_standard_threshold(self):
        detector = EndpointDetector(16000, silence_ms=1100, min_speech_ms=250, fast_silence_ms=450, extended_silence_ms=2000)
        self.assertEqual(self._utterance_then_pause(detector, 2.0), 1100)
        self.assertEqual(detector.last_end_mode, "standard")

    def test_verdict_is_discarded_once_speech_resumes(self):
        detector = EndpointDetector(16000, silence_ms=1100, min_speech_ms=250, fast_silence_ms=450, extended_silence_ms=2000)
        for frame in self._frames(amplitude=20, seconds=0.5):
            detector.process(frame)
        for frame in self._frames(amplitude=4000, seconds=1.0):
            detector.process(frame)
        for frame in self._frames(amplitude=20, seconds=0.3):
            detector.process(frame)
        stale_pause = detector.pause_id
        for frame in self._frames(amplitude=4000, seconds=0.5):  # caller keeps talking
            detector.process(frame)
        detector.mark_turn_complete(stale_pause)  # verdict for the old pause arrives late
        events = [detector.process(frame) for frame in self._frames(amplitude=20, seconds=0.8)]
        self.assertNotIn("utterance_end", events)  # must not end on the fast threshold


class FakeVoice:
    """Stands in for VoiceService so the turn-orchestration logic can be
    tested without a real Whisper/Ollama/TTS round trip."""

    def __init__(self, transcript: str, reply_tokens: list[str], language: str = "en", confidence: float = 0.0, low_confidence: bool = False) -> None:
        self.transcript = transcript
        self.reply_tokens = reply_tokens
        self.synthesize_calls: list[str] = []
        self.language = language
        self.confidence = confidence
        self.low_confidence = low_confidence

    def transcribe_pcm(self, pcm: bytes, sample_rate: int, language: str | None, fast: bool = False) -> dict[str, object]:
        return {"text": self.transcript, "language": self.language, "language_probability": self.confidence, "low_confidence": self.low_confidence}

    @staticmethod
    def language_instruction(language: str) -> str:
        return "Answer in the same language." if language == "auto" else f"Answer only in {language}."

    @staticmethod
    def resolve_target_language(requested: str, detected: str | None) -> str:
        if requested and requested != "auto":
            return requested
        return detected or "en"

    @staticmethod
    def requested_reply_language(text: str) -> str | None:
        return "te" if "telugu" in text.lower() else None

    @staticmethod
    def is_turn_complete(text: str) -> bool | None:
        return True

    @staticmethod
    def detect_mixed_language(text: str) -> str | None:
        return "hi" if "hindi-mixed-marker" in text else None

    def stream_chat(self, request):
        for index, token in enumerate(self.reply_tokens):
            yield {"token": token, "done": index == len(self.reply_tokens) - 1, "model": "fake"}

    @staticmethod
    def split_sentences(text: str) -> list[str]:
        import re

        return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]

    def stream_answer(self, question, target_code, history=(), source_code=None):
        assistant_text = ""
        spoken_length = 0
        for chunk in self.stream_chat(None):
            assistant_text += chunk.get("token", "")
            unspoken = assistant_text[spoken_length:]
            sentences = self.split_sentences(unspoken)
            if len(sentences) > 1:
                cursor = 0
                for sentence in sentences[:-1]:
                    found_at = unspoken.find(sentence, cursor)
                    if found_at == -1:
                        break
                    cursor = found_at + len(sentence)
                    yield {"sentence": sentence}
                spoken_length += cursor
            if chunk.get("done"):
                break
        remainder = assistant_text[spoken_length:].strip()
        if remainder:
            yield {"sentence": remainder}
        yield {"done": True, "full_text": assistant_text}

    def synthesize(self, request) -> dict[str, object]:
        self.synthesize_calls.append(request.text)
        return {"audioBase64": "AA==", "contentType": "audio/wav"}


class FakeRepository:
    def __init__(self) -> None:
        self.saved_calls: list[object] = []

    def add_call(self, call, transcript=None, latency_ms=None) -> None:
        self.saved_calls.append((call, transcript, latency_ms))


class InterimCaptionTests(unittest.TestCase):
    """Interim (live-caption) transcription used to re-decode the whole growing
    utterance every ~400ms -- cost rose the longer someone talked, for no benefit.
    These cover the fix: a confirmed pause locks in stable text, and later interim
    passes only add to it instead of re-transcribing everything again."""

    def test_interim_transcript_combines_confirmed_prefix_with_new_text(self):
        voice = FakeVoice(transcript="today please", reply_tokens=[])
        session = LiveSession(voice=voice, repository=FakeRepository(), language="auto", voice_name="default", sample_rate=16000)
        sent: list[dict] = []

        async def send_json(payload):
            sent.append(payload)

        asyncio.run(session._send_interim_transcript(b"\x00\x00" * 100, "what time do you open", send_json))
        captions = [m["text"] for m in sent if m["type"] == "transcript_interim"]
        self.assertEqual(captions, ["what time do you open today please"])

    def test_interim_transcript_with_no_prefix_is_just_the_new_text(self):
        voice = FakeVoice(transcript="hello there", reply_tokens=[])
        session = LiveSession(voice=voice, repository=FakeRepository(), language="auto", voice_name="default", sample_rate=16000)
        sent: list[dict] = []

        async def send_json(payload):
            sent.append(payload)

        asyncio.run(session._send_interim_transcript(b"\x00\x00" * 100, "", send_json))
        captions = [m["text"] for m in sent if m["type"] == "transcript_interim"]
        self.assertEqual(captions, ["hello there"])

    def test_confirmed_pause_locks_in_prefix_for_later_interim_passes(self):
        voice = FakeVoice(transcript="what time do you open", reply_tokens=[])
        session = LiveSession(voice=voice, repository=FakeRepository(), language="auto", voice_name="default", sample_rate=16000)
        # Simulate speech having reached a pause partway through the utterance.
        for _ in range(session.detector._pause_onset_frames + session.detector._min_speech_frames):
            session.detector._speaking = True
        session.detector._silence_frames = session.detector._pause_onset_frames

        async def send_json(payload):
            pass

        pcm = b"\x00\x00" * 48000  # arbitrary audio "up to this pause"

        async def run():
            session._launch_pause_check(pcm, send_json)
            await session._pause_stt_task
            await asyncio.sleep(0)  # let the classify task (created inside _launch_pause_check) run

        asyncio.run(run())
        self.assertEqual(session._confirmed_caption, "what time do you open")
        self.assertEqual(session._confirmed_bytes, len(pcm))

    def test_garbled_pause_transcript_does_not_get_locked_in(self):
        voice = FakeVoice(transcript="बार बार बार बार बार बार बार बार", reply_tokens=[])
        session = LiveSession(voice=voice, repository=FakeRepository(), language="auto", voice_name="default", sample_rate=16000)
        session.detector._speaking = True
        session.detector._silence_frames = session.detector._pause_onset_frames

        async def send_json(payload):
            pass

        async def run():
            session._launch_pause_check(b"\x00\x00" * 48000, send_json)
            await session._pause_stt_task
            await asyncio.sleep(0)

        asyncio.run(run())
        self.assertEqual(session._confirmed_caption, "")
        self.assertEqual(session._confirmed_bytes, 0)


class LiveSessionTurnTests(unittest.TestCase):
    def test_full_turn_emits_transcript_and_per_sentence_audio(self):
        from backend import config

        config.settings.tts_enabled = True  # exercise the audio path with the fake synthesize()
        live_session_module._speech_cache.clear()  # otherwise a filler cached by an earlier test is never synthesized here
        try:
            voice = FakeVoice(transcript="What are your hours?", reply_tokens=["We're open ", "9 to 5. ", "See you soon!"])
            repository = FakeRepository()
            session = LiveSession(voice=voice, repository=repository, language="auto", voice_name="default", sample_rate=16000)

            sent: list[dict] = []

            async def send_json(payload):
                sent.append(payload)

            async def send_bytes(_data):
                sent.append({"type": "__audio_bytes__"})

            asyncio.run(session._run_turn(b"\x00\x00" * 48000, send_json, send_bytes))

            types = [item["type"] for item in sent]
            self.assertIn("transcript", types)
            self.assertIn("assistant_text", types)
            self.assertIn("audio", types)
            # A short acknowledgment is spoken first, then every reply sentence in order.
            self.assertIn(voice.synthesize_calls[0], live_session_module._FILLER_PHRASES["en"])
            self.assertEqual(voice.synthesize_calls[1:], ["We're open 9 to 5.", "See you soon!"])
            self.assertTrue(sent[-1]["final"] is True or sent[-2]["final"] is True)
        finally:
            config.settings.tts_enabled = False

    def test_garbled_transcript_asks_to_repeat_instead_of_answering(self):
        from backend import config

        config.settings.tts_enabled = True
        live_session_module._speech_cache.clear()
        try:
            voice = FakeVoice(transcript="बद्याँ बार बार बार बार बार बार बार", reply_tokens=["should not be reached."])
            session = LiveSession(voice=voice, repository=FakeRepository(), language="auto", voice_name="default", sample_rate=16000)
            sent: list[dict] = []

            async def send_json(payload):
                sent.append(payload)

            async def send_bytes(_data):
                pass

            asyncio.run(session._run_turn(b"\x00\x00" * 48000, send_json, send_bytes))
            replies = [item["text"] for item in sent if item["type"] == "assistant_text" and item.get("final")]
            self.assertEqual(len(replies), 1)
            self.assertIn(replies[0], live_session_module._REPEAT_REQUESTS["en"])
            self.assertNotIn("should not be reached.", voice.synthesize_calls)
            self.assertEqual(session._turns, [])  # kept out of the LLM's conversation history
        finally:
            config.settings.tts_enabled = False

    def test_low_confidence_transcript_asks_to_repeat_even_if_text_looks_clean(self):
        # A transcript can read as a perfectly normal sentence (no repeated words/
        # phrases) and still be wrong -- Whisper's own avg_logprob/no_speech_prob can
        # say so even when the text pattern checks find nothing. This is that case:
        # text_looks_garbled() alone would say False, but low_confidence=True must
        # still stop it from reaching the LLM.
        from backend import config

        config.settings.tts_enabled = True
        live_session_module._speech_cache.clear()
        try:
            voice = FakeVoice(transcript="the weather is quite nice today", reply_tokens=["should not be reached."], low_confidence=True)
            session = LiveSession(voice=voice, repository=FakeRepository(), language="auto", voice_name="default", sample_rate=16000)
            sent: list[dict] = []

            async def send_json(payload):
                sent.append(payload)

            async def send_bytes(_data):
                pass

            asyncio.run(session._run_turn(b"\x00\x00" * 48000, send_json, send_bytes))
            replies = [item["text"] for item in sent if item["type"] == "assistant_text" and item.get("final")]
            self.assertEqual(len(replies), 1)
            self.assertIn(replies[0], live_session_module._REPEAT_REQUESTS["en"])
            self.assertNotIn("should not be reached.", voice.synthesize_calls)
        finally:
            config.settings.tts_enabled = False

    def test_language_request_switches_and_sticks(self):
        from backend import config

        config.settings.tts_enabled = True
        live_session_module._speech_cache.clear()
        try:
            voice = FakeVoice(transcript="can you talk in telugu", reply_tokens=["should not be reached."])
            session = LiveSession(voice=voice, repository=FakeRepository(), language="auto", voice_name="default", sample_rate=16000)
            sent: list[dict] = []

            async def send_json(payload):
                sent.append(payload)

            async def send_bytes(_data):
                pass

            asyncio.run(session._run_turn(b"\x00\x00" * 48000, send_json, send_bytes))
            final = [item["text"] for item in sent if item["type"] == "assistant_text" and item.get("final")]
            self.assertEqual(final, [live_session_module._LANGUAGE_SWITCH_CONFIRMATIONS["te"]])
            self.assertNotIn("should not be reached.", voice.synthesize_calls)  # no LLM round trip
            self.assertEqual(session._established_language, "te")

            # A later ordinary question, detected as English but with zero confidence
            # (FakeVoice reports none), still stays in Telugu -- exactly the low-confidence
            # case established-language is for.
            voice.transcript, voice.reply_tokens = "what time do you open?", ["We open at 11."]
            sent.clear()
            answers = []
            original_stream = voice.stream_answer

            def recording_stream(question, target_code, history=(), source_code=None):
                answers.append(target_code)
                return original_stream(question, target_code, history, source_code)

            voice.stream_answer = recording_stream
            asyncio.run(session._run_turn(b"\x00\x00" * 48000, send_json, send_bytes))
            self.assertEqual(answers, ["te"])
        finally:
            config.settings.tts_enabled = False

    def test_language_request_does_not_permanently_lock_out_other_languages(self):
        # The actual bug reported live: after asking for Telugu once, a later question
        # clearly and confidently spoken in English still came back in Telugu
        # ("detection (en, 0.56)... replying in 'te'") -- an explicit request must not
        # override every later turn regardless of how clearly the caller then speaks
        # a different language. A *confident* detection should switch normally, same
        # as it would for any other established language.
        from backend import config

        config.settings.tts_enabled = True
        live_session_module._speech_cache.clear()
        try:
            voice = FakeVoice(transcript="can you talk in telugu", reply_tokens=["unused"])
            session = LiveSession(voice=voice, repository=FakeRepository(), language="auto", voice_name="default", sample_rate=16000)

            async def send_json(payload):
                pass

            async def send_bytes(_data):
                pass

            asyncio.run(session._run_turn(b"\x00\x00" * 48000, send_json, send_bytes))
            self.assertEqual(session._established_language, "te")

            # A later question, confidently detected as English (above the revert
            # threshold) -- must be answered in English, not stuck in Telugu.
            voice.transcript, voice.reply_tokens, voice.language, voice.confidence = "what time do you open?", ["We open at 11."], "en", 0.95
            answers = []
            original_stream = voice.stream_answer
            voice.stream_answer = lambda q, t, h=(), s=None: (answers.append(t), original_stream(q, t, h, s))[1]
            asyncio.run(session._run_turn(b"\x00\x00" * 48000, send_json, send_bytes))
            self.assertEqual(answers, ["en"])
        finally:
            config.settings.tts_enabled = False

    def test_established_non_english_does_not_flip_on_a_shaky_cross_language_guess(self):
        # The bug reported live: a caller speaking Telugu throughout had the session
        # flip to Hindi mid-conversation on a single turn Whisper misheard -- the same
        # "one shaky guess overrides an established session" failure already fixed for
        # reverting to English, just between two non-English languages instead. A
        # detection between the two confidence bars (above _MIN_LANGUAGE_CONFIDENCE,
        # below _LANGUAGE_SWITCH_CONFIDENCE) must not override the established language.
        from backend import config

        config.settings.tts_enabled = True
        live_session_module._speech_cache.clear()
        try:
            voice = FakeVoice(transcript="can you talk in telugu", reply_tokens=["unused"])
            session = LiveSession(voice=voice, repository=FakeRepository(), language="auto", voice_name="default", sample_rate=16000)

            async def send_json(payload):
                pass

            async def send_bytes(_data):
                pass

            asyncio.run(session._run_turn(b"\x00\x00" * 48000, send_json, send_bytes))
            self.assertEqual(session._established_language, "te")

            voice.transcript, voice.reply_tokens, voice.language, voice.confidence = "meeru ela unnaru ivala", ["Reply."], "hi", 0.65
            answers = []
            original_stream = voice.stream_answer
            voice.stream_answer = lambda q, t, h=(), s=None: (answers.append(t), original_stream(q, t, h, s))[1]
            asyncio.run(session._run_turn(b"\x00\x00" * 48000, send_json, send_bytes))
            self.assertEqual(answers, ["te"])
            self.assertEqual(session._established_language, "te")
        finally:
            config.settings.tts_enabled = False

    def test_established_non_english_does_switch_on_a_confident_cross_language_detection(self):
        # The flip side of the test above: a genuinely confident detection of a
        # different non-English language must still switch normally, same as any
        # other established language.
        from backend import config

        config.settings.tts_enabled = True
        live_session_module._speech_cache.clear()
        try:
            voice = FakeVoice(transcript="can you talk in telugu", reply_tokens=["unused"])
            session = LiveSession(voice=voice, repository=FakeRepository(), language="auto", voice_name="default", sample_rate=16000)

            async def send_json(payload):
                pass

            async def send_bytes(_data):
                pass

            asyncio.run(session._run_turn(b"\x00\x00" * 48000, send_json, send_bytes))
            self.assertEqual(session._established_language, "te")

            voice.transcript, voice.reply_tokens, voice.language, voice.confidence = "aap kaise hain aaj", ["Reply."], "hi", 0.9
            answers = []
            original_stream = voice.stream_answer
            voice.stream_answer = lambda q, t, h=(), s=None: (answers.append(t), original_stream(q, t, h, s))[1]
            asyncio.run(session._run_turn(b"\x00\x00" * 48000, send_json, send_bytes))
            self.assertEqual(answers, ["hi"])
            self.assertEqual(session._established_language, "hi")
        finally:
            config.settings.tts_enabled = False

    def test_code_mixed_question_tagged_english_still_replies_in_the_mix(self):
        # STT tags a whole utterance with one language, and for genuinely code-mixed
        # speech it often picks "en" even though the words themselves are clearly
        # mixed (e.g. "restaurant ka phone number kya hai") -- without correcting
        # this, the mixed-reply logic never ran at all and the reply came back in
        # plain English no matter how clearly mixed the question was.
        from backend import config

        config.settings.tts_enabled = True
        try:
            voice = FakeVoice(
                transcript="question with hindi-mixed-marker inside it",
                reply_tokens=["The answer is here."],
                language="en", confidence=0.9,
            )
            session = LiveSession(voice=voice, repository=FakeRepository(), language="auto", voice_name="default", sample_rate=16000)
            answers = []
            original_stream = voice.stream_answer
            voice.stream_answer = lambda q, t, h=(), s=None: (answers.append(t), original_stream(q, t, h, s))[1]

            async def send_json(payload):
                pass

            async def send_bytes(_data):
                pass

            asyncio.run(session._run_turn(b"\x00\x00" * 48000, send_json, send_bytes))
            self.assertEqual(answers, ["hi"])  # not "en", despite STT's own language tag
        finally:
            config.settings.tts_enabled = False

    def test_empty_transcript_ends_turn_without_calling_llm(self):
        voice = FakeVoice(transcript="   ", reply_tokens=["should not be reached"])
        repository = FakeRepository()
        session = LiveSession(voice=voice, repository=repository, language="auto", voice_name="default", sample_rate=16000)

        sent: list[dict] = []

        async def send_json(payload):
            sent.append(payload)

        async def send_bytes(_data):
            pass

        asyncio.run(session._run_turn(b"\x00\x00" * 48000, send_json, send_bytes))

        self.assertEqual(sent, [{"type": "empty_transcript"}])
        self.assertEqual(session._turns, [])


if __name__ == "__main__":
    unittest.main()
