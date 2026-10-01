import unittest

from backend.services.voice import VoiceService, speech_rate_implausible, transcript_looks_garbled


class GarbledTranscriptTests(unittest.TestCase):
    """Feeding a hallucinated transcript to the LLM produced confidently wrong
    answers instead of asking the caller to repeat themselves -- these guard the
    two decoding-loop shapes actually seen live: one word/syllable repeating, and
    (the gap found afterward) a multi-word phrase repeating."""

    def test_single_word_loop_is_garbled(self):
        self.assertTrue(transcript_looks_garbled("बार बार बार बार बार बार बार बार"))

    def test_repeated_multiword_phrase_is_garbled(self):
        self.assertTrue(transcript_looks_garbled("mi rupe del vulo a insertista del vulo a insertista"))

    def test_short_phrase_needs_three_repeats(self):
        self.assertFalse(transcript_looks_garbled("very very good, thank you"))
        self.assertTrue(transcript_looks_garbled("please help me please help me please help me"))

    def test_ordinary_sentences_are_not_garbled(self):
        for text in [
            "miru telugu lo answer esthara?",
            "What time does the restaurant open today?",
            "can you repeat what you said about the parking and about the delivery options",
            "The average cost for two people is 1,200 rupees for dinner and 800 rupees for lunch.",
        ]:
            self.assertFalse(transcript_looks_garbled(text), text)

    def test_short_dominant_word_loop_is_garbled(self):
        # Whisper's whole output for a 3-word Telugu question -- "మెనూలో ఏమి ఉంది?"
        # ("what's in the menu?") -- came back as this, below the 4-repeat regex
        # threshold but still clearly a decoding loop, not a real answer.
        self.assertTrue(transcript_looks_garbled("మార్లు మార్లు మార్"))
        self.assertTrue(transcript_looks_garbled("restaurant restaurant restaurant"))

    def test_short_legitimate_exclamations_are_not_garbled(self):
        for text in ["yes yes", "no no no", "okay okay", "one two three"]:
            self.assertFalse(transcript_looks_garbled(text), text)

    def test_empty_text_is_not_garbled(self):
        self.assertFalse(transcript_looks_garbled(""))
        self.assertFalse(transcript_looks_garbled("   "))


class ImplausibleSpeechRateTests(unittest.TestCase):
    """A fluent, non-repetitive, normal-confidence transcript can still be a
    hallucination -- caught live: "I'm gonna take her down." (5 words) came from
    well under a second of mostly-silent audio (two brief energy bursts, no
    sustained speech), which is physically too little audio for those words at any
    real speaking rate. transcript_looks_garbled and low_confidence both miss this
    shape since the text itself looks completely ordinary."""

    def test_real_captured_hallucination_is_implausible(self):
        # The exact case: ~700ms of pause-check audio (the snapshot taken at the
        # moment of the pause, before the full 1.00s including trailing silence).
        self.assertTrue(speech_rate_implausible("I'm gonna take her down.", 700))

    def test_ordinary_pace_is_plausible(self):
        # A real 4-word question said at a natural pace easily clears the bar.
        self.assertFalse(speech_rate_implausible("What are your hours?", 1400))

    def test_fast_but_humanly_possible_speech_is_plausible(self):
        # Generous margin for a genuinely fast talker -- must not false-positive here.
        self.assertFalse(speech_rate_implausible("Can you tell me your address please", 1200))

    def test_single_word_is_never_flagged(self):
        # One word has no internal pacing to judge implausibility from.
        self.assertFalse(speech_rate_implausible("Hello", 50))
        self.assertFalse(speech_rate_implausible("Yes", 10))

    def test_zero_duration_is_not_flagged(self):
        self.assertFalse(speech_rate_implausible("What are your hours?", 0))


class SentenceSplittingTests(unittest.TestCase):
    """Sentence-by-sentence splitting is what lets the live pipeline start
    speaking the first sentence while the rest of the reply is still
    streaming in — this is the core of the low-latency design."""

    def test_splits_on_terminal_punctuation(self):
        sentences = VoiceService.split_sentences("Hello there. How can I help you today? Sure thing!")
        self.assertEqual(sentences, ["Hello there.", "How can I help you today?", "Sure thing!"])

    def test_splits_on_hindi_devanagari_terminator(self):
        sentences = VoiceService.split_sentences("नमस्ते। आप कैसे हैं?")
        self.assertEqual(sentences, ["नमस्ते।", "आप कैसे हैं?"])

    def test_incomplete_trailing_sentence_is_kept_as_is(self):
        sentences = VoiceService.split_sentences("The appointment is confirmed. Please arrive by")
        self.assertEqual(sentences, ["The appointment is confirmed.", "Please arrive by"])

    def test_empty_input_returns_no_sentences(self):
        self.assertEqual(VoiceService.split_sentences(""), [])
        self.assertEqual(VoiceService.split_sentences("   "), [])


class SpeakableChunkTests(unittest.TestCase):
    """A long sentence is spoken clause by clause so the caller hears the start of
    the answer before the whole sentence has been generated/translated/synthesized."""

    def test_long_unfinished_sentence_releases_its_first_clause(self):
        from backend.services.voice import _take_speakable

        reply = "I can understand and respond to questions written in Hindi, English, Spanish"
        first = None
        for end in range(1, len(reply) + 1):  # streamed a character at a time
            ready, _ = _take_speakable(reply[:end])
            if ready:
                first = (ready, reply[:end])
                break
        # Released as soon as the first full clause exists -- long before the sentence ends.
        self.assertEqual(first[0], ["I can understand and respond to questions written in Hindi,"])
        self.assertTrue(first[1].endswith("Hindi, "))

    def test_short_clauses_are_not_chopped(self):
        from backend.services.voice import _take_speakable

        ready, consumed = _take_speakable("Yes, we do offer parking. And")
        self.assertEqual(ready, ["Yes, we do offer parking."])

    def test_numbers_with_commas_are_not_split(self):
        from backend.services.voice import _clauses

        self.assertEqual(_clauses("The average cost for two people is 1,200 rupees tonight."), ["The average cost for two people is 1,200 rupees tonight."])

    def test_streaming_token_by_token_never_loses_or_repeats_text(self):
        from backend.services.voice import _clauses, _take_speakable

        reply = ("Spice Garden is open from 11 AM to 11 PM on weekdays, and from 10 AM to midnight on weekends. "
                 "You can reserve by phone, online, or at the counter, and large groups should book two days ahead. Thanks!")
        spoken: list[str] = []
        text, done_up_to = "", 0
        for index in range(0, len(reply), 3):  # tokens of a few characters, like an LLM stream
            text = reply[: index + 3]
            ready, consumed = _take_speakable(text[done_up_to:])
            spoken.extend(ready)
            done_up_to += consumed
        spoken.extend(_clauses(text[done_up_to:].strip()))
        self.assertEqual(" ".join(spoken).split(), reply.split())
        self.assertGreater(len(spoken), 3)  # the long sentences really were broken up


if __name__ == "__main__":
    unittest.main()
