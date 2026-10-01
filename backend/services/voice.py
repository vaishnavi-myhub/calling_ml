import asyncio
import base64
import json
import mimetypes
import io
import os
import re
import tempfile
import time
import urllib.error
import urllib.request
import wave
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import lru_cache
from collections.abc import Iterator

from ..config import SUPPORTED_LANGUAGES, settings
from ..logging_setup import get_logger
from ..schemas import ChatRequest, SpeechRequest, TranscriptionRequest, VoiceTurnRequest
from .documents import document_rag
from . import translation

logger = get_logger("voice")

# A handful of Whisper's ~100 detectable language codes use non-standard alpha-2 codes
# that pycountry's ISO 639 database doesn't recognize directly.
_ISO_CODE_OVERRIDES = {"jw": "Javanese", "yue": "Cantonese", "haw": "Hawaiian"}


@lru_cache(maxsize=128)
def _iso_language_name(code: str) -> str | None:
    if code in _ISO_CODE_OVERRIDES:
        return _ISO_CODE_OVERRIDES[code]
    try:
        import pycountry

        language = pycountry.languages.get(alpha_2=code) or pycountry.languages.get(alpha_3=code)
        # pycountry annotates some entries as e.g. "Swahili (macrolanguage)" -- the
        # parenthetical is a linguistic classification detail, not part of the name.
        return language.name.split(" (")[0] if language else None
    except Exception:
        return None

_SCRIPT_RANGES: tuple[tuple[int, int, str], ...] = (
    (0x0980, 0x09FF, "bn"),
    (0x0A00, 0x0A7F, "pa"),
    (0x0A80, 0x0AFF, "gu"),
    (0x0B80, 0x0BFF, "ta"),
    (0x0C00, 0x0C7F, "te"),
    (0x0C80, 0x0CFF, "kn"),
    (0x0D00, 0x0D7F, "ml"),
    (0x0600, 0x06FF, "ur"),
    (0x0900, 0x097F, "hi"),
)

# Common function/particle words for typing an Indian language phonetically in Latin
# script ("Hinglish"/"Tenglish"-style, e.g. "miku telugu lo answer chepthara?") -- a
# very common way Indian-language speakers type casually, which native-script
# detection above cannot see at all. This is a lightweight heuristic, not real
# language-ID, so it only uses words distinctive enough to rarely appear in genuine
# English text (deliberately excludes short/ambiguous ones like "ki" or the language
# name itself, e.g. "telugu", since "do you know telugu?" is plain English *about*
# Telugu, not written *in* it).
_ROMANIZED_MARKERS: dict[str, frozenset[str]] = {
    "te": frozenset({
        "meeku", "miku", "meku", "meeru", "miru", "meru", "naaku", "naku", "vachu", "vacha", "ochu",
        "chepthara", "cheppandi", "cheppagalara", "cheppagalaru", "cheppu", "unnara", "unnaru",
        "unnav", "gurinchi", "bagunnara", "chestunnav", "chestunnaru", "kavali", "enti", "emiti",
        # Added after real conversation testing showed these common narrative/statement forms
        # (as opposed to the mostly question-phrased markers above) went undetected entirely,
        # e.g. "restaurant eh time ki open avuthundhi?" and "vecil parking undha?" got answered
        # in plain English because none of the words above matched.
        "avuthundhi", "authundi", "unnae", "unnai", "undha", "undi", "memu", "manam", "mugguram",
        "unnam", "cheyyadaniki", "vachindi", "kadha", "endhuku", "enduku", "esthunnaru",
        "chestharu", "nundi", "varaku", "untundi", "untada", "kosam",
    }),
    "hi": frozenset({
        "kaise", "aapko", "aapka", "mujhe", "nahi", "haan", "kripya", "dhanyavad", "namaste",
        "kya", "karo", "karenge", "chahiye", "accha",
        # Added after real testing showed the most basic, unavoidable Hindi grammar
        # words ("ka", "hai", "mein"...) were missing -- not just an input-detection
        # gap like the Telugu one above, but also silently breaking output validation:
        # a genuinely well-formed code-mix rewrite ("Spice Garden Restaurant ka phone
        # number +91 9876543210 hai.") was being rejected as "not mixed" because none
        # of these words were recognized, forcing a fallback that then produced a
        # mistranslation on that retry. These are unambiguous Hindi function words,
        # not real English words, so the collision risk is low.
        "hai", "hain", "mein", "se", "aur", "hoon", "raha", "rahi", "rahe", "liye",
        "saath", "yaha", "wahan", "kaisa", "kaisi", "kar", "ka", "ki", "ke",
    }),
    "ta": frozenset({
        "eppadi", "irukku", "irukka", "vanakkam", "nandri", "venum", "illa", "enakku", "unga",
    }),
    "kn": frozenset({
        "hegiddira", "namaskara", "beku", "yenu", "chennagideera", "nimma",
    }),
}


def _detect_romanized_language(text: str) -> str | None:
    words = set(re.findall(r"[a-z]+", text.lower()))
    for language, markers in _ROMANIZED_MARKERS.items():
        if words & markers:
            return language
    return None


# Concrete worked examples for the code-mix rewrite instruction below -- tested and found
# that giving qwen3 an abstract instruction alone ("respond in a mix") was unreliable (it just
# answered in plain English), but a concrete example of the exact expected style is a much
# stronger, more specific signal for a small model to actually follow.
_CODE_MIX_EXAMPLES: dict[str, str] = {
    "te": (
        'Example -- English sentence: "The restaurant is open from 11 AM to 11 PM on weekdays, '
        'and 10 AM to 12 AM on weekends." Natural Telugu-English mixed rewrite: "Restaurant '
        'weekdays lo 11 AM nundi 11 PM varaku, weekends lo 10 AM nundi 12 AM varaku open ga '
        'untundi."'
    ),
    "hi": (
        'Example -- English sentence: "The restaurant is open from 11 AM to 11 PM on weekdays, '
        'and 10 AM to 12 AM on weekends." Natural Hindi-English mixed rewrite: "Restaurant '
        'weekdays mein 11 AM se 11 PM tak, aur weekends mein 10 AM se 12 AM tak open rehta hai."'
    ),
}


def _code_mix_instruction(language_name: str, target_code: str) -> str:
    example = _CODE_MIX_EXAMPLES.get(target_code, "")
    return (
        f"Rewrite the given English sentence in natural, casual, code-mixed {language_name}-English "
        f"-- exactly the way a bilingual {language_name} speaker types when mixing the two languages "
        f"in the same sentence: written in Latin letters (romanized {language_name}), using "
        f"{language_name} grammar, connecting words, and verbs, while keeping English nouns, names, "
        f"numbers, and technical terms in English. {example} Do not add, remove, or change any fact "
        f"from the original sentence -- only change how it is expressed. Output only the rewritten "
        f"sentence itself, with no explanation, quotes, or labels."
    )


# Semantic end-of-turn check for the live call pipeline (see VoiceService.is_turn_complete).
# Few-shot, not an abstract instruction: the abstract version was measured answering
# "cut off" for 12/12 test utterances, complete ones included; this version scored 28/30
# on a held-out set spanning English, Hindi and Telugu (native and romanized script), at
# ~60-125ms per call on the already-warm model.
_TURN_COMPLETE_PROMPT = (
    "Is the following phone-call utterance a complete sentence (a full question, request, greeting or "
    "statement that makes sense on its own), or is it cut off in the middle so the speaker must still be "
    "going to say more? The text may be English, Hindi, Telugu, or a mix, and may contain speech-to-text "
    "errors. Reply with exactly one word: complete or cutoff.\n\n"
    "Examples:\n"
    "\"what are your opening hours\" -> complete\n"
    "\"do you deliver\" -> complete\n"
    "\"hello how are you\" -> complete\n"
    "\"thanks that's all\" -> complete\n"
    "\"menu lo em unnayi\" -> complete\n"
    "\"i wanted to ask about the\" -> cutoff\n"
    "\"can i book a table for\" -> cutoff\n"
    "\"what is the price of the\" -> cutoff\n"
    "\"and also\" -> cutoff"
)
# Spellings of each language's name as they actually come out of STT -- in Latin,
# Devanagari, Telugu and Urdu script, including garbled forms seen live ("तिलगू",
# "تلگو" for Telugu). Only a mention of one of these triggers the (LLM) check for an
# explicit "please speak <language>" request, so ordinary turns pay nothing for it.
_LANGUAGE_NAME_VARIANTS: dict[str, tuple[str, ...]] = {
    "en": ("english", "angrezi", "angreji", "अंग्रेजी", "अंग्रेज़ी", "इंग्लिश", "ఇంగ్లీష్", "ఇంగ్లీషు", "ఇంగ్లిష్", "انگریزی", "انگلش"),
    "hi": ("hindi", "हिंदी", "हिन्दी", "హిందీ", "ہندی"),
    "te": ("telugu", "telgu", "telagu", "తెలుగు", "तेलुगु", "तेलुगू", "तेलगु", "तेलगू", "तिलगू", "तिलगु", "تلگو", "تیلگو"),
}
_LANGUAGE_REQUEST_YES_NO = (
    "A caller on a phone call said the sentence below (speech-to-text; it may contain errors and may be in "
    "English, Hindi, Telugu, Urdu script or a mix). Is the caller asking the assistant to speak, reply or "
    "continue the conversation in {name} -- including asking whether it can speak {name}? "
    "Reply with exactly one word: yes or no.\n\n"
    "Examples:\n"
    "\"can you talk to me in {name}\" -> yes\n"
    "\"do you know {name}\" -> yes\n"
    "\"{name} lo cheppandi\" -> yes\n"
    "\"is the menu available in {name}\" -> no\n"
    "\"do you serve {name} style meals\" -> no\n"
    "\"my friend only speaks {name}, can he book a table\" -> no"
)
# Only for sentences naming more than one language ("you know Telugu, so why are you
# replying in English?") -- keywords can't tell which one is being asked for there.
_LANGUAGE_REQUEST_WHICH = (
    "A caller on a phone call said the sentence below (speech-to-text, may contain errors and may be in "
    "English, Hindi, Telugu, Urdu script or a mix). Is the caller asking the assistant to speak, reply or "
    "continue the conversation in a particular language -- including asking whether it can speak that "
    "language? Reply with exactly one word: english, hindi, telugu, or none.\n\n"
    "Examples:\n"
    "\"can you talk to me in telugu\" -> telugu\n"
    "\"do you know hindi\" -> hindi\n"
    "\"you speak telugu so why reply in english\" -> telugu\n"
    "\"is the menu available in hindi\" -> none"
)
_LANGUAGE_WORDS = {"english": "en", "hindi": "hi", "telugu": "te"}
# STT garbles "Telugu" differently every time -- "तिलगू", "तिलगु", then "तिल्गु" in
# consecutive live runs -- so a spelling list always lags. What stays stable is the
# consonant skeleton once vowel marks are ignored (त-ल-ग / త-ల-గ), plus a loose Latin
# pattern. A loose match here only costs one extra ~100ms LLM yes/no check; it can't
# switch the language on its own.
_TELUGU_SKELETONS = ("तलग", "తలగ")
_TELUGU_LATIN = re.compile(r"\bt[aeiy]*l+[aeiuy]*g+[uoa]*\b")


def _consonant_skeleton(text: str) -> str:
    import unicodedata

    return "".join(ch for ch in text if unicodedata.category(ch) not in ("Mn", "Mc"))


def _mentioned_languages(text: str) -> list[str]:
    lowered = text.lower()
    mentioned = [code for code, variants in _LANGUAGE_NAME_VARIANTS.items() if any(v in lowered for v in variants)]
    if "te" not in mentioned and (
        _TELUGU_LATIN.search(lowered) or any(s in _consonant_skeleton(lowered) for s in _TELUGU_SKELETONS)
    ):
        mentioned.append("te")
    return mentioned
# Whisper closes whatever audio it's handed with punctuation, even audio that stops
# mid-sentence (measured: "I would like to know the restaurant." and "...opening hours
# end?" for two genuinely cut-off utterances) -- so the classifier judges the words only.
_TURN_PUNCTUATION = str.maketrans("", "", ".,?!;:\"।॥…")


def _is_code_mixed(text: str) -> bool:
    """True when a single utterance genuinely blends English with an Indian language
    -- e.g. "restuarant location cheppagalara?" (English + romanized Telugu), or the
    same thing with the Telugu portion in its own native script -- as opposed to the
    text being just one language written unusually (e.g. Telugu typed phonetically
    start to finish, with no real English content). This matters because IndicTrans2
    is a strict monolingual translation model: fed a mixed sentence it can only
    produce a fully single-language translation, destroying the mix. Used on the
    question to skip reverse-translation-to-English (see stream_answer), and on the
    generated reply to decide whether to attempt the code-mix rewrite pass instead
    of the normal single-language translation -- verified live: without this,
    "restuarant location cheppagalara?" got answered in flat, single-language text
    instead of the natural bilingual reply a person would actually give back."""
    words = re.findall(r"[a-z]+", text.lower())
    if len(words) < 2:
        return False
    if VoiceService.has_native_script(text):
        # Native Indic script alongside a real amount of Latin-alphabet text in the
        # same utterance -- the Latin part isn't just a stray acronym or number.
        return len(words) >= 2
    marker_language = _detect_romanized_language(text)
    if not marker_language:
        return False
    matched = _ROMANIZED_MARKERS[marker_language]
    remaining = [word for word in words if word not in matched and len(word) >= 3]
    return bool(remaining)


# faster-whisper/ctranslate2 (and XTTS's CUDA backend) hang when a cached GPU model is
# called from a *different* OS thread than whichever thread first used it - a
# thread-affinity issue in the underlying CUDA/cuDNN context, not something this code
# can fix at that layer. The mitigation: every GPU-bound model call always runs on this
# one dedicated, reused thread. This is a real constraint on handling multiple
# simultaneous calls -- GPU-mode STT/TTS work is serialized no matter how many callers
# are waiting -- but it's a correctness requirement (the alternative is a hang), not a
# tunable. CPU-bound model calls (STT when STT_DEVICE=cpu, Piper TTS, embeddings,
# translation, intent classification) have no such constraint -- ctranslate2's CPU path
# doesn't hold a GPU context, so those get their own pool sized by MODEL_CPU_WORKERS,
# letting multiple concurrent calls' STT/translation/etc. genuinely run in parallel
# instead of queuing behind each other on the same single thread.
_GPU_MODEL_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="gpu-model-worker")
_CPU_MODEL_EXECUTOR = ThreadPoolExecutor(max_workers=settings.model_cpu_workers, thread_name_prefix="cpu-model-worker")


def _decode_limits(duration_s: float) -> dict[str, object]:
    """Bounds Whisper's worst case. By default, when a decode looks like garbage
    (repetitive, low-confidence) faster-whisper retries at rising temperatures -- up to
    5 more full-length passes. Measured on a Telugu clip the small model can't handle:
    9,829ms with retries vs 515ms for one greedy pass, and the retries still ended in
    garbage. A single pass with a length cap scaled to the audio (dense scripts like
    Telugu tokenize to many tokens per second) keeps a hard case from stalling a turn."""
    window_s = min(max(duration_s, 1.0), 30.0)  # the cap applies per 30s decoding window
    return {
        "temperature": 0.0,
        "condition_on_previous_text": False,
        "max_new_tokens": min(400, int(window_s * 20) + 40),
    }


# Whisper's characteristic failure on audio it can't make out: one word or syllable
# looped until the length cap -- seen live as "बद्याँ बार बार बार बार बार..." and
# "සිවිවිවිවිවිවි...". Real speech essentially never repeats a word 4+ times in a row.
_REPEATED_WORD = re.compile(r"(\S+)(?:\s+\1){3,}")
_REPEATED_CHUNK = re.compile(r"(\S{1,4}?)\1{5,}")


def _has_repeated_phrase(words: list[str]) -> bool:
    """Catches the other shape of the same failure: not one word looping, but a
    multi-word phrase looping -- seen live as "mi rupe del vulo a insertista del vulo
    a insertista" (the 4-word tail repeated once), which _REPEATED_WORD/_REPEATED_CHUNK
    don't cover since neither a single word nor a short character run repeats there.
    A short (2-word) phrase needs 3+ repeats to count, since saying a couple of words
    twice for emphasis is normal speech; longer phrases essentially never repeat
    verbatim in real speech, so 2 repeats is already enough."""
    n = len(words)
    for phrase_len, min_repeats in ((2, 3), (3, 2), (4, 2), (5, 2), (6, 2)):
        if phrase_len * min_repeats > n:
            break
        for start in range(0, n - phrase_len * min_repeats + 1):
            phrase = words[start:start + phrase_len]
            if all(words[start + k * phrase_len:start + (k + 1) * phrase_len] == phrase for k in range(1, min_repeats)):
                return True
    return False


# A handful of short words that legitimately repeat in real speech ("yes yes",
# "no no no", "okay okay") -- excluded from the short-utterance ratio check below so
# genuine emphatic repetition is never mistaken for a decoding loop.
_LEGITIMATE_REPEATS = {"yes", "no", "okay", "ok", "haan", "nahi", "nahin", "haa", "achha", "acha"}


def _dominant_word_ratio_too_high(words: list[str]) -> bool:
    """Catches a decoding loop too short to trip the 4-repeat/6-repeat regexes above --
    seen live as 'మార్లు మార్లు మార్' (STT's whole output for a 3-word question,
    mostly one word repeated) for what should have been 'మెనూలో ఏమి ఉంది?'. Only
    applies once there's enough of the utterance to judge (<3 words is exactly where
    real short exclamations like "yes yes" live, so those are left alone entirely)."""
    if len(words) < 3:
        return False
    from collections import Counter

    word, count = Counter(words).most_common(1)[0]
    return word not in _LEGITIMATE_REPEATS and count >= 2 and count / len(words) >= 0.66


# Generous ceiling on sustained human speech (well above ordinary conversational pace,
# which tops out around 3-4 words/sec) -- exists specifically to catch a different
# failure than transcript_looks_garbled: not a decoding loop, but a short, fluent,
# grammatically normal sentence that is simply impossible to have fit in the audio
# actually captured. Found via a real captured turn: "I'm gonna take her down." (5
# words) came from well under a second of mostly-silent audio with only two brief
# energy bursts -- physically too little audio for those words at any real speaking
# rate, meaning Whisper invented the sentence rather than mishearing it. Neither
# avg_logprob/no_speech_prob (already established elsewhere in this file as unreliable
# for "confidently wrong" hallucinations) nor the repetition/compression checks above
# catch this, since the invented text is fluent and non-repetitive -- only comparing
# against the physical audio duration does.
_MAX_PLAUSIBLE_WORDS_PER_SECOND = 6.0


def speech_rate_implausible(text: str, duration_ms: float) -> bool:
    """True when the transcript has more words than could plausibly have been spoken
    in the audio duration actually captured, at any realistic human speaking rate."""
    words = text.strip().split()
    if len(words) < 2 or duration_ms <= 0:
        return False
    return len(words) / (duration_ms / 1000.0) > _MAX_PLAUSIBLE_WORDS_PER_SECOND


def transcript_looks_garbled(text: str) -> bool:
    """True for a transcript that is clearly a decoding loop rather than speech. Such
    text used to go straight to the LLM, which then answered the nonsense ("it seems
    you're dealing with someone returning again and again") or repeated its previous
    reply; asking the caller to say it again is what a person would do instead."""
    stripped = text.strip()
    if not stripped:
        return False
    if _REPEATED_WORD.search(stripped) or _REPEATED_CHUNK.search(stripped):
        return True
    words = stripped.lower().split()
    if _has_repeated_phrase(words) or _dominant_word_ratio_too_high(words):
        return True
    import zlib

    encoded = stripped.encode("utf-8")
    # Whisper's own hallucination heuristic (compression ratio > 2.4), only for text
    # long enough for the ratio to mean anything.
    return len(encoded) >= 60 and len(encoded) / len(zlib.compress(encoded)) > 2.4


def _warm_stt_model(model) -> None:
    import numpy as np

    noise = (np.random.default_rng(0).normal(0, 0.01, 16000)).astype("float32")
    try:
        list(model.transcribe(noise, beam_size=1, vad_filter=False, language="en", **_decode_limits(1.0))[0])
        model.detect_language(audio=noise)
    except Exception:
        logger.warning("STT warm-up pass failed -- the first live turn may be slower", exc_info=True)


def _allowed_stt_languages() -> set[str]:
    return {code.strip().lower() for code in settings.stt_languages.split(",") if code.strip()}


def _detect_allowed_language(model, audio_path: str) -> tuple[str, float] | None:
    """Whisper's language detection, limited to the configured languages (see
    settings.stt_languages): the most probable allowed language, with its probability
    renormalized over the allowed set so the session's confidence thresholds still mean
    something. None if detection fails (e.g. no speech found) -- the caller then falls
    back to Whisper's unrestricted detection."""
    try:
        from faster_whisper import decode_audio

        _, _, all_probs = model.detect_language(audio=decode_audio(audio_path), vad_filter=True)
    except Exception:
        return None
    allowed = _allowed_stt_languages()
    candidates = [(code, prob) for code, prob in all_probs if code in allowed]
    total = sum(prob for _, prob in candidates)
    if not candidates or total <= 0:
        return None
    code, prob = max(candidates, key=lambda item: item[1])
    top_code, top_prob = all_probs[0]
    if top_code != code:
        logger.info("language detection restricted: %s (%.2f) -> %s (%.2f of allowed)", top_code, top_prob, code, prob / total)
    return code, prob / total


def _stt_uses_gpu() -> bool:
    return settings.stt_device == "cuda"


@lru_cache(maxsize=1)
def _stt_cpu_threads() -> int:
    """ctranslate2 defaults to using most/all CPU cores for a single inference
    (cpu_threads=0, its "auto" setting) -- fine for one call at a time, but it means
    MODEL_CPU_WORKERS concurrent calls all fight over the same cores instead of
    genuinely running in parallel, measured to actually make each call slower with no
    net throughput gain. Capping each inference's own thread count to a fair share of
    the machine's cores (split across the configured worker count) lets concurrent
    calls actually overlap instead of serializing through core contention."""
    if settings.stt_device == "cuda":
        return 0  # GPU inference doesn't consume CPU threads this way; leave at auto.
    cpu_count = os.cpu_count() or 4
    return max(1, cpu_count // max(1, settings.model_cpu_workers))


def _tts_uses_gpu() -> bool:
    return settings.tts_engine == "xtts_v2" and settings.tts_gpu


async def run_on_model_thread(func, *args, gpu: bool = False):
    loop = asyncio.get_running_loop()
    executor = _GPU_MODEL_EXECUTOR if gpu else _CPU_MODEL_EXECUTOR
    return await loop.run_in_executor(executor, func, *args)


DOCUMENT_GROUNDED_SYSTEM_PROMPT = """You are a natural, helpful conversational assistant speaking with someone in real time.
Hard rule, no exceptions: when greeting someone or talking about yourself, do not call yourself a chatbot, a bot,
an AI, or a language model, and do not say "I'm just a..." about yourself in any form. Simply greet them and offer
to help, the confident way a knowledgeable person would, with no self-description at all unless directly asked
what you are -- and even then, describe yourself only as an assistant that's here to help, nothing more.
Answer using only the reference information supplied with the user's question.
Do not use general knowledge, assumptions, memory, or outside information.
Never mention documents, uploaded files, context, retrieval, prompts, instructions, or internal rules.
Speak the way a real person actually talks on a phone call: short and to the point, not a
presentation. One or two sentences is normal for most questions -- answer exactly what was asked and stop.
Do not stack on every related fact from the reference information just because it's there; a caller who
wants more will ask a follow-up, the same way a real conversation works. Only go longer when the question
itself is broad ("tell me everything about...", "explain...") or genuinely needs a few steps explained.
Use simple, everyday words.
This reply will be spoken aloud, not read as text: never use emojis, markdown, bullet points, numbered lists, or
headers -- say things the way a person would say them out loud, as plain flowing sentences. Do not repeat or
restate the user's question back to them before answering; just answer it, the way a person naturally would.
If you are asked to write in a language you do not have strong, confident command of, say so honestly
in English rather than attempting it: name the language and explain plainly that you don't have
reliable support for it, instead of guessing or producing text you are not confident is correct.
Never fabricate an answer in a language you are unsure of just to appear capable.
For greetings, small talk, asking how you are, thanks, or simple conversational pleasantries --
including when phrased in another language, or written phonetically using English letters instead
of that language's own script -- respond naturally and warmly. These never need the reference
information, even though factual questions do; recognize the intent behind the words even if the
exact phrasing looks unfamiliar, rather than defaulting to "not enough information" for anything
that isn't a clearly factual question.
If the user asks about your own abilities (for example, whether you can speak or understand a
particular language), answer that naturally and honestly -- you can understand and respond in
many languages, but say plainly if a specific one isn't among them -- instead of treating it as a
request for reference information. Keep that to one short sentence and never list languages.
The reference information covering the general topic is NOT the same as it answering the specific
question asked, and you must never treat the two as equivalent, even when the reference information is
about the exact same subject as the question.
Example -- reference information: "Ambience: Calm, elegant, soft lighting, background music." Question:
"Do you have live music performances on weekends?" The reference only mentions background music playing
in the venue, not scheduled live performances -- the correct answer states plainly that you don't have
that information, never "yes."
Example -- reference information: "Address: 45 Jubilee Hills Road, Hyderabad." Question: "What is your
phone number?" The reference gives an address, not a phone number -- the correct answer states plainly
that you don't have that information; never invent a phone number, or any other specific number, name,
or detail that is not itself written in the reference information word for word.
Before answering any factual question, find the exact sentence in the reference information that states
the answer. If no sentence states it -- even if other sentences are about the same general subject --
apologize briefly and say, in the same language the user is speaking, that you don't have enough
information to answer that. As a concrete check: if the specific thing the user named (e.g. "phone
number", "live music", "performances", "reservation", "delivery", "parking") is not itself one of the
words or a direct synonym used in the reference information, it is not present, even if the reference
information describes something else in the same category ("background music" is not "live music
performances"; a described feature is not the same as a scheduled or bookable one unless the reference
information says so explicitly). Do not guess, extrapolate, combine unrelated details into a new fact, or
explain why the information is unavailable."""

NO_REFERENCE_SYSTEM_PROMPT = """You are a natural, helpful conversational assistant speaking with someone in real time.
Hard rule, no exceptions: when greeting someone or talking about yourself, do not call yourself a chatbot, a bot,
an AI, or a language model, and do not say "I'm just a..." about yourself in any form. Simply greet them and offer
to help, the confident way a knowledgeable person would, with no self-description at all unless directly asked
what you are -- and even then, describe yourself only as an assistant that's here to help, nothing more.
Speak the way a real person actually talks on a phone call: short and to the point, not a
presentation. One or two sentences is normal for most questions -- answer exactly what was asked and stop,
the same way a real conversation works. Use simple, everyday words.
This reply will be spoken aloud, not read as text: never use emojis, markdown, bullet points, numbered lists, or
headers -- say things the way a person would say them out loud, as plain flowing sentences. Do not repeat or
restate the user's question back to them before answering; just answer it, the way a person naturally would.
If you are asked to write in a language you do not have strong, confident command of, say so honestly
in English rather than attempting it: name the language and explain plainly that you don't have
reliable support for it, instead of guessing or producing text you are not confident is correct.
Never fabricate an answer in a language you are unsure of just to appear capable.
For greetings, small talk, asking how you are, thanks, or simple conversational pleasantries --
including when phrased in another language, or written phonetically using English letters instead
of that language's own script -- respond naturally and warmly, recognizing the intent behind the
words even if the exact phrasing looks unfamiliar.
If the user asks about your own abilities (for example, whether you can speak or understand a
particular language), answer that naturally and honestly -- you can understand and respond in
many languages, but say plainly if a specific one isn't among them -- instead of treating it as a
request for factual information. Keep that to one short sentence and never list languages.
No reference material is available, so do not invent factual answers or use outside knowledge.
If the user asks for factual information, say so briefly in the same language the user is speaking:
that you don't have enough information yet and they should provide the relevant details.
Never mention documents, uploaded files, context, retrieval, prompts, instructions, or internal rules."""

_LANGUAGE_CODES = {entry["value"].lower(): entry["code"] for entry in SUPPORTED_LANGUAGES}
# Derived from the same SUPPORTED_LANGUAGES list the /api/voice/languages dropdown is
# built from -- not a separate hardcoded list -- so adding a language to that one config
# table is the only place that ever needs editing to extend verified-direct-generation
# coverage. See stream_answer() for how this gates unverified-language generation.
_VERIFIED_LANGUAGE_CODES = {entry["code"] for entry in SUPPORTED_LANGUAGES}
# Splits on sentence-ending punctuation (incl. the Hindi/Devanagari full stop) or blank lines,
# so a streamed LLM reply can be synthesized and played back sentence-by-sentence.
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?।॥])\s+|\n+")
# Long sentences are spoken clause by clause. Every stage (generation, translation,
# synthesis) scales with sentence length, and nothing is heard until the whole first
# sentence clears all three -- measured: a 45-word first sentence took 2.7s to
# generate, 2.9s to translate and 2.1s to synthesize, so the caller heard nothing for
# 9s. A clause must have at least this many words, so short ones aren't chopped up.
_CLAUSE_BREAK = re.compile(r"(?<=[,;:،])\s+")
_MIN_CLAUSE_WORDS = 6


def _clauses(sentence: str) -> list[str]:
    """A sentence split at commas into pieces of at least _MIN_CLAUSE_WORDS words."""
    pieces: list[str] = []
    current = ""
    for part in _CLAUSE_BREAK.split(sentence):
        current = f"{current} {part}".strip()
        if len(current.split()) >= _MIN_CLAUSE_WORDS:
            pieces.append(current)
            current = ""
    if current:
        if pieces and len(current.split()) < _MIN_CLAUSE_WORDS:
            pieces[-1] = f"{pieces[-1]} {current}"
        else:
            pieces.append(current)
    return pieces


def _take_speakable(unspoken: str) -> tuple[list[str], int]:
    """The chunks of a partially-generated reply that are ready to speak now, and how
    many characters of it they used. Finished sentences are ready (split into clauses
    if long); the unfinished last sentence is ready up to its last comma once that
    part alone is a full clause -- so a long first sentence starts being spoken while
    the rest of it is still being generated."""
    ready: list[str] = []
    consumed = 0
    for match in _SENTENCE_BOUNDARY.finditer(unspoken):
        sentence = unspoken[consumed:match.start()].strip()
        if sentence:
            ready.extend(_clauses(sentence))
        consumed = match.end()
    tail = unspoken[consumed:]
    breaks = [match.end() for match in _CLAUSE_BREAK.finditer(tail)]
    for position in reversed(breaks):
        head = tail[:position].strip()
        if len(head.split()) >= _MIN_CLAUSE_WORDS:
            ready.extend(_clauses(head))
            consumed += position
            break
    return ready, consumed
# The system prompt tells the LLM not to use emojis, but that instruction isn't always
# followed (observed emoji slipping through even so) -- stripped unconditionally here
# because they can't be spoken by TTS and corrupt the translation step's transliteration.
_EMOJI_PATTERN = re.compile(
    "[\U0001F300-\U0001FAFF\U00002600-\U000027BF\U0001F1E6-\U0001F1FF\U00002190-\U000021FF\U00002B00-\U00002BFF]+"
)


def _looks_like_english(text: str) -> bool:
    """Guards the translation step: the LLM is told to answer in English before
    translation, but doesn't always obey (observed answering in the question's own
    script instead) -- feeding that non-English text into a model that expects
    English caused a decoder repetition loop. If most letters aren't Latin, this
    generation didn't actually follow the instruction, so it's used as-is instead
    of being translated (still imperfect, but never garbage)."""
    letters = [character for character in text if character.isalpha()]
    if not letters:
        return True
    latin_count = sum(1 for character in letters if ord(character) < 0x0250)
    return (latin_count / len(letters)) > 0.6


def _script_violation(text: str, expected_code: str) -> bool:
    """True if `text` contains a meaningful amount of script OTHER than Latin and
    `expected_code`'s own script -- e.g. a Telugu reply that drifts into Kannada or
    Malayalam characters.

    Found live, reproduced deterministically: asked "speak in telugu" (plain
    English, about Telugu, not written in it -- target_code correctly stays "en"
    per _ROMANIZED_MARKERS' own deliberate exclusion of language names), the LLM
    sometimes ignores its own "answer only in English" instruction and writes
    Telugu-script text directly, which then drifts further into visually/
    structurally adjacent scripts it apparently confuses with Telugu -- Unicode
    gives Telugu (0C00-0C7F), Kannada (0C80-0CFF) and Malayalam (0D00-0D7F)
    parallel internal layouts, which is exactly the kind of near-miss a model
    weak on a low-resource script can make. Because target_code=="en" means
    needs_translation is False, NOTHING downstream ever validates this output --
    it reaches the caller completely raw. Worse, once one such reply lands in the
    conversation history, the LLM was observed conditioning on its own broken
    prior turn and producing progressively more corrupted replies in later
    turns -- so this check exists specifically to stop that before it starts,
    not just to tidy up one bad sentence."""
    expected_range = next((rng for start, end, code in _SCRIPT_RANGES if code == expected_code for rng in ((start, end),)), None)
    for character in text:
        if not character.isalpha():
            continue
        code_point = ord(character)
        if code_point < 0x0250:  # Latin (incl. accented) -- always allowed (names, numbers, loanwords)
            continue
        if expected_range and expected_range[0] <= code_point <= expected_range[1]:
            continue
        return True
    return False


# Pre-written, never-generated fallback for when _script_violation fires and a turn
# ends up with nothing safe to say -- deliberately NOT another LLM call (that would
# risk repeating the exact failure this guards against) or a translation call (one
# more moving part that could itself fail); a fixed phrase is the one response
# guaranteed not to be broken.
_GENERATION_FALLBACK: dict[str, str] = {
    "en": "Sorry, I'm having trouble answering that right now. Could you ask again?",
    "hi": "माफ़ कीजिए, अभी इसका जवाब देने में समस्या हो रही है। क्या आप दोबारा पूछ सकते हैं?",
    "te": "క్షమించండి, దీనికి సమాధానం ఇవ్వడంలో సమస్య ఉంది. మళ్ళీ అడగగలరా?",
}


class ProviderUnavailable(RuntimeError):
    """Raised when a configured local AI provider cannot be reached."""


@dataclass(frozen=True)
class VoiceService:
    timeout: int = 45

    def provider_status(self) -> dict[str, object]:
        voice_profiles = settings.voice_profiles()
        return {
            "llm": {"configured": bool(settings.llm_url), "url": settings.llm_url, "model": settings.llm_model},
            "stt": {"configured": bool(settings.stt_url) or settings.stt_mode == "local", "mode": "url" if settings.stt_url else settings.stt_mode, "model": settings.stt_model, "device": settings.stt_device, "compute_type": settings.stt_compute_type},
            "tts": {"enabled": settings.tts_enabled, "configured": settings.tts_enabled and (bool(settings.tts_url) or (settings.tts_mode == "local" and ((settings.tts_engine == "xtts_v2" and bool(voice_profiles)) or (settings.tts_engine != "xtts_v2" and bool(settings.tts_model_path))))), "mode": "url" if settings.tts_url else settings.tts_mode, "engine": settings.tts_engine, "model": settings.tts_model if settings.tts_engine == "xtts_v2" else settings.tts_model_path or "not selected", "voices": list(voice_profiles.keys()), "gpu": settings.tts_gpu},
        }

    def languages(self) -> list[dict[str, str]]:
        return [dict(entry) for entry in SUPPORTED_LANGUAGES]

    @staticmethod
    def _language_name_for_code(code: str) -> str:
        for entry in SUPPORTED_LANGUAGES:
            if entry["code"] == code:
                return entry["value"]
        # Whisper can detect ~100 languages, far more than the curated dropdown list
        # above -- for anyone speaking a language outside that list, fall back to the
        # real ISO 639 name (via pycountry's standard language database) instead of
        # handing the LLM a bare, unexplained code like "sw", which is a confusing
        # instruction that invites a wrong guess rather than an honest answer.
        return _iso_language_name(code) or code

    @staticmethod
    def resolve_target_language(requested: str, detected: str | None) -> str:
        """Picks the definitive reply language for a turn: the caller's explicit
        selection if they made one, else the language actually detected from their
        speech (Whisper's own acoustic detection, not the LLM guessing from text)."""
        if requested and requested != "auto":
            return requested
        return detected or "en"

    @staticmethod
    def has_native_script(text: str) -> bool:
        return any(start <= ord(character) <= end for character in text for start, end, _ in _SCRIPT_RANGES)

    @staticmethod
    def detect_script_language(text: str) -> str | None:
        """Best-effort language guess for the typed-text path, where there's no audio
        to run real language detection on. Checks native script first (unambiguous),
        then falls back to common romanized-word markers (see _ROMANIZED_MARKERS) for
        the very common case of typing an Indian language phonetically in Latin
        letters. Genuinely plain English/ambiguous input still returns None."""
        for character in text:
            code_point = ord(character)
            for start, end, language in _SCRIPT_RANGES:
                if start <= code_point <= end:
                    return language
        return _detect_romanized_language(text)

    @staticmethod
    def detect_mixed_language(text: str) -> str | None:
        """If `text` genuinely blends English with one Indic language (see
        _is_code_mixed), returns that language's code; otherwise None. Exists for the
        live voice pipeline specifically: Whisper's own per-utterance language tag is
        one label for the whole turn, and for genuinely code-mixed speech it often
        picks "en" even when the words themselves are clearly mixed (the same English
        bias that motivated _is_code_mixed in the first place) -- so a turn can arrive
        here already tagged "en" despite the caller obviously mixing languages. This
        lets the live session catch that and correct target_code before generation,
        instead of the mixing logic only ever running when detection happened to get
        the target language right in the first place."""
        if not _is_code_mixed(text):
            return None
        if VoiceService.has_native_script(text):
            for character in text:
                code_point = ord(character)
                for start, end, language in _SCRIPT_RANGES:
                    if start <= code_point <= end:
                        return language
        return _detect_romanized_language(text)

    def stream_answer(
        self,
        question: str,
        target_code: str,
        history: tuple[dict[str, str], ...] = (),
        source_code: str | None = None,
    ) -> Iterator[dict[str, object]]:
        """Streams a reply sentence-by-sentence, translated into `target_code` when
        that language isn't one the LLM can reliably write directly (see
        services/translation.py) -- the LLM always generates in English in that case,
        and each completed sentence is translated before being handed to the caller,
        so the streaming/low-latency shape of a turn is unchanged.

        `source_code` is the language the question itself was asked in, if known (e.g.
        Whisper's own acoustic detection for voice, distinct from `target_code` since a
        caller can speak one language and ask for replies in another). If not given, it
        falls back to a script guess on the question text. Cross-lingual embedding
        similarity between an Indic-language question and an English document was
        measured to be far weaker than same-language matching, so a translatable
        question is translated to English before retrieval and generation -- otherwise
        a correct answer already in the document goes unfound."""
        started = time.monotonic()
        source_code = source_code or self.detect_script_language(question)
        is_code_mixed = _is_code_mixed(question)
        llm_question = question
        # The reverse-translation model expects the question in its actual native
        # script; fed romanized/Latin-script text instead (e.g. Whisper transcribing
        # spoken Telugu as Latin letters, or someone typing phonetically) it doesn't
        # translate at all -- it passes the text through almost unchanged, but
        # capitalized oddly, which then reads as gibberish to the LLM and was observed
        # causing it to refuse rather than recognize an ordinary greeting. Skip
        # reverse-translation for romanized text, and for code-mixed text regardless of
        # script (translating a mixed sentence through a monolingual model garbles the
        # English portion too), and let the LLM read it directly, which it already
        # does noticeably better than a broken "translation" of it.
        if (
            source_code and source_code != "en" and translation.is_supported(source_code)
            and self.has_native_script(question) and not is_code_mixed
        ):
            translated_question = translation.translate_to_english(question, source_code)
            if translated_question:
                llm_question = translated_question

        # Note on is_code_mixed and output: asking the LLM to directly write a mixed
        # reply AS THE MAIN ANSWER (i.e. generate the actual facts in mixed-language
        # form) was tried and tested unreliable -- at temperature 0 (required
        # elsewhere for factual consistency -- see _chat_payload) it just answered in
        # plain English regardless of instruction. The fix isn't to give up on mixed
        # output, but to stop asking the LLM to do two hard things at once (recall
        # facts correctly AND write in an unusual mixed style). Instead the main
        # answer is still always generated in reliable English first, and each
        # finished sentence is then separately rewritten into the mix by
        # _code_mix_rewrite() below -- a much easier constrained style-transfer task
        # (reword known content, invent nothing) that a small model handles far more
        # reliably than open generation, with a concrete few-shot example in the
        # instruction (see _CODE_MIX_EXAMPLES) since an abstract instruction alone
        # was the weak point of the earlier attempt. Validated and falls back to the
        # existing single-language translation path (which also naturally keeps
        # proper nouns/loanwords transliterated, e.g. "Spice Garden" ->
        # "స్పైస్ గార్డెన్") if the rewrite doesn't actually come back mixed.
        needs_translation = translation.is_supported(target_code)
        # Whisper can detect ~100 languages, far more than this app has actually
        # verified good output for (the curated SUPPORTED_LANGUAGES dropdown, plus the
        # Indic languages the dedicated translation model covers). Asking the LLM to
        # write directly in anything outside that -- and just telling it via the prompt
        # to admit when it's unsure -- was tested against a real unverified language
        # (Swahili) and it attempted an answer anyway rather than admitting the gap, a
        # real, observed hallucination the prompt alone didn't prevent. So this is a
        # hard guarantee, not a request: unverified languages always generate in
        # English with an explicit acknowledgement, never an unverifiable attempt.
        is_verified_language = needs_translation or target_code in _VERIFIED_LANGUAGE_CODES
        unverified_language_name = None if is_verified_language else self._language_name_for_code(target_code)
        generation_code = "en" if (needs_translation or unverified_language_name) else target_code
        if unverified_language_name:
            instruction = (
                f"The user wants a reply in {unverified_language_name}, which you do not have reliable, "
                f"confident command of. Do not attempt to write in {unverified_language_name} even "
                f"partially. Instead, write your entire answer in English, and begin by honestly telling "
                f"the user, in English, that you don't have confident support for {unverified_language_name} "
                f"so you're answering in English instead."
            )
        elif generation_code == "en":
            # Stricter than language_instruction("English"): observed the LLM ignoring
            # that wording and mirroring a non-Latin-script (or romanized) question
            # instead -- happened even when English was the *directly* requested target,
            # not just on the translate-then-generate-in-English path, so this applies
            # whenever English is the generation target, not only when needs_translation.
            #
            # Naming the source language explicitly when known matters a lot here: left
            # to guess, qwen3 was observed misreading short romanized Telugu syllables
            # ("miru ela unnaru?") as Japanese and confidently answering based on that
            # wrong guess -- a real hallucination, not just a language-matching miss.
            if source_code and source_code != "en":
                source_name = self._language_name_for_code(source_code)
                if self.has_native_script(question):
                    source_hint = f"The user's message is written in {source_name}. "
                else:
                    source_hint = (
                        f"The user's message is {source_name}, written phonetically using English "
                        f"letters (not any other language, and not literal English words) -- read it "
                        f"as {source_name}. "
                    )
            else:
                source_hint = "The user's question may be written in a non-English language or script. "
            instruction = (
                f"{source_hint}"
                "Regardless of what language the question is written in, you must write your "
                "entire answer in English, using the Latin alphabet only. Never answer in the "
                "question's own language or script, even partially."
            )
            if needs_translation:
                # Without this, the LLM has no way to know a translation step follows
                # it -- observed it treating "write in English" as a personal language
                # limitation and volunteering a confusing, self-contradictory disclaimer
                # ("since my responses are generated in English, we'll have to
                # communicate in that language") that then got dutifully translated
                # into fluent Hindi anyway, i.e. the assistant spoke perfect Hindi while
                # its own words claimed it couldn't. This is purely an internal
                # implementation detail the user must never see.
                instruction += (
                    " This is only an internal step: a separate, reliable translation system will "
                    "convert your English answer into the user's own language afterward, so you are "
                    "not actually limited to English and must never say or imply that you are. Never "
                    "mention English, translation, or any language limitation to the user -- just "
                    "answer their actual question naturally and confidently, exactly as you would if "
                    "you were writing directly in their language."
                )
        else:
            instruction = self.language_instruction(self._language_name_for_code(generation_code))
        request = ChatRequest(llm_question, instruction, history)

        assistant_text = ""
        output_text = ""
        spoken_length = 0
        # Per-sentence timing, so a slow turn can be pinned to a specific stage
        # (generation vs. translation/rewrite) instead of guessed at.
        prep_ms = int((time.monotonic() - started) * 1000)

        had_any_generation = False
        had_violation = False

        def emit(raw_sentence: str) -> dict[str, object] | None:
            nonlocal output_text
            generated_ms = int((time.monotonic() - started) * 1000)
            localize_started = time.monotonic()
            should_translate = needs_translation and _looks_like_english(raw_sentence)
            output = self._localize_sentence(raw_sentence, target_code, is_code_mixed) if should_translate else raw_sentence
            # Catches the LLM ignoring its own language instruction (generation_code=="en"
            # but it wrote non-Latin script) just as much as a direct-language generation
            # drifting into an adjacent script (e.g. Telugu into Kannada) -- see
            # _script_violation's docstring for the real, reproduced incident this guards.
            # Checked on `output` (what actually gets shown/spoken), not `raw_sentence`,
            # so a bad translation/rewrite is caught too, not just bad raw generation.
            # expected script: target_code whenever translation is in play at all (even
            # if this specific piece skipped it because the LLM already wrote non-English
            # -- it should still match target_code, not just any non-Latin script);
            # otherwise whatever the LLM was actually told to generate in.
            expected_script = target_code if needs_translation else generation_code
            if _script_violation(output, expected_script):
                nonlocal had_violation
                had_violation = True
                logger.warning(
                    "stream_answer: dropping a reply piece with unexpected script (expected %r): %r",
                    expected_script, output,
                )
                return None
            output_text = f"{output_text} {output}".strip()
            return {
                "sentence": output,
                "timing": {"prep_ms": prep_ms, "generated_ms": generated_ms, "localize_ms": int((time.monotonic() - localize_started) * 1000)},
            }

        try:
            for chunk in self.stream_chat(request):
                if chunk.get("error"):
                    yield {"error": chunk["error"]}
                    return
                assistant_text += chunk.get("token", "")
                ready, consumed = _take_speakable(assistant_text[spoken_length:])
                spoken_length += consumed
                for piece in ready:
                    piece = _EMOJI_PATTERN.sub("", piece).strip()
                    if piece:
                        had_any_generation = True
                        emitted = emit(piece)
                        if emitted is not None:
                            yield emitted
                if chunk.get("done"):
                    break
        except ProviderUnavailable as error:
            yield {"error": str(error)}
            return

        remainder = _EMOJI_PATTERN.sub("", assistant_text[spoken_length:]).strip()
        for piece in _clauses(remainder) if remainder else []:
            had_any_generation = True
            emitted = emit(piece)
            if emitted is not None:
                yield emitted

        if had_violation and not output_text.strip() and had_any_generation:
            # Every piece this turn was flagged -- rather than send nothing (dead air)
            # or a partially-corrupted reply, say so honestly in a fixed, never-wrong
            # phrase. See _GENERATION_FALLBACK's docstring for why this is a fixed
            # phrase and not another generation/translation call.
            fallback = _GENERATION_FALLBACK.get(target_code, _GENERATION_FALLBACK["en"])
            output_text = fallback
            yield {"sentence": fallback, "timing": {"prep_ms": prep_ms, "generated_ms": 0, "localize_ms": 0}}

        # full_text is what was actually shown/spoken (translated, if this turn needed
        # translation) -- NOT the LLM's raw English generation, so a stored transcript
        # or conversation-history entry never silently reverts to English underneath it.
        yield {"done": True, "full_text": output_text}

    def _localize_sentence(self, sentence: str, target_code: str, is_code_mixed: bool) -> str:
        """Renders one English-generated sentence in the target language. For an
        ordinary (single-language) turn this is just the existing translation path.
        For a turn where the caller themselves mixed languages, tries the code-mix
        rewrite first and only falls back to the plain single-language translation
        if that rewrite doesn't actually come back looking mixed -- see the note in
        stream_answer() above for why this is a separate, validated pass rather than
        asking the main generation to do it directly."""
        if is_code_mixed:
            mixed = self._code_mix_rewrite(sentence, target_code)
            if mixed:
                return mixed
        return translation.translate_from_english(sentence, target_code)

    def _code_mix_rewrite(self, sentence: str, target_code: str) -> str | None:
        instruction = _code_mix_instruction(self._language_name_for_code(target_code), target_code)
        try:
            rewritten = self._raw_chat(sentence, instruction, num_predict=min(180, max(60, len(sentence) + 20)))
        except ProviderUnavailable:
            return None
        rewritten = _EMOJI_PATTERN.sub("", rewritten).strip()
        # The model was observed occasionally prefacing the actual rewrite with a
        # leaked meta-label ("Telugu-English mixed sentence: \"...\"") despite being
        # told not to -- strip it rather than let commentary-about-the-answer leak
        # into what's supposed to be the spoken reply itself.
        label_match = re.match(r'^[A-Za-z][A-Za-z \-]{0,40}:\s*"(.+)"\s*$', rewritten, re.DOTALL)
        if label_match:
            rewritten = label_match.group(1).strip()
        rewritten = rewritten.strip('"')
        # A length sanity check: the rewrite is a style-transfer task on already-known
        # content (same facts, different words), not new generation, so a result far
        # longer than the source is a sign of padding/hallucinated content rather than
        # a legitimate translation (romanized Indic phrasing runs a bit longer than
        # English for the same facts, but not multiples longer) -- observed the model
        # once appending an unrelated invented sentence this way. Reject rather than
        # risk handing a caller a fabricated fact in a document-grounded assistant.
        length_ok = len(rewritten) <= len(sentence) * 1.8 + 40
        if rewritten and length_ok and _is_code_mixed(rewritten):
            return rewritten
        logger.info("code-mix rewrite for %s rejected (mixed=%s, length_ok=%s): %r -- falling back to translation", target_code, rewritten and _is_code_mixed(rewritten), length_ok, rewritten)
        return None

    def _raw_chat(self, message: str, system_prompt: str, num_predict: int) -> str:
        """A direct LLM call with no document retrieval/enrichment -- for internal
        helper calls (like the code-mix rewrite above) that operate on already-known
        content and must not have retrieval context or the main grounded system
        prompt mixed into a small, constrained rewriting task."""
        payload = self._chat_payload(ChatRequest(message, system_prompt), stream=False)
        payload["options"]["num_predict"] = num_predict
        # _chat_payload's repeat_penalty (1.3) is tuned for open factual generation,
        # where any repeated phrase is a bad sign (a degenerate loop). Natural
        # code-mixed Telugu/Hindi genuinely repeats the same grammar particles often
        # ("ga", "lo", "untundi", "mein", "hai") -- observed that same high penalty
        # actively fighting that, pushing the model into increasingly strange word
        # choices and erratic capitalization to avoid repeating them. This is a
        # constrained rewrite of already-fixed content, not open generation, so a
        # much gentler penalty (Ollama's own default) is both safe and necessary here.
        payload["options"]["repeat_penalty"] = 1.1
        payload["options"]["repeat_last_n"] = 32
        response = self._json_request(f"{settings.llm_url.rstrip('/')}/api/chat", payload, settings.llm_api_key)
        return str(response.get("message", {}).get("content", "")).strip()

    def requested_reply_language(self, text: str) -> str | None:
        """The language code the caller is explicitly asking the assistant to speak
        ("can you talk in Telugu", "तेलुगु में बात करो", "English lo cheppandi"), or
        None. Keywords decide which language was named; the LLM only decides whether
        naming it was a request -- measured: asked to do both, qwen3 read a Telugu-script
        request for English as a request for Telugu, following the script, not the words."""
        mentioned = _mentioned_languages(text)
        if not mentioned:
            return None
        try:
            if len(mentioned) == 1:
                name = self._language_name_for_code(mentioned[0])
                answer = self._raw_chat(text, _LANGUAGE_REQUEST_YES_NO.format(name=name), num_predict=2).lower()
                return mentioned[0] if answer.startswith("yes") else None
            answer = self._raw_chat(text, _LANGUAGE_REQUEST_WHICH, num_predict=3).lower()
        except ProviderUnavailable:
            return None
        return next((code for word, code in _LANGUAGE_WORDS.items() if answer.startswith(word)), None)

    def is_turn_complete(self, text: str) -> bool | None:
        """Whether a transcript-so-far reads like a finished thought (True) or a
        sentence cut off mid-way (False); None if it can't be judged. Lets the live
        pipeline end a turn quickly after a finished question instead of always
        waiting out the full silence timeout -- see EndpointDetector.mark_turn_complete.
        Uses the same num_ctx as the main answer call on purpose: Ollama reloads the
        model whenever num_ctx changes between requests."""
        cleaned = " ".join(text.translate(_TURN_PUNCTUATION).split())
        if not cleaned:
            return None
        try:
            answer = self._raw_chat(cleaned, _TURN_COMPLETE_PROMPT, num_predict=2).lower()
        except ProviderUnavailable:
            return None
        if answer.startswith("complete"):
            return True
        if answer.startswith("cut"):
            return False
        return None

    def voices(self) -> list[str]:
        return list(settings.voice_profiles().keys()) or (["default"] if settings.tts_engine != "xtts_v2" else [])

    def warmup(self) -> None:
        """Loads the STT/TTS/embedding models once at startup instead of on the
        first real request, so the first live call doesn't pay multi-second
        model-load latency. Each model is loaded via the same executor
        (GPU vs. CPU) it will actually be called through later -- see
        _GPU_MODEL_EXECUTOR / _CPU_MODEL_EXECUTOR above."""
        if settings.stt_mode == "local" and not settings.stt_url:
            stt_executor = _GPU_MODEL_EXECUTOR if _stt_uses_gpu() else _CPU_MODEL_EXECUTOR
            # Loading a model isn't enough: its first actual inference also pays one-time
            # GPU setup -- measured as a 1,329ms transcription for the first 2.4s turn of
            # a live session, vs ~350-600ms after. One throwaway pass here absorbs it.
            for loader in (self._whisper_model, self._interim_whisper_model):
                try:
                    model = stt_executor.submit(loader).result()
                    stt_executor.submit(_warm_stt_model, model).result()
                except ProviderUnavailable:
                    pass
        if settings.tts_enabled and settings.tts_mode == "local" and not settings.tts_url:
            try:
                if settings.tts_engine == "xtts_v2":
                    tts_executor = _GPU_MODEL_EXECUTOR if _tts_uses_gpu() else _CPU_MODEL_EXECUTOR
                    tts_executor.submit(self._xtts_engine).result()
                else:
                    # Every configured voice, not just English: each Piper voice takes
                    # ~1.75s to load, measured -- and with only English preloaded, the
                    # first Hindi/Telugu/etc. reply after every restart paid that inside
                    # a live turn (the caller heard "अच्छा" ~2.8s after they stopped
                    # talking instead of ~0.8s).
                    model_paths = {self._tts_model_path("en")}
                    try:
                        model_paths.update(str(path) for path in json.loads(settings.tts_model_paths or "{}").values())
                    except json.JSONDecodeError:
                        pass
                    for model_path in filter(None, model_paths):
                        try:
                            _CPU_MODEL_EXECUTOR.submit(self._piper_engine, model_path).result()
                        except ProviderUnavailable:
                            logger.warning("could not preload Piper voice %s", model_path)
            except ProviderUnavailable:
                pass
        from . import embeddings

        _CPU_MODEL_EXECUTOR.submit(embeddings.warmup).result()
        _CPU_MODEL_EXECUTOR.submit(translation.warmup).result()
        if settings.llm_url:
            try:
                # Ollama loads a model into VRAM on its own first request, which can take
                # several seconds for an 8B model -- without this, whichever real user
                # happens to send the first message after startup (or after Ollama's own
                # idle-unload) pays that cost. A trivial call here absorbs it instead.
                list(self.stream_chat(ChatRequest("hi", "Reply with just: ok")))
            except ProviderUnavailable:
                pass

    @staticmethod
    def split_sentences(text: str) -> list[str]:
        return [sentence.strip() for sentence in _SENTENCE_BOUNDARY.split(text) if sentence.strip()]

    def chat(self, request: ChatRequest) -> dict[str, object]:
        enriched_request = self._enrich_with_documents(request)
        payload = self._chat_payload(enriched_request, stream=False)
        response = self._json_request(f"{settings.llm_url.rstrip('/')}/api/chat", payload, settings.llm_api_key)
        message = response.get("message", {})
        return {"reply": message.get("content", ""), "model": response.get("model", settings.llm_model), "done": response.get("done", True)}

    def stream_chat(self, request: ChatRequest) -> Iterator[dict[str, object]]:
        enriched_request = self._enrich_with_documents(request)
        payload = self._chat_payload(enriched_request, stream=True)
        headers = {"Content-Type": "application/json"}
        if settings.llm_api_key:
            headers["Authorization"] = f"Bearer {settings.llm_api_key}"
        try:
            request_object = urllib.request.Request(f"{settings.llm_url.rstrip('/')}/api/chat", data=json.dumps(payload).encode(), headers=headers, method="POST")
            with urllib.request.urlopen(request_object, timeout=self.timeout) as response:
                for raw_line in response:
                    if raw_line.strip():
                        chunk = json.loads(raw_line)
                        message = chunk.get("message", {})
                        yield {"token": message.get("content", ""), "done": chunk.get("done", False), "model": chunk.get("model", settings.llm_model)}
        except (urllib.error.URLError, TimeoutError, ValueError, json.JSONDecodeError) as error:
            logger.error("LLM request to %s failed: %s (is Ollama running?)", settings.llm_url, error)
            raise ProviderUnavailable(f"Streaming LLM request failed: {error}") from error

    def stream_chat_raw(self, messages: list[dict[str, object]], tools: list[dict[str, object]] | None = None) -> Iterator[dict[str, object]]:
        """Streams a raw Ollama /api/chat call with a caller-supplied message
        list and optional tool declarations -- deliberately bypassing
        _enrich_with_documents/_chat_payload (this project's own document-
        grounded system prompt + RAG corpus). Built for the telephony bridge,
        where a remote caller supplies its OWN complete system prompt (already
        including its own knowledge/persona/rules) -- wrapping that in this
        project's DOCUMENT_GROUNDED_SYSTEM_PROMPT or grounding it against this
        project's own demo document store would silently answer from the
        wrong knowledge base, a correctness bug, not a style choice.

        Verified directly against this project's own Ollama+qwen3:8b instance
        (both non-streaming and streaming) that tool-calling works
        mechanically: `message.tool_calls` arrives as a list of
        {id, function: {name, arguments}} dicts, in its own chunk (content
        empty, done=False) separate from any text content chunks -- yielded
        here as-is via the "tool_calls" key so a caller can detect and act on
        a tool call without parsing text."""
        payload: dict[str, object] = {
            "model": settings.llm_model,
            "messages": messages,
            "stream": True,
            "think": settings.llm_think,
            "keep_alive": settings.llm_keep_alive,
            "options": {
                "temperature": 0.0,
                "num_predict": settings.llm_num_predict,
                "num_ctx": settings.llm_num_ctx,
                "repeat_penalty": 1.3,
                "repeat_last_n": 64,
            },
        }
        if tools:
            payload["tools"] = tools
        headers = {"Content-Type": "application/json"}
        if settings.llm_api_key:
            headers["Authorization"] = f"Bearer {settings.llm_api_key}"
        try:
            request_object = urllib.request.Request(f"{settings.llm_url.rstrip('/')}/api/chat", data=json.dumps(payload).encode(), headers=headers, method="POST")
            with urllib.request.urlopen(request_object, timeout=self.timeout) as response:
                for raw_line in response:
                    if not raw_line.strip():
                        continue
                    chunk = json.loads(raw_line)
                    message = chunk.get("message", {})
                    yield {
                        "token": message.get("content", ""),
                        "tool_calls": message.get("tool_calls") or None,
                        "done": chunk.get("done", False),
                        "model": chunk.get("model", settings.llm_model),
                    }
        except (urllib.error.URLError, TimeoutError, ValueError, json.JSONDecodeError) as error:
            logger.error("LLM request to %s failed: %s (is Ollama running?)", settings.llm_url, error)
            raise ProviderUnavailable(f"Streaming LLM request failed: {error}") from error

    @staticmethod
    def language_instruction(language: str) -> str:
        """Single source of truth for the language-matching instruction given to the LLM
        -- used by the live voice pipeline, the typed-question REST path, and the one-shot
        turn endpoint alike, so all three behave identically instead of drifting apart."""
        if language == "auto":
            return (
                "Detect the language the user just spoke in -- including if they mixed two languages "
                "in the same sentence (e.g. Telugu-English or Hindi-English code-mixing) -- and answer "
                "in that same language, or the same mix, using its native script. Match their tone."
            )
        return (
            f"Answer only in {language}, in its native script. If the user's question mixes {language} "
            f"with another language in the same sentence, respond naturally using that same mix rather "
            f"than switching entirely to one language. Preserve their natural conversational tone."
        )

    @staticmethod
    def _chat_payload(request: ChatRequest, stream: bool) -> dict[str, object]:
        messages = []
        if request.system_prompt:
            messages.append({"role": "system", "content": request.system_prompt})
        messages.extend(request.history)
        messages.append({"role": "user", "content": request.message})
        return {
            "model": settings.llm_model,
            "messages": messages,
            "stream": stream,
            "think": settings.llm_think,
            "keep_alive": settings.llm_keep_alive,
            "options": {
                # Zero: this is a document-grounded assistant that should answer the same
                # real question the same way, not creatively vary -- observed the same
                # ambiguous romanized query hallucinating a different wrong answer (once
                # inventing hours, once treating the question itself as the restaurant's
                # name) across repeated calls even at temperature 0.1; 0.0 was
                # empirically far more stable for this kind of grounded Q&A.
                "temperature": 0.0,
                "num_predict": settings.llm_num_predict,
                "num_ctx": settings.llm_num_ctx,
                # Guards against the small model falling into a degenerate word-repeat loop
                # (e.g. "kala kala kala...") on inputs it finds confusing, such as romanized
                # Indian-language text -- higher than Ollama's 1.1 default, still low enough
                # not to visibly distort normal replies.
                "repeat_penalty": 1.3,
                "repeat_last_n": 64,
            },
        }

    @staticmethod
    def _enrich_with_documents(request: ChatRequest) -> ChatRequest:
        # Always retrieve, regardless of how the question looks -- an earlier version
        # of this skipped retrieval for questions an embedding-based classifier judged
        # to be small talk, as a latency optimization. Measured directly: it
        # misclassified a real, short, Telugu-English code-mixed question ("restuarant
        # eh time ki open?") as small talk and skipped retrieval, producing a wrong "I
        # don't have enough information" reply for a question the document actually
        # answers. get_context()'s own relevance threshold already correctly returns
        # nothing for genuine greetings on its own (verified: "hi"/"hello"/"thanks"
        # score 0.07-0.18 against a 0.30 threshold) -- so the classifier was never
        # needed for correctness, only saved a few milliseconds, and cost real answers
        # on exactly the kind of informal, code-mixed input this app needs to handle.
        context = document_rag.get_context(request.message, max_chunks=2)
        has_documents = document_rag.count() > 0
        system_prompt = DOCUMENT_GROUNDED_SYSTEM_PROMPT if has_documents else NO_REFERENCE_SYSTEM_PROMPT
        if request.system_prompt:
            system_prompt = f"{system_prompt}\n\n{request.system_prompt}"
        document_context = context or "No relevant reference information is available."
        enriched_prompt = (
            f"Reference information:\n{document_context}\n\n"
            f"User question:\n{request.message}"
        )
        return ChatRequest(enriched_prompt, system_prompt, request.history)

    def answer_with_audio(self, request: VoiceTurnRequest) -> dict[str, object]:
        language_instruction = self.language_instruction(request.language)
        system_prompt = f"{request.system_prompt}\n{language_instruction}" if request.system_prompt else language_instruction
        answer = self.chat(ChatRequest(request.question, system_prompt))
        result: dict[str, object] = {"question": request.question, "reply": answer["reply"], "model": answer["model"], "audioBase64": None, "contentType": None}
        if settings.tts_enabled:
            try:
                speech = self.synthesize(SpeechRequest(str(answer["reply"]), request.voice, request.language))
                result["audioBase64"] = speech["audioBase64"]
                result["contentType"] = speech["contentType"]
                result["language"] = request.language
            except ProviderUnavailable as error:
                result["audioError"] = str(error)
        return result

    def test_qwen_to_tts(self, prompt: str, language: str, voice: str) -> dict[str, object]:
        instruction = f"Write one short natural sentence in {language} for a text-to-speech test. Do not add labels or explanations. User request: {prompt}"
        answer = self.chat(ChatRequest(instruction))
        speech = self.synthesize(SpeechRequest(str(answer["reply"]), voice, language))
        return {"model": answer["model"], "generatedText": answer["reply"], "language": language, "audioBase64": speech["audioBase64"], "contentType": speech["contentType"]}

    def transcribe(self, request: TranscriptionRequest) -> dict[str, object]:
        audio = base64.b64decode(request.audio_base64, validate=True)
        if not settings.stt_url:
            return self._transcribe_local(request, audio)
        boundary = "----AuraVoiceBoundary"
        content_type = mimetypes.guess_type(request.filename)[0] or "audio/webm"
        body = self._multipart(boundary, request, audio, content_type)
        headers = {"Content-Type": f"multipart/form-data; boundary={boundary}"}
        if settings.stt_api_key:
            headers["Authorization"] = f"Bearer {settings.stt_api_key}"
        response = self._request(settings.stt_url, body, headers)
        parsed = json.loads(response)
        return {"text": parsed.get("text", ""), "language": parsed.get("language", request.language)}

    def transcribe_pcm(self, pcm_bytes: bytes, sample_rate: int, language: str | None = None, fast: bool = False) -> dict[str, object]:
        """Transcribes raw 16-bit PCM audio (used by the live WebSocket session,
        which streams mic frames directly instead of an uploaded file).

        `fast=True` uses the smaller, quicker stt_interim_model instead of the
        configured stt_model -- for in-progress live-caption previews only (see
        live_session.py), never for the final transcript that drives the LLM's
        answer, since it trades accuracy for speed."""
        if settings.stt_mode != "local":
            raise ProviderUnavailable("Live transcription requires STT_MODE=local (STT_URL streaming is not supported yet).")
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temporary:
            with wave.open(temporary, "wb") as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(sample_rate)
                wav_file.writeframes(pcm_bytes)
            audio_path = temporary.name
        try:
            model = self._interim_whisper_model() if fast else self._whisper_model()
            resolved_language = None if language in (None, "", "auto") else self._language_code(language)
            duration_s = len(pcm_bytes) / 2 / sample_rate
            restricted = None
            if resolved_language is None and _allowed_stt_languages():
                restricted = _detect_allowed_language(model, audio_path)
                if restricted:
                    resolved_language = restricted[0]
            segments, info = model.transcribe(
                audio_path, language=resolved_language, beam_size=1, vad_filter=True, **_decode_limits(duration_s),
            )
            segments = list(segments)
            text = " ".join(segment.text.strip() for segment in segments).strip()
            # language_probability is only meaningful when a detection actually ran --
            # Whisper's own, or the restricted one above (renormalized over the allowed
            # languages); when the caller forced a language there was nothing to be
            # confident or unconfident about.
            if restricted:
                probability = restricted[1]
            else:
                probability = float(info.language_probability) if resolved_language is None else 1.0
            # Whisper's own per-segment confidence, thrown away until now -- only the
            # text was ever looked at. _decode_limits() above deliberately disables
            # faster-whisper's own retry-on-low-quality behavior (retrying is what
            # caused the 9.8s hallucination loops), which also means its own low_conf
            # segment filtering never runs anymore; recomputing it here (same combined
            # rule faster-whisper itself uses: no_speech_prob > 0.6 AND avg_logprob <
            # -1.0) recovers that signal without paying for the retries.
            low_confidence = bool(segments) and any(
                s.no_speech_prob > 0.6 and s.avg_logprob < -1.0 for s in segments
            )
            return {
                "text": text, "language": info.language or language or "auto", "language_probability": probability,
                "low_confidence": low_confidence,
            }
        except Exception as error:
            logger.exception("local STT (%s) failed", settings.stt_interim_model if fast else settings.stt_model)
            raise ProviderUnavailable(f"Local STT failed: {error}") from error
        finally:
            try:
                os.remove(audio_path)
            except OSError:
                pass

    def synthesize(self, request: SpeechRequest) -> dict[str, object]:
        if not settings.tts_url:
            return self._synthesize_local(request)
        payload = {"input": request.text}
        if request.voice:
            payload["voice"] = request.voice
        if request.language != "auto":
            payload["language"] = request.language
        headers = {"Content-Type": "application/json"}
        if settings.tts_api_key:
            headers["Authorization"] = f"Bearer {settings.tts_api_key}"
        audio = self._request(settings.tts_url, json.dumps(payload).encode(), headers)
        return {"audioBase64": base64.b64encode(audio).decode("ascii"), "contentType": "audio/mpeg"}

    def _transcribe_local(self, request: TranscriptionRequest, audio: bytes) -> dict[str, object]:
        if settings.stt_mode != "local":
            raise ProviderUnavailable("STT_URL is not configured.")
        try:
            from faster_whisper import WhisperModel  # noqa: F401  (import check only; model loaded via cache below)
        except ImportError as error:
            raise ProviderUnavailable("Install faster-whisper or configure STT_URL.") from error

        suffix = os.path.splitext(request.filename)[1] or ".webm"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temporary:
            temporary.write(audio)
            audio_path = temporary.name
        try:
            model = self._whisper_model()
            language = None if request.language in (None, "", "auto") else self._language_code(request.language)
            segments, info = model.transcribe(audio_path, language=language, beam_size=1, vad_filter=True, **_decode_limits(30.0))
            text = " ".join(segment.text.strip() for segment in segments).strip()
            return {"text": text, "language": info.language or request.language}
        except Exception as error:
            raise ProviderUnavailable(f"Local STT failed: {error}") from error
        finally:
            try:
                os.remove(audio_path)
            except OSError:
                pass

    @staticmethod
    @lru_cache(maxsize=1)
    def _whisper_model() -> object:
        try:
            from faster_whisper import WhisperModel
            return WhisperModel(settings.stt_model, device=settings.stt_device, compute_type=settings.stt_compute_type, cpu_threads=_stt_cpu_threads())
        except Exception as error:
            raise ProviderUnavailable(f"Local STT model failed to load: {error}") from error

    @staticmethod
    @lru_cache(maxsize=1)
    def _interim_whisper_model() -> object:
        try:
            from faster_whisper import WhisperModel
            return WhisperModel(settings.stt_interim_model, device=settings.stt_device, compute_type=settings.stt_compute_type, cpu_threads=_stt_cpu_threads())
        except Exception as error:
            raise ProviderUnavailable(f"Local interim STT model failed to load: {error}") from error

    @staticmethod
    @lru_cache(maxsize=1)
    def _xtts_engine() -> object:
        try:
            from TTS.api import TTS
        except ImportError as error:
            raise ProviderUnavailable("Install coqui-tts (and its PyTorch dependency) for local XTTS v2 speech.") from error
        try:
            return TTS(settings.tts_model, progress_bar=False, gpu=settings.tts_gpu)
        except Exception as error:
            raise ProviderUnavailable(f"Failed to load XTTS v2 model: {error}") from error

    @staticmethod
    @lru_cache(maxsize=8)
    def _piper_engine(model_path: str) -> object:
        try:
            from piper import PiperVoice
        except ImportError as error:
            raise ProviderUnavailable("Install piper-tts or configure TTS_URL.") from error
        try:
            return PiperVoice.load(model_path)
        except Exception as error:
            raise ProviderUnavailable(f"Failed to load Piper model {model_path}: {error}") from error

    def _synthesize_local(self, request: SpeechRequest) -> dict[str, object]:
        if settings.tts_mode != "local":
            raise ProviderUnavailable("TTS_URL is not configured.")
        if settings.tts_engine == "xtts_v2":
            return self._synthesize_xtts(request)
        if request.voice == "cloned":
            raise ProviderUnavailable("Piper does not provide voice cloning. Set TTS_ENGINE=xtts_v2.")
        model_path = self._tts_model_path(request.language)
        if not model_path:
            raise ProviderUnavailable("Set TTS_MODEL_PATH or TTS_MODEL_PATHS for local Piper speech.")
        try:
            output = io.BytesIO()
            with wave.open(output, "wb") as wav_file:
                self._piper_engine(model_path).synthesize_wav(request.text, wav_file)
            return {"audioBase64": base64.b64encode(output.getvalue()).decode("ascii"), "contentType": "audio/wav"}
        except ProviderUnavailable:
            raise
        except Exception as error:
            raise ProviderUnavailable(f"Local TTS failed: {error}") from error

    def _synthesize_xtts(self, request: SpeechRequest) -> dict[str, object]:
        voice_profiles = settings.voice_profiles()
        if not voice_profiles:
            raise ProviderUnavailable("Set TTS_VOICE_PROFILES (or TTS_SPEAKER_WAV) to at least one reference voice recording for XTTS v2.")
        speaker_wav = voice_profiles.get(request.voice or "default") or voice_profiles.get("default") or next(iter(voice_profiles.values()))
        if not os.path.exists(speaker_wav):
            raise ProviderUnavailable(f"Voice reference WAV does not exist: {speaker_wav}")
        language = "en" if request.language == "auto" else self._language_code(request.language)
        try:
            engine = self._xtts_engine()
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temporary:
                output_path = temporary.name
            engine.tts_to_file(text=request.text, file_path=output_path, speaker_wav=speaker_wav, language=language)
            with open(output_path, "rb") as audio_file:
                audio = audio_file.read()
            return {"audioBase64": base64.b64encode(audio).decode("ascii"), "contentType": "audio/wav"}
        except ProviderUnavailable:
            raise
        except Exception as error:
            raise ProviderUnavailable(f"Local XTTS v2 failed: {error}") from error
        finally:
            if "output_path" in locals():
                try:
                    os.remove(output_path)
                except OSError:
                    pass

    @staticmethod
    def _language_code(language: str) -> str:
        return _LANGUAGE_CODES.get(language.lower(), language.lower()[:2])

    @staticmethod
    def _tts_model_path(language: str) -> str:
        if settings.tts_model_paths:
            try:
                paths = json.loads(settings.tts_model_paths)
                if language in paths:
                    return str(paths[language])
            except json.JSONDecodeError:
                pass
        return settings.tts_model_path

    def _json_request(self, url: str, payload: dict[str, object], api_key: str = "") -> dict[str, object]:
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        response = self._request(url, json.dumps(payload).encode(), headers)
        return json.loads(response)

    def _request(self, url: str, body: bytes, headers: dict[str, str]) -> bytes:
        try:
            request = urllib.request.Request(url, data=body, headers=headers, method="POST")
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return response.read()
        except (urllib.error.URLError, TimeoutError, ValueError) as error:
            raise ProviderUnavailable(f"AI provider request failed: {error}") from error

    @staticmethod
    def _multipart(boundary: str, request: TranscriptionRequest, audio: bytes, content_type: str = "audio/webm") -> bytes:
        parts = [
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{request.filename}\"\r\nContent-Type: {content_type}\r\n\r\n".encode() + audio,
        ]
        if request.language:
            parts.append(f"--{boundary}\r\nContent-Disposition: form-data; name=\"language\"\r\n\r\n{request.language}".encode())
        return b"\r\n".join(parts) + f"\r\n--{boundary}--\r\n".encode()
