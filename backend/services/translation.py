"""English -> Indic-language translation via a local IndicTrans2 ONNX model.

Why this exists: asking the conversational LLM to directly *write* fluent
Telugu/Tamil/etc. produced garbled, script-mixed output (verified against the
real model -- see project notes). Dedicated translation models are far more
reliable for this than a generalist chat LLM, so the pipeline now always
generates its answer in English and, for languages this model covers,
translates the finished text instead of asking the LLM to write it directly.

The bundle only needs to be downloaded once (see README for the source URLs);
if it's missing, translation is silently unavailable and callers fall back to
asking the LLM to answer directly in the target language (the previous,
less-reliable behavior for that language, but no crash).
"""

import re
from functools import lru_cache
from typing import TYPE_CHECKING

from ..config import settings
from ..logging_setup import get_logger

if TYPE_CHECKING:
    from .indictrans_onnx import IndicTransONNX

logger = get_logger("translation")

# Emoji confuse the transliteration step (observed producing stray garbage
# characters in the output) and can't be spoken by TTS either way, so they're
# stripped before translation regardless of whether the system prompt already
# told the LLM not to use them.
_EMOJI_PATTERN = re.compile(
    "[\U0001F300-\U0001FAFF\U00002600-\U000027BF\U0001F1E6-\U0001F1FF\U00002190-\U000021FF\U00002B00-\U00002BFF]+"
)

# The model was trained on English plus these 10 Indic languages (confirmed from
# its published language metadata) -- not the full 22 scheduled Indian languages.
# Any SUPPORTED_LANGUAGES code outside this map falls back to direct LLM generation.
_FLORES_TARGETS: dict[str, str] = {
    "hi": "hin_Deva",
    "bn": "ben_Beng",
    "ta": "tam_Taml",
    "te": "tel_Telu",
    "mr": "mar_Deva",
    "gu": "guj_Gujr",
    "kn": "kan_Knda",
    "ml": "mal_Mlym",
    "pa": "pan_Guru",
    "ur": "urd_Arab",
}


def is_supported(language_code: str) -> bool:
    return language_code in _FLORES_TARGETS


@lru_cache(maxsize=1)
def _engine() -> "IndicTransONNX | None":
    model_dir = settings.indic_translation_model_path
    if not model_dir:
        return None
    try:
        # Imported lazily: this pulls in onnxruntime/tokenizers/indicnlp/sacremoses,
        # which may not be installed in every environment this backend runs in (they're
        # optional -- translation is simply unavailable without them, not a hard crash).
        from .indictrans_onnx import IndicTransONNX

        return IndicTransONNX(model_dir)
    except Exception:
        return None


@lru_cache(maxsize=1)
def _reverse_engine() -> "IndicTransONNX | None":
    model_dir = settings.indic_translation_reverse_model_path
    if not model_dir:
        return None
    try:
        from .indictrans_onnx import IndicTransONNX

        return IndicTransONNX(model_dir)
    except Exception:
        return None


def warmup() -> None:
    _engine()
    _reverse_engine()


def translate_from_english(text: str, target_code: str) -> str:
    flores_tag = _FLORES_TARGETS.get(target_code)
    engine = _engine()
    text = _EMOJI_PATTERN.sub("", text).strip()
    if flores_tag is None or engine is None or not text:
        return text
    try:
        translated = engine.translate(text, src_lang="eng_Latn", tgt_lang=flores_tag)
        logger.info("translated to %s: %r -> %r", target_code, text, translated)
        return translated
    except Exception:
        logger.exception("translation to %s failed for %r -- returning untranslated English", target_code, text)
        return text


def translate_to_english(text: str, source_code: str) -> str | None:
    """Translates a question written in an Indic language into English, so document
    retrieval (embedding similarity search) and the LLM both work against English text
    instead of doing unreliable cross-lingual matching directly against the document.
    Returns None (not the original text) on failure/unavailability, so callers can tell
    "translation didn't happen" apart from "translated to an empty string"."""
    flores_tag = _FLORES_TARGETS.get(source_code)
    engine = _reverse_engine()
    text = text.strip()
    if flores_tag is None or engine is None or not text:
        return None
    try:
        translated = engine.translate(text, src_lang=flores_tag, tgt_lang="eng_Latn")
        logger.info("translated from %s to English: %r -> %r", source_code, text, translated)
        return translated
    except Exception:
        logger.exception("translation from %s failed for %r", source_code, text)
        return None
