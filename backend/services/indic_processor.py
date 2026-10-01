"""Pure-Python reimplementation of AI4Bharat's IndicTransToolkit ``IndicProcessor``.

The upstream ``IndicTransToolkit`` package only ships a Cython source
distribution (no prebuilt wheel), and building it locally needs both a C++
compiler (not installed here) and triggers this machine's Windows Application
Control policy when pip builds fresh compiled extensions in its isolated
build environment. The Cython file itself (processor.pyx) is pure
text-processing logic with no algorithmic dependency on being compiled --
every method is plain Python/regex/dict work wrapped in `cdef` purely for
speed. This module is that same logic with the Cython type annotations
stripped, so IndicTrans2 can be used without needing a compiler at all.
"""

import sys
import types
from queue import Queue

# indic-nlp-library imports pandas at module load time in two files
# (script/indic_scripts.py, transliterate/unicode_transliterate.py), purely to
# read two CSV lookup tables inside an init() function this codebase never
# calls -- UnicodeIndicTransliterator.transliterate(), the only entry point
# used below, is self-contained Unicode-offset arithmetic (see its source)
# and never touches that data. But the bare `import pandas` statement alone
# eagerly loads pandas' compiled internals (pandas._libs...), which this
# machine's Windows Application Control policy blocks outright -- confirmed
# via direct traceback -- the same class of restriction that already ruled
# out sentence-transformers for embeddings (see services/embeddings.py). A
# stub module lets indicnlp's imports succeed without ever loading real
# pandas; if indic-nlp-library ever actually calls a pandas function through
# it, this raises a clear, loud error instead of silently returning nonsense.
if "pandas" not in sys.modules:
    def _pandas_unavailable(*_args, **_kwargs):
        raise RuntimeError(
            "pandas is not available in this environment (blocked by Windows Application "
            "Control) and indic_processor.py only stubs it out to satisfy indic-nlp-library's "
            "unused CSV-loading code path -- something now depends on real pandas functionality."
        )

    _pandas_stub = types.ModuleType("pandas")
    _pandas_stub.__getattr__ = _pandas_unavailable
    sys.modules["pandas"] = _pandas_stub

import regex as re
from indicnlp.normalize.indic_normalize import IndicNormalizerFactory
from indicnlp.tokenize import indic_detokenize, indic_tokenize
from indicnlp.transliterate.unicode_transliterate import UnicodeIndicTransliterator
from sacremoses import MosesDetokenizer, MosesPunctNormalizer, MosesTokenizer

_FLORES_CODES = {
    "asm_Beng": "as", "awa_Deva": "hi", "ben_Beng": "bn", "bho_Deva": "hi",
    "brx_Deva": "hi", "doi_Deva": "hi", "eng_Latn": "en", "gom_Deva": "kK",
    "gon_Deva": "hi", "guj_Gujr": "gu", "hin_Deva": "hi", "hne_Deva": "hi",
    "kan_Knda": "kn", "kas_Arab": "ur", "kas_Deva": "hi", "kha_Latn": "en",
    "lus_Latn": "en", "mag_Deva": "hi", "mai_Deva": "hi", "mal_Mlym": "ml",
    "mar_Deva": "mr", "mni_Beng": "bn", "mni_Mtei": "hi", "npi_Deva": "ne",
    "ory_Orya": "or", "pan_Guru": "pa", "san_Deva": "hi", "sat_Olck": "or",
    "snd_Arab": "ur", "snd_Deva": "hi", "tam_Taml": "ta", "tel_Telu": "te",
    "urd_Arab": "ur", "unr_Deva": "hi",
}

_DIGIT_GROUPS = {
    "0": "\u09e6\u0ae6\u0ce6\u0966\u0660\uabf0\u0b66\u0a66\u1c50\u06f0",
    "1": "\u09e7\u0ae7\u0967\u0ce7\u06f1\uabf1\u0b67\u0a67\u1c51\u0c67",
    "2": "\u09e8\u0ae8\u0968\u0ce8\u06f2\uabf2\u0b68\u0a68\u1c52\u0c68",
    "3": "\u09e9\u0ae9\u0969\u0ce9\u06f3\uabf3\u0b69\u0a69\u1c53\u0c69",
    "4": "\u09ea\u0aea\u096a\u0cea\u06f4\uabf4\u0b6a\u0a6a\u1c54\u0c6a",
    "5": "\u09eb\u0aeb\u096b\u0ceb\u06f5\uabf5\u0b6b\u0a6b\u1c55\u0c6b",
    "6": "\u09ec\u0aec\u096c\u0cec\u06f6\uabf6\u0b6c\u0a6c\u1c56\u0c6c",
    "7": "\u09ed\u0aed\u096d\u0ced\u06f7\uabf7\u0b6d\u0a6d\u1c57\u0c6d",
    "8": "\u09ee\u0aee\u096e\u0cee\u06f8\uabf8\u0b6e\u0a6e\u1c58\u0c6e",
    "9": "\u09ef\u0aef\u096f\u0cef\u06f9\uabf9\u0b6f\u0a6f\u1c59\u0c6f",
}

_INDIC_FAILURE_CASES = [
    "آی ڈی ", "ꯑꯥꯏꯗꯤ", "आईडी", "आई . डी . ", "आई . डी .", "आई. डी. ", "आई. डी.",
    "आय. डी. ", "आय. डी.", "आय . डी . ", "आय . डी ." "आइ . डी . ", "आइ . डी .",
    "आइ. डी. ", "आइ. डी.", "ऐटि", "آئی ڈی ", "ᱟᱭᱰᱤ ᱾", "आयडी", "ऐडि", "आइडि", "ᱟᱭᱰᱤ",
]

_URL_PATTERN = re.compile(r"\b(?<![\w/.])(?:(?:https?|ftp)://)?(?:(?:[\w-]+\.)+(?!\.))(?:[\w/\-?#&=%.]+)+(?!\.\w+)\b")
_NUMERAL_PATTERN = re.compile(
    r"(~?\d+\.?\d*\s?%?\s?-?\s?~?\d+\.?\d*\s?%|~?\d+%|\d+[-\/.,:']\d+[-\/.,:'+]\d+(?:\.\d+)?|\d+[-\/.:'+]\d+(?:\.\d+)?)"
)
_EMAIL_PATTERN = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}")
_OTHER_PATTERN = re.compile(r"[A-Za-z0-9]*[#|@]\w+")
_MULTISPACE_REGEX = re.compile(r"[ ]{2,}")
_DIGIT_SPACE_PERCENT = re.compile(r"(\d) %")
_DOUBLE_QUOT_PUNC = re.compile(r"\"([,\.]+)")
_DIGIT_NBSP_DIGIT = re.compile(r"(\d) (\d)")
_END_BRACKET_SPACE_PUNC_REGEX = re.compile(r"\) ([\.!:?;,])")

_PUNC_REPLACEMENTS = [
    (re.compile(r"\r"), ""),
    (re.compile(r"\(\s*"), "("),
    (re.compile(r"\s*\)"), ")"),
    (re.compile(r"\s:\s?"), ":"),
    (re.compile(r"\s;\s?"), ";"),
    (re.compile("[`´‘‚’]"), "'"),
    (re.compile("[„“”«»]"), '"'),
    (re.compile("[–—]"), "-"),
    (re.compile(r"\.\.\."), "..."),
    (re.compile(r" %"), "%"),
    (re.compile(r" [?!;]"), lambda m: m.group(0).strip()),
    (re.compile(r", "), ", "),
]


class IndicProcessor:
    """Drop-in, pure-Python equivalent of IndicTransToolkit's Cython IndicProcessor."""

    def __init__(self, inference: bool = True) -> None:
        self.inference = inference
        self._digits_translation_table = {}
        for digit, chars in _DIGIT_GROUPS.items():
            for char in chars:
                self._digits_translation_table[ord(char)] = digit
        for code in range(ord("0"), ord("9") + 1):
            self._digits_translation_table[code] = chr(code)

        self._placeholder_entity_maps: Queue = Queue()
        self._en_tok = MosesTokenizer(lang="en")
        self._en_normalizer = MosesPunctNormalizer()
        self._en_detok = MosesDetokenizer(lang="en")
        self._xliterator = UnicodeIndicTransliterator()

    def _apply_punc_replacements(self, text: str) -> str:
        for pattern, replacement in _PUNC_REPLACEMENTS:
            text = pattern.sub(replacement, text)
        return text

    def _punc_norm(self, text: str) -> str:
        text = self._apply_punc_replacements(text)
        text = _MULTISPACE_REGEX.sub(" ", text)
        text = _END_BRACKET_SPACE_PUNC_REGEX.sub(r")\1", text)
        text = _DIGIT_SPACE_PERCENT.sub(r"\1%", text)
        text = _DOUBLE_QUOT_PUNC.sub(r'\1"', text)
        text = _DIGIT_NBSP_DIGIT.sub(r"\1.\2", text)
        return text.strip()

    def _wrap_with_placeholders(self, text: str) -> str:
        serial_no = 1
        placeholder_entity_map: dict[str, str] = {}
        patterns = [_EMAIL_PATTERN, _URL_PATTERN, _NUMERAL_PATTERN, _OTHER_PATTERN]

        for pattern in patterns:
            matches = set(pattern.findall(text))
            for match in matches:
                if pattern is _URL_PATTERN and len(match.replace(".", "")) < 4:
                    continue
                if pattern is _NUMERAL_PATTERN and len(match.replace(" ", "").replace(".", "").replace(":", "")) < 4:
                    continue

                base_placeholder = f"<ID{serial_no}>"
                for form in (
                    f"<ID{serial_no}>", f"< ID{serial_no} >", f"[ID{serial_no}]", f"[ ID{serial_no} ]",
                    f"[ID {serial_no}]", f"<ID{serial_no}]", f"< ID{serial_no}]", f"<ID{serial_no} ]",
                    f"<id{serial_no}>", f"< id{serial_no} >", f"[id{serial_no}]", f"[ id{serial_no} ]",
                    f"[id {serial_no}]", f"<id{serial_no}]", f"< id{serial_no}]", f"<id{serial_no} ]",
                ):
                    placeholder_entity_map[form] = match
                for indic_case in _INDIC_FAILURE_CASES:
                    for form in (
                        f"<{indic_case}{serial_no}>", f"< {indic_case}{serial_no} >", f"< {indic_case} {serial_no} >",
                        f"<{indic_case} {serial_no}]", f"< {indic_case} {serial_no} ]", f"[{indic_case}{serial_no}]",
                        f"[{indic_case} {serial_no}]", f"[ {indic_case}{serial_no} ]", f"[ {indic_case} {serial_no} ]",
                        f"{indic_case} {serial_no}", f"{indic_case}{serial_no}",
                    ):
                        placeholder_entity_map[form] = match

                text = text.replace(match, base_placeholder)
                serial_no += 1

        text = re.sub(r"\s+", " ", text).replace(">/", ">").replace("]/", "]")
        self._placeholder_entity_maps.put(placeholder_entity_map)
        return text

    def _normalize(self, text: str) -> str:
        text = text.translate(self._digits_translation_table)
        if self.inference:
            text = self._wrap_with_placeholders(text)
        return text

    def _do_indic_tokenize_and_transliterate(self, sentence: str, normalizer, iso_lang: str, transliterate: bool) -> str:
        normed = normalizer.normalize(sentence.strip())
        tokens = indic_tokenize.trivial_tokenize(normed, iso_lang)
        joined = " ".join(tokens)
        if not transliterate:
            return joined
        xlated = self._xliterator.transliterate(joined, iso_lang, "hi")
        return xlated.replace(" \u0964 ", "\u0964")

    def _preprocess(self, sent: str, src_lang: str, tgt_lang: str | None, normalizer, is_target: bool) -> str:
        iso_lang = _FLORES_CODES.get(src_lang, "hi")
        script_part = src_lang.split("_")[1]
        do_transliterate = script_part not in ("Arab", "Aran", "Olck", "Mtei", "Latn")

        sent = self._punc_norm(sent)
        sent = self._normalize(sent)

        if iso_lang == "en":
            e_norm = self._en_normalizer.normalize(sent.strip())
            processed_sent = " ".join(self._en_tok.tokenize(e_norm, escape=False))
        else:
            processed_sent = self._do_indic_tokenize_and_transliterate(sent, normalizer, iso_lang, do_transliterate)

        processed_sent = processed_sent.strip()
        if not is_target:
            return f"{src_lang} {tgt_lang} {processed_sent}"
        return processed_sent

    def _postprocess(self, sent, lang: str, placeholder_entity_map: dict[str, str] | None = None) -> str:
        if isinstance(sent, (tuple, list)):
            sent = sent[0]
        if placeholder_entity_map is None:
            placeholder_entity_map = self._placeholder_entity_maps.get()

        lang_code, script_code = lang.split("_", 1)
        iso_lang = _FLORES_CODES.get(lang, "hi")

        if script_code in ("Arab", "Aran"):
            sent = (
                sent.replace(" \u061f", "\u061f")
                .replace(" \u06d4", "\u06d4")
                .replace(" \u060c", "\u060c")
                .replace("\u066e\u06ea", "\u06e0")
            )
        if lang_code == "ory":
            sent = sent.replace("\u0b2f\u0b3c", "\u0b5f")

        for placeholder, original in placeholder_entity_map.items():
            sent = sent.replace(placeholder, original)

        if lang == "eng_Latn":
            return self._en_detok.detokenize(sent.split(" "))
        xlated = self._xliterator.transliterate(sent, "hi", iso_lang)
        return indic_detokenize.trivial_detokenize(xlated, iso_lang)

    def preprocess_batch(self, batch: list[str], src_lang: str, tgt_lang: str | None = None, is_target: bool = False) -> list[str]:
        normalizer = None
        if src_lang != "eng_Latn":
            iso_code = _FLORES_CODES.get(src_lang, "hi")
            normalizer = IndicNormalizerFactory().get_normalizer(iso_code)
        return [self._preprocess(s, src_lang, tgt_lang, normalizer, is_target) for s in batch]

    def postprocess_batch(self, sents: list[str], lang: str = "hin_Deva") -> list[str]:
        results = []
        placeholder_maps = [self._placeholder_entity_maps.get() for _ in range(len(sents))]
        for sent, current_map in zip(sents, placeholder_maps):
            results.append(self._postprocess(sent, lang, current_map))
        self._placeholder_entity_maps.queue.clear()
        return results
