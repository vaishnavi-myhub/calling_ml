"""Local multilingual embedding model for semantic document retrieval.

Runs directly through ONNX Runtime + the standalone `tokenizers` library
instead of sentence-transformers. That package (and even a bare
`transformers` import, since AutoModel's generation utilities pull it in as
a side effect) drags in scikit-learn's compiled Cython extensions — some
locked-down Windows environments (Application Control / WDAC policies)
refuse to load those unsigned native DLLs outright. onnxruntime and
tokenizers ship their own native code but don't hit that specific block, so
this path is what actually runs everywhere sentence-transformers can't.

Model files are the same ones sentence-transformers would use (the model
repo ships its own onnx/model.onnx alongside the tokenizer), just loaded
and run directly. Cached process-wide, same lru_cache pattern as the
Whisper model in services/voice.py, so it's downloaded/loaded once.
"""

from functools import lru_cache

import numpy as np

from ..config import settings


class EmbeddingUnavailable(RuntimeError):
    """Raised when the local embedding model can't be loaded."""


class _OnnxEmbedder:
    def __init__(self, tokenizer: object, session: object) -> None:
        self._tokenizer = tokenizer
        self._session = session

    def encode(self, texts: list[str]) -> np.ndarray:
        encodings = self._tokenizer.encode_batch(texts)
        max_len = max((len(encoding.ids) for encoding in encodings), default=1) or 1
        batch = len(texts)
        input_ids = np.zeros((batch, max_len), dtype="int64")
        attention_mask = np.zeros((batch, max_len), dtype="int64")
        token_type_ids = np.zeros((batch, max_len), dtype="int64")
        for row, encoding in enumerate(encodings):
            length = len(encoding.ids)
            input_ids[row, :length] = encoding.ids
            attention_mask[row, :length] = encoding.attention_mask

        (token_embeddings,) = self._session.run(
            ["last_hidden_state"],
            {"input_ids": input_ids, "attention_mask": attention_mask, "token_type_ids": token_type_ids},
        )
        mask = attention_mask[:, :, None].astype("float32")
        summed = (token_embeddings * mask).sum(axis=1)
        counts = np.clip(mask.sum(axis=1), 1e-9, None)
        pooled = summed / counts
        norms = np.clip(np.linalg.norm(pooled, axis=1, keepdims=True), 1e-9, None)
        return (pooled / norms).astype("float32")


@lru_cache(maxsize=1)
def _embedder() -> _OnnxEmbedder:
    try:
        import onnxruntime
        from huggingface_hub import hf_hub_download
        from tokenizers import Tokenizer
    except ImportError as error:
        raise EmbeddingUnavailable("Install huggingface_hub, tokenizers, and onnxruntime to enable semantic document search.") from error
    try:
        tokenizer_path = hf_hub_download(settings.embedding_model, "tokenizer.json")
        model_path = hf_hub_download(settings.embedding_model, "onnx/model.onnx")
        tokenizer = Tokenizer.from_file(tokenizer_path)
        tokenizer.enable_truncation(max_length=256)
        session = onnxruntime.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        return _OnnxEmbedder(tokenizer, session)
    except Exception as error:
        raise EmbeddingUnavailable(f"Failed to load embedding model {settings.embedding_model}: {error}") from error


def embed(texts: list[str]) -> np.ndarray:
    """Returns L2-normalized float32 embeddings, one row per input text."""
    if not texts:
        return np.zeros((0, 0), dtype="float32")
    return _embedder().encode(texts)


def warmup() -> None:
    try:
        _embedder()
    except EmbeddingUnavailable:
        pass
