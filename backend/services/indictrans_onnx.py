"""IndicTrans2 ONNX inference (encoder/decoder + KV-cached decoding loop).

Ported from the reference implementation published alongside the ONNX weights
at https://huggingface.co/remiai3/TRANSLATION_MODELS (MIT-licensed,
model.SRC/model.TGT dict.SRC.json - see NOTICE.md in the downloaded bundle),
substituting this project's pure-Python `IndicProcessor`
(services/indic_processor.py) for the upstream IndicTransToolkit package,
which cannot be installed here (see indic_processor.py's module docstring).
"""

import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

from .indic_processor import IndicProcessor


def _past_feed(past_outputs: list[np.ndarray], num_layers: int) -> dict[str, np.ndarray]:
    feed: dict[str, np.ndarray] = {}
    for i in range(num_layers):
        base = i * 4
        feed[f"past_key_values.{i}.decoder.key"] = past_outputs[base]
        feed[f"past_key_values.{i}.decoder.value"] = past_outputs[base + 1]
        feed[f"past_key_values.{i}.encoder.key"] = past_outputs[base + 2]
        feed[f"past_key_values.{i}.encoder.value"] = past_outputs[base + 3]
    return feed


class IndicTransONNX:
    def __init__(self, model_dir: str, providers: list[str] | None = None) -> None:
        snap = Path(model_dir)
        self._providers = providers or ["CPUExecutionProvider"]
        self._ip = IndicProcessor(inference=True)

        self._src_tok = Tokenizer.from_file(str(snap / "tokenizer_src.json"))
        self._tgt_tok = Tokenizer.from_file(str(snap / "tokenizer_tgt.json"))
        self._meta: dict = json.loads((snap / "tokenizer_meta.json").read_text(encoding="utf-8"))

        gen_cfg: dict = {}
        gen_config_path = snap / "generation_config.json"
        if gen_config_path.exists():
            gen_cfg = json.loads(gen_config_path.read_text(encoding="utf-8"))
        self._decoder_start_id: int = int(gen_cfg.get("decoder_start_token_id", 2))
        self._eos_id: int = int(gen_cfg.get("eos_token_id", 2))

        self._enc = ort.InferenceSession(str(snap / "encoder_model.onnx"), providers=self._providers)
        self._dec = ort.InferenceSession(str(snap / "decoder_model.onnx"), providers=self._providers)
        self._dec_past = ort.InferenceSession(str(snap / "decoder_with_past_model.onnx"), providers=self._providers)
        self._num_layers: int = (len(self._dec.get_outputs()) - 1) // 4

    def translate(self, text: str, src_lang: str, tgt_lang: str, max_new_tokens: int = 128) -> str:
        self._ip._placeholder_entity_maps.queue.clear()
        prefixed = self._ip.preprocess_batch([text], src_lang=src_lang, tgt_lang=tgt_lang)[0]

        encoded = self._src_tok.encode(prefixed)
        input_ids = np.array(
            [[i if i < self._meta["src_dict_size"] else self._meta["unk_id"] for i in encoded.ids]],
            dtype=np.int64,
        )
        attn_mask = np.array([encoded.attention_mask], dtype=np.int64)

        enc_out = self._enc.run(["last_hidden_state"], {"input_ids": input_ids, "attention_mask": attn_mask})[0]

        decoder_input_ids = np.array([[self._decoder_start_id]], dtype=np.int64)
        output_ids: list[int] = [self._decoder_start_id]
        past_outputs: list[np.ndarray] | None = None

        for step in range(max_new_tokens):
            if step == 0:
                dec_out = self._dec.run(
                    None,
                    {"input_ids": decoder_input_ids, "encoder_hidden_states": enc_out, "encoder_attention_mask": attn_mask},
                )
            else:
                dec_out = self._dec_past.run(
                    None,
                    {
                        "input_ids": decoder_input_ids,
                        "encoder_attention_mask": attn_mask,
                        **_past_feed(past_outputs, self._num_layers),
                    },
                )

            logits = dec_out[0]
            past_outputs = list(dec_out[1:])
            next_id = int(np.argmax(logits[0, -1, :]))
            output_ids.append(next_id)
            if next_id == self._eos_id:
                break
            decoder_input_ids = np.array([[next_id]], dtype=np.int64)

        safe_ids = [i if i < self._meta["tgt_dict_size"] else self._meta["unk_id"] for i in output_ids]
        raw_decoded = self._tgt_tok.decode(safe_ids, skip_special_tokens=True)
        return self._ip.postprocess_batch([raw_decoded], lang=tgt_lang)[0]
