"""Pipeline: M1/tokenizer freeze. Convert DECIMER's pickled Keras tokenizer to a stable
`vocab.json` used by the ONNX runtime (no pickle/TF at inference).

Keras tokenizers are 1-based in `word_index` and reserve 0 for padding; the published
DECIMER tokenizer has 156 entries (including <unk>, <start>, <end>). We preserve the
original ids and record them explicitly so the ONNX decoder and detokenization stay
byte/token-identical to upstream.
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

SPECIAL = {"<pad>", "<unk>", "<start>", "<end>"}
# DECIMER digit remapping (utils.replacer/decoder) applied to R-group/placeholder digits.
DIGIT_REMAP = str.maketrans("1234567890", "!$^<>?£¢€§")
DIGIT_UNMAP = str.maketrans("!$^<>?£¢€§", "1234567890")


def load_pickled_tokenizer(path: Path) -> Any:
    with path.open("rb") as f:
        return pickle.load(f)


def freeze_tokenizer(tokenizer_pkl: Path, out_path: Path) -> dict[str, Any]:
    tok = load_pickled_tokenizer(tokenizer_pkl)
    word_index = getattr(tok, "word_index", None) or {}
    index_word = getattr(tok, "index_word", None) or {}
    # normalize keys to ints for stable JSON
    id_to_token = {int(k): str(v) for k, v in index_word.items()}
    token_to_id = {str(k): int(v) for k, v in word_index.items()}
    vocab = {
        "format": "decimer-tokenizer-v1",
        "source": str(tokenizer_pkl),
        "vocab_size": len(token_to_id),
        "pad_token": "<pad>",
        "pad_id": token_to_id.get("<pad>", 0),
        "unk_token": "<unk>",
        "start_token": "<start>",
        "end_token": "<end>",
        "start_id": token_to_id.get("<start>"),
        "end_id": token_to_id.get("<end>"),
        "unk_id": token_to_id.get("<unk>"),
        "token_to_id": token_to_id,
        "id_to_token": {str(k): v for k, v in sorted(id_to_token.items())},
        "digit_remap": {"1234567890": "!$^<>?£¢€§"},
        "notes": [
            "Keras tokenizer ids preserved (1-based entries; pad is 0).",
            "Decoder output is softmax over vocab_size classes in the model's id space; verify alignment in M0/M1.",
        ],
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(vocab, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return vocab
