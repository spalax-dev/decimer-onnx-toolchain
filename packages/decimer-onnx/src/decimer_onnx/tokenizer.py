"""Runtime tokenizer for DECIMER ONNX (pipeline: runtime/M3, no TF).

Loads a frozen vocab.json and decodes model token ids to SMILES with DECIMER's digit
remapping for R-group placeholders.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

DIGIT_UNMAP = str.maketrans("!$^<>?£¢€§", "1234567890")


@dataclass
class Vocab:
    token_to_id: dict[str, int]
    id_to_token: dict[int, str]
    pad_id: int
    start_id: int
    end_id: int
    unk_id: int
    vocab_size: int

    @classmethod
    def load(cls, path: str | Path) -> "Vocab":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        id_to_token = {int(k): v for k, v in data["id_to_token"].items()}
        token_to_id = {k: int(v) for k, v in data["token_to_id"].items()}
        return cls(
            token_to_id=token_to_id,
            id_to_token=id_to_token,
            pad_id=int(data["pad_id"]),
            start_id=int(data["start_id"]),
            end_id=int(data["end_id"]),
            unk_id=int(data.get("unk_id", 1)),
            vocab_size=int(data["vocab_size"]),
        )

    def decode_ids(self, ids: list[int]) -> str:
        special = {self.pad_id, self.start_id, self.end_id}
        tokens = [self.id_to_token.get(i, "<unk>") for i in ids if i not in special]
        return "".join(tokens).translate(DIGIT_UNMAP)
