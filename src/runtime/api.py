"""Public runtime API for DECIMER ONNX (pipeline: runtime/M3)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnxruntime as ort

from runtime.decode import DecodeResult, greedy_decode
from runtime.preprocess import preprocess_image
from runtime.tokenizer import Vocab


@dataclass
class Prediction:
    smiles: str
    token_ids: list[int]
    confidences: list[float]
    incomplete: bool


class DecimerONNX:
    def __init__(self, encoder_path: str | Path, decoder_path: str | Path, vocab_path: str | Path):
        opts = ort.SessionOptions()
        self.encoder = ort.InferenceSession(str(encoder_path), opts, providers=["CPUExecutionProvider"])
        self.decoder = ort.InferenceSession(str(decoder_path), opts, providers=["CPUExecutionProvider"])
        self.vocab = Vocab.load(vocab_path)

    def predict_preprocessed(self, preprocessed: np.ndarray) -> Prediction:
        res: DecodeResult = greedy_decode(
            self.encoder,
            self.decoder,
            preprocessed,
            start_id=self.vocab.start_id,
            end_id=self.vocab.end_id,
        )
        return Prediction(
            smiles=self.vocab.decode_ids(res.token_ids),
            token_ids=res.token_ids,
            confidences=res.confidences,
            incomplete=res.incomplete,
        )

    def predict(self, image: str | Path | np.ndarray) -> Prediction:
        arr = preprocess_image(image)
        return self.predict_preprocessed(arr)
