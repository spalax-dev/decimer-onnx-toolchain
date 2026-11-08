"""Greedy decoder over DECIMER ONNX graphs (pipeline: runtime/M3).

Runs encoder once and autoregressively calls decoder_step(tokens, enc_out)->probs until
<end> or max_len. No TensorFlow dependency.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class DecodeResult:
    token_ids: list[int]
    confidences: list[float]
    incomplete: bool


def greedy_decode(
    encoder_session,
    decoder_session,
    preprocessed: np.ndarray,
    *,
    start_id: int,
    end_id: int,
    max_len: int = 302,
) -> DecodeResult:
    enc_name = encoder_session.get_inputs()[0].name
    dec_inputs = {i.name: i for i in decoder_session.get_inputs()}
    tok_name = next((n for n in ("tokens", "input_ids") if n in dec_inputs), decoder_session.get_inputs()[0].name)
    enc_in_name = next((n for n in ("enc_out", "encoder_output") if n in dec_inputs), decoder_session.get_inputs()[1].name)

    x = np.asarray(preprocessed, dtype=np.float32)
    if x.ndim == 3:
        x = x[None, ...]
    enc_out = encoder_session.run(None, {enc_name: x})[0]

    ids = [int(start_id)]
    confs: list[float] = []
    for _ in range(max_len):
        tokens = np.array([ids], dtype=np.int32)
        feed = {tok_name: tokens, enc_in_name: enc_out}
        # legacy exports may still carry extra inputs; supply zeros
        for name, inp in dec_inputs.items():
            if name in feed:
                continue
            shape = [d if isinstance(d, int) and d > 0 else 1 for d in inp.shape]
            feed[name] = np.zeros(shape, dtype=np.float32)
        probs = decoder_session.run(None, feed)[0]
        last = probs[0, -1]
        pred = int(np.argmax(last))
        conf = float(last[pred])
        ids.append(pred)
        confs.append(conf)
        if pred == end_id:
            break
    return DecodeResult(token_ids=ids, confidences=confs, incomplete=(ids[-1] != end_id))
