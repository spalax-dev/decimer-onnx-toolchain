"""Pipeline stages 3/4/5 — Validation.

Compare TF reference vs ONNX variants: numeric diffs, token parity, and (later) golden-set
chemistry gates. This module contains a minimal greedy decoder over the exported ONNX graphs
to support M0 parity checks.

Known M0 issue: the exported decoder exposes fixed positional-encoding tensors as extra
inputs; the runner supplies the canonical constants. Baking them as ONNX initializers is a
follow-up fix.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

START_ID = 13
END_ID = 14
PAD_ID = 0
MAX_LEN = 302


@dataclass
class DecodeResult:
    token_ids: list[int]
    confidences: list[float]
    incomplete: bool


def _decoder_feed(dec_session, tokens: np.ndarray, enc_out: np.ndarray) -> dict[str, np.ndarray]:
    feed = {"tokens": tokens.astype(np.int32), "enc_out": enc_out.astype(np.float32)}
    # Tolerate legacy exports that exposed positional encodings as inputs.
    for inp in dec_session.get_inputs():
        if inp.name in feed:
            continue
        from toolchain.graph import positional_encoding_1d, positional_encoding_2d

        if inp.shape == [1, 302, 512]:
            feed[inp.name] = positional_encoding_1d()
        elif inp.shape == [1, 256, 512]:
            feed[inp.name] = positional_encoding_2d()
        else:
            shape = [d if isinstance(d, int) and d > 0 else 1 for d in inp.shape]
            feed[inp.name] = np.zeros(shape, dtype=np.float32)
    return feed


def greedy_decode_onnx(enc_session, dec_session, preprocessed: np.ndarray, max_len: int = MAX_LEN) -> DecodeResult:
    enc_out = enc_session.run(None, {"image": preprocessed.astype(np.float32)})[0]
    ids = [START_ID]
    confs: list[float] = []
    for _ in range(max_len):
        tokens = np.array([ids], dtype=np.int32)
        probs = dec_session.run(None, _decoder_feed(dec_session, tokens, enc_out))[0]
        last = probs[0, -1]
        pred = int(np.argmax(last))
        conf = float(last[pred])
        ids.append(pred)
        confs.append(conf)
        if pred == END_ID:
            break
    incomplete = ids[-1] != END_ID
    return DecodeResult(token_ids=ids, confidences=confs, incomplete=incomplete)


def decode_tf_full(saved_model_dir: Path, preprocessed: np.ndarray) -> DecodeResult:
    import os

    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    import tensorflow as tf

    m = tf.saved_model.load(str(saved_model_dir))
    x = np.asarray(preprocessed, dtype=np.float32)
    if x.ndim == 4 and x.shape[0] == 1:
        x = x[0]  # SavedModel __call__ expects (512,512,3)
    out, conf = m(tf.constant(x, dtype=tf.float32))
    ids = out.numpy().astype(int).reshape(-1).tolist()
    # trim padding/trailing if any
    if PAD_ID in ids:
        ids = ids[: ids.index(PAD_ID) + 1]
        while ids and ids[-1] == PAD_ID:
            ids.pop()
    cvals = conf.numpy().astype(float).reshape(-1).tolist()
    return DecodeResult(token_ids=ids, confidences=cvals, incomplete=(not ids or ids[-1] != END_ID))


def compare_tf_onnx(saved_model_dir: Path, onnx_dir: Path, preprocessed: np.ndarray) -> dict[str, Any]:
    import onnxruntime as ort

    enc = ort.InferenceSession(str(onnx_dir / "encoder.fp32.onnx"), providers=["CPUExecutionProvider"])
    dec = ort.InferenceSession(str(onnx_dir / "decoder_step.fp32.onnx"), providers=["CPUExecutionProvider"])
    onnx_res = greedy_decode_onnx(enc, dec, preprocessed)
    tf_res = decode_tf_full(saved_model_dir, preprocessed)
    # compare generated sequence (exclude initial start for token parity clarity)
    t = tf_res.token_ids
    o = onnx_res.token_ids
    match = t == o
    first_diff = next((i for i, (a, b) in enumerate(zip(t, o)) if a != b), None)
    if first_diff is None and len(t) != len(o):
        first_diff = min(len(t), len(o))
    return {
        "match": match,
        "tf_len": len(t),
        "onnx_len": len(o),
        "first_diff": first_diff,
        "tf_ids": t,
        "onnx_ids": o,
        "tf_incomplete": tf_res.incomplete,
        "onnx_incomplete": onnx_res.incomplete,
    }
