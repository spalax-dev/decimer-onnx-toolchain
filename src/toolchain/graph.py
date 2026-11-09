"""Pipeline stage 3 — FP32 export wrappers (M0/M2).

Build clean tf.function wrappers around the restored DECIMER encoder/decoder for ONNX
conversion. The published SavedModel runs the full greedy loop; ONNX export uses two graphs:
- encoder(image) -> enc_out [1,256,512]
- decoder(tokens, enc_out) -> probs [1,T,156]

Dynamic sequence length T is preserved. Causal mask and positional encodings are computed
inside the decoder wrapper so they become graph constants (not external inputs).
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

# Must match DECIMER Transformer_decoder/config (printed model): max_len 302, d_model 512, 16x16.
MAX_LEN = 302
D_MODEL = 512
ROWS = 16
COLS = 16


def _angles(pos, i, d_model):
    rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    return pos * rates


def positional_encoding_1d(position: int = MAX_LEN, d_model: int = D_MODEL) -> np.ndarray:
    angle_rads = _angles(
        np.arange(position)[:, np.newaxis], np.arange(d_model)[np.newaxis, :], d_model
    )
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    return angle_rads[np.newaxis, ...].astype(np.float32)


def positional_encoding_2d(row: int = ROWS, col: int = COLS, d_model: int = D_MODEL) -> np.ndarray:
    assert d_model % 2 == 0
    row_pos = np.repeat(np.arange(row), col)[:, np.newaxis]
    col_pos = np.repeat(np.expand_dims(np.arange(col), 0), row, axis=0).reshape(-1, 1)
    angle_rads_row = _angles(row_pos, np.arange(d_model // 2)[np.newaxis, :], d_model // 2)
    angle_rads_col = _angles(col_pos, np.arange(d_model // 2)[np.newaxis, :], d_model // 2)
    angle_rads_row[:, 0::2] = np.sin(angle_rads_row[:, 0::2])
    angle_rads_row[:, 1::2] = np.cos(angle_rads_row[:, 1::2])
    angle_rads_col[:, 0::2] = np.sin(angle_rads_col[:, 0::2])
    angle_rads_col[:, 1::2] = np.cos(angle_rads_col[:, 1::2])
    pos = np.concatenate([angle_rads_row, angle_rads_col], axis=1)[np.newaxis, ...]
    return pos.astype(np.float32)


@dataclass
class ExportGraphs:
    encoder_fn: Any
    decoder_fn: Any
    encoder_spec: Any
    decoder_spec: Any


def load_saved_model(model_dir: Path):
    import tensorflow as tf

    return tf.saved_model.load(str(model_dir))


def build_export_graphs(loaded: Any) -> ExportGraphs:
    """Build export wrappers.

    Note: the restored decoder computes positional encodings internally from its own weights
    attributes. We keep the wrapper minimal (tokens, enc_out -> probs) and rely on the traced
    decoder graph; causal mask is computed inside the wrapper. If positional encodings appear
    as graph inputs after export, the export path must be revisited (M0 risk).
    """
    import tensorflow as tf

    enc = loaded.DECIMER.encoder
    dec = loaded.DECIMER.transformer
    # Inject deterministic positional encodings so tracing captures constants, not inputs.
    pe1 = tf.constant(positional_encoding_1d(), dtype=tf.float32)
    pe2 = tf.constant(positional_encoding_2d(), dtype=tf.float32)
    for obj in (dec, getattr(dec, "keras_api", None)):
        if obj is None:
            continue
        for name, val in (("pos_encoding_1d", pe1), ("pos_encoding_2d", pe2)):
            try:
                setattr(obj, name, val)
            except Exception:
                pass

    @tf.function(input_signature=[
        tf.TensorSpec(shape=[1, 512, 512, 3], dtype=tf.float32, name="image"),
    ])
    def encoder(image):
        out = enc(image, training=False)
        return tf.identity(out, name="enc_out")

    @tf.function(input_signature=[
        tf.TensorSpec(shape=[1, None], dtype=tf.int32, name="tokens"),
        tf.TensorSpec(shape=[1, 256, 512], dtype=tf.float32, name="enc_out"),
    ])
    def decoder(tokens, enc_out):
        t = tf.shape(tokens)[1]
        # Upstream: mask is 0=allowed, 1=masked; attention adds mask * -1e9.
        mask = 1 - tf.linalg.band_part(tf.ones((t, t), dtype=tf.float32), -1, 0)
        mask = mask[tf.newaxis, tf.newaxis, :, :]
        probs = dec(tokens, enc_out, False, mask, None)
        return tf.identity(probs, name="probs")

    return ExportGraphs(
        encoder_fn=encoder,
        decoder_fn=decoder,
        encoder_spec=encoder.input_signature,
        decoder_spec=decoder.input_signature,
    )


def get_concrete_functions(model_dir: Path):
    loaded = load_saved_model(model_dir)
    g = build_export_graphs(loaded)
    enc_cf = g.encoder_fn.get_concrete_function()
    dec_cf = g.decoder_fn.get_concrete_function()
    return g, enc_cf, dec_cf
