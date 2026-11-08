"""Pipeline stage 3 — FP32 ONNX export (M0/M2).

Converts the wrapped encoder/decoder tf.functions to ONNX via tf2onnx. Records basic graph
metadata; numeric/token parity validation is handled by parity.py.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class OnnxExportResult:
    encoder_path: str | None = None
    decoder_path: str | None = None
    opset: int = 17
    metadata: dict[str, Any] | None = None
    errors: list[str] | None = None


def _bake_decoder_positional_encodings(path: Path) -> dict[str, Any]:
    """Bake leaked positional-encoding inputs into ONNX initializers (M0 interface fix)."""
    import onnx
    from onnx import numpy_helper

    from decimer_onnx_export.graph import positional_encoding_1d, positional_encoding_2d

    model = onnx.load(str(path))
    keep = []
    baked = []
    for inp in model.graph.input:
        dims = tuple(d.dim_value if d.dim_value else d.dim_param for d in inp.type.tensor_type.shape.dim)
        arr = None
        if inp.name in {"tokens", "enc_out"}:
            keep.append(inp)
            continue
        if dims == (1, 302, 512):
            arr = positional_encoding_1d()
        elif dims == (1, 256, 512):
            arr = positional_encoding_2d()
        if arr is not None:
            model.graph.initializer.append(numpy_helper.from_array(arr.astype(np.float32), name=inp.name))
            baked.append(inp.name)
        else:
            keep.append(inp)
    del model.graph.input[:]
    model.graph.input.extend(keep)
    onnx.checker.check_model(model)
    onnx.save(model, str(path))
    return {
        "baked_inputs": baked,
        "inputs": [f"{i.name}:{[d.dim_value or d.dim_param for d in i.type.tensor_type.shape.dim]}" for i in model.graph.input],
    }


def export_fp32_onnx(model_dir: Path, out_dir: Path, opset: int = 17) -> OnnxExportResult:
    import tf2onnx

    from decimer_onnx_export.graph import build_export_graphs, load_saved_model

    out_dir.mkdir(parents=True, exist_ok=True)
    res = OnnxExportResult(opset=opset, metadata={}, errors=[])

    loaded = load_saved_model(model_dir)
    g = build_export_graphs(loaded)

    enc_path = out_dir / "encoder.fp32.onnx"
    try:
        model_proto, _ = tf2onnx.convert.from_function(
            g.encoder_fn,
            input_signature=g.encoder_spec,
            opset=opset,
            output_path=str(enc_path),
        )
        res.encoder_path = str(enc_path)
        res.metadata["encoder"] = {
            "inputs": [f"{i.name}:{i.type.tensor_type.elem_type}" for i in model_proto.graph.input],
            "outputs": [o.name for o in model_proto.graph.output],
            "nodes": len(model_proto.graph.node),
        }
    except Exception as e:  # noqa: BLE001
        res.errors.append(f"encoder export failed: {e}")

    dec_path = out_dir / "decoder_step.fp32.onnx"
    try:
        model_proto, _ = tf2onnx.convert.from_function(
            g.decoder_fn,
            input_signature=g.decoder_spec,
            opset=opset,
            output_path=str(dec_path),
        )
        onnx.save(model_proto, str(dec_path))
        baked = _bake_decoder_positional_encodings(dec_path)
        model_proto = onnx.load(str(dec_path))
        res.decoder_path = str(dec_path)
        res.metadata["decoder"] = {
            "inputs": [f"{i.name}:{i.type.tensor_type.elem_type}" for i in model_proto.graph.input],
            "outputs": [o.name for o in model_proto.graph.output],
            "nodes": len(model_proto.graph.node),
            **baked,
        }
    except Exception as e:  # noqa: BLE001
        res.errors.append(f"decoder export failed: {e}")

    meta_path = out_dir / "export.fp32.meta.json"
    meta_path.write_text(
        json.dumps(
            {
                "opset": opset,
                "encoder_path": res.encoder_path,
                "decoder_path": res.decoder_path,
                "metadata": res.metadata,
                "errors": res.errors,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    res.metadata["meta_path"] = str(meta_path)
    return res
