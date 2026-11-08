"""Pipeline stage 3 — FP32 ONNX export (M0/M2).

Converts the wrapped encoder/decoder tf.functions to ONNX via tf2onnx. Records basic graph
metadata; numeric/token parity validation is handled by parity.py (next steps).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class OnnxExportResult:
    encoder_path: str | None = None
    decoder_path: str | None = None
    opset: int = 17
    metadata: dict[str, Any] | None = None
    errors: list[str] | None = None


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
        res.decoder_path = str(dec_path)
        res.metadata["decoder"] = {
            "inputs": [f"{i.name}:{i.type.tensor_type.elem_type}" for i in model_proto.graph.input],
            "outputs": [o.name for o in model_proto.graph.output],
            "nodes": len(model_proto.graph.node),
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
