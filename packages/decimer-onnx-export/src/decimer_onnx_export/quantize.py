"""Pipeline stage 4 — Quantize (M4).

Build quantization candidates and support selecting the best by measured quality/latency/size.
Candidates (per plan): E1 dynamic INT8, E2 static INT8 QDQ, E3 hybrid, E4 FP16.
This module starts with E1 (no calibration) as the baseline INT8 experiment.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any


@dataclass
class QuantizeResult:
    variant: str
    encoder_path: str | None = None
    decoder_path: str | None = None
    metadata: dict[str, Any] | None = None
    errors: list[str] | None = None


def _size(path: str | None) -> int | None:
    if not path:
        return None
    p = Path(path)
    return p.stat().st_size if p.exists() else None


def _ort_dynamic():
    try:
        from onnxruntime.quantization.quant_utils import QuantType
        from onnxruntime.quantization.quantize import quantize_dynamic
        return QuantType, quantize_dynamic
    except ImportError:
        import sys
        import types

        if "onnxruntime.quantization.shape_inference" not in sys.modules:
            m = types.ModuleType("onnxruntime.quantization.shape_inference")
            m.quant_pre_process = lambda *a, **k: None
            sys.modules["onnxruntime.quantization.shape_inference"] = m
        from onnxruntime.quantization.quant_utils import QuantType
        from onnxruntime.quantization.quantize import quantize_dynamic
        return QuantType, quantize_dynamic


def quantize_e1_dynamic(fp32_dir: Path, out_dir: Path) -> QuantizeResult:
    """E1: dynamic INT8. Encoder ConvInteger + decoder MatMulInteger, no calibration."""
    QuantType, quantize_dynamic = _ort_dynamic()

    out_dir.mkdir(parents=True, exist_ok=True)
    res = QuantizeResult(variant="e1-dynamic-int8", metadata={}, errors=[])
    enc_in = fp32_dir / "encoder.fp32.onnx"
    dec_in = fp32_dir / "decoder_step.fp32.onnx"
    enc_out = out_dir / "encoder.e1.dynamic.onnx"
    dec_out = out_dir / "decoder_step.e1.dynamic.onnx"

    try:
        quantize_dynamic(
            model_input=str(enc_in),
            model_output=str(enc_out),
            op_types_to_quantize=["Conv", "MatMul", "Gemm"],
            per_channel=True,
            reduce_range=False,
            weight_type=QuantType.QInt8,
        )
        res.encoder_path = str(enc_out)
    except Exception as e:  # noqa: BLE001
        res.errors.append(f"encoder e1 failed: {e}")

    try:
        quantize_dynamic(
            model_input=str(dec_in),
            model_output=str(dec_out),
            op_types_to_quantize=["MatMul", "Gemm"],
            per_channel=True,
            reduce_range=False,
            weight_type=QuantType.QInt8,
        )
        res.decoder_path = str(dec_out)
    except Exception as e:  # noqa: BLE001
        res.errors.append(f"decoder e1 failed: {e}")

    res.metadata = {
        "variant": res.variant,
        "encoder_bytes": _size(res.encoder_path),
        "decoder_bytes": _size(res.decoder_path),
        "fp32_encoder_bytes": _size(str(enc_in)),
        "fp32_decoder_bytes": _size(str(dec_in)),
    }
    (out_dir / "e1.dynamic.meta.json").write_text(json.dumps(asdict(res), indent=2) + "\n", encoding="utf-8")
    return res
