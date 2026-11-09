"""Pipeline stage 4 — Quantize (M4).

Build quantization candidates and support selecting the best by measured quality/latency/size.
Candidates: E1 dynamic INT8, E2 static INT8 QDQ, E3 hybrid (static encoder + dynamic decoder), E4 FP16.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np


@dataclass
class QuantizeResult:
    variant: str
    encoder_path: str | None = None
    decoder_path: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)


def _size(path: str | None) -> int | None:
    if not path:
        return None
    p = Path(path)
    return p.stat().st_size if p.exists() else None




def _preprocess_export(image_path: Path) -> np.ndarray:
    """Preprocess for calibration in export env (prefers DECIMER/TF reference, falls back to runtime)."""
    try:
        import DECIMER.pre_process as pre
        tensor = pre.decode_image(str(image_path))
        arr = tensor.numpy() if hasattr(tensor, "numpy") else np.array(tensor)
        return np.asarray(arr, dtype=np.float32)
    except Exception:
        from runtime.preprocess import preprocess_image
        return preprocess_image(image_path).astype(np.float32)

def _stub_shape_inference() -> None:
    import sys
    import types

    if "onnxruntime.quantization.shape_inference" not in sys.modules:
        m = types.ModuleType("onnxruntime.quantization.shape_inference")
        m.quant_pre_process = lambda *a, **k: None
        sys.modules["onnxruntime.quantization.shape_inference"] = m


def _ort_dynamic():
    _stub_shape_inference()
    from onnxruntime.quantization.quant_utils import QuantType
    from onnxruntime.quantization.quantize import quantize_dynamic

    return QuantType, quantize_dynamic


def _ort_static():
    _stub_shape_inference()
    from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static

    return CalibrationDataReader, QuantFormat, QuantType, quantize_static


def _meta(res: QuantizeResult, enc_in: Path, dec_in: Path, extra: dict[str, Any] | None = None) -> None:
    res.metadata = {
        "variant": res.variant,
        "encoder_bytes": _size(res.encoder_path),
        "decoder_bytes": _size(res.decoder_path),
        "fp32_encoder_bytes": _size(str(enc_in)),
        "fp32_decoder_bytes": _size(str(dec_in)),
        **(extra or {}),
    }


class EncoderCalibrationReader:
    """Feeds preprocessed images [1,512,512,3] to the encoder for static calibration."""

    def __init__(self, image_paths: Iterable[Path], input_name: str = "image"):
        self.image_paths = list(image_paths)
        self.input_name = input_name
        self._i = 0

    def get_next(self):
        if self._i >= len(self.image_paths):
            return None
        arr = _preprocess_export(self.image_paths[self._i]).astype(np.float32)[None, ...]
        self._i += 1
        return {self.input_name: arr}

    def rewind(self):
        self._i = 0


class ClassifierCalibrationReader:
    """Feeds preprocessed images + required extra inputs to the image classifier for static calibration."""

    def __init__(self, image_paths: Iterable[Path], model_path: Path):
        import onnxruntime as ort
        from toolchain.benchmark_run import _classifier_preprocess_np
        self.image_paths = list(image_paths)
        self._pre = _classifier_preprocess_np
        sess = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        self._feed_names = [i.name for i in sess.get_inputs()]
        self._feeds_meta = {i.name: (i.shape, i.type) for i in sess.get_inputs()}
        self._i = 0

    def get_next(self):
        if self._i >= len(self.image_paths):
            return None
        x = self._pre(self.image_paths[self._i]).astype(np.float32)  # NHWC [1,224,224,3]
        self._i += 1
        feed = {}
        for name in self._feed_names:
            if name in ("input_2", "image", "images", "input_1", "input") or "input" in name.lower():
                feed[name] = x
            else:
                # extra inputs observed as [1,1,1,3] float zeros
                shape, _ = self._feeds_meta[name]
                shp = [1 if (d is None or isinstance(d, str) or d <= 0) else int(d) for d in shape]
                if not shp:
                    shp = [1, 1, 1, 3]
                feed[name] = np.zeros(shp, dtype=np.float32)
        return feed

    def rewind(self):
        self._i = 0


class DecoderCalibrationReader:
    """Feeds (tokens, enc_out) pairs to the decoder for static calibration.

    Generates representative prefixes by running the FP32 encoder and sampling greedy-like
    token lengths from the FP32 decoder.
    """

    def __init__(
        self,
        image_paths: Iterable[Path],
        fp32_enc_path: Path,
        fp32_dec_path: Path,
        *,
        token_name: str = "tokens",
        enc_name: str = "enc_out",
        max_samples_per_image: int = 12,
    ):
        import onnxruntime as ort

        self.token_name = token_name
        self.enc_name = enc_name
        self.samples: list[dict[str, np.ndarray]] = []
        enc = ort.InferenceSession(str(fp32_enc_path), providers=["CPUExecutionProvider"])
        dec = ort.InferenceSession(str(fp32_dec_path), providers=["CPUExecutionProvider"])
        enc_in = enc.get_inputs()[0].name
        for img_path in image_paths:
            x = _preprocess_export(img_path).astype(np.float32)[None, ...]
            enc_out = enc.run(None, {enc_in: x})[0]
            ids = [13]  # <start>
            steps = 0
            # collect prefixes at varied lengths
            take_at = {1, 2, 3, 5, 8, 13, 21, 34, 55}
            while steps < 120:
                tokens = np.array([ids], dtype=np.int32)
                self._maybe_add(tokens, enc_out, steps, take_at, max_samples_per_image)
                probs = dec.run(None, {self.token_name: tokens, self.enc_name: enc_out})[0]
                pred = int(np.argmax(probs[0, -1]))
                ids.append(pred)
                steps += 1
                if pred == 14:  # <end>
                    tokens = np.array([ids], dtype=np.int32)
                    self._maybe_add(tokens, enc_out, steps, take_at, max_samples_per_image)
                    break
        # dedupe by (len, last few tokens) to cap size
        seen = set()
        uniq = []
        for s in self.samples:
            t = s[self.token_name][0]
            key = (len(t), tuple(t[-4:].tolist()))
            if key not in seen:
                seen.add(key)
                uniq.append(s)
        self.samples = uniq[: max_samples_per_image * max(1, len(list(image_paths)))]
        self._i = 0

    def _maybe_add(self, tokens, enc_out, steps, take_at, limit):
        if len(self.samples) >= limit * 4:
            return
        if steps in take_at or steps % 10 == 0:
            self.samples.append(
                {
                    self.token_name: tokens.astype(np.int32),
                    self.enc_name: enc_out.astype(np.float32),
                }
            )

    def get_next(self):
        if self._i >= len(self.samples):
            return None
        s = self.samples[self._i]
        self._i += 1
        return s

    def rewind(self):
        self._i = 0


def _calibration_images(data_dir: Path | None = None) -> list[Path]:
    root = Path(__file__).resolve().parents[2]
    candidates = []
    for base in [root / "tests" / "fixtures", root / "data" / "calibration"]:
        if base.exists():
            candidates.extend(sorted(base.glob("*.png")))
            candidates.extend(sorted(base.glob("*.jpg")))
    if data_dir and data_dir.exists():
        candidates.extend(sorted(data_dir.glob("*.png")))
        candidates.extend(sorted(data_dir.glob("*.jpg")))
        candidates.extend(sorted(data_dir.glob("**/*.png")))
        candidates.extend(sorted(data_dir.glob("**/*.jpg")))
        candidates.extend(sorted(data_dir.glob("**/*.tif")))
        candidates.extend(sorted(data_dir.glob("**/*.TIF")))
    # unique, existing
    seen = set()
    out = []
    for p in candidates:
        rp = p.resolve() if p.exists() else p
        if p.exists() and rp not in seen:
            seen.add(rp)
            out.append(p)
    return out


def _classifier_holdout_calibration_images(limit: int = 128, seed: int = 20260308) -> list[Path]:
    """Sample calibration images from classifier frozen non-eval splits (holdout, no leakage)."""
    import json, random
    root = Path(__file__).resolve().parents[2]
    man = root / "data/manifests/frozen/classifier.frozen.json"
    if not man.exists():
        return []
    m = json.loads(man.read_text(encoding="utf-8"))
    splits = m.get("splits") or {}
    eval_set = set(splits.get("eval") or [])
    pool = [iid for iid in (splits.get("frozen") or []) if iid not in eval_set]
    if not pool:
        pool = list(splits.get("remainder") or [])
    rng = random.Random(seed)
    rng.shuffle(pool)
    base = root / "data/benchmark/classifier-curated/classified"
    out = []
    for iid in pool:
        # portable: classifier::<folder>/<file>
        rel = iid.split("::", 1)[1] if "::" in iid else iid
        p = base / rel
        # handle rest->? folder name as stored
        if not p.exists():
            # try alternate rest folder naming
            alt = base / rel.replace("other/", "rest/")
            p = alt if alt.exists() else p
        if p.exists():
            out.append(p)
        if len(out) >= limit:
            break
    return out


def quantize_e1_dynamic(fp32_dir: Path, out_dir: Path) -> QuantizeResult:
    """E1: dynamic INT8. No calibration. Supports OCSR (enc+dec) and classifier (single file)."""
    QuantType, quantize_dynamic = _ort_dynamic()
    out_dir.mkdir(parents=True, exist_ok=True)
    clf_in = fp32_dir / "classifier.fp32.onnx"
    if clf_in.exists():
        res = QuantizeResult(variant="e1-dynamic-int8-classifier")
        out = out_dir / "classifier.e1.dynamic.onnx"
        try:
            quantize_dynamic(str(clf_in), str(out), op_types_to_quantize=["MatMul", "Gemm", "Conv"], per_channel=True, reduce_range=False, weight_type=QuantType.QInt8)
            res.encoder_path = str(out)
        except Exception as e:  # noqa: BLE001
            res.errors.append(f"classifier e1 failed: {e}")
        _meta(res, clf_in, clf_in, {"calibration_images": 0, "mode": "classifier"})
        (out_dir / "e1.dynamic.meta.json").write_text(json.dumps(asdict(res), indent=2) + "\n", encoding="utf-8")
        return res
    res = QuantizeResult(variant="e1-dynamic-int8")
    enc_in = fp32_dir / "encoder.fp32.onnx"
    dec_in = fp32_dir / "decoder_step.fp32.onnx"
    enc_out = out_dir / "encoder.e1.dynamic.onnx"
    dec_out = out_dir / "decoder_step.e1.dynamic.onnx"
    try:
        quantize_dynamic(str(enc_in), str(enc_out), op_types_to_quantize=["Conv", "MatMul", "Gemm"], per_channel=True, reduce_range=False, weight_type=QuantType.QInt8)
        res.encoder_path = str(enc_out)
    except Exception as e:  # noqa: BLE001
        res.errors.append(f"encoder e1 failed: {e}")
    try:
        quantize_dynamic(str(dec_in), str(dec_out), op_types_to_quantize=["MatMul", "Gemm"], per_channel=True, reduce_range=False, weight_type=QuantType.QInt8)
        res.decoder_path = str(dec_out)
    except Exception as e:  # noqa: BLE001
        res.errors.append(f"decoder e1 failed: {e}")
    _meta(res, enc_in, dec_in, {"calibration_images": 0})
    (out_dir / "e1.dynamic.meta.json").write_text(json.dumps(asdict(res), indent=2) + "\n", encoding="utf-8")
    return res


def quantize_e2_static(fp32_dir: Path, out_dir: Path, calibration_dir: Path | None = None) -> QuantizeResult:
    """E2: static INT8 QDQ with calibration. Supports classifier (single file, holdout calib) and OCSR (enc+dec)."""
    CalibrationDataReader, QuantFormat, QuantType, quantize_static = _ort_static()
    _ = CalibrationDataReader
    out_dir.mkdir(parents=True, exist_ok=True)
    clf_in = fp32_dir / "classifier.fp32.onnx"
    if clf_in.exists():
        res = QuantizeResult(variant="e2-static-int8-qdq-classifier")
        out = out_dir / "classifier.e2.static.onnx"
        images = _calibration_images(calibration_dir) if calibration_dir else []
        if not images:
            images = _classifier_holdout_calibration_images(limit=128)
        if not images:
            res.errors.append("no calibration images found for classifier (holdout/frozen or calibration_dir)")
            _meta(res, clf_in, clf_in, {"calibration_images": 0, "mode": "classifier"})
            (out_dir / "e2.static.meta.json").write_text(json.dumps(asdict(res), indent=2) + "\n", encoding="utf-8")
            return res

        class _ClfReader(CalibrationDataReader):
            def __init__(self):
                self.r = ClassifierCalibrationReader(images, clf_in)

            def get_next(self):
                return self.r.get_next()

            def rewind(self):
                self.r.rewind()

        try:
            quantize_static(
                str(clf_in),
                str(out),
                _ClfReader(),
                quant_format=QuantFormat.QDQ,
                per_channel=True,
                weight_type=QuantType.QInt8,
                activation_type=QuantType.QUInt8,
                op_types_to_quantize=["Conv", "MatMul", "Gemm", "Add"],
            )
            res.encoder_path = str(out)
        except Exception as e:  # noqa: BLE001
            res.errors.append(f"classifier e2 failed: {e}")
        _meta(res, clf_in, clf_in, {"calibration_images": len(images), "mode": "classifier", "calibration": "holdout_frozen_non_eval" if not calibration_dir else "dir"})
        (out_dir / "e2.static.meta.json").write_text(json.dumps(asdict(res), indent=2) + "\n", encoding="utf-8")
        return res

    res = QuantizeResult(variant="e2-static-int8-qdq")
    enc_in = fp32_dir / "encoder.fp32.onnx"
    dec_in = fp32_dir / "decoder_step.fp32.onnx"
    enc_out = out_dir / "encoder.e2.static.onnx"
    dec_out = out_dir / "decoder_step.e2.static.onnx"
    images = _calibration_images(calibration_dir)
    if not images:
        res.errors.append("no calibration images found (tests/fixtures or data/calibration)")
        _meta(res, enc_in, dec_in, {"calibration_images": 0})
        return res

    class _EncReader(CalibrationDataReader):
        def __init__(self):
            self.r = EncoderCalibrationReader(images, input_name="image")

        def get_next(self):
            return self.r.get_next()

        def rewind(self):
            self.r.rewind()

    try:
        quantize_static(
            str(enc_in),
            str(enc_out),
            _EncReader(),
            quant_format=QuantFormat.QDQ,
            per_channel=True,
            weight_type=QuantType.QInt8,
            activation_type=QuantType.QUInt8,
            op_types_to_quantize=["Conv", "MatMul", "Gemm", "Add"],
        )
        res.encoder_path = str(enc_out)
    except Exception as e:  # noqa: BLE001
        res.errors.append(f"encoder e2 failed: {e}")

    class _DecReader(CalibrationDataReader):
        def __init__(self):
            self.r = DecoderCalibrationReader(images, enc_in, dec_in, token_name="tokens", enc_name="enc_out")

        def get_next(self):
            return self.r.get_next()

        def rewind(self):
            self.r.rewind()

    try:
        quantize_static(
            str(dec_in),
            str(dec_out),
            _DecReader(),
            quant_format=QuantFormat.QDQ,
            per_channel=True,
            weight_type=QuantType.QInt8,
            activation_type=QuantType.QUInt8,
            op_types_to_quantize=["MatMul", "Gemm", "Add"],
        )
        res.decoder_path = str(dec_out)
    except Exception as e:  # noqa: BLE001
        res.errors.append(f"decoder e2 failed: {e}")

    _meta(res, enc_in, dec_in, {"calibration_images": len(images)})
    (out_dir / "e2.static.meta.json").write_text(json.dumps(asdict(res), indent=2) + "\n", encoding="utf-8")
    return res


def quantize_e3_hybrid(fp32_dir: Path, out_dir: Path, calibration_dir: Path | None = None) -> QuantizeResult:
    """E3: hybrid = static INT8 QDQ encoder + dynamic INT8 decoder."""
    CalibrationDataReader, QuantFormat, QuantType, quantize_static = _ort_static()
    _, quantize_dynamic = _ort_dynamic()
    out_dir.mkdir(parents=True, exist_ok=True)
    res = QuantizeResult(variant="e3-hybrid-staticenc-dyndec")
    enc_in = fp32_dir / "encoder.fp32.onnx"
    dec_in = fp32_dir / "decoder_step.fp32.onnx"
    enc_out = out_dir / "encoder.e3.static.onnx"
    dec_out = out_dir / "decoder_step.e3.dynamic.onnx"
    images = _calibration_images(calibration_dir)
    if not images:
        res.errors.append("no calibration images found (tests/fixtures or data/calibration)")

    class _EncReader(CalibrationDataReader):
        def __init__(self):
            self.r = EncoderCalibrationReader(images, input_name="image")

        def get_next(self):
            return self.r.get_next()

        def rewind(self):
            self.r.rewind()

    try:
        if images:
            quantize_static(
                str(enc_in),
                str(enc_out),
                _EncReader(),
                quant_format=QuantFormat.QDQ,
                per_channel=True,
                weight_type=QuantType.QInt8,
                activation_type=QuantType.QUInt8,
                op_types_to_quantize=["Conv", "MatMul", "Gemm", "Add"],
            )
            res.encoder_path = str(enc_out)
    except Exception as e:  # noqa: BLE001
        res.errors.append(f"encoder e3 failed: {e}")

    try:
        quantize_dynamic(str(dec_in), str(dec_out), op_types_to_quantize=["MatMul", "Gemm"], per_channel=True, reduce_range=False, weight_type=QuantType.QInt8)
        res.decoder_path = str(dec_out)
    except Exception as e:  # noqa: BLE001
        res.errors.append(f"decoder e3 failed: {e}")

    _meta(res, enc_in, dec_in, {"calibration_images": len(images)})
    (out_dir / "e3.hybrid.meta.json").write_text(json.dumps(asdict(res), indent=2) + "\n", encoding="utf-8")
    return res


def _attention_node_names(model_path: Path) -> list[str]:
    """Return node names matching attention patterns (for selective quantization exclusion)."""
    import onnx

    m = onnx.load(str(model_path))
    pats = ("multi_head", "attention", "self_attention", "cross_attention", "SelfAttention", "MultiHead")
    out = []
    for n in m.graph.node:
        blob = (n.name or "") + " " + " ".join(n.input) + " " + " ".join(n.output)
        low = blob.lower()
        if any(p.lower() in low for p in pats):
            out.append(n.name)
    # nodes_to_exclude expects node names; keep only non-empty unique
    return sorted({x for x in out if x})


def quantize_e1s_selective(fp32_dir: Path, out_dir: Path) -> QuantizeResult:
    """E1s: selective dynamic INT8.
    - classifier: quantize MatMul/Gemm only (exclude Conv) to avoid E1-all regression.
    - OCSR: encoder Conv+MatMul+Gemm dynamic; decoder MatMul/Gemm dynamic excluding attention nodes.
    """
    QuantType, quantize_dynamic = _ort_dynamic()
    out_dir.mkdir(parents=True, exist_ok=True)
    clf_in = fp32_dir / "classifier.fp32.onnx"
    if clf_in.exists():
        res = QuantizeResult(variant="e1s-selective-dynamic-int8-classifier")
        out = out_dir / "classifier.e1s.dynamic.onnx"
        try:
            quantize_dynamic(
                str(clf_in), str(out),
                op_types_to_quantize=["MatMul", "Gemm"],  # exclude Conv (E1-all regressed badly)
                per_channel=True, reduce_range=False, weight_type=QuantType.QInt8,
            )
            res.encoder_path = str(out)
        except Exception as e:  # noqa: BLE001
            res.errors.append(f"classifier e1s failed: {e}")
        _meta(res, clf_in, clf_in, {"calibration_images": 0, "mode": "classifier", "op_types_to_quantize": ["MatMul", "Gemm"], "excluded": "Conv"})
        (out_dir / "e1s.dynamic.meta.json").write_text(json.dumps(asdict(res), indent=2) + "\n", encoding="utf-8")
        return res

    res = QuantizeResult(variant="e1s-selective-dynamic-int8-noattn")
    enc_in = fp32_dir / "encoder.fp32.onnx"
    dec_in = fp32_dir / "decoder_step.fp32.onnx"
    enc_out = out_dir / "encoder.e1s.dynamic.onnx"
    dec_out = out_dir / "decoder_step.e1s.dynamic.onnx"
    try:
        quantize_dynamic(
            str(enc_in), str(enc_out),
            op_types_to_quantize=["Conv", "MatMul", "Gemm"],
            per_channel=True, reduce_range=False, weight_type=QuantType.QInt8,
        )
        res.encoder_path = str(enc_out)
    except Exception as e:  # noqa: BLE001
        res.errors.append(f"encoder e1s failed: {e}")
    dec_excl = _attention_node_names(dec_in)
    try:
        quantize_dynamic(
            str(dec_in), str(dec_out),
            op_types_to_quantize=["MatMul", "Gemm"],
            nodes_to_exclude=dec_excl or None,
            per_channel=True, reduce_range=False, weight_type=QuantType.QInt8,
        )
        res.decoder_path = str(dec_out)
    except Exception as e:  # noqa: BLE001
        res.errors.append(f"decoder e1s failed: {e}")
    _meta(res, enc_in, dec_in, {"calibration_images": 0, "decoder_nodes_excluded_attention": len(dec_excl), "decoder_nodes_excluded": dec_excl[:40]})
    (out_dir / "e1s.dynamic.meta.json").write_text(json.dumps(asdict(res), indent=2) + "\n", encoding="utf-8")
    return res


def quantize_e4_fp16(fp32_dir: Path, out_dir: Path) -> QuantizeResult:
    """E4: FP16 conversion (safety net). Known CPU runtime issue in OCSR decoder attention (see E4 notes)."""
    import onnx
    from onnxruntime.transformers.float16 import convert_float_to_float16

    out_dir.mkdir(parents=True, exist_ok=True)
    clf_in = fp32_dir / "classifier.fp32.onnx"
    if clf_in.exists():
        res = QuantizeResult(variant="e4-fp16-classifier")
        out = out_dir / "classifier.e4.fp16.onnx"
        try:
            model = onnx.load(str(clf_in))
            model_fp16 = convert_float_to_float16(model, keep_io_types=True)
            onnx.save(model_fp16, str(out))
            res.encoder_path = str(out)
        except Exception as e:  # noqa: BLE001
            res.errors.append(f"classifier e4 failed: {e}")
        _meta(res, clf_in, clf_in, {"mode": "classifier"})
        (out_dir / "e4.fp16.meta.json").write_text(json.dumps(asdict(res), indent=2) + "\n", encoding="utf-8")
        return res

    res = QuantizeResult(variant="e4-fp16")
    enc_in = fp32_dir / "encoder.fp32.onnx"
    dec_in = fp32_dir / "decoder_step.fp32.onnx"
    enc_out = out_dir / "encoder.e4.fp16.onnx"
    dec_out = out_dir / "decoder_step.e4.fp16.onnx"

    for name, src, dst in [("encoder", enc_in, enc_out), ("decoder", dec_in, dec_out)]:
        try:
            model = onnx.load(str(src))
            model_fp16 = convert_float_to_float16(model, keep_io_types=True)
            onnx.save(model_fp16, str(dst))
            if name == "encoder":
                res.encoder_path = str(dst)
            else:
                res.decoder_path = str(dst)
        except Exception as e:  # noqa: BLE001
            res.errors.append(f"{name} e4 failed: {e}")

    _meta(res, enc_in, dec_in)
    (out_dir / "e4.fp16.meta.json").write_text(json.dumps(asdict(res), indent=2) + "\n", encoding="utf-8")
    return res
