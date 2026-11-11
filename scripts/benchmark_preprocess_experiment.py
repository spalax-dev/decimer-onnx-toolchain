#!/usr/bin/env python3
"""Experiment: preprocessing strategies/filters for the 3 selected DECIMER ONNX models.

Like the Imago experiment: tests alternative preprocessing configurations against
frozen benchmarks and records results (not part of official report).

Models (selected): ocsr-printed FP32, ocsr-handdrawn FP32, image-classifier E4 FP16.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from runtime.preprocess import (
    central_square_image,
    delete_empty_borders,
    get_bnw_image,
    get_resize,
    increase_brightness,
    increase_contrast,
    preprocess_image as runtime_preprocess,
    remove_transparent,
    _IMAGENET_MEAN,
    _IMAGENET_STD,
)
from toolchain.frozen import load_frozen, resolve_frozen_rows
from toolchain.benchmark_run import (
    benchmark_classifier,
    benchmark_ocsr,
    make_onnx_classifier_predict_fn,
    make_onnx_ocsr_predict_fn,
    _classifier_preprocess_np,
    CLASSIFIER_THRESHOLD,
)
from toolchain.benchmark_data import ensure_benchmark_data


# ---------- OCSR preprocessing variants ----------

def _finalize(arr_img: Image.Image) -> np.ndarray:
    rgb = arr_img.convert("RGB").resize((512, 512), resample=Image.Resampling.LANCZOS)
    arr = np.asarray(rgb, dtype=np.float32)
    arr = (arr / 255.0 - _IMAGENET_MEAN) / _IMAGENET_STD
    return arr.astype(np.float32)


def pp_runtime(image) -> np.ndarray:
    return runtime_preprocess(image)


def pp_decimer_tf(image) -> np.ndarray:
    """Use DECIMER 2.8.1 reference preprocessing (TF gaussian resize + efn preprocess)."""
    import DECIMER.pre_process as pre
    t = pre.decode_image(str(image) if not isinstance(image, np.ndarray) else image)
    return t.numpy().astype(np.float32)


def pp_no_binarize(image) -> np.ndarray:
    img = remove_transparent(image)
    img = increase_contrast(img)
    # skip get_bnw_image; keep RGB->L-like but preserve grayscale info differently
    img = img.convert("L")
    img = get_resize(img)
    img = delete_empty_borders(img)
    img = central_square_image(img)
    img = increase_brightness(img)
    return _finalize(img)


def pp_no_contrast_boost(image) -> np.ndarray:
    img = remove_transparent(image)
    # skip increase_contrast
    img = get_bnw_image(img)
    img = get_resize(img)
    img = delete_empty_borders(img)
    img = central_square_image(img)
    img = increase_brightness(img)
    return _finalize(img)


def pp_no_brightness_boost(image) -> np.ndarray:
    img = remove_transparent(image)
    img = increase_contrast(img)
    img = get_bnw_image(img)
    img = get_resize(img)
    img = delete_empty_borders(img)
    img = central_square_image(img)
    # skip increase_brightness
    return _finalize(img)


def pp_autocontrast(image) -> np.ndarray:
    img = remove_transparent(image)
    img = ImageOps.autocontrast(img.convert("RGB"), cutoff=1)
    img = get_bnw_image(img)
    img = get_resize(img)
    img = delete_empty_borders(img)
    img = central_square_image(img)
    img = increase_brightness(img)
    return _finalize(img)


def pp_invert(image) -> np.ndarray:
    # helpful if input has dark background/light strokes in some cases
    img = remove_transparent(image)
    img = increase_contrast(img)
    img = get_bnw_image(img)
    img = ImageOps.invert(img.convert("L"))
    img = get_resize(img)
    img = delete_empty_borders(img)
    img = central_square_image(img)
    img = increase_brightness(img)
    return _finalize(img)


def pp_sharpen(image) -> np.ndarray:
    img = remove_transparent(image)
    img = increase_contrast(img)
    img = get_bnw_image(img)
    img = img.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))
    img = get_resize(img)
    img = delete_empty_borders(img)
    img = central_square_image(img)
    img = increase_brightness(img)
    return _finalize(img)


def pp_denoise_median(image) -> np.ndarray:
    img = remove_transparent(image)
    img = increase_contrast(img)
    img = get_bnw_image(img)
    img = img.filter(ImageFilter.MedianFilter(size=3))
    img = get_resize(img)
    img = delete_empty_borders(img)
    img = central_square_image(img)
    img = increase_brightness(img)
    return _finalize(img)


def pp_stronger_contrast(image) -> np.ndarray:
    img = remove_transparent(image)
    img = increase_contrast(img)
    img = get_bnw_image(img)
    # extra contrast
    img = ImageEnhance.Contrast(img).enhance(2.2)
    img = get_resize(img)
    img = delete_empty_borders(img)
    img = central_square_image(img)
    img = increase_brightness(img)
    return _finalize(img)


def pp_no_border_crop(image) -> np.ndarray:
    img = remove_transparent(image)
    img = increase_contrast(img)
    img = get_bnw_image(img)
    img = get_resize(img)
    # skip delete_empty_borders
    img = central_square_image(img)
    img = increase_brightness(img)
    return _finalize(img)


OCSR_VARIANTS: dict[str, Callable] = {
    "runtime_default": pp_runtime,
    "decimer_tf_ref": pp_decimer_tf,
    "no_binarize": pp_no_binarize,
    "no_contrast_boost": pp_no_contrast_boost,
    "no_brightness_boost": pp_no_brightness_boost,
    "autocontrast": pp_autocontrast,
    "invert": pp_invert,
    "sharpen_unsharp": pp_sharpen,
    "denoise_median3": pp_denoise_median,
    "stronger_contrast": pp_stronger_contrast,
    "no_border_crop": pp_no_border_crop,
}


# ---------- Classifier preprocessing variants ----------

def _cls_base_pil(path: Path):
    img = Image.open(path)
    if img.mode == "RGBA":
        bg = Image.new("RGB", img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[3])
        img = bg
    return img


def _cls_resize_pad(img: Image.Image, desired_size: int = 224) -> Image.Image:
    img = img.convert("L").convert("RGB")
    old = img.size
    if old[0] != desired_size or old[1] != desired_size:
        ratio = float(desired_size) / max(old)
        new_size = (int(old[0] * ratio), int(old[1] * ratio))
        try:
            resample = Image.Resampling.LANCZOS
        except Exception:
            resample = Image.LANCZOS
        img = img.resize(new_size, resample)
    else:
        new_size = old
    canvas = Image.new("RGB", (desired_size, desired_size), "white")
    canvas.paste(img, ((desired_size - new_size[0]) // 2, (desired_size - new_size[1]) // 2))
    return canvas


def cls_default(path: Path) -> np.ndarray:
    return _classifier_preprocess_np(path)


def cls_autocontrast(path: Path) -> np.ndarray:
    img = _cls_base_pil(path)
    img = ImageOps.autocontrast(img.convert("RGB"), cutoff=1)
    canvas = _cls_resize_pad(img)
    return np.expand_dims(np.asarray(canvas, dtype=np.float32), 0)


def cls_sharpen(path: Path) -> np.ndarray:
    img = _cls_base_pil(path)
    canvas = _cls_resize_pad(img)
    canvas = canvas.filter(ImageFilter.UnsharpMask(radius=1.5, percent=120, threshold=3))
    return np.expand_dims(np.asarray(canvas, dtype=np.float32), 0)


def cls_invert(path: Path) -> np.ndarray:
    img = _cls_base_pil(path)
    canvas = _cls_resize_pad(img)
    canvas = ImageOps.invert(canvas.convert("RGB"))
    return np.expand_dims(np.asarray(canvas, dtype=np.float32), 0)


def cls_higher_contrast(path: Path) -> np.ndarray:
    img = _cls_base_pil(path)
    canvas = _cls_resize_pad(img)
    canvas = ImageEnhance.Contrast(canvas).enhance(1.5)
    return np.expand_dims(np.asarray(canvas, dtype=np.float32), 0)


def cls_equalize(path: Path) -> np.ndarray:
    img = _cls_base_pil(path)
    canvas = _cls_resize_pad(img)
    canvas = ImageOps.equalize(canvas.convert("RGB"))
    return np.expand_dims(np.asarray(canvas, dtype=np.float32), 0)


CLS_VARIANTS: dict[str, Callable] = {
    "default": cls_default,
    "autocontrast": cls_autocontrast,
    "sharpen": cls_sharpen,
    "invert": cls_invert,
    "higher_contrast": cls_higher_contrast,
    "equalize": cls_equalize,
}


# ---------- Predict wrappers ----------

def make_ocsr_predict_with_preprocess(encoder, decoder, vocab, preprocess_fn):
    from runtime.api import DecimerONNX
    model = DecimerONNX(encoder, decoder, vocab)
    def _fn(img: Path):
        arr = preprocess_fn(img)
        p = model.predict_preprocessed(arr)
        return {"smiles": p.smiles, "incomplete": p.incomplete, "token_ids": p.token_ids}
    return _fn


def make_classifier_predict_with_preprocess(model_path: Path, preprocess_fn, threshold: float = CLASSIFIER_THRESHOLD):
    import onnxruntime as ort
    sess = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    inputs = sess.get_inputs()
    def _fn(img: Path):
        x = preprocess_fn(img)
        feed: dict[str, Any] = {}
        for i in inputs:
            shape = [d if isinstance(d, int) and d > 0 else 1 for d in i.shape]
            if i.name == "input_2" or shape[-3:] == [224, 224, 3]:
                feed[i.name] = x
            elif i.type == "tensor(float)" or "float" in str(i.type):
                feed[i.name] = np.zeros(shape, dtype=np.float32)
            else:
                feed[i.name] = np.zeros(shape)
        logit = sess.run(None, feed)[0]
        score = 1.0 / (1.0 + np.exp(-float(np.asarray(logit).reshape(-1)[0])))
        label = "structure" if score <= threshold else "no-structure"
        return {"label": label, "score": score, "logit": float(np.asarray(logit).reshape(-1)[0])}
    return _fn


def run_ocsr(model_key: str, encoder: Path, decoder: Path, vocab: Path, manifest: Path, split: str, variant_name: str, pp_fn, out_path: Path):
    frozen = load_frozen(manifest)
    ensure_benchmark_data(frozen.get('model') or frozen.get('dossier') or '', registry=None)
    rows, labels_path = resolve_frozen_rows(frozen, split=split)
    predict = make_ocsr_predict_with_preprocess(encoder, decoder, vocab, pp_fn)
    report = benchmark_ocsr(labels_path, rows=rows, predict_fn=predict, tf_predict_fn=None, limit=None, cache=None, model_key=model_key, variant_key=f"preproc-{variant_name}")
    report["preprocess_variant"] = variant_name
    report["split"] = split
    report["frozen_manifest"] = str(manifest)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    cr = report.get("canonical_match_rate")
    print(f"[OCSR {model_key} {variant_name} {split}] n={report['count']} canonical={report.get('canonical_match')}/{report.get('canonical_comparable')} ({cr}) exact={report.get('exact_match_rate')} valid={report.get('valid_rate')} p50={report['latency_ms']['p50']}", flush=True)
    return report


def run_classifier(model_path: Path, manifest: Path, split: str, variant_name: str, pp_fn, out_path: Path):
    frozen = load_frozen(manifest)
    ensure_benchmark_data(frozen.get('model') or frozen.get('dossier') or '', registry=None)
    rows, labels_path = resolve_frozen_rows(frozen, split=split)
    predict = make_classifier_predict_with_preprocess(model_path, pp_fn)
    report = benchmark_classifier(labels_path, rows=rows, predict_fn=predict, limit=None, cache=None, model_key="image-classifier", variant_key=f"preproc-{variant_name}")
    report["preprocess_variant"] = variant_name
    report["split"] = split
    report["frozen_manifest"] = str(manifest)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    cm = report.get("binary_confusion_matrix") or {}
    print(f"[CLS {variant_name} {split}] n={report.get('total_rows')} acc={cm.get('accuracy')} p50={report['latency_ms']['p50']}", flush=True)
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["ocsr-printed", "ocsr-handdrawn", "classifier", "all"], default="all")
    ap.add_argument("--split", choices=["pilot", "eval"], default="pilot")
    ap.add_argument("--variants", nargs="*", help="Subset of variant names (default: all for model)")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "data/benchmark/experiments/preprocess")
    args = ap.parse_args()

    split = args.split
    suffix = "pilot300" if split == "pilot" else "eval1000"
    out_dir = args.out_dir / suffix
    results = []

    if args.model in ("ocsr-printed", "all"):
        enc = ROOT/"data/artifacts/fp32/ocsr-printed/encoder.fp32.onnx"
        dec = ROOT/"data/artifacts/fp32/ocsr-printed/decoder_step.fp32.onnx"
        vocab = ROOT/"data/artifacts/tokenizer_SMILES.vocab.json"
        man = ROOT/"data/manifests/frozen/printed.frozen.json"
        names = args.variants or list(OCSR_VARIANTS.keys())
        for name in names:
            fn = OCSR_VARIANTS[name]
            out = out_dir / f"ocsr-printed.{name}.{suffix}.json"
            try:
                results.append(run_ocsr("ocsr-printed", enc, dec, vocab, man, split, name, fn, out))
            except Exception as e:
                print(f"ERROR ocsr-printed {name}: {e}", file=sys.stderr)

    if args.model in ("ocsr-handdrawn", "all"):
        enc = ROOT/"data/artifacts/fp32/ocsr-handdrawn/encoder.fp32.onnx"
        dec = ROOT/"data/artifacts/fp32/ocsr-handdrawn/decoder_step.fp32.onnx"
        vocab = ROOT/"data/artifacts/tokenizer_handdrawn.vocab.json"
        man = ROOT/"data/manifests/frozen/handdrawn.frozen.json"
        names = args.variants or list(OCSR_VARIANTS.keys())
        for name in names:
            fn = OCSR_VARIANTS[name]
            out = out_dir / f"ocsr-handdrawn.{name}.{suffix}.json"
            try:
                results.append(run_ocsr("ocsr-handdrawn", enc, dec, vocab, man, split, name, fn, out))
            except Exception as e:
                print(f"ERROR ocsr-handdrawn {name}: {e}", file=sys.stderr)

    if args.model in ("classifier", "all"):
        model_path = ROOT/"data/artifacts/quant/e4/image-classifier/classifier.e4.fp16.onnx"
        man = ROOT/"data/manifests/frozen/classifier.frozen.json"
        names = args.variants or list(CLS_VARIANTS.keys())
        for name in names:
            fn = CLS_VARIANTS[name]
            out = out_dir / f"classifier.{name}.{suffix}.json"
            try:
                results.append(run_classifier(model_path, man, split, name, fn, out))
            except Exception as e:
                print(f"ERROR classifier {name}: {e}", file=sys.stderr)

    print(f"done. {len(results)} runs -> {out_dir}")


if __name__ == "__main__":
    main()
