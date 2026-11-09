"""Pipeline stage 5 — benchmark/validation harness.

Evaluates ONNX variants (and optionally TF baseline) against labeled benchmark suites.
Supports frozen manifests/splits and prediction caching by (model_key, variant_key, image_id).
OCSR: reports exact SMILES match, parity vs TF (token/SMILES), valid/incomplete, latency.
Classifier: binary confusion matrix over mapped categories; reactions reported separately.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

from decimer_onnx_export.benchmark_import import CLASSIFIER_CATEGORY_MAP
from decimer_onnx_export.frozen import PredictionCache


def _load_labels(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def _resolve_image(row: dict, labels_path: Path | None = None) -> Path:
    img = row["image"]
    p = Path(img)
    if p.is_absolute():
        return p
    bases = []
    if row.get("_labels_path"):
        bases.append(Path(row["_labels_path"]).parent)
    if labels_path is not None:
        bases.append(labels_path.parent)
        bases.append(labels_path.parent.parent)
    for base in bases:
        cand = base / p
        if cand.exists():
            return cand
    return (bases[0] / p) if bases else p


def _confusion(binary_rows: list[tuple[str, str]]) -> dict[str, Any]:
    tp = sum(1 for e, p in binary_rows if e == "structure" and p == "structure")
    tn = sum(1 for e, p in binary_rows if e == "no-structure" and p == "no-structure")
    fp = sum(1 for e, p in binary_rows if e == "no-structure" and p == "structure")
    fn = sum(1 for e, p in binary_rows if e == "structure" and p == "no-structure")
    total = len(binary_rows)
    acc = (tp + tn) / total if total else 0.0
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn, "total": total, "accuracy": acc,
            "labels": ["structure", "no-structure"],
            "matrix_rows_expected_cols_predicted": [[tp, fn], [fp, tn]]}


def _image_id(row: dict) -> str:
    return str(row.get("_frozen_key") or row.get("id") or row.get("image"))


def _cached_call(cache: PredictionCache | None, model_key: str | None, variant_key: str | None,
                 image_id: str, fn: Callable, img: Path, *, time_it: bool = True):
    if cache and model_key and variant_key:
        cached = cache.load(model_key, variant_key)
        if image_id in cached and cached[image_id] is not None:
            pred = cached[image_id]
            return pred, 0.0, True
    t0 = time.perf_counter()
    pred = fn(img)
    dt = (time.perf_counter() - t0) * 1000 if time_it else 0.0
    if cache and model_key and variant_key:
        cache.append(model_key, variant_key, image_id, pred)
    return pred, dt, False


def benchmark_classifier(labels_path: Path | None = None, *, rows: list[dict[str, Any]] | None = None,
                         predict_fn, limit: int | None = None, cache: PredictionCache | None = None,
                         model_key: str | None = None, variant_key: str | None = None) -> dict[str, Any]:
    if rows is None:
        rows = _load_labels(labels_path)
    if limit:
        rows = rows[:limit]
    binary_pairs: list[tuple[str, str]] = []
    reactions_report: list[dict[str, Any]] = []
    details = []
    cache_hits = 0
    for row in rows:
        cat = row.get("category")
        mapping = CLASSIFIER_CATEGORY_MAP.get(cat) or row
        img = _resolve_image(row, labels_path)
        iid = _image_id(row)
        pred, dt, hit = _cached_call(cache, model_key, variant_key, iid, predict_fn, img)
        cache_hits += int(hit)
        pred_label = pred["label"] if isinstance(pred, dict) else str(pred)
        rec = {"id": iid, "image": row.get("image"), "category": cat, "predicted": pred_label, "latency_ms": round(dt, 2), "cache_hit": hit}
        if cat == "reactions" or not mapping.get("included_in_binary", row.get("included_in_binary", False)):
            rec["excluded_from_binary"] = True
            rec["report_as"] = "ambiguous_excluded"
            reactions_report.append(rec)
        else:
            expected = mapping.get("binary") or row.get("binary_label")
            rec["expected_binary"] = expected
            rec["included_in_binary"] = True
            binary_pairs.append((str(expected), pred_label))
        details.append(rec)
    res = {
        "task": "image-classifier-binary",
        "mapping": CLASSIFIER_CATEGORY_MAP,
        "binary_confusion_matrix": _confusion(binary_pairs),
        "binary_note": "Computed only over one_molecule, several_molecules, other.",
        "reactions_separate_metric": {
            "count": len(reactions_report), "excluded_from_binary": True, "report_as": "ambiguous",
            "predicted_structure": sum(1 for r in reactions_report if r["predicted"] == "structure"),
            "predicted_no_structure": sum(1 for r in reactions_report if r["predicted"] == "no-structure"),
            "items": reactions_report,
        },
        "total_rows": len(rows), "cache_hits": cache_hits, "details": details,
    }
    if model_key or variant_key:
        res["model_key"] = model_key; res["variant_key"] = variant_key
    return res


def benchmark_ocsr(labels_path: Path | None = None, *, rows: list[dict[str, Any]] | None = None,
                   predict_fn, tf_predict_fn=None, limit: int | None = None,
                   cache: PredictionCache | None = None, model_key: str | None = None,
                   variant_key: str | None = None, tf_variant_key: str | None = "tf-baseline") -> dict[str, Any]:
    """Evaluate OCSR predictions. Supports rows (frozen/multi-source) and prediction cache."""
    if rows is None:
        rows = _load_labels(labels_path)
    if limit:
        rows = rows[:limit]
    out_rows = []
    exact = tfm = valid = incomplete_n = 0
    latencies = []
    cache_hits = tf_cache_hits = 0
    for row in rows:
        img = _resolve_image(row, labels_path)
        iid = _image_id(row)
        expected = (row.get("smiles") or row.get("SMILES") or "").strip()
        pred, dt, hit = _cached_call(cache, model_key, variant_key, iid, predict_fn, img)
        cache_hits += int(hit)
        if not hit:
            latencies.append(dt)
        got = (pred.get("smiles") if isinstance(pred, dict) else str(pred) or "").strip()
        is_exact = got == expected
        is_incomplete = bool(pred.get("incomplete")) if isinstance(pred, dict) else False
        is_valid = bool(got) and not is_incomplete
        exact += int(is_exact); incomplete_n += int(is_incomplete); valid += int(is_valid)
        rec = {"id": iid, "expected": expected, "got": got,
               "exact_match": is_exact, "valid": is_valid, "incomplete": is_incomplete,
               "latency_ms": round(dt, 2), "cache_hit": hit}
        if tf_predict_fn is not None:
            tfp, _, tfhit = _cached_call(cache, model_key, tf_variant_key, iid, tf_predict_fn, img, time_it=False)
            tf_cache_hits += int(tfhit)
            tf_sm = (tfp.get("smiles") if isinstance(tfp, dict) else str(tfp) or "").strip()
            rec["tf_smiles"] = tf_sm
            rec["tf_match"] = got == tf_sm
            rec["tf_cache_hit"] = tfhit
            tfm += int(rec["tf_match"])
        out_rows.append(rec)
    n = len(rows) or 1
    res: dict[str, Any] = {
        "task": "ocsr", "count": len(rows),
        "exact_match": exact, "exact_match_rate": exact / n,
        "valid": valid, "valid_rate": valid / n,
        "incomplete": incomplete_n, "incomplete_rate": incomplete_n / n,
        "latency_ms": {"p50": float(np.percentile(latencies, 50)) if latencies else None,
                       "p95": float(np.percentile(latencies, 95)) if latencies else None,
                       "mean": float(np.mean(latencies)) if latencies else None,
                       "n_measured": len(latencies)},
        "cache_hits": cache_hits, "details": out_rows,
    }
    if model_key or variant_key:
        res["model_key"] = model_key; res["variant_key"] = variant_key
    if tf_predict_fn is not None:
        res["tf_match"] = tfm; res["tf_match_rate"] = tfm / n; res["tf_cache_hits"] = tf_cache_hits
    return res


def make_onnx_ocsr_predict_fn(encoder: Path, decoder: Path, vocab: Path):
    from decimer_onnx.api import DecimerONNX
    model = DecimerONNX(encoder, decoder, vocab)
    def _fn(img: Path):
        p = model.predict(img)
        return {"smiles": p.smiles, "incomplete": p.incomplete, "token_ids": p.token_ids}
    return _fn


def make_tf_ocsr_predict_fn(model_dir: Path, vocab_path: Path):
    """TF full SavedModel baseline (export env; uses DECIMER preprocessing for reference)."""
    import os
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    import tensorflow as tf
    import DECIMER.pre_process as pre
    vocab = json.loads(Path(vocab_path).read_text())
    id2tok = {int(k): v for k, v in vocab["id_to_token"].items()}
    fromto = str.maketrans("!$^<>?£¢€§", "1234567890")
    m = tf.saved_model.load(str(model_dir))
    def _fn(img: Path):
        arr = pre.decode_image(str(img)).numpy()
        out, conf = m(tf.constant(arr, dtype=tf.float32))
        ids = out.numpy().astype(int).reshape(-1).tolist()
        if 0 in ids:
            ids = ids[: ids.index(0) + 1]
            while ids and ids[-1] == 0:
                ids.pop()
        smiles = "".join(id2tok.get(i, "") for i in ids if i not in (0, 13, 14)).translate(fromto)
        return {"smiles": smiles, "incomplete": (not ids or ids[-1] != 14), "token_ids": ids}
    return _fn


def _classifier_preprocess_np(path: Path, desired_size: int = 224) -> np.ndarray:
    """Match DECIMER Image Classifier preprocessing (grayscale->RGB, aspect resize+pad white, 0-255 float)."""
    from PIL import Image
    img = Image.open(path)
    if img.mode == "RGBA":
        bg = Image.new("RGB", img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[3])
        img = bg
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
    arr = np.asarray(canvas, dtype=np.float32)  # efficientnet.preprocess_input is pass-through in this env
    return np.expand_dims(arr, 0)


CLASSIFIER_THRESHOLD = 0.000089  # upstream: score <= threshold => chemical structure


def make_onnx_classifier_predict_fn(model_path: Path, threshold: float = CLASSIFIER_THRESHOLD):
    import onnxruntime as ort
    sess = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0].name
    def _fn(img: Path):
        x = _classifier_preprocess_np(img)
        logit = sess.run(None, {inp: x})[0]
        score = 1.0 / (1.0 + np.exp(-float(np.asarray(logit).reshape(-1)[0])))
        label = "structure" if score <= threshold else "no-structure"
        return {"label": label, "score": score, "logit": float(np.asarray(logit).reshape(-1)[0])}
    return _fn


def make_tf_classifier_predict_fn(model_dir: Path, threshold: float = CLASSIFIER_THRESHOLD):
    import os
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    import tensorflow as tf
    from PIL import Image
    # resolve model dir
    sm = model_dir / "saved_model.pb"
    if not sm.exists() and (model_dir / "model" / "saved_model.pb").exists():
        model_dir = model_dir / "model"
    m = tf.saved_model.load(str(model_dir))
    sig = m.signatures.get("serving_default")
    def _fn(img: Path):
        # match upstream exactly for baseline
        pil = Image.open(img)
        if pil.mode == "RGBA":
            bg = Image.new("RGB", pil.size, (255,255,255)); bg.paste(pil, mask=pil.split()[3]); pil = bg
        pil = pil.convert("L").convert("RGB")
        old = pil.size; desired=224
        if old[0]!=desired or old[1]!=desired:
            ratio = desired/max(old); new_size=(int(old[0]*ratio),int(old[1]*ratio))
            try: resample=Image.Resampling.LANCZOS
            except Exception: resample=Image.LANCZOS
            pil = pil.resize(new_size, resample)
        else: new_size=old
        canvas = Image.new("RGB",(desired,desired),"white")
        canvas.paste(pil, ((desired-new_size[0])//2,(desired-new_size[1])//2))
        arr = np.asarray(canvas, dtype=np.float32)[None, ...]
        arr = tf.keras.applications.efficientnet.preprocess_input(arr)
        out = sig(input_2=tf.constant(arr))
        logit = list(out.values())[0] if isinstance(out, dict) else out
        score = float(tf.nn.sigmoid(logit).numpy().reshape(-1)[0])
        label = "structure" if score <= threshold else "no-structure"
        return {"label": label, "score": score, "logit": float(np.asarray(logit).numpy().reshape(-1)[0])}
    return _fn
