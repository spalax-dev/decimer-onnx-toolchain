"""Pipeline stage 5 — benchmark/validation harness.

Evaluates ONNX variants (and optionally TF baseline) against labeled benchmark suites.
OCSR: reports exact SMILES match, parity vs TF (token/SMILES), valid/incomplete, latency.
Classifier: reports binary confusion matrix over mapped categories; reactions reported separately
(see benchmark_import.CLASSIFIER_CATEGORY_MAP).
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from decimer_onnx_export.benchmark_import import CLASSIFIER_CATEGORY_MAP


def _load_labels(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def _resolve_image(row: dict, labels_path: Path) -> Path:
    img = row["image"]
    p = Path(img)
    if p.is_absolute():
        return p
    # relative to labels file dir or suite root
    cand = labels_path.parent / p
    if cand.exists():
        return cand
    cand = labels_path.parent.parent / p
    if cand.exists():
        return cand
    return cand


def _confusion(binary_rows: list[tuple[str, str]]) -> dict[str, int]:
    # rows of (expected_binary, predicted_binary) where values in {structure,no-structure}
    tp = sum(1 for e, p in binary_rows if e == "structure" and p == "structure")
    tn = sum(1 for e, p in binary_rows if e == "no-structure" and p == "no-structure")
    fp = sum(1 for e, p in binary_rows if e == "no-structure" and p == "structure")
    fn = sum(1 for e, p in binary_rows if e == "structure" and p == "no-structure")
    total = len(binary_rows)
    acc = (tp + tn) / total if total else 0.0
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn, "total": total, "accuracy": acc,
            "labels": ["structure", "no-structure"],
            "matrix_rows_expected_cols_predicted": [[tp, fn], [fp, tn]]}


def benchmark_classifier(labels_path: Path, *, predict_fn, limit: int | None = None) -> dict[str, Any]:
    """Evaluate classifier with explicit mapping.

    Binary confusion matrix computed ONLY over categories where included_in_binary is true
    (one_molecule, several_molecules, other). Reactions are excluded from binary metrics and
    reported separately as ambiguous.
    """
    rows = _load_labels(labels_path)
    if limit:
        rows = rows[:limit]
    binary_pairs: list[tuple[str, str]] = []
    reactions_report: list[dict[str, Any]] = []
    details = []
    for row in rows:
        cat = row.get("category")
        mapping = CLASSIFIER_CATEGORY_MAP.get(cat) or row
        img = _resolve_image(row, labels_path)
        pred = predict_fn(img)  # expected to return "structure"|"no-structure" (or dict with label)
        pred_label = pred["label"] if isinstance(pred, dict) else str(pred)
        rec = {"image": row.get("image"), "category": cat, "predicted": pred_label}
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
    return {
        "task": "image-classifier-binary",
        "mapping": CLASSIFIER_CATEGORY_MAP,
        "binary_confusion_matrix": _confusion(binary_pairs),
        "binary_note": "Computed only over one_molecule, several_molecules, other.",
        "reactions_separate_metric": {
            "count": len(reactions_report),
            "excluded_from_binary": True,
            "report_as": "ambiguous",
            "predicted_structure": sum(1 for r in reactions_report if r["predicted"] == "structure"),
            "predicted_no_structure": sum(1 for r in reactions_report if r["predicted"] == "no-structure"),
            "items": reactions_report,
        },
        "total_rows": len(rows),
        "details": details,
    }


def benchmark_ocsr(labels_path: Path, *, predict_fn, tf_predict_fn=None, limit: int | None = None) -> dict[str, Any]:
    """Evaluate OCSR predictions.

    predict_fn(image_path)->{smiles, incomplete, token_ids?, latency_ms?}
    tf_predict_fn optional, same shape, for parity (tf_match).
    """
    rows = _load_labels(labels_path)
    if limit:
        rows = rows[:limit]
    out_rows = []
    exact = tfm = valid = incomplete_n = 0
    latencies = []
    for row in rows:
        img = _resolve_image(row, labels_path)
        expected = (row.get("smiles") or row.get("SMILES") or "").strip()
        t0 = time.perf_counter()
        pred = predict_fn(img)
        dt = (time.perf_counter() - t0) * 1000
        latencies.append(dt)
        got = (pred.get("smiles") if isinstance(pred, dict) else str(pred) or "").strip()
        is_exact = got == expected
        is_incomplete = bool(pred.get("incomplete")) if isinstance(pred, dict) else False
        # validity: non-empty and not obviously broken (RDKit-free heuristic)
        is_valid = bool(got) and not is_incomplete
        exact += int(is_exact)
        incomplete_n += int(is_incomplete)
        valid += int(is_valid)
        rec = {"id": row.get("id") or row.get("image"), "expected": expected, "got": got,
               "exact_match": is_exact, "valid": is_valid, "incomplete": is_incomplete, "latency_ms": round(dt, 2)}
        if tf_predict_fn is not None:
            tfp = tf_predict_fn(img)
            tf_sm = (tfp.get("smiles") if isinstance(tfp, dict) else str(tfp) or "").strip()
            rec["tf_smiles"] = tf_sm
            rec["tf_match"] = got == tf_sm
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
                       "mean": float(np.mean(latencies)) if latencies else None},
        "details": out_rows,
    }
    if tf_predict_fn is not None:
        res["tf_match"] = tfm
        res["tf_match_rate"] = tfm / n
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
    import json, os
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
