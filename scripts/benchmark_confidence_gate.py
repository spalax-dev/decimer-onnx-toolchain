#!/usr/bin/env python3
"""Experiment: confidence-based rejection gate (selective prediction) for the 3 selected models.

Like the Imago/preprocessing experiments: NOT part of the official report.

Question: does prediction confidence identify errors, so chemistry-draw can reject
low-confidence outputs ("ask for another image") instead of returning them?

- OCSR (printed/hand-drawn FP32): per-token decoder confidences from runtime
  Prediction.confidences -> summary stats (mean/min/gmean).
- Classifier (E4 FP16): sigmoid score; ranking confidence = |logit - logit(threshold)|.

Outputs: data/benchmark/experiments/confidence/<model>.<split>.json
Prints: risk-coverage tables (accuracy vs coverage) + confidence-discriminative AUC.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from toolchain.frozen import load_frozen, resolve_frozen_rows
from toolchain.benchmark_data import ensure_benchmark_data
from toolchain.benchmark_run import (
    CLASSIFIER_THRESHOLD,
    _canonical_smiles,
    _image_id,
    _resolve_image,
)

COVERAGES = [1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.6, 0.5]


# ---------- per-item confidence summaries ----------

def _ocsr_conf_stats(confs: list[float]) -> dict[str, float]:
    if not confs:
        return {"mean": 0.0, "min": 0.0, "gmean": 0.0, "n": 0}
    a = np.asarray(confs, dtype=np.float64)
    a = np.clip(a, 1e-12, 1.0)
    return {
        "mean": float(a.mean()),
        "min": float(a.min()),
        "gmean": float(np.exp(np.log(a).mean())),
        "n": int(a.size),
    }


def _cls_margin(logit: float) -> float:
    thr = CLASSIFIER_THRESHOLD
    logit_thr = math.log(thr / (1.0 - thr))
    return abs(float(logit) - logit_thr)


# ---------- risk-coverage + AUC ----------

def risk_coverage(
    items: list[dict[str, Any]],
    conf_key: str,
    correct_key: str,
    coverages: list[float] = COVERAGES,
) -> list[dict[str, Any]]:
    """Keep the most-confident fraction of items; report accuracy on what is kept.

    Items where correct is None (OCSR non-comparable) count as failures in the
    production view (handled by correct_key already being a bool)."""
    ranked = sorted(items, key=lambda r: r[conf_key], reverse=True)
    n = len(ranked)
    out = []
    for cov in coverages:
        k = max(1, int(round(cov * n)))
        kept = ranked[:k]
        hits = sum(1 for r in kept if r[correct_key])
        out.append({
            "coverage": cov,
            "n_kept": k,
            "n_rejected": n - k,
            "accuracy": hits / k if k else None,
            "hits": hits,
            "min_conf_kept": min(r[conf_key] for r in kept),
        })
    return out


def auc_confidence(items: list[dict[str, Any]], conf_key: str, correct_key: str) -> float | None:
    """Rank-based AUC of confidence for predicting correctness (Mann-Whitney, avg ranks for ties)."""
    pairs = [(r[conf_key], 1 if r[correct_key] else 0) for r in items]
    pos = [c for c, y in pairs if y]
    neg = [c for c, y in pairs if not y]
    if not pos or not neg:
        return None
    vals = sorted(c for c, _ in pairs)
    # average rank
    rank: dict[float, float] = {}
    i = 0
    while i < len(vals):
        j = i
        while j + 1 < len(vals) and vals[j + 1] == vals[i]:
            j += 1
        avg = (i + j) / 2 + 1  # 1-based average rank
        rank[vals[i]] = avg
        i = j + 1
    rank_pos = sum(rank[c] for c in pos)
    n1, n2 = len(pos), len(neg)
    return (rank_pos - n1 * (n1 + 1) / 2) / (n1 * n2)


# ---------- runners ----------

def run_ocsr(model_key: str, encoder: Path, decoder: Path, vocab: Path,
             manifest: Path, split: str, out_path: Path) -> dict[str, Any]:
    from runtime.api import DecimerONNX

    frozen = load_frozen(manifest)
    ensure_benchmark_data(frozen.get("model") or frozen.get("dossier") or "", registry=None)
    rows, labels_path = resolve_frozen_rows(frozen, split=split)
    model = DecimerONNX(encoder, decoder, vocab)

    items: list[dict[str, Any]] = []
    conf_min, conf_max = 1.0, 0.0
    t_start = time.time()
    for i, row in enumerate(rows):
        img = _resolve_image(row, labels_path)
        iid = _image_id(row)
        expected = (row.get("smiles") or row.get("SMILES") or "").strip()
        p = model.predict(img)
        got = (p.smiles or "").strip()
        ce, cg = _canonical_smiles(expected), _canonical_smiles(got)
        can = (ce == cg) if (ce is not None and cg is not None) else None
        stats = _ocsr_conf_stats(p.confidences)
        if p.confidences:
            conf_min = min(conf_min, min(p.confidences))
            conf_max = max(conf_max, max(p.confidences))
        items.append({
            "id": iid,
            "canonical_match": can,
            "correct_prod": bool(can),            # non-comparable = failure
            "correct_comp": can,                  # None when non-comparable
            "valid": bool(got) and not p.incomplete,
            "incomplete": p.incomplete,
            "conf_mean": stats["mean"],
            "conf_min": stats["min"],
            "conf_gmean": stats["gmean"],
            "n_tokens": stats["n"],
        })
        if (i + 1) % 100 == 0:
            print(f"  [{model_key} {split}] {i+1}/{len(rows)} ({time.time()-t_start:.0f}s)", flush=True)

    res = {
        "experiment": "confidence_gate",
        "task": "ocsr",
        "model_key": model_key,
        "split": split,
        "count": len(items),
        "confidence_source": "decoder per-token probs (Prediction.confidences); summaries mean/min/gmean",
        "observed_token_conf_range": [conf_min, conf_max],
        "items": items,
    }
    _post_ocsr(res, items, out_path)
    return res


def _post_ocsr(res: dict[str, Any], items: list[dict[str, Any]], out_path: Path) -> None:
    for key in ("conf_mean", "conf_min", "conf_gmean"):
        res[f"risk_coverage_prod_{key}"] = risk_coverage(items, key, "correct_prod")
        res[f"auc_{key}"] = auc_confidence(items, key, "correct_prod")
    # comparable-basis coverage (production view is primary; comparable shown for reference)
    comp = [r for r in items if r["correct_comp"] is not None]
    res["risk_coverage_comp_conf_mean"] = risk_coverage(comp, "conf_mean", "correct_comp") if comp else []
    res["auc_comp_conf_mean"] = auc_confidence(comp, "conf_mean", "correct_comp") if comp else None
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")
    print_summary(res, items, "correct_prod", "conf_mean")


def run_classifier(manifest: Path, split: str, model_path: Path, out_path: Path) -> dict[str, Any]:
    import onnxruntime as ort
    from toolchain.benchmark_run import CLASSIFIER_CATEGORY_MAP, _classifier_preprocess_np

    frozen = load_frozen(manifest)
    ensure_benchmark_data(frozen.get("model") or frozen.get("dossier") or "", registry=None)
    rows, labels_path = resolve_frozen_rows(frozen, split=split)

    sess = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    inputs = sess.get_inputs()

    items: list[dict[str, Any]] = []
    t_start = time.time()
    for i, row in enumerate(rows):
        # replicate benchmark_classifier inclusion/label logic exactly
        cat = row.get("category")
        mapping = CLASSIFIER_CATEGORY_MAP.get(cat) or row
        if cat == "reactions" or not mapping.get("included_in_binary", row.get("included_in_binary", False)):
            continue
        expected = str(mapping.get("binary") or row.get("binary_label"))
        img = _resolve_image(row, labels_path)
        iid = _image_id(row)
        x = _classifier_preprocess_np(img)
        feed: dict[str, Any] = {}
        for inp in inputs:
            shape = [d if isinstance(d, int) and d > 0 else 1 for d in inp.shape]
            if inp.name == "input_2" or shape[-3:] == [224, 224, 3]:
                feed[inp.name] = x
            elif "float" in str(inp.type):
                feed[inp.name] = np.zeros(shape, dtype=np.float32)
            else:
                feed[inp.name] = np.zeros(shape)
        logit = float(np.asarray(sess.run(None, feed)[0]).reshape(-1)[0])
        score = 1.0 / (1.0 + math.exp(-logit))
        predicted = "structure" if score <= CLASSIFIER_THRESHOLD else "no-structure"
        items.append({
            "id": iid,
            "expected": expected,
            "predicted": predicted,
            "correct_prod": predicted == expected,
            "score": score,
            "logit": logit,
            "margin": _cls_margin(logit),
        })
        if (i + 1) % 200 == 0:
            print(f"  [classifier {split}] {i+1}/{len(rows)} ({time.time()-t_start:.0f}s)", flush=True)

    res = {
        "experiment": "confidence_gate",
        "task": "classifier",
        "model_key": "image-classifier",
        "split": split,
        "count": len(items),
        "confidence_source": "|logit - logit(CLASSIFIER_THRESHOLD)| (margin)",
        "threshold": CLASSIFIER_THRESHOLD,
        "items": items,
    }
    for key in ("margin",):
        res[f"risk_coverage_prod_{key}"] = risk_coverage(items, key, "correct_prod")
        res[f"auc_{key}"] = auc_confidence(items, key, "correct_prod")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")
    print_summary(res, items, "correct_prod", "margin")


def print_summary(res: dict[str, Any], items: list[dict[str, Any]], correct_key: str, conf_key: str) -> None:
    n = len(items)
    base = sum(1 for r in items if r[correct_key]) / n if n else 0.0
    print(f"\n== {res['model_key']} ({res['split']}, n={n}) base {correct_key}={base:.4f} ==")
    for row in res.get(f"risk_coverage_prod_{conf_key}", []):
        print(f"  cov={row['coverage']:.2f} kept={row['n_kept']:4d} rej={row['n_rejected']:4d} "
              f"acc={row['accuracy']:.4f} minconf={row['min_conf_kept']:.4f}")
    print(f"  AUC(conf->correct) = {res.get(f'auc_{conf_key}')}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["ocsr-printed", "ocsr-handdrawn", "classifier", "all"], default="all")
    ap.add_argument("--split", choices=["pilot", "eval"], default="eval")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "data/benchmark/experiments/confidence")
    ap.add_argument("--dossier", choices=["standard", "colored"], default="standard",
                    help="standard = in-domain frozen dossiers (default); colored = stress dossier "
                         "(both OCSR FP32 models, colored.frozen.json, split=eval n=200)")
    args = ap.parse_args()

    if args.dossier == "colored":
        # Stress set cross-check: apply the confidence gate to the colored-background
        # dossier (both selected OCSR FP32 models). No classifier (gate already rejected).
        man = ROOT / "data/manifests/frozen/colored.frozen.json"
        out_col = args.out_dir / "colored"
        for mk, enc, dec, vocab in (
            ("ocsr-printed", "data/artifacts/fp32/ocsr-printed/encoder.fp32.onnx",
             "data/artifacts/fp32/ocsr-printed/decoder_step.fp32.onnx",
             "data/artifacts/tokenizer_SMILES.vocab.json"),
            ("ocsr-handdrawn", "data/artifacts/fp32/ocsr-handdrawn/encoder.fp32.onnx",
             "data/artifacts/fp32/ocsr-handdrawn/decoder_step.fp32.onnx",
             "data/artifacts/tokenizer_handdrawn.vocab.json"),
        ):
            run_ocsr(mk, ROOT / enc, ROOT / dec, ROOT / vocab, man, "eval",
                     out_col / f"{mk}.evalcolored.json")
        return

    suffix = "pilot300" if args.split == "pilot" else "eval1000"
    out_dir = args.out_dir / suffix

    if args.model in ("ocsr-printed", "all"):
        run_ocsr(
            "ocsr-printed",
            ROOT / "data/artifacts/fp32/ocsr-printed/encoder.fp32.onnx",
            ROOT / "data/artifacts/fp32/ocsr-printed/decoder_step.fp32.onnx",
            ROOT / "data/artifacts/tokenizer_SMILES.vocab.json",
            ROOT / "data/manifests/frozen/printed.frozen.json",
            args.split, out_dir / f"ocsr-printed.{suffix}.json",
        )
    if args.model in ("ocsr-handdrawn", "all"):
        run_ocsr(
            "ocsr-handdrawn",
            ROOT / "data/artifacts/fp32/ocsr-handdrawn/encoder.fp32.onnx",
            ROOT / "data/artifacts/fp32/ocsr-handdrawn/decoder_step.fp32.onnx",
            ROOT / "data/artifacts/tokenizer_handdrawn.vocab.json",
            ROOT / "data/manifests/frozen/handdrawn.frozen.json",
            args.split, out_dir / f"ocsr-handdrawn.{suffix}.json",
        )
    if args.model in ("classifier", "all"):
        run_classifier(
            ROOT / "data/manifests/frozen/classifier.frozen.json",
            args.split,
            ROOT / "data/artifacts/quant/e4/image-classifier/classifier.e4.fp16.onnx",
            out_dir / f"image-classifier.{suffix}.json",
        )


if __name__ == "__main__":
    main()
