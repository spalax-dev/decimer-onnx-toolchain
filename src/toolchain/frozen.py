"""Frozen benchmark sets and prediction cache.

Generates/loads deterministic frozen ID manifests (per dossier) and caches predictions by
(model, variant, image_id) to avoid recomputation.
"""
from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[2]
FROZEN_DIR = ROOT / "data" / "manifests" / "frozen"
CACHE_DIR = ROOT / "data" / "cache" / "predictions"
SEED_DEFAULT = 20260308


def load_labels(path: Path) -> list[dict[str, Any]]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def load_frozen(path: Path | str) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def resolve_frozen_rows(frozen: dict[str, Any], *, split: str = "eval") -> tuple[list[dict[str, Any]], Path]:
    """Return (rows, primary_labels_path) for a frozen split. Supports printed (multi-source) and single-source manifests."""
    want = set(frozen["splits"][split])
    sources = frozen.get("sources") or []
    rows: list[dict[str, Any]] = []
    by_key: dict[str, dict[str, Any]] = {}
    for src in sources:
        lp = ROOT / src["labels"] if not Path(src["labels"]).is_absolute() else Path(src["labels"])
        subset = src.get("subset")
        for r in load_labels(lp):
            rid = r.get("id") or r.get("image")
            if subset == "hdm":
                key = f"hdm::{r['id']}"
            elif subset in ("clef", "jpo"):
                key = f"{subset}::{r['id']}"
            else:
                # portable/curated labels may already carry a full stable key as id
                if isinstance(rid, str) and rid.startswith(f"{subset or frozen.get('dossier','x')}::"):
                    key = r.get("_frozen_key") or rid
                else:
                    key = r.get("_frozen_key") or f"{subset or frozen.get('dossier','x')}::{rid}"
            r = dict(r)
            r["_frozen_key"] = key
            r["_labels_path"] = str(lp)
            by_key[key] = r
    # preserve frozen order
    for k in frozen["splits"][split]:
        if k in by_key:
            rows.append(by_key[k])
    labels_path = ROOT / sources[0]["labels"] if sources else ROOT / "data/benchmark/labels.jsonl"
    return rows, labels_path


def freeze_classifier(labels_path: Path, out_path: Path, *, target_n: int = 10000, seed: int = SEED_DEFAULT) -> dict[str, Any]:
    """Stratified freeze: rare classes overrepresented (floor + sqrt-weighted remainder)."""
    rows = load_labels(labels_path)
    by_cat: dict[str, list[dict]] = {}
    for r in rows:
        cat = r.get("category") or "unknown"
        key = f"classifier::{r.get('id') or r.get('image')}"
        r = dict(r); r["_frozen_key"] = key
        by_cat.setdefault(cat, []).append(r)
    cats = sorted(by_cat)
    avail = {c: len(by_cat[c]) for c in cats}
    rng = random.Random(seed)
    for c in cats:
        rng.shuffle(by_cat[c])
    total_avail = sum(avail.values())
    target = min(target_n, total_avail)
    # floor to overrepresent rare/smaller classes
    floor_each = min(max(500, math.ceil(target / max(1, len(cats)))), target)
    alloc = {}
    remaining = target
    # first pass: give floor where possible (prioritize smaller classes by assigning floor)
    for c in sorted(cats, key=lambda x: avail[x]):
        take = min(floor_each, avail[c], remaining)
        alloc[c] = take
        remaining -= take
    # second pass: distribute remaining by sqrt(available - alloc) weight
    if remaining > 0:
        weights = {c: math.sqrt(max(0, avail[c] - alloc[c])) for c in cats}
        wsum = sum(weights.values()) or 1.0
        extras = {c: 0 for c in cats}
        for c in cats:
            extras[c] = min(avail[c] - alloc[c], int(math.floor(remaining * weights[c] / wsum)))
        # rounding remainder
        left = remaining - sum(extras.values())
        for c in sorted(cats, key=lambda x: (-weights[x], x)):
            if left <= 0:
                break
            can = avail[c] - alloc[c] - extras[c]
            if can > 0:
                extras[c] += 1; left -= 1
        for c in cats:
            alloc[c] += extras[c]
    selected = []
    for c in cats:
        selected.extend(by_cat[c][:alloc[c]])
    rng.shuffle(selected)  # deterministic final order
    frozen_ids = [r["_frozen_key"] for r in selected]
    eval_n = min(1000, len(frozen_ids))
    pilot_n = min(300, eval_n)
    manifest = {
        "dossier": "classifier", "model": "image-classifier", "seed": seed,
        "method": "stratified by 4 classes; floor boost for rare classes + sqrt(capacity)-weighted remainder; deterministic shuffle",
        "source_labels": str(labels_path),
        "class_available": avail, "class_allocated": alloc,
        "available_n": total_avail, "frozen_n": len(frozen_ids), "eval_n": eval_n, "pilot_n": pilot_n,
        "remainder_n": max(0, len(frozen_ids) - eval_n),
        "sources": [{"subset": "classifier", "labels": str(labels_path), "n": total_avail}],
        "splits": {"pilot": frozen_ids[:pilot_n], "eval": frozen_ids[:eval_n], "frozen": frozen_ids, "remainder": frozen_ids[eval_n:]},
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest


class PredictionCache:
    """Cache predictions by (model, variant, image_id) in JSONL shards."""
    def __init__(self, cache_dir: Path | None = None):
        self.root = cache_dir or CACHE_DIR

    def _path(self, model: str, variant: str) -> Path:
        safe_m = "".join(c if c.isalnum() or c in "._-" else "_" for c in model)
        safe_v = "".join(c if c.isalnum() or c in "._-" else "_" for c in variant)
        return self.root / safe_m / f"{safe_v}.jsonl"

    def load(self, model: str, variant: str) -> dict[str, Any]:
        p = self._path(model, variant)
        out = {}
        if not p.exists():
            return out
        for line in p.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                r = json.loads(line); out[str(r["image_id"])] = r.get("prediction")
            except Exception:
                pass
        return out

    def append(self, model: str, variant: str, image_id: str, prediction: Any) -> None:
        p = self._path(model, variant)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"image_id": image_id, "prediction": prediction}, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--classifier-labels", type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--target-n", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=SEED_DEFAULT)
    a = ap.parse_args()
    if a.classifier_labels:
        m = freeze_classifier(a.classifier_labels, a.out or (FROZEN_DIR / "classifier.frozen.json"), target_n=a.target_n, seed=a.seed)
        print(json.dumps({k: m[k] for k in ["frozen_n","eval_n","pilot_n","class_available","class_allocated"]}, indent=2))
