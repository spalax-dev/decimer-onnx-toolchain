"""Pipeline: import external benchmark datasets into the local manifest format.

Supports:
- HF datasets-server rows API (e.g. Keylab/OCSR-Benchmarks CLEF/JPO) without extra deps.
- Local directory + CSV/TSV/JSONL manifest (generic).

Records per-subset provenance/license notes in the manifest. For HF aggregates with
mixed licenses, verify the original subset terms before redistribution.
"""
from __future__ import annotations

import csv
import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

from decimer_onnx_export.fetch import MANIFESTS_DIR, file_checksum

DEFAULT_BENCH_DIR = Path("data/benchmark")
HF_ROWS_API = "https://datasets-server.huggingface.co/rows"
HF_DATASET_COMMIT = "2fdcdc59067acb36d84ddc82c1ace24c91d5ad71"  # Keylab/OCSR-Benchmarks, observed

# Per-subset provenance notes (recorded in manifest). HF aggregate is mixed-see-sources.
HF_OCSR_SUBSETS: dict[str, dict[str, Any]] = {
    "CLEF": {
        "hf_dataset": "Keylab/OCSR-Benchmarks", "config": "CLEF", "split": "test", "expected_rows": 992,
        "provenance": "CLEF 2012 OCSR benchmark test set (as distributed in Keylab/OCSR-Benchmarks).",
        "license_note": "HF aggregate license is mixed-see-sources; verify original CLEF 2012 benchmark terms before redistribution.",
    },
    "JPO": {
        "hf_dataset": "Keylab/OCSR-Benchmarks", "config": "JPO", "split": "test", "expected_rows": 449,
        "provenance": "JPO subset from ChemInfty/OSRA context (as distributed in Keylab/OCSR-Benchmarks).",
        "license_note": "HF aggregate license is mixed-see-sources; verify original JPO/ChemInfty/OSRA subset terms before redistribution.",
    },
    "UOB": {
        "hf_dataset": "Keylab/OCSR-Benchmarks", "config": "UOB", "split": "test", "expected_rows": 5740,
        "provenance": "University of Birmingham/MolRec OCSR benchmark (as distributed in Keylab/OCSR-Benchmarks).",
        "license_note": "HF aggregate license is mixed-see-sources; verify original UOB/MolRec terms before redistribution.",
    },
    "USPTO": {
        "hf_dataset": "Keylab/OCSR-Benchmarks", "config": "USPTO", "split": "test", "expected_rows": 5719,
        "provenance": "USPTO-based OCSR benchmark (OSRA context, as distributed in Keylab/OCSR-Benchmarks).",
        "license_note": "HF aggregate license is mixed-see-sources; verify original USPTO/OSRA subset terms before redistribution.",
    },
}


def _http_json(url: str) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "decimer-onnx-toolchain/0.1"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode("utf-8"))


def _download_bytes(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "decimer-onnx-toolchain/0.1"})
    with urllib.request.urlopen(req, timeout=120) as r, dest.open("wb") as f:
        f.write(r.read())


def import_hf_ocsr(config: str, data_dir: Path = DEFAULT_BENCH_DIR, *, model: str = "ocsr-printed",
                   force: bool = False, limit: int | None = None, page_size: int = 100) -> dict:
    if config not in HF_OCSR_SUBSETS:
        raise KeyError(f"unsupported HF OCSR subset {config!r}; supported: {', '.join(sorted(HF_OCSR_SUBSETS))}")
    meta = HF_OCSR_SUBSETS[config]
    suite = f"hf-{config.lower()}"
    out = data_dir / suite
    img_dir = out / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    print(f"import-hf: {meta['hf_dataset']} config={config} -> {out}", file=sys.stderr)
    print(f"license/provenance: {meta['license_note']}", file=sys.stderr)

    labels: list[dict[str, Any]] = []
    offset = 0
    total = None
    while True:
        params = {"dataset": meta["hf_dataset"], "config": meta["config"], "split": meta["split"],
                  "offset": offset, "length": page_size}
        url = HF_ROWS_API + "?" + urllib.parse.urlencode(params)
        data = _http_json(url)
        rows = data.get("rows", [])
        if total is None:
            total = data.get("num_rows_total")
            print(f"  rows: {total}", file=sys.stderr)
        if not rows:
            break
        for item in rows:
            row = item.get("row", {})
            image_id = str(row.get("image_id") or item.get("row_idx") or len(labels))
            smiles = str(row.get("smiles") or row.get("SMILES") or "").strip()
            img = row.get("image") or {}
            src = img.get("src")
            if not src or not smiles:
                continue
            # sanitize filename
            safe = "".join(c if c.isalnum() or c in "._-" else "_" for c in image_id)[:180] or f"row_{len(labels)}"
            dest = img_dir / f"{safe}.jpg"
            if force or not dest.exists():
                _download_bytes(src, dest)
            labels.append({
                "id": image_id, "image": str(dest.relative_to(out)), "smiles": smiles,
                "model": model, "source": f"{meta['hf_dataset']}:{config}", "hf_row": item.get("row_idx"),
            })
            if limit and len(labels) >= limit:
                break
        if limit and len(labels) >= limit:
            break
        offset += len(rows)
        if total is not None and offset >= total:
            break
        time.sleep(0.05)
    labels_path = out / "labels.jsonl"
    labels_path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in labels), encoding="utf-8")
    manifest = {
        "suite": suite, "title": f"HF import {meta['hf_dataset']} {config}",
        "hf_dataset": meta["hf_dataset"], "hf_dataset_commit_observed": HF_DATASET_COMMIT,
        "config": config, "split": meta["split"], "model": model,
        "provenance": meta["provenance"], "license_note": meta["license_note"],
        "aggregate_license": "mixed-see-sources",
        "labels_path": str(labels_path), "labels_count": len(labels),
        "expected_rows": meta["expected_rows"],
    }
    MANIFESTS_DIR.mkdir(parents=True, exist_ok=True)
    mpath = MANIFESTS_DIR / f"benchmark-{suite}.import.json"
    mpath.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"labels: {labels_path} ({len(labels)})  manifest: {mpath}", file=sys.stderr)
    return manifest


# ---- Classifier mapping (13378718 / ChemIC-style 4-class -> DECIMER binary) ----
CLASSIFIER_CATEGORY_MAP = {
    "one_molecule": {"binary": "structure", "included_in_binary": True},
    "several_molecules": {"binary": "structure", "included_in_binary": True},
    "other": {"binary": "no-structure", "included_in_binary": True},
    "reactions": {"binary": None, "included_in_binary": False, "report_as": "ambiguous_excluded"},
}


def import_classifier_13378718(root: Path, data_dir: Path = DEFAULT_BENCH_DIR, *, split: str = "classified",
                               force: bool = False) -> dict:
    """Import Zenodo 13378718 (dataset_for_image_classifier.zip, extracted) with explicit mapping.

    Mapping (stored in manifest):
      one_molecule -> structure
      several_molecules -> structure
      other -> no-structure
      reactions -> excluded from binary metrics, reported separately as ambiguous.

    `root` should point to the extracted zip root (containing `classified/` and/or `for_model/`).
    `split`: 'classified' (4*4000 manual labels), 'for_model/test', 'for_model/val', 'for_model/train'.
    """
    suite = f"classifier-13378718-{split.replace('/', '-')}"
    out = data_dir / "classifier-13378718"
    if split == "classified":
        base = root / "classified"
    else:
        base = root / split
    if not base.exists():
        raise FileNotFoundError(f"classifier split not found: {base}")
    labels: list[dict[str, Any]] = []
    for cat, mapping in CLASSIFIER_CATEGORY_MAP.items():
        cat_dir = base / cat
        if not cat_dir.exists():
            print(f"  warn: missing category dir {cat_dir}", file=sys.stderr)
            continue
        for img in sorted(cat_dir.iterdir()):
            if img.suffix.lower() not in {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".gif"}:
                continue
            labels.append({
                "image": str(img.resolve()), "category": cat,
                "binary_label": mapping["binary"],
                "included_in_binary": mapping["included_in_binary"],
                "model": "image-classifier", "source": "Zenodo:13378718",
            })
    labels_path = out / f"{suite}.labels.jsonl"
    out.mkdir(parents=True, exist_ok=True)
    labels_path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in labels), encoding="utf-8")
    manifest = {
        "suite": suite, "title": "Classifier eval from Zenodo 13378718 (explicit 4->binary mapping)",
        "record": "https://zenodo.org/records/13378718",
        "upstream_license": "MIT (dataset record)",
        "source_root": str(root), "split": split,
        "category_map": CLASSIFIER_CATEGORY_MAP,
        "mapping_rationale": "DECIMER classifier is binary (structure vs no-structure). one/several_molecule mapped to structure; other to no-structure; reactions excluded from binary confusion matrix and reported separately as ambiguous.",
        "labels_path": str(labels_path), "labels_count": len(labels),
        "binary_count": sum(1 for r in labels if r["included_in_binary"]),
        "reactions_count": sum(1 for r in labels if r["category"] == "reactions"),
    }
    MANIFESTS_DIR.mkdir(parents=True, exist_ok=True)
    mpath = MANIFESTS_DIR / f"benchmark-{suite}.import.json"
    mpath.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"labels: {labels_path} ({len(labels)}, binary={manifest['binary_count']}, reactions={manifest['reactions_count']})", file=sys.stderr)
    return manifest
