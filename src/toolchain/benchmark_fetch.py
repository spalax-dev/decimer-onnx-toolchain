"""Pipeline: fetch benchmark/golden datasets for evaluating DECIMER ONNX variants.

Fetches small, pinned, openly-licensed benchmark suites with checksum verification.
Large/external datasets (HF aggregates, 6670746 test split) are supported via importers
rather than wholesale download here.
"""
from __future__ import annotations

import csv
import json
import sys
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path

from toolchain.fetch import MANIFESTS_DIR, _download, _extract, _parse_checksum, verify

DEFAULT_BENCH_DIR = Path("data/benchmark")
DECIMER_REPO_COMMIT = "9d0686127d21b572d5e78783874462b542836c6c"
DECIMER_RAW = f"https://raw.githubusercontent.com/Kohulan/DECIMER-Image_Transformer/{DECIMER_REPO_COMMIT}"


@dataclass(frozen=True)
class BenchFile:
    name: str
    url: str
    size: int
    checksum: str
    extract: bool = False
    dest: str | None = None


def _fetch_files(suite: str, files: list[BenchFile], out_dir: Path, *, record: str, license_: str,
                 title: str, labels: list[dict] | None = None, force: bool = False) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"fetch-benchmark: {suite} ({title})", file=sys.stderr)
    print(f"source: {record}  license: {license_}", file=sys.stderr)
    fetched = []
    for bf in files:
        rel = Path(bf.dest) if bf.dest else Path(bf.name)
        out_path = out_dir / rel
        if force or not out_path.exists():
            _download(bf.url, out_path, bf.size)
        else:
            print(f"  cached: {out_path}", file=sys.stderr)
        verify(out_path, bf.checksum)
        print(f"  verified: {bf.checksum}", file=sys.stderr)
        if bf.extract and out_path.suffix == ".zip":
            marker = out_path.with_suffix(out_path.suffix + ".extracted")
            if force or not marker.exists():
                _extract(out_path, out_dir)
                marker.write_text(bf.checksum + "\n", encoding="utf-8")
        fetched.append({"file": asdict(bf), "path": str(out_path)})
    if labels is not None:
        labels_path = out_dir / "labels.jsonl"
        labels_path.write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in labels), encoding="utf-8"
        )
        fetched.append({"labels": str(labels_path), "count": len(labels)})
    MANIFESTS_DIR.mkdir(parents=True, exist_ok=True)
    manifest = {
        "suite": suite,
        "title": title,
        "record": record,
        "license": license_,
        "labels": labels or [],
        "fetched": fetched,
    }
    mpath = MANIFESTS_DIR / f"benchmark-{suite}.fetch.json"
    mpath.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"manifest: {mpath}", file=sys.stderr)
    return manifest


def fetch_decimer_smoke(data_dir: Path = DEFAULT_BENCH_DIR, *, force: bool = False) -> dict:
    """Printed OCSR smoke: 4 official labeled images from DECIMER-Image_Transformer@9d06861 (MIT)."""
    suite = "decimer-smoke"
    suite_dir = data_dir / suite
    out = suite_dir / "images"
    files = [
        BenchFile("DECIMER_test_0.png", f"{DECIMER_RAW}/Benchmark/test_images/DECIMER_test_0.png", 4743,
                  "sha256:f464ae34ff2a82759a640b95181b1a9306e086063cbcbe39d317af7bdb4724f2"),
        BenchFile("DECIMER_test_1.png", f"{DECIMER_RAW}/Benchmark/test_images/DECIMER_test_1.png", 12486,
                  "sha256:73d940ad62648e82e404cd4f363dbc787d46178116e0bb221a16991163630ba1"),
        BenchFile("github_issue_test.png", f"{DECIMER_RAW}/Benchmark/test_images/github_issue_test.png", 30464,
                  "sha256:4a20ba7aa00d9ae178e3bd39e936e0d25c29de55b9ff522395db1929af22f260"),
        BenchFile("caffeine_upstream.png", f"{DECIMER_RAW}/tests/caffeine.png", 24609,
                  "sha256:31df300ab9cae7a4761a278a952f22881d058ec6c4a39baba537d370e528ac8b"),
        BenchFile("TEST.txt", f"{DECIMER_RAW}/Benchmark/TEST.txt", 169,
                  "sha256:5133bdd7756962345ed67867cc8c5552cb094729d62ebb02ede740228efc6089"),
    ]
    labels = [
        {"image": "images/DECIMER_test_0.png", "smiles": "C1=CC=C(C=C1)CC2=C(C3=CC=C(C=C3)OS(=O)(=O)C(F)(F)F)C4=C(C=CC=C4)O2", "source": "DECIMER-Image_Transformer/Benchmark/TEST.txt", "model": "ocsr-printed"},
        {"image": "images/DECIMER_test_1.png", "smiles": "CCCCC1=CC=CC=C1", "source": "DECIMER-Image_Transformer/Benchmark/TEST.txt", "model": "ocsr-printed"},
        {"image": "images/github_issue_test.png", "smiles": "CC(CC1=CC2=C(C=C1)OCO2)NC", "source": "DECIMER-Image_Transformer/Benchmark/TEST.txt", "model": "ocsr-printed"},
        {"image": "images/caffeine_upstream.png", "smiles": "CN1C=NC2=C1C(=O)N(C)C(=O)N2C", "source": "DECIMER-Image_Transformer/tests/test_functions.py", "model": "ocsr-printed"},
    ]
    manifest = _fetch_files(
        suite, files, out, record=f"https://github.com/Kohulan/DECIMER-Image_Transformer/tree/{DECIMER_REPO_COMMIT}",
        license_="MIT (DECIMER-Image_Transformer)", title="DECIMER printed OCSR smoke (4 labeled images)",
        labels=None, force=force,
    )
    labels_path = suite_dir / "labels.jsonl"
    labels_path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in labels), encoding="utf-8")
    manifest["labels"] = labels
    manifest["labels_path"] = str(labels_path)
    manifest["labels_count"] = len(labels)
    (MANIFESTS_DIR / f"benchmark-{suite}.fetch.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def fetch_handdrawn_hdm(data_dir: Path = DEFAULT_BENCH_DIR, *, force: bool = False) -> dict:
    """Hand-drawn OCSR benchmark: DECIMER HDM dataset (5088 images + SMILES), Zenodo 7617107, CC-BY-4.0."""
    suite = "handdrawn-hdm"
    out = data_dir / suite
    files = [
        BenchFile("DECIMER_HDM_Dataset_Images.zip",
                  "https://zenodo.org/api/records/7617107/files/DECIMER_HDM_Dataset_Images.zip/content",
                  79109007, "md5:7c67bb16bb9819b5bf2699e216ace087", extract=True),
        BenchFile("DECIMER_HDM_Dataset_SMILES.tsv",
                  "https://zenodo.org/api/records/7617107/files/DECIMER_HDM_Dataset_SMILES.tsv/content",
                  234153, "md5:3b643bc381be7fe005a7541e9884ca8d"),
    ]
    manifest = _fetch_files(
        suite, files, out, record="https://zenodo.org/records/7617107",
        license_="CC-BY-4.0", title="DECIMER Hand-Drawn Molecules benchmark (5088 images)",
        labels=None, force=force,
    )
    # build labels.jsonl from TSV (IDs\tSMILES) mapping to extracted image paths
    tsv = out / "DECIMER_HDM_Dataset_SMILES.tsv"
    labels = _parse_hdm_tsv(tsv, out)
    labels_path = out / "labels.jsonl"
    labels_path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in labels), encoding="utf-8")
    manifest["labels_count"] = len(labels)
    manifest["labels_path"] = str(labels_path)
    (MANIFESTS_DIR / f"benchmark-{suite}.fetch.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"labels: {labels_path} ({len(labels)} rows)", file=sys.stderr)
    return manifest


def _parse_hdm_tsv(tsv: Path, root: Path) -> list[dict]:
    rows = []
    with tsv.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        # normalize header names
        fields = [c.strip() for c in (reader.fieldnames or [])]
        id_key = next((k for k in fields if k.lower() in {"ids", "id", "image_id", "filename", "image"}), fields[0] if fields else "IDs")
        smi_key = next((k for k in fields if k.lower() in {"smiles", "smile", "canonical_smiles"}), fields[1] if len(fields) > 1 else "SMILES")
        f.seek(0)
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            rid = (row.get(id_key) or row.get(fields[0]) or "").strip()
            smi = (row.get(smi_key) or row.get(fields[1] if len(fields) > 1 else "SMILES") or "").strip()
            if not rid or not smi:
                continue
            img = _resolve_hdm_image(root, rid)
            rows.append({"id": rid, "image": str(img.relative_to(root)) if img else f"images/{rid}.png", "smiles": smi, "model": "ocsr-handdrawn", "source": "Zenodo:7617107"})
    return rows


def _resolve_hdm_image(root: Path, rid: str) -> Path | None:
    # try common layouts/extensions produced by DECIMER_HDM_Dataset_Images.zip
    for ext in (".png", ".PNG", ".jpg", ".jpeg"):
        for base in [root / "images" / f"{rid}{ext}", root / f"{rid}{ext}", root / "DECIMER_HDM_Dataset_Images" / f"{rid}{ext}"]:
            if base.exists():
                return base
    # fallback: search by stem
    for p in root.rglob(f"{rid}.*"):
        if p.suffix.lower() in {".png", ".jpg", ".jpeg"}:
            return p
    return None


BENCHMARK_SUITES = {
    "decimer-smoke": (fetch_decimer_smoke, "ocsr-printed"),
    "handdrawn-hdm": (fetch_handdrawn_hdm, "ocsr-handdrawn"),
}


def fetch_benchmark(suite: str, data_dir: Path = DEFAULT_BENCH_DIR, *, force: bool = False) -> dict:
    if suite not in BENCHMARK_SUITES:
        raise KeyError(f"unknown benchmark suite {suite!r}; available: {', '.join(sorted(BENCHMARK_SUITES))}")
    fn, _ = BENCHMARK_SUITES[suite]
    return fn(data_dir, force=force)
