"""Curated benchmark registry, ensure/decompress, and packaging helpers.

Benchmark pipeline stages call ensure_benchmark_data() so compressed curated archives
(committed under data/curated/) are auto-extracted to their fixed decompression
location when needed.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import tarfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
REGISTRY_PATH = ROOT / "data" / "manifests" / "benchmark_registry.json"


def load_registry(path: Path | None = None) -> dict[str, Any]:
    p = path or REGISTRY_PATH
    if not p.exists():
        return {"version": 1, "benchmarks": []}
    return json.loads(p.read_text(encoding="utf-8"))


def get_entry(benchmark_id: str, registry: dict[str, Any] | None = None) -> dict[str, Any] | None:
    reg = registry or load_registry()
    for b in reg.get("benchmarks", []):
        if b.get("id") == benchmark_id or b.get("dossier") == benchmark_id or b.get("model") == benchmark_id:
            return b
    return None


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1024 * 1024), b""):
            h.update(c)
    return h.hexdigest()


def is_available(entry: dict[str, Any], root: Path | None = None) -> bool:
    base = root or ROOT
    provides = entry.get("provides") or []
    if not provides:
        return False
    for rel in provides:
        p = base / rel
        if not p.exists():
            return False
    return True


def decompress(entry: dict[str, Any], root: Path | None = None, *, verify: bool = True) -> Path:
    base = root or ROOT
    archive = base / entry["curated_archive"]
    if not archive.exists():
        url = entry.get("curated_url") or ""
        raise FileNotFoundError(
            f"Curated archive missing: {archive}. "
            + (f"Download from {url} and place it there." if url else "Place the curated archive in the repo path shown.")
        )
    expected = (entry.get("sha256") or "").strip()
    if verify and expected:
        got = _sha256(archive)
        if got != expected:
            raise ValueError(f"sha256 mismatch for {archive}: got {got}, expected {expected}")
    dest = base / entry.get("decompress_to", "data/benchmark")
    dest.mkdir(parents=True, exist_ok=True)
    fmt = entry.get("format") or archive.suffix.lstrip(".")
    if fmt in ("tar.zst", "zst", "tzst") or archive.name.endswith(".tar.zst"):
        zstd = shutil.which("zstd")
        tar = shutil.which("tar")
        if tar and zstd:
            subprocess.run([tar, "--zstd", "-xf", str(archive), "-C", str(dest)], check=True)
        elif zstd:
            # fallback: zstd -dc | tar -x
            p1 = subprocess.Popen([zstd, "-dc", str(archive)], stdout=subprocess.PIPE)
            subprocess.run([shutil.which("tar") or "tar", "-xf", "-", "-C", str(dest)], stdin=p1.stdout, check=True)
            if p1.stdout:
                p1.stdout.close()
            p1.wait()
        else:
            raise RuntimeError("zstd/tar required to extract .tar.zst curated benchmark archive")
    elif fmt in ("tar.xz", "xz", "txz") or archive.name.endswith(".tar.xz"):
        with tarfile.open(archive, "r:xz") as tf:
            tf.extractall(dest, filter="data")
    elif fmt in ("tar.gz", "gz", "tgz") or archive.name.endswith((".tar.gz", ".tgz")):
        with tarfile.open(archive, "r:gz") as tf:
            tf.extractall(dest, filter="data")
    else:
        with tarfile.open(archive, "r:*") as tf:
            tf.extractall(dest, filter="data")
    return dest


def ensure_benchmark_data(benchmark_id: str, root: Path | None = None, *, registry: dict[str, Any] | None = None) -> dict[str, Any] | None:
    """Ensure curated benchmark data is available; decompress archive if needed.

    Returns the registry entry if found, else None. No-op if entry not in registry.
    """
    entry = get_entry(benchmark_id, registry)
    if entry is None:
        return None
    if is_available(entry, root):
        return entry
    archive = (root or ROOT) / entry["curated_archive"]
    if archive.exists():
        print(f"decompressing curated benchmark: {entry['id']} -> {entry.get('decompress_to')}")
        decompress(entry, root)
        return entry
    # not available and no archive - caller may fetch/import via normal path
    return entry
