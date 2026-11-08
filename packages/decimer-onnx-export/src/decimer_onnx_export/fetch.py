"""Pipeline stage 1 — Fetch: download pinned DECIMER weights from Zenodo with checksum verification.

This is the first pipeline stage. It caches artifacts under data/weights/, verifies integrity,
extracts archives when needed, and records a small manifest for reproducibility.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import sys
import urllib.request
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

DEFAULT_DATA_DIR = Path("data/weights")
MANIFESTS_DIR = Path("data/manifests")


@dataclass(frozen=True)
class ModelFile:
    name: str
    url: str
    size: int
    checksum: str  # "md5:<hex>" or "sha256:<hex>"
    extract: bool = False
    dest: str | None = None  # relative path inside model dir; defaults to name


@dataclass(frozen=True)
class ModelSpec:
    key: str
    record: str
    title: str
    license: str
    files: tuple[ModelFile, ...]


CLASSIFIER_COMMIT = "d21f4cb005f20571bb5e75b08ca375f56655884b"
CLASSIFIER_RAW = (
    f"https://raw.githubusercontent.com/Iagea/DECIMER-Image-Classifier/{CLASSIFIER_COMMIT}"
)

MODELS: dict[str, ModelSpec] = {
    "ocsr-printed": ModelSpec(
        key="ocsr-printed",
        record="https://zenodo.org/records/8300489",
        title="DECIMER V2 Models (printed OCSR)",
        license="CC-BY-4.0",
        files=(
            ModelFile(
                name="models.zip",
                url="https://zenodo.org/api/records/8300489/files/models.zip/content",
                size=298662033,
                checksum="md5:de78c966b0d63b290cd3f70bb81a91e7",
                extract=True,
            ),
        ),
    ),
    "ocsr-handdrawn": ModelSpec(
        key="ocsr-handdrawn",
        record="https://zenodo.org/records/10781330",
        title="DECIMER V2 hand-drawn specialist OCSR model",
        license="CC-BY-4.0",
        files=(
            ModelFile(
                name="DECIMER_HandDrawn_model.zip",
                url="https://zenodo.org/api/records/10781330/files/DECIMER_HandDrawn_model.zip/content",
                size=298515112,
                checksum="md5:1fece9813549417440f7c45d0c51603a",
                extract=True,
            ),
        ),
    ),
    "image-classifier": ModelSpec(
        key="image-classifier",
        record="https://github.com/Iagea/DECIMER-Image-Classifier (Zenodo DOI 10.5281/zenodo.6670746)",
        title="DECIMER Image Classifier (EfficientNetB0, structure vs non-structure)",
        license="MIT (code); see upstream for model/data terms",
        files=(
            ModelFile(
                name="saved_model.pb",
                url=f"{CLASSIFIER_RAW}/decimer_image_classifier/model/saved_model.pb",
                size=5845477,
                checksum="sha256:1f1f192e3a63c8985f20d8a0777a558458ef829379e5373fdad25bc0b4161978",
                dest="model/saved_model.pb",
            ),
            ModelFile(
                name="keras_metadata.pb",
                url=f"{CLASSIFIER_RAW}/decimer_image_classifier/model/keras_metadata.pb",
                size=769952,
                checksum="sha256:4c6f6149a612a52736ecc808434645242f746c57f7b38bb9eea2851112ee3cc6",
                dest="model/keras_metadata.pb",
            ),
            ModelFile(
                name="variables.index",
                url=f"{CLASSIFIER_RAW}/decimer_image_classifier/model/variables/variables.index",
                size=19547,
                checksum="sha256:bc74ce41019f517f4f766868d6b037566dbdf9da307af2414eb28ec7263680d6",
                dest="model/variables/variables.index",
            ),
            ModelFile(
                name="variables.data-00000-of-00001",
                url=f"{CLASSIFIER_RAW}/decimer_image_classifier/model/variables/variables.data-00000-of-00001",
                size=19694056,
                checksum="sha256:d4153f669cb35242739d0b901f5b9083d31acd5fcdb8d2c46d2a3589f9d1ed5a",
                dest="model/variables/variables.data-00000-of-00001",
            ),
        ),
    ),
}


def _parse_checksum(spec: str) -> tuple[str, str]:
    if ":" not in spec:
        raise ValueError(f"unsupported checksum format: {spec!r}")
    algo, value = spec.split(":", 1)
    algo = algo.lower()
    if algo not in {"md5", "sha256"}:
        raise ValueError(f"unsupported checksum algorithm: {algo}")
    return algo, value.lower()


def file_checksum(path: Path, algo: str) -> str:
    h = hashlib.new(algo)
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def verify(path: Path, checksum: str) -> None:
    algo, expected = _parse_checksum(checksum)
    actual = file_checksum(path, algo)
    if actual != expected:
        raise ValueError(
            f"checksum mismatch for {path.name}: expected {algo}:{expected}, got {algo}:{actual}"
        )


def _download(url: str, dest: Path, expected_size: int | None = None) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "decimer-onnx-toolchain/0.1"})
    with urllib.request.urlopen(req) as resp, tmp.open("wb") as out:
        total = expected_size or int(resp.headers.get("Content-Length") or 0)
        downloaded = 0
        last_pct = -1
        while True:
            chunk = resp.read(1024 * 1024)
            if not chunk:
                break
            out.write(chunk)
            downloaded += len(chunk)
            if total:
                pct = int(downloaded * 100 / total)
                if pct != last_pct and pct % 5 == 0:
                    print(f"  {dest.name}: {pct}% ({downloaded}/{total} bytes)", file=sys.stderr)
                    last_pct = pct
    tmp.replace(dest)
    if total:
        print(f"  {dest.name}: done ({downloaded} bytes)", file=sys.stderr)


def _extract(zip_path: Path, out_dir: Path) -> None:
    print(f"  extracting {zip_path.name} -> {out_dir}", file=sys.stderr)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(out_dir)


def fetch_model(
    key: str,
    data_dir: Path = DEFAULT_DATA_DIR,
    *,
    force: bool = False,
    keep_archives: bool = True,
) -> dict:
    if key not in MODELS:
        raise KeyError(f"unknown model {key!r}; available: {', '.join(sorted(MODELS))}")
    spec = MODELS[key]
    target_dir = data_dir / spec.key
    target_dir.mkdir(parents=True, exist_ok=True)
    print(f"fetch: {spec.key} ({spec.title})", file=sys.stderr)
    print(f"source: {spec.record}  license: {spec.license}", file=sys.stderr)

    fetched: list[dict] = []
    for mf in spec.files:
        rel = Path(mf.dest) if mf.dest else Path(mf.name)
        out_path = target_dir / rel
        if force or not out_path.exists():
            _download(mf.url, out_path, mf.size)
        else:
            print(f"  cached: {out_path}", file=sys.stderr)
        verify(out_path, mf.checksum)
        print(f"  verified: {mf.checksum}", file=sys.stderr)
        if mf.extract and out_path.suffix == ".zip":
            marker = out_path.with_suffix(out_path.suffix + ".extracted")
            if force or not marker.exists():
                _extract(out_path, target_dir)
                marker.write_text(mf.checksum + "\n", encoding="utf-8")
            else:
                print(f"  extracted (cached): {target_dir}", file=sys.stderr)
            if not keep_archives:
                out_path.unlink(missing_ok=True)
        fetched.append({"file": asdict(mf), "path": str(out_path)})

    MANIFESTS_DIR.mkdir(parents=True, exist_ok=True)
    manifest = {
        "model": asdict(spec) | {"files": [asdict(f) for f in spec.files]},
        "fetched": fetched,
    }
    manifest_path = MANIFESTS_DIR / f"{spec.key}.fetch.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"manifest: {manifest_path}", file=sys.stderr)
    return manifest


def available_models() -> Iterable[str]:
    return sorted(MODELS)
