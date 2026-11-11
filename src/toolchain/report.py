"""Generate a neutral benchmark report (glossary, acceptance criteria, machine info, results).

Reads benchmark JSON reports/manifests under data/ and writes docs/benchmark-report.md.
No judgments or decisions are emitted about results.
"""
from __future__ import annotations

import json
import platform
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]


def _run(cmd: str) -> str | None:
    try:
        return subprocess.check_output(cmd, shell=True, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def collect_machine_info() -> dict[str, Any]:
    info: dict[str, Any] = {}
    info["collected_at_utc"] = datetime.now(timezone.utc).isoformat()
    info["hostname"] = platform.node()
    info["os"] = f"{platform.system()} {platform.release()}"
    info["os_version"] = platform.version()
    info["arch"] = platform.machine()
    info["python"] = platform.python_version()
    lscpu = _run("lscpu")
    if lscpu:
        def g(pat: str) -> str | None:
            m = re.search(pat, lscpu, re.M)
            return m.group(1).strip() if m else None
        info["cpu_model"] = g(r"Model name:\s*(.+)")
        info["cpu_sockets"] = g(r"Socket\(s\):\s*(\S+)")
        info["cpu_cores_per_socket"] = g(r"Core\(s\) per socket:\s*(\S+)")
        info["cpu_threads_per_core"] = g(r"Thread\(s\) per core:\s*(\S+)")
        info["cpu_logical"] = g(r"CPU\(s\):\s*(\S+)")
        info["cpu_max_mhz"] = g(r"CPU max MHz:\s*(\S+)")
        info["cpu_min_mhz"] = g(r"CPU min MHz:\s*(\S+)")
        info["l1d"] = g(r"L1d cache:\s*(.+)")
        info["l1i"] = g(r"L1i cache:\s*(.+)")
        info["l2"] = g(r"L2 cache:\s*(.+)")
        info["l3"] = g(r"L3 cache:\s*(.+)")
        flags = g(r"Flags:\s*(.+)")
        info["cpu_flags_relevant"] = " ".join(f for f in (flags or "").split() if f in {"avx","avx2","fma","avx512f","avx512_vnni","sse4_2"}) or None
    mem = _run("free -b")
    if mem:
        m = re.search(r"Mem:\s+(\d+)\s+(\d+)\s+(\d+)\s+\d+\s+\d+\s+(\d+)", mem)
        if m:
            info["ram_total_bytes"] = int(m.group(1))
            info["ram_available_at_collect_bytes"] = int(m.group(4))
    try:
        import importlib.metadata as md
        pkgs = {}
        for p in ["onnxruntime","tensorflow","tensorflow-cpu","numpy","tf2onnx","onnx","Pillow","decimer","typer","sympy","rdkit"]:
            try:
                pkgs[p] = md.version(p)
            except Exception:
                pass
        info["packages"] = pkgs
    except Exception:
        pass
    return info


def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _fmt_rate(n: int | None, d: int | None) -> str:
    if n is None or d in (None, 0):
        return "—"
    return f"{n}/{d} ({n/d:.4f})"


def _fmt_ms(v: Any) -> str:
    if v is None:
        return "—"
    try:
        return f"{float(v):.2f}"
    except Exception:
        return "—"


def _delta_pp(var_path: Path, fp32_path: Path, field: str) -> str:
    v = _load_json(var_path); f = _load_json(fp32_path)
    if not v or not f:
        return "—"
    vd = {str(d.get("id")): d for d in v.get("details", [])}
    fd = {str(d.get("id")): d for d in f.get("details", [])}
    ids = [i for i in vd.keys() if i in fd]
    if not ids:
        # fallback to aggregate if same count and no details overlap
        vr, fr = v.get(f"{field}_rate"), f.get(f"{field}_rate")
        if vr is not None and fr is not None and v.get("count") == f.get("count"):
            return f"{(vr-fr)*100:+.2f} pp (agg)"
        return "—"
    vb = sum(1 for i in ids if vd[i].get(field) is True)
    fb = sum(1 for i in ids if fd[i].get(field) is True)
    dpp = (vb - fb) / len(ids) * 100
    note = "" if len(ids) == v.get("count") == f.get("count") else f", n={len(ids)}"
    return f"{dpp:+.2f} pp{note}"


def _artifact_size_mb(report_path: Path) -> str:
    """Best-effort artifact size (MB) inferred from report name/variant and known layout.
    Returns '—' when artifacts are not resolvable locally (e.g. not built/present)."""
    name = report_path.name.lower()
    full = str(report_path).lower()
    d = _load_json(report_path) or {}
    model = str(d.get("model_key") or d.get("model") or "").lower()
    vk = str(d.get("variant_key") or "").lower()

    def sz(*parts: str | None) -> str | None:
        paths = [ROOT / p for p in parts if p]
        found = [p for p in paths if p.exists()]
        if not found:
            return None
        return f"{sum(p.stat().st_size for p in found)/1024/1024:.1f}"

    # classifier (single graph)
    if "classifier" in name or "image-classifier" in model:
        if "e4-fp16" in name or "fp16" in vk:
            return sz("data/artifacts/quant/e4/image-classifier/classifier.e4.fp16.onnx") or "—"
        if "e2-static" in name or ("e2" in vk and "static" in vk):
            return sz("data/artifacts/quant/e2/image-classifier/classifier.e2.static.onnx") or "—"
        if "e1s-sel" in name or "e1s" in vk:
            return sz("data/artifacts/quant/e1s/image-classifier/classifier.e1s.dynamic.onnx") or "—"
        if "int8-dyn-all" in name or vk == "int8-dyn-all":
            return sz("data/artifacts/quant/e1/image-classifier/classifier.e1.dynamic.onnx") or "—"
        if "fp32" in name or vk == "fp32":
            return sz("data/artifacts/fp32/image-classifier/classifier.fp32.onnx") or "—"

    # OCSR dossier/model
    if "handdrawn" in name or "handdrawn" in model:
        dossier = "ocsr-handdrawn"
    else:
        dossier = "ocsr-printed"

    fp32_enc = f"data/artifacts/fp32/{dossier}/encoder.fp32.onnx"
    fp32_dec = f"data/artifacts/fp32/{dossier}/decoder_step.fp32.onnx"
    e1_enc = f"data/artifacts/quant/e1/{dossier}/encoder.e1.dynamic.onnx"
    e1_dec = f"data/artifacts/quant/e1/{dossier}/decoder_step.e1.dynamic.onnx"
    e1s_dec = f"data/artifacts/quant/e1s/{dossier}/decoder_step.e1s.dynamic.onnx"
    e1s_enc = f"data/artifacts/quant/e1s/{dossier}/encoder.e1s.dynamic.onnx"
    e3_enc = f"data/artifacts/quant/e3/{dossier}/encoder.e3.static.onnx"
    e3_dec = f"data/artifacts/quant/e3/{dossier}/decoder_step.e3.dynamic.onnx"

    # mixed/selective enc + dec
    for tag, encp in [("selmid", "data/artifacts/quant/exp/encoder.selMid.dynamic.onnx"),
                      ("selg", "data/artifacts/quant/exp/encoder.selG.dynamic.onnx"),
                      ("self", "data/artifacts/quant/exp/encoder.selF.dynamic.onnx"),
                      ("sele", "data/artifacts/quant/exp/encoder.selE.dynamic.onnx"),
                      ("seld", "data/artifacts/quant/exp/encoder.selD.dynamic.onnx"),
                      ("selc", "data/artifacts/quant/exp/encoder.selC.dynamic.onnx"),
                      ("selb", "data/artifacts/quant/exp/encoder.selB.dynamic.onnx"),
                      ("sela", "data/artifacts/quant/exp/encoder.selA.dynamic.onnx")]:
        if f"{tag}enc-e1sdec" in name or vk == f"{tag}enc-e1sdec":
            return sz(encp, e1s_dec) or "—"
    if "fp32enc-e1sdec-rr" in name or vk == "fp32enc-e1sdec-rr":
        return sz(fp32_enc, "data/artifacts/quant/exp/decoder_step.e1s.rr.dynamic.onnx") or "—"
    if "fp32enc-e1sdec" in name or vk == "fp32enc-e1sdec":
        return sz(fp32_enc, e1s_dec) or "—"
    if "fp32enc-e1dec-rr" in name or vk == "fp32enc-e1dec-rr":
        return sz(fp32_enc, "data/artifacts/quant/exp/decoder_step.e1.rr.dynamic.onnx") or "—"
    if "fp32enc-e1dec" in name or vk == "fp32enc-e1dec":
        return sz(fp32_enc, e1_dec) or "—"
    if "e1enc-fp32dec" in name or vk == "e1enc-fp32dec":
        return sz(e1_enc, fp32_dec) or "—"

    # standard variants
    if "e1s-sel-noattn" in name or ("e1s" in vk and "e1s-sel" not in name):
        return sz(e1s_enc, e1s_dec) or "—"
    if "int8-dyn-all" in name or vk == "int8-dyn-all" or (".e1." in name) or vk == "e1":
        # note: .e1. in historical report names
        if "e1s" in name:  # handled above
            pass
        else:
            return sz(e1_enc, e1_dec) or "—"
    if ".e3." in name or "e3" in vk:
        return sz(e3_enc, e3_dec) or "—"
    if ".fp32." in name or vk == "fp32":
        return sz(fp32_enc, fp32_dec) or "—"

    return "—"


def _row(label: str, report_path: Path, *, tf_join_note: str = "", fp32_path: Path | None = None) -> str | None:
    r = _load_json(report_path)
    if not r:
        return None
    n = r.get("count")
    ex = r.get("exact_match"); exr = r.get("exact_match_rate")
    cm = r.get("canonical_match"); ccomp = r.get("canonical_comparable"); cmr = r.get("canonical_match_rate")
    tfm = r.get("tf_match"); tfmr = r.get("tf_match_rate")
    tcm, tcc, tcr = _tf_canonical_stats(r)
    tf_cell = _fmt_rate(tfm, n)
    if tfm is None:
        tf_cell = "—"
    elif tf_join_note:
        tf_cell = f"{tfm}/{n} ({tfmr:.4f}) {tf_join_note}" if tfmr is not None else f"{tfm}/{n} {tf_join_note}"
    if tcm is not None and tcc:
        tf_cell = f"{tf_cell} / can {tcm}/{tcc} ({tcr:.3f})"
    valid = r.get("valid"); vr = r.get("valid_rate"); inc = r.get("incomplete")
    p50, p95, note = _latency_from_details(r)
    size = _artifact_size_mb(report_path)
    if fp32_path and report_path != fp32_path and fp32_path.exists():
        fr = _load_json(fp32_path) or {}
        dtf = _delta_pp(report_path, fp32_path, "tf_match") if tfm is not None and fr.get("tf_match") is not None else "—"
        dex = _delta_pp(report_path, fp32_path, "exact_match")
        dcan = f"{(cmr-fr['canonical_match_rate'])*100:+.2f} pp" if (cmr is not None and fr.get("canonical_match_rate") is not None) else "—"
    elif fp32_path and report_path == fp32_path:
        dtf = dex = dcan = "(ref)"
    else:
        dtf = dex = dcan = "—"
    rep = str(report_path.relative_to(ROOT)) if report_path.is_relative_to(ROOT) else str(report_path)
    mark = " ✅≥70%" if cmr is not None and cmr >= 0.70 else (" ❌<70%" if cmr is not None else "")
    can_cell = f"**{cm}/{ccomp} ({cmr:.4f})**{mark}" if (cm is not None and ccomp) else "—"
    return (
        f"| {label} | {n} | {can_cell} | {ex} ({exr:.4f}) | {tf_cell} | {dcan} | {dtf} | {dex} | {valid} ({vr:.4f}) | {inc} | "
        f"{p50}{note} | {p95} | {size} | `{rep}` |"
    )


GLOSSARY = r"""## 1. Glossary

Metrics are computed with the same harness over the same frozen image/ID set when applicable (paired comparison), unless cache is indicated.

| Term / column | What it measures | How to read it | Important nuance |
|---|---|---|---|
| **OCSR** | Optical Chemical Structure Recognition: recognizing a chemical structure from an image and producing a representation (here, SMILES). | — | — |
| **SMILES** | Simplified Molecular Input Line Entry System: a text string representing a molecule. | — | The same molecule can be written with different SMILES. |
| **TF model (baseline)** | Original DECIMER model in TensorFlow (SavedModel), used as behavior reference. | — | Reference for `tf_match`; not absolute "chemical truth". |
| **ONNX model** | Model converted to ONNX (encoder + `decoder_step`) executed with ONNX Runtime (ORT). | — | — |
| **FP32** | ONNX export in 32-bit floating point precision (fidelity control/reference). | — | Baseline for computing Δ vs FP32. |
| **INT8 dynamic (all)** | Dynamic INT8 quantization applied to supported ops (e.g. E1), without calibration. | — | Can reduce size; **does not** guarantee lower CPU latency. |
| **INT8 dynamic (no attention)** | Dynamic variant excluding attention or selected ops. | — | `[depends on artifact/availability]`. |
| **FP16** | Weights/compute in half precision where applicable. | — | `[depends on viability/runtime]`. |
| **Static INT8 (QDQ/calibration)** | Static quantization with calibration data (e.g. E3 hybrid). | — | Sensitive to calibration/distribution; evaluate on holdout per protocol. |
| **`canonical_match` RDKit (primary, vs label)** | **Primary OCSR metric (chemical quality):** equality of **RDKit canonical SMILES** between prediction and label, counted only when **both parse** (`correct/comparable (rate)`). | `210/292 (0.7192)` = 210 correct over 292 comparable. | Preferred for representation equivalence. Depends on RDKit/parsing; denominator is `canonical_comparable` (may be < n). Non-comparable cases are interpreted together with `valid`/`incomplete`. |
| **`exact_match` string (secondary, vs label)** | **Exact text** match `predicted == expected` (no canonicalization). | `121 (0.4033)` = 121/300 identical strings. | **Secondary** for chemistry: equivalent SMILES may differ and count as non-matches. Useful as raw output/text consistency signal. |
| **`tf_match` ONNX↔TF (fidelity, string)** | Conversion fidelity: **ONNX** output identical (string) to **TF baseline** for the same image. `tf_canonical` (when reported) is fidelity in canonical form. | `279/300 (0.9300)`. | Measures **ONNX ≈ TF parity**, **not** chemical correctness vs label. |
| **Δ vs FP32 (pp): Δcanonical / Δtf / Δexact** | Absolute difference in **percentage points** vs FP32, over the **same metric and same IDs** (paired). | `-1.03 pp` canonical = 1.03 pp lower than FP32. `(ref)` = FP32. | **Primary for quality:** `Δcanonical`. “pp” is absolute difference. |
| **`valid` / `incomplete`** | Harness heuristics (no RDKit): `valid_rate` = fraction marked valid by simple rules (e.g. non-empty/non-heuristically-invalid); `incomplete` = sequences that did not finish (e.g. missing end token). | `0.9967 / 1` = 99.67% heuristically valid, 1 incomplete out of 300. | Approximation; **does not** guarantee real chemical validity. |
| **Latency p50 / p95 (ms)** | Time per image (ms) on same hardware and execution provider (here, ORT CPU): **p50** = median (50% ≤ t), **p95** = tail (95% ≤ t). | `3455 / 4676` = median ~3.46 s, p95 tail ~4.68 s. | Do not use mean only. Not comparable across machines/configurations without control. If there are `cache_hits`, latency is not fresh/comparable (sometimes `n/a`). |
| **Size (MB)** | On-disk size of variant artifacts (e.g. encoder+decoder). | — | `[if computed]`. |
| **Throughput** | Images/s or batches/s if evaluated in batch. | — | `[if applicable]`. |
| **Classifier (binary + reactions)** | Binary metric over 3 mapped classes (`one/several_molecule→structure`, `other→no-structure`); `reactions` **excluded** from binary and reported separately as `ambiguous`. | Matrix `[[TP,FN],[FP,TN]]`, binary accuracy, TF parity, reactions→prediction counts. | Binary does not include `reactions`; interpret separately. |
| **`cache_hits`** | Predictions reused from cache `(model, variant, image_id)`. | — | Speeds reruns; if all are cache hits, no fresh latency is measured. |
| **McNemar** | Paired test over discordances (A correct/B wrong vs inverse) on the same set. | — | `[pending implementation in report/harness]`. |"""

CRITERIA = r"""## 2. Acceptance criteria (defined for evaluation)

| Criterion | Definition / acceptance condition |
|---|---|
| **Absolute floor (mandatory, all)** | **Accuracy ≥ 70%** on the dossier **primary metric**, on the **frozen `eval` split** (same IDs). **OCSR:** RDKit `canonical_match_rate` (over comparable pairs). **Classifier:** `binary accuracy` (non-reaction). Variants < 70% **are not accepted**. |
| **Paired comparison** | All variants are measured on the **same frozen set/IDs** (pilot ⊂ eval ⊂ pool). Δ vs FP32 in pp over same IDs. |
| **Fidelity (rate / strict)** | `tf_match_rate`/`tf_canonical_match_rate` (ONNX≈TF). Strict: 100%. Numeric threshold (`≥ [TBD]%`) is an external parameter if defined. |
| **Δ vs FP32 (accuracy/fidelity)** | Absolute difference in pp vs FP32, same metric and IDs. Threshold `≤ [TBD] pp` `[TBD]`. |
| **OCSR accuracy (primary/secondary)** | **Primary:** RDKit canonical (`canonical_match`). **Secondary:** `exact_match` string (no canonicalization), informational. |
| **Validity / completeness** | `valid_rate`, `incomplete_rate`. External thresholds if defined. |
| **Latency** | **p50/p95** ms, same hardware and execution provider (ORT CPU). Not mean only. p95 threshold `[TBD]`. Dynamic INT8 does not imply lower latency. |
| **Size / throughput** | Disk MB (artifacts); throughput (img/s, batch) if applicable. |
| **Classifier** | Binary over 3 mapped classes; `reactions` separate/excluded from binary. |
| **Reproducibility** | Source/pin/manifest/checksums, frozen IDs (seed), hardware/provider/versions. |"""

PROVENANCE = r"""## 3. Data sources and provenance (facts)

| Suite / use | Target model | Source | Identifier / pin | License / provenance (per manifest/source) | Local artifacts |
|---|---|---|---|---|---|
| Printed golden CLEF | `ocsr-printed` | `Keylab/OCSR-Benchmarks` (HF, rows API) | observed dataset commit `2fdcdc59`; config `CLEF`, split `test`, 992 | HF aggregate **mixed-see-sources**; verify original CLEF 2012 terms | `data/benchmark/hf-clef/labels.jsonl`; `data/manifests/benchmark-hf-clef.import.json` |
| Printed golden JPO | `ocsr-printed` | `Keylab/OCSR-Benchmarks` (HF, rows API) | config `JPO`, split `test`, 449 | HF aggregate **mixed-see-sources**; verify original JPO/ChemInfty/OSRA terms | `data/benchmark/hf-jpo/labels.jsonl`; `data/manifests/benchmark-hf-jpo.import.json` |
| Hand-drawn HDM | `ocsr-handdrawn` | Zenodo DECIMER Hand-drawn molecule images | record `7617107`, DOI 10.5281/zenodo.7617107, 5088 | CC-BY-4.0 | `data/benchmark/handdrawn-hdm/labels.jsonl`; `data/manifests/benchmark-handdrawn-hdm.fetch.json` |
| Classifier (prepared) | `image-classifier` | Zenodo 13378718 | record `13378718`, zip md5 `e7e47f2aa4af4b79874a19382cec06da` (Zenodo API) | Record indicates MIT; mixed image origins | `decimer-export import-benchmark --source classifier-13378718` (requires local extraction) |
| Classifier 6670746 | `image-classifier` | Zenodo 6670746 | single file 126.6 GB, md5 `9e847917a75ac77d7fbe1f50b2818a5b` | CC-BY-4.0 | Not downloaded. |

**Classifier mapping (13378718, binary):** `one_molecule`→`structure`, `several_molecules`→`structure`, `other`→`no-structure` (included in binary matrix); `reactions`→excluded from binary, separate metric `ambiguous_excluded`."""


def _latency_from_details(d: dict[str, Any]) -> tuple[str, str, str]:
    lat = d.get("latency_ms") or {}
    p50, p95 = lat.get("p50"), lat.get("p95")
    if p50 is None or p95 is None:
        vals = [r.get("latency_ms") for r in d.get("details", []) if not r.get("cache_hit") and r.get("latency_ms") is not None]
        vals = [float(v) for v in vals]
        if vals:
            vals.sort()
            p50 = vals[len(vals) // 2]
            idx = max(0, int(len(vals) * 0.95) - 1)
            p95 = vals[idx]
    cache_hits = d.get("cache_hits")
    n = d.get("count") or d.get("total_rows") or 0
    note = ""
    if cache_hits and n and int(cache_hits) >= int(n):
        note = " (cache, sin latencia fresca)"
        return "—", "—", note
    if p50 is None and cache_hits:
        return "—", "—", f" (cache_hits={cache_hits})"
    return _fmt_ms(p50), _fmt_ms(p95), note


def _tf_canonical_stats(r: dict[str, Any]) -> tuple[int | None, int | None, float | None]:
    """Top-level tf_canonical aggregate or computed from details."""
    if r.get("tf_canonical_comparable") is not None:
        return r.get("tf_canonical_match"), r.get("tf_canonical_comparable"), r.get("tf_canonical_match_rate")
    det = r.get("details") or []
    if not det or not any("tf_canonical_match" in d for d in det[:3]):
        return None, None, None
    comp = [d for d in det if d.get("tf_canonical_match") is not None]
    if not comp:
        return (None, 0, None) if any(d.get("tf_smiles") is not None for d in det) else (None, None, None)
    m = sum(1 for d in comp if d.get("tf_canonical_match"))
    return m, len(comp), m / len(comp)


def _pilot_ocsr_row(label: str, report_path: Path, *, fp32_path: Path | None = None) -> str | None:
    r = _load_json(report_path)
    if not r:
        return None
    n = r.get("count")
    ex = r.get("exact_match"); exr = r.get("exact_match_rate")
    cm = r.get("canonical_match"); ccomp = r.get("canonical_comparable"); cmr = r.get("canonical_match_rate")
    tfm = r.get("tf_match"); tfmr = r.get("tf_match_rate")
    tcm, tcc, tcr = _tf_canonical_stats(r)
    valid = r.get("valid"); vr = r.get("valid_rate"); inc = r.get("incomplete")
    p50, p95, note = _latency_from_details(r)
    size = _artifact_size_mb(report_path)
    if fp32_path and report_path != fp32_path and fp32_path.exists():
        dtf = _delta_pp(report_path, fp32_path, "tf_match") if tfm is not None else "—"
        dex = _delta_pp(report_path, fp32_path, "exact_match")
        fr = _load_json(fp32_path) or {}
        dcan = "—"
        if cmr is not None and fr.get("canonical_match_rate") is not None:
            dcan = f"{(cmr-fr['canonical_match_rate'])*100:+.2f} pp"
    elif fp32_path and report_path == fp32_path:
        dtf = dex = dcan = "(ref)"
    else:
        dtf = dex = dcan = "—"
    rep = str(report_path.relative_to(ROOT)) if report_path.is_relative_to(ROOT) else str(report_path)
    tf_cell = _fmt_rate(tfm, n) if tfm is not None else "—"
    if tcm is not None and tcc:
        tf_cell = f"{tf_cell} / can {tcm}/{tcc} ({tcr:.3f})"
    mark = " ✅≥70%" if cmr is not None and cmr >= 0.70 else (" ❌<70%" if cmr is not None else "")
    can_cell = f"**{cm}/{ccomp} ({cmr:.4f})**{mark}" if (cm is not None and ccomp) else "—"
    # Primary = canonical RDKit (floor >=70%)
    return (
        f"| {label} | {n} | {can_cell} | {ex} ({exr:.4f}) | {tf_cell} | {dcan} | {dtf} | {dex} | {valid} ({vr:.4f}) | {inc} | "
        f"{p50}{note} | {p95} | {size} | `{rep}` |"
    )


def _classifier_row(report_path: Path, *, fp32_path: Path | None = None) -> str | None:
    d = _load_json(report_path)
    if not d:
        return None
    cm = d.get("binary_confusion_matrix") or {}
    rx = d.get("reactions_separate_metric") or {}
    tfp = d.get("tf_parity") or {}
    acc = cm.get("accuracy")
    total = cm.get("total")
    mat = cm.get("matrix_rows_expected_cols_predicted")
    amark = " ✅≥70%" if acc is not None and acc >= 0.70 else (" ❌<70%" if acc is not None else "")
    acc_s = f"{acc:.4f} (n={total}){amark}" if acc is not None else "—"
    mat_s = f"`{mat}`" if mat else "—"
    tf_s = _fmt_rate(tfp.get("agree"), tfp.get("total")) if tfp else "—"
    rx_s = f"{rx.get('count','—')} → struct {rx.get('predicted_structure','—')} / no-struct {rx.get('predicted_no_structure','—')}" if rx else "—"
    p50, p95, note = _latency_from_details(d)
    variant = d.get("variant_key") or report_path.stem
    # Δ vs FP32 on binary accuracy (aggregate; same frozen split implied)
    dacc = "—"
    if fp32_path and report_path != fp32_path and fp32_path.exists():
        fd = _load_json(fp32_path) or {}
        facc = ((fd.get("binary_confusion_matrix") or {}).get("accuracy"))
        if acc is not None and facc is not None:
            dacc = f"{(acc-facc)*100:+.2f} pp"
    elif fp32_path and report_path == fp32_path:
        dacc = "(ref)"
    rep = str(report_path.relative_to(ROOT)) if report_path.is_relative_to(ROOT) else str(report_path)
    split = d.get("split") or "—"
    return f"| `{variant}` (`{split}`) | {d.get('total_rows','—')} | {acc_s} | {mat_s} | {dacc} | {tf_s} | {rx_s} | {p50}{note} | {p95} | `{rep}` |"


def _ocsr_variant_label(path: Path, data: dict[str, Any] | None) -> str:
    d = data or {}
    vk = (d.get("variant_key") or path.stem or "").lower()
    name = path.name
    if "fp32enc-e1sdec-rr" in name or vk == "fp32enc-e1sdec-rr": return "FP32enc + e1sdec (no-attn) + reduce_range"
    if "fp32enc-e1sdec" in name or vk == "fp32enc-e1sdec": return "FP32enc + e1sdec (no-attn, mixed dec)"
    if "fp32enc-e1dec-rr" in name or vk == "fp32enc-e1dec-rr": return "FP32enc + E1dec + reduce_range"
    if "fp32enc-e1dec" in name or vk == "fp32enc-e1dec": return "FP32enc + E1dec (dyn)"
    if "e1enc-fp32dec" in name or vk == "e1enc-fp32dec": return "E1enc (dyn) + FP32dec"
    if "selbenc-e1sdec" in name or vk == "selbenc-e1sdec": return "selBenc (blocks 40..56 FP32) + e1sdec"
    if "selaenc-e1sdec" in name or vk == "selaenc-e1sdec": return "selAenc (stem+0..5+50..56 FP32) + e1sdec"
    if "selcenc-e1sdec" in name or vk == "selcenc-e1sdec": return "selCenc (stem+0..15 FP32) + e1sdec"
    if "e1s-sel-noattn" in name or "e1s" in vk and "noattn" in name: return "E1s selective dyn (decoder without attention)"
    if "int8-dyn-all" in name or vk == "int8-dyn-all" or ("e1" in vk and "dyn" in vk): return "E1 dynamic INT8 all"
    if "e3" in name or "e3" in vk: return "E3 hybrid (static-enc QDQ + dyn-dec)"
    if "e4" in name or "fp16" in vk: return "E4 FP16"
    if "fp32" in name or vk == "fp32": return "FP32 (ref)"
    if "tf-baseline" in vk or vk == "tf": return "TF baseline"
    return d.get("variant_key") or path.stem


def _render_ocsr(lines: list[str], pending: list[str]) -> None:
    configs = [
        ("`ocsr-printed`", "printed", ROOT/"data/manifests/frozen/printed.frozen.json"),
        ("`ocsr-handdrawn`", "handdrawn", ROOT/"data/manifests/frozen/handdrawn.frozen.json"),
    ]
    header = "| Variant | n | **RDKit canonical (primary, comp.)** | exact string | tf_match string (+ tf_canonical) | Δcanonical pp | Δtf pp | Δexact pp | valid | incomplete | p50 ms | p95 ms | size MB | Report |"
    sep = "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|"
    splits = [
        ("eval-1000", "split `eval` (gate, n=1000)", "eval1000"),
        ("remainder", "split `remainder` (held-out, no selection)", "remainder"),
        ("pilot", "split `pilot` (signal/selection, n=300)", "pilot300"),
    ]
    for title, prefix, fman in configs:
        man = _load_json(fman) or {}
        meta = ""
        if man:
            meta = f" (frozen n={man.get('frozen_n')}, eval={man.get('eval_n')}, pilot={man.get('pilot_n')}, seed={man.get('seed')}, source=`{fman.relative_to(ROOT)}`)"
        for dirname, splitlabel, suffix in splits:
            d = ROOT / "data/benchmark" / dirname
            if not d.exists():
                continue
            files = sorted([p for p in d.glob(f"{prefix}.*.{suffix}.json") if "imago" not in p.name.lower()])
            if not files:
                continue
            # fp32 ref for this split
            fp32_ref = d / f"{prefix}.fp32.{suffix}.json"
            if not fp32_ref.exists():
                fp32_ref = files[0]
            rows = []
            for p in files:
                data = _load_json(p)
                lbl = _ocsr_variant_label(p, data)
                rows.append(_pilot_ocsr_row(lbl, p, fp32_path=fp32_ref))
            got = [r for r in rows if r]
            if not got:
                continue
            marker = "5.E" if dirname == "eval-1000" else ("5.R" if dirname == "remainder" else "5.P")
            lines.append(f"### {marker} {title} — **{splitlabel}**{meta}")
            lines.append("")
            lines.append(header); lines.append(sep); lines.extend(got); lines.append("")
    if (ROOT / "data/benchmark/eval-1000/printed.fp32.eval1000.json").exists():
        pending = [p for p in pending if "eval-1000 pending" not in p.lower()]


def _classifier_label(path: Path, data: dict[str, Any] | None) -> str:
    d = data or {}
    vk = (d.get("variant_key") or path.stem or "").lower()
    name = path.name.lower()
    if "e4-fp16" in name or "fp16" in vk: return "E4 FP16 (selected, ≥50% compression, acc ≥70%)"
    if "e2-static" in name or ("e2" in vk and "static" in vk): return "E2 static QDQ (holdout calib)"
    if "e1s-sel" in name or ("e1s" in vk): return "E1s selective dyn (no Conv)"
    if "int8-dyn-all" in name or vk == "int8-dyn-all": return "E1 dynamic INT8 all"
    if "fp32" in name or vk == "fp32": return "FP32 (ref)"
    return d.get("variant_key") or path.stem


def _render_classifier(lines: list[str], pending: list[str]) -> None:
    for dirname, splitlabel in [("eval-1000", "eval-1000"), ("pilot", "pilot")]:
        d = ROOT / "data/benchmark" / dirname
        if not d.exists():
            continue
        files = sorted([p for p in d.glob("classifier.*.json") if "imago" not in p.name.lower()])
        if not files:
            continue
        fp32_ref = d / ("classifier.fp32.eval1000.json" if dirname == "eval-1000" else "classifier.fp32.pilot300.json")
        got2 = []
        for p in files:
            data = _load_json(p)
            lbl = _classifier_label(p, data)
            row = _classifier_row(p, fp32_path=fp32_ref)
            if not row:
                continue
            parts = row.split("|")
            if len(parts) > 2:
                parts[1] = f" {lbl} (`{splitlabel}`) "
                row = "|".join(parts)
            got2.append(row)
        if not got2:
            continue
        lines.append(f"### 5.5 `image-classifier` (Zenodo 13378718, congelado) — {splitlabel}")
        lines.append("")
        fman = _load_json(ROOT/"data/manifests/frozen/classifier.frozen.json") or {}
        if fman and dirname == "eval-1000":
            lines.append(f"Frozen: n={fman.get('frozen_n')}, eval={fman.get('eval_n')}, pilot={fman.get('pilot_n')}, seed={fman.get('seed')}, classes {fman.get('class_allocated')}. Binary over `one/several_molecule→structure`, `other→no-structure` (threshold `0.000089`); `reactions` excluded from binary and reported separately. Goal: **accuracy ≥70%** while improving compression. **All** results for this suite are recorded (including non-selected).")
            lines.append("")
        lines.append("| Variant (split) | total rows | Binary acc (non-reaction) | Matrix `[[TN,FP],[FN,TP]]` | Δacc vs FP32 | TF parity (incl. reactions) | reactions (separate) | p50 ms | p95 ms | Report |")
        lines.append("|---|---:|---:|---|---:|---:|---:|---:|---:|---|")
        lines.extend(got2)
        lines.append("")
    if (ROOT / "data/benchmark/eval-1000/classifier.e4-fp16.eval1000.json").exists():
        lines.append("> **Decision (classifier, for now):** `E4 FP16` meets the goal (acc ≈ FP32, ≥70%; **-49% size**; high TF parity). INT8 on Conv (E1-all/E2 static holdout) drops below 70%; E1s preserves accuracy but does not compress.")
        lines.append("")
    lines.append("| Suite | Status | Note |")
    lines.append("|---|---|---|")
    lines.append("| Zenodo 6670746 | OUT OF SCOPE | Do not download/use (126.6 GB). |")
    lines.append("| Zenodo 13378718 | All runs recorded in this report | 4→binary mapped; `reactions` separate. Imago excluded (curiosity experiment, `docs/experiments/`). |")
    lines.append("")
def render_machine(info: dict[str, Any]) -> str:
    ram = info.get("ram_total_bytes")
    ram_gb = f"{ram/1024**3:.2f} GiB ({ram} bytes)" if ram else "—"
    ram_av = info.get("ram_available_at_collect_bytes")
    ram_av_s = f"{ram_av/1024**3:.2f} GiB" if ram_av else "—"
    pkgs = info.get("packages") or {}
    pkg_rows = "\n".join(f"| `{k}` | `{v}` |" for k, v in sorted(pkgs.items())) or "| — | — |"
    return f"""## 4. Machine characteristics (automatic collection)

Automatically collected at `{info.get('collected_at_utc','')}` using `platform`, `lscpu`, and `free`.

| Field | Value |
|---|---|
| Hostname | `{info.get('hostname','')}` |
| OS | `{info.get('os','')}` |
| Kernel/OS version | `{info.get('os_version','')}` |
| Architecture | `{info.get('arch','')}` |
| Python | `{info.get('python','')}` |
| CPU model | `{info.get('cpu_model','')}` |
| Sockets / cores per socket / threads per core | `{info.get('cpu_sockets','')}` / `{info.get('cpu_cores_per_socket','')}` / `{info.get('cpu_threads_per_core','')}` |
| Logical CPUs | `{info.get('cpu_logical','')}` |
| CPU max/min frequency (MHz) | `{info.get('cpu_max_mhz','')}` / `{info.get('cpu_min_mhz','')}` |
| Caches L1d/L1i/L2/L3 | `{info.get('l1d','')}` / `{info.get('l1i','')}` / `{info.get('l2','')}` / `{info.get('l3','')}` |
| Relevant CPU flags | `{info.get('cpu_flags_relevant','')}` |
| Total RAM | {ram_gb} |
| Available RAM (at collection) | {ram_av_s} |

### Package versions (execution environment)

| Package | Version |
|---|---|
{pkg_rows}
"""


def render_results() -> tuple[str, list[str]]:
    b = ROOT / "data" / "benchmark"
    pending: list[str] = []
    lines = [
        "## 5. Results",
        "",
        "This report covers **three distinct models** evaluated in independent dossiers: `ocsr-printed`, `ocsr-handdrawn`, and `image-classifier` (no ranking across models).",
        "",
        "Protocol: variants compared on **same IDs** when applicable (paired, pilot ⊂ eval ⊂ frozen pool). Latency **p50/p95** (same hardware/provider), not mean only; dynamic INT8 may not be faster. **Δ vs FP32** = difference in pp over common IDs (same metric). `—` = not measured/not available.",
        "",
        "> **Execution note (reality):** some runs (pilots/remainders) were executed with CPU contention, so their `p50/p95` may not be comparable (accuracy is unaffected); prefer solo measurements for latency comparison. McNemar/bootstrap CI are not automated in this report (rates/Δpp/`n` are reported). There were retries due to transient ONNX load failures and runtime/quantization limitations documented in results/selections. See glossary (§1) and `docs/benchmark-spec.md` (§7) for details.",
        "",
        "> **Domain/specialization (under evaluation):** `ocsr-printed` and `ocsr-handdrawn` are distinct models (different weights/tokenizers/origins) but **not exclusive specialists** based on observed cross-evaluation evidence (e.g. 2×2 pilot FP32: handdrawn 0.712 on printed vs 0.711 printed in-domain; 0.719 on hand-drawn, while printed drops to 0.677 on hand-drawn). Additional tests (cross eval-1000 and 1:1 printed+hand-drawn mix) are in progress/pending consolidation to confirm differences.",
        "",
    ]

    def table(title: str, rows: list[str | None]):
        got = [r for r in rows if r]
        if not got:
            return
        lines.append(f"### {title}")
        lines.append("")
        lines.append("| Variant | n | **RDKit canonical (primary, comp.)** | exact string | tf_match string (+ tf_canonical) | Δcanonical pp | Δtf pp | Δexact pp | valid | incomplete | p50 ms | p95 ms | size MB (enc+dec) | Report |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|")
        lines.extend(got)
        lines.append("")

    # Pilotos congelados (tres dossiers)
    _render_ocsr(lines, pending)

    table("5.1 `ocsr-printed` — golden CLEF (Keylab/OCSR-Benchmarks, n=992, historical evidence)", [
        _row("E1 dynamic INT8 (+TF)", b/"hf-clef/report.e1.tf.json"),
        _row("E3 static-enc + dyn-dec (tf_match por join a TF de E1)", b/"hf-clef/report.e3.tfjoined.json", tf_join_note="[join]"),
    ])
    table("5.2 `ocsr-printed` — golden JPO (Keylab/OCSR-Benchmarks, n=449, historical evidence)", [
        _row("E1 dynamic INT8 (+TF)", b/"hf-jpo/report.e1.tf.json"),
        _row("E3 static-enc + dyn-dec (tf_match por join a TF de E1)", b/"hf-jpo/report.e3.tfjoined.json", tf_join_note="[join]"),
    ])
    hdm_fp32 = b/"handdrawn-hdm/report.fp32.tf.limit500.json"
    table("5.3 `ocsr-handdrawn` — HDM Zenodo 7617107 (ad hoc sample `--limit 500`, not a formal decision)", [
        _row("FP32 (+TF)", hdm_fp32, fp32_path=hdm_fp32),
        _row("E1 dynamic INT8 (+TF)", b/"handdrawn-hdm/report.e1.tf.limit500.json", fp32_path=hdm_fp32),
        _row("E3 static-enc + dyn-dec (+TF, calib 32 HDM)", b/"handdrawn-hdm/report.e3.tf.limit500.json", fp32_path=hdm_fp32),
    ])
    full = b/"handdrawn-hdm/report.e1.tf.full5088.json"
    if full.exists():
        table("5.4 `ocsr-handdrawn` — full HDM (5088)", [
            _row("E1 dynamic INT8 (+TF)", full),
        ])
    else:
        pending.append(f"`ocsr-handdrawn` full HDM n=5088, E1+TF: `{full.relative_to(ROOT)}` (not found at generation time).")
        lines.append("### 5.4 `ocsr-handdrawn` — full HDM (5088)")
        lines.append("")
        lines.append("| Variant | n | Status | Expected report |")
        lines.append("|---|---:|---|---|")
        lines.append(f"| E1 dynamic INT8 (+TF) | 5088 | Pending / not available at generation time | `{full.relative_to(ROOT)}` |")
        lines.append("")

    _render_classifier(lines, pending)
    return "\n".join(lines), pending


LIMITATIONS = r"""## 6. Measurement limitations (methodological facts)

- `exact_match` (string) compares `predicted == expected` **without RDKit canonicalization**. `canonical_match` (RDKit) compares canonical SMILES when both parse (integrated in OCSR benchmark; may be absent in historical reports). Equivalent representation differences may count as non-exact in `exact_match`.
- `tf_match` measures **ONNX↔TF** fidelity according to the harness baseline/pipeline; it is not canonical chemical correctness against the label.
- `valid`/`incomplete` are heuristics without RDKit (approximation).
- Latencies come from the execution environment/harness (machine/provider/load/threads/versions). Report **p50/p95**; do not use mean only. `cache_hits` invalidate fresh latency. Dynamic INT8 can be slower than FP32 on CPU.
- Δ in pp computed over the same metric and IDs when paired data is available; if pairing is missing, an aggregate may be indicated.
- HF CLEF/JPO subsets: aggregate `mixed-see-sources`; verify original terms before redistribution.
- Static quantization (E3/QDQ) strongly depends on the calibration set (size/distribution/holdout); results do not generalize without a controlled protocol.
- Classifier: binary only over 3 mapped classes; `reactions` excluded and reported separately (`ambiguous_excluded`)."""


def generate_report(targets: list[Path] | None = None, machine_info: dict[str, Any] | None = None) -> str:
    if targets is None:
        targets = [ROOT / "docs/benchmark-report.md"]
    info = machine_info or collect_machine_info()
    results, pending = render_results()
    now = datetime.now(timezone.utc).isoformat()
    parts = [
        "# DECIMER → ONNX benchmark report",
        "",
        "> Document auto-generated from JSON reports/manifests in `data/`. It does not include decisions or judgments about results; it reports measurements, definitions, and acceptance criteria defined for evaluation.",
        "",
        f"**Generated (UTC):** `{now}`  ",
        "**Generator:** `toolchain.report.generate_report` / `dtx report` (alias: `decimer-export report`)  ",
        "**Relevant branch/commits:** `main` (workspace fetch/export/runtime/quantization/benchmark).",
        "",
        "---",
        "",
        GLOSSARY,
        "",
        CRITERIA,
        "",
        PROVENANCE,
        "",
        render_machine(info),
        results,
        "## 7. Reference files",
        "",
        "- Fetch/import manifests (tracked): `data/manifests/benchmark-*.json`.",
        "- Detailed reports (gitignored, generated locally): `data/benchmark/**/*.json`.",
        "- Printed golden summary (tracked): `data/artifacts/selection/ocsr-printed.hf-golden.json`.",
        "- Printed selection (tracked, may contain interpretation): `data/artifacts/selection/ocsr-printed.selection.json`.",
        "",
        LIMITATIONS,
        "",
    ]
    if pending:
        parts += ["## 8. Pending items at generation time", ""] + [f"- {p}" for p in pending] + [""]
    text = "\n".join(parts)
    for t in targets:
        t.parent.mkdir(parents=True, exist_ok=True)
        t.write_text(text, encoding="utf-8")
    return text


if __name__ == "__main__":
    generate_report()
    print("ok")
