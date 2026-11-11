# Experiment: external Imago baseline (printed OCSR) — reference data

> **Scope:** curiosity/comparison experiment. **Not part of the official report** (`docs/benchmark-report.md`) and not part of the Imago build pipeline. It is recorded as a reference metric because Imago is a classic OCSR tool that, like the plan for DECIMER printed/hand-drawn, can be **embedded/distributed in software** (it is not an ML model).

## Tool
- **Imago OCR** (EPAM). Rule-based/C++ OCSR recognition.
- Package used: `epam.imago==2.0.0rc1` (PyPI, pre-release; Apache-2.0 in the current repo). Installed ad hoc in the venv (`uv pip install --pre`), not declared in `pyproject.toml`.
- Measured flow: image → `imago.Imago().load_image_from_file/recognize()` → `molecule` (MDL molfile) → **RDKit canonical SMILES** (`Chem.MolFromSmiles` + `MolToSmiles(canonical=True)`). Compared using the **primary OCSR metric: RDKit canonical SMILES**.

## Benchmark (same protocol as the printed dossier)
- Target dossier/model: `ocsr-printed` (DECIMER v2 printed, golden CLEF+JPO).
- Frozen set: `data/manifests/frozen/printed.frozen.json`, **split `eval`, n=1000**, seed `20260308` (same IDs as FP32/E1/TF).
- No TF parity baseline (not applicable to an external tool); cache `(ocsr-printed, imago-2.0.0rc1)`.
- Resilient execution: `scripts/benchmark_imago_resilient.py` (per-image subprocess, timeout/recovery) due to observed native crashes (`free(): invalid pointer`) after 107/1000 in in-process runs. 39 isolated errors/crashes; 107 cache hits from the initial run.
- Artifact (gitignored): `data/benchmark/eval-1000/printed.imago.eval1000.json`.

## Results (printed eval-1000)

| Metric | Imago 2.0.0rc1 |
|---|---:|
| **RDKit canonical (primary)** | **69/582 = 0.1186** (582/1000 comparable; both parse) — **❌ < 70%** |
| Exact string (secondary) | 50/1000 = 0.0500 |
| valid / incomplete | 608 / 392 |
| errors (crash/timeout/parse/no-mol) | 39 (out of 893 measured in resilient runner) |
| Latency (n=893 measured) | p50 **347.4 ms**, p95 **962.8 ms** |

## Comparison (same printed eval, primary canonical metric)

| System | Canonical (primary) | ≥70% | Exact string |
|---|---:|:---:|---:|
| **TF baseline (DECIMER printed, reference)** | **627/884 = 0.7093** | ✅ | 12/1000 (0.0120) |
| FP32 ONNX | 614/872 = 0.7041 | ✅ | 12/1000 (0.0120) |
| E1 dyn INT8 ONNX | 597/881 = 0.6776 | ❌ | 12/1000 (0.0120) |
| **Imago 2.0.0rc1 (external)** | **69/582 = 0.1186** | ❌ | 50/1000 (0.0500) |

**Paired (same 1000 IDs, canonical hit vs label):** TF correct only **575**, Imago only **17**, both 52, neither 356. Δ ≈ **+59.1 pp** in favor of TF (over respective comparable sets). On this set, **DECIMER printed (TF/FP32) is far superior to Imago** in canonical accuracy; the difference is large and paired (conclusive on this eval). Imago also produces many non-comparable/incomplete outputs.

## Caveats
- Old/rule-based tool; performance depends on image type/cleanliness. These results apply to **this frozen printed eval** (curated CLEF+JPO) and do not generalize without further evidence.
- Only 582/1000 outputs were canonically comparable (mol/SMILES parsing on both sides). Completeness/parsing rate is a relevant part of practical performance.
- Imago showed native instability in this environment; metrics use the resilient runner. Measured latency includes per-image subprocess overhead (not 1:1 comparable to in-process ORT/TF inference).
- Reference/embeddability data point; **not** included in `docs/benchmark-report.md`.
