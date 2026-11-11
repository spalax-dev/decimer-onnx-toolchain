# Dossier — Image classifier (`image-classifier`)

> Independent dossier. Not compared with printed/hand-drawn OCSR.

## Purpose
Validate DECIMER Image Classifier → ONNX conversion: fidelity vs baseline (if applicable) and structure/no-structure classification quality; close ONNX export + wiring + variant selection.

## Suite
| Source | n | License/provenance | Status |
|---|---:|---|---|
| Zenodo **13378718** (`dataset_for_image_classifier.zip`) | per extracted split (e.g. `classified` 4×4000, `for_model/test` 1604) | record indicates MIT; mixed image origins (verify) | **IN** — requires local extraction |
| Zenodo **6670746** | 126.6 GB | CC-BY-4.0 | **OUT OF SCOPE (formalized)** — do not download or use (single file, no exposed separate test set) |

Importer: `decimer-export import-benchmark --source classifier-13378718 --root <extracted_zip> --split <classified|for_model/test|...>`.

## Label mapping (contract, explicit)
| Original category | Binary | In binary matrix | Reporting |
|---|---|:---:|---|
| `one_molecule` | `structure` | Yes | 4×4 + binary |
| `several_molecules` | `structure` | Yes | 4×4 + binary |
| `other` | `no-structure` | Yes | 4×4 + binary |
| `reactions` | — | **No** | separate metric `ambiguous_excluded` (not in TP/TN/FP/FN) |

Report **4×4** and **collapsed binary**; exclusion of `reactions` is explicit.

## `n` / IDs (frozen)
- **Target frozen pool n=10000**, **stratified by the 4 classes, rare classes overrepresented** (seed `20260308`, deterministic method: per-class floor + remainder weighted by sqrt(capacity); see `data/manifests/frozen/classifier.frozen.spec.json`).
- **Status:** `pending_data` until 13378718 labels are imported (`decimer-export import-benchmark --source classifier-13378718 --root <extracted_zip>`). Generate: `python -m decimer_onnx_export.frozen --classifier-labels <labels.jsonl> --out data/manifests/frozen/classifier.frozen.json`.
- **Eval n=1000**, **Pilot n=300** = first 300 of eval (fixed frozen order), same IDs for all variants. Remaining pool for extended analysis/reporting as defined.
- Prediction cache: `(model, variant, image_id)`.

## Metrics (paired protocol)
- 4×4 + binary matrix (only 3 mapped categories), accuracy/precision/recall/F1 `[TBD]`. `reactions` separate.
- ONNX↔baseline/TF fidelity if applicable `[TBD]`. Latency **p50/p95** (same hardware, same provider), **size MB**, **throughput** if batches. Not mean only.
- **Absolute Δ in pp vs FP32** (same metric, same frozen IDs) + absolute value. Paired comparison: **McNemar** + bootstrap CI where applicable; `n`+CI. Pilot ⊂ eval ⊂ frozen pool. Cache `(model,variant,image_id)`.
- Note: dynamic INT8 may not be faster; measure latency.

## Acceptance criteria (placeholders `[TBD]`)
| Criterion | Threshold |
|---|---|
| Main binary metric | `[TBD]` |
| Export fidelity (if applicable) | `[TBD]` |
| p95 latency | `≤ [TBD] ms` |

## Variants to compare (measurement order)
1. **FP32** (control)
2. **INT8 dynamic (all)**
3. **INT8 dynamic (no attention / selective MLP)** `[pending generation]`
4. **FP16**

Record per variant: `model_key=image-classifier`, stable `variant_key`. Pending ONNX export + predict wiring.

## Status
- Weights: fetch available (`--model image-classifier`, pinned commit). **ONNX export not performed.** Predict wiring in CLI/harness pending.
- Logical harness/mapping implemented in `benchmark_run`/importer; e2e execution over 13378718 pending.
- Immediate blocker: export classifier to ONNX and define iteration split/`n` (manifest).
