# Dossier — Hand-drawn OCSR (`ocsr-handdrawn`)

> Independent dossier. Not compared with printed or classifier.

## Purpose
Validate DECIMER hand-drawn → ONNX conversion: fidelity vs TF and SMILES quality on the hand-drawn domain; select variant (FP32/E1/E3) and calibration policy.

## Suite
| Source | n | Domain | License/provenance | Manifest/labels |
|---|---:|---|---|---|
| `handdrawn-hdm` (Zenodo 7617107) | 5088 | hand-drawn | CC-BY-4.0, verified checksums | `data/manifests/benchmark-handdrawn-hdm.fetch.json`, `data/benchmark/handdrawn-hdm/labels.jsonl` |

Stratify by subsource if identifiable from IDs/original structure.

## `n` / IDs (frozen)
- **Frozen pool n=1400** (seed `20260308`, deterministic selection from 5088 HDM) → `data/manifests/frozen/handdrawn.frozen.json`.
- **Eval n=1000** = first 1000 of the frozen pool (fixed order). **Pilot n=300** = first 300 of eval (same IDs for all variants).
- **Full dossier n=5088** reserved for final reporting. Previous `--limit 500` = ad hoc sample (do not use as formal decision basis). All compared variants run on the same IDs of the split (paired).
- Prediction cache: `(model, variant, image_id)` in `data/cache/predictions/` (gitignored).

## Metrics (paired protocol)
- `tf_match` (ONNX==TF, fidelity). RDKit **canonical SMILES accuracy** `[pending integration]` + `exact_match` string. `valid`, `incomplete`.
- Latency **p50/p95** (same hardware, ORT `CPUExecutionProvider`), **size MB**, **throughput** if batches. Not mean only.
- **Absolute Δ in pp vs FP32** (same metric, same IDs) + absolute value. **McNemar** + Δ pp with **bootstrap CI**; `n`+CI. Pilot ⊂ eval ⊂ frozen pool; ad hoc samples are not used for formal decisions. Cache `(model,variant,image_id)`.
- Note: dynamic INT8 may not be faster; measure latency.

## Acceptance criteria (placeholders `[TBD]`)
| Criterion | Threshold |
|---|---|
| `tf_match` fidelity | `≥ [TBD]%` |
| Canonical Δ vs TF/label | `≤ [TBD]` pp / Tanimoto `[TBD]` |
| `valid` / `incomplete` | `≥ [TBD]%` / `≤ [TBD]%` |
| p95 latency | `≤ [TBD] ms` (fixed hardware) |

## Variants to compare (measurement order)
1. **FP32** (fidelity control)
2. **INT8 dynamic (all)** — (current: E1 artifacts)
3. **INT8 dynamic (no attention)** `[pending generation]`
4. **FP16** `[pending; evaluate CPU ORT/target viability]`
5. **Static INT8 (if 2–4 fail criteria)** — holdout `[conditional]`

Record per variant: `model_key=ocsr-handdrawn`, stable `variant_key`, artifacts and config.

## Calibration (E3)
Policy `[TBD]`: fixed source, `n`, method; evaluation on **holdout**. Current evidence: E3 calibrated with 32 HDM showed `tf_match` 0/500 and 203 incompletes on an ad hoc sample — recalibrate/evaluate under holdout protocol before concluding.

## Status
- Weights: `data/weights/ocsr-handdrawn/DECIMER_HandDrawn_model`. Tokenizer: `data/artifacts/tokenizer_handdrawn.vocab.json` (`token_to_id` identical to printed).
- Measured: FP32/E1/E3+TF `--limit 500` (ad hoc sample). **E1+TF full 5088** may be running → `data/benchmark/handdrawn-hdm/report.e1.tf.full5088.json`.
- Pending: freeze iteration IDs (manifest), FP32 control on subset/full as cost allows, RDKit, set `[TBD]`, E3 holdout.
