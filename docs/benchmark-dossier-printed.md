# Dossier — Printed OCSR (`ocsr-printed`)

> Independent dossier. Not compared with hand-drawn or classifier.

## Purpose
Validate DECIMER printed → ONNX conversion: fidelity vs TF and SMILES quality on the printed domain; select variant (FP32/E1/E3) and calibration policy.

## Suite
| Source | n | Domain | License/provenance | Manifest/labels |
|---|---:|---|---|---|
| `hf-clef` | 992 | printed | Keylab/OCSR-Benchmarks, HF mixed-see-sources; verify CLEF 2012 terms | `data/manifests/benchmark-hf-clef.import.json` |
| `hf-jpo` | 449 | printed | same; verify JPO/ChemInfty/OSRA terms | `data/manifests/benchmark-hf-jpo.import.json` |
| **Total dossier (intra-dossier)** | **1441** | printed | — | `data/benchmark/hf-clef/labels.jsonl`, `data/benchmark/hf-jpo/labels.jsonl` |

## `n` / IDs (frozen)
- **Frozen pool n=1400** (seed `20260308`, deterministic selection from 1441 available; 41 excluded recorded by method) → `data/manifests/frozen/printed.frozen.json`.
- **Eval n=1000** = first 1000 of the frozen pool (fixed order). **Pilot n=300** = first 300 of eval (same IDs for all variants).
- All compared variants run on the **same IDs** of the split (paired). View by primary source + labeled intra-dossier pooled (CLEF+JPO within the split).
- Prediction cache: `(model, variant, image_id)` in `data/cache/predictions/` (gitignored), reusable via `--model-key/--variant-key/--cache-dir`.

## Metrics (paired protocol)
- `tf_match` (ONNX==TF, end-to-end fidelity; preprocessing contract recorded). RDKit **canonical SMILES accuracy** `[pending integration]` + `exact_match` string as data. `valid`, `incomplete`.
- Latency **p50/p95** (same hardware, ORT `CPUExecutionProvider`), **size MB** (encoder+decoder), **throughput** if batches. Not mean only.
- Report **absolute Δ in pp vs FP32** (same metric, same IDs) + absolute value. Paired comparison: **McNemar** + Δ pp with **bootstrap CI**; `n`+CI. Pilot ⊂ eval ⊂ frozen pool. Cache `(model,variant,image_id)`.
- Note: dynamic INT8 may not be faster (activation overhead); measure latency.

## Acceptance criteria (placeholders `[TBD]`)
| Criterion | Threshold |
|---|---|
| `tf_match` fidelity | `≥ [TBD]%` (strict 100% or defined tolerance) |
| Canonical Δ vs TF/label | `≤ [TBD]` pp / Tanimoto `[TBD]` |
| `valid` / `incomplete` | `≥ [TBD]%` / `≤ [TBD]%` |
| p95 latency | `≤ [TBD] ms` (fixed hardware) |

## Variants to compare (measurement order)
1. **FP32** (fidelity control)
2. **INT8 dynamic (all)** — weights/ops dynamically quantized (current: E1 artifacts)
3. **INT8 dynamic (no attention)** — dynamic variant excluding attention `[pending generation/export if applicable]`
4. **FP16** `[pending: E4 decoder showed runtime error on CPU ORT; evaluate viability/target runtime]`
5. **Static INT8 (if 2–4 fail criteria)** — fixed/holdout calibration, documented method `[conditional]`

Record per variant: `model_key=ocsr-printed`, stable `variant_key` (e.g. `fp32`, `int8-dyn-all`, `int8-dyn-noattn`, `fp16`, `int8-static`), artifacts and config.

## Calibration (E3)
Policy `[TBD]`: fixed source, `n`, method; evaluation on **holdout** (not in-sample). Current evidence: E3 calibrated with 2 fixtures showed strong `tf_match` regression on CLEF/JPO (see reports/selection) — define new calibration before reconsidering.

## Status
- Weights: `data/weights/ocsr-printed/DECIMER_model`. Tokenizer: `data/artifacts/tokenizer_SMILES.vocab.json`.
- Measured: E1+TF and E3 (TF join) on CLEF/JPO. Pending: FP32+TF control on golden, RDKit canonical, set `[TBD]`, recalibrate E3 on holdout if applicable.
- Reports: `data/benchmark/hf-{clef,jpo}/report.*.json` (gitignored).
