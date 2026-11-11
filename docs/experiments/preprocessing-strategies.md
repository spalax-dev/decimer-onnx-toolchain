# Experiment: preprocessing strategies and DECIMER API behavior (3 selected models)

> **Scope:** exploratory experiment (like Imago baseline). **Not part of the official report** (`docs/benchmark-report.md`). Goal: test whether alternative preprocessing/filters improve benchmark results for the 3 selected ONNX models, and document what the DECIMER API does to improve results.

## Models (selected artifacts)
- `ocsr-printed` — FP32 (`data/artifacts/fp32/ocsr-printed/{encoder,decoder_step}.fp32.onnx`, vocab `tokenizer_SMILES.vocab.json`)
- `ocsr-handdrawn` — FP32 (`data/artifacts/fp32/ocsr-handdrawn/{encoder,decoder_step}.fp32.onnx`, vocab `tokenizer_handdrawn.vocab.json`)
- `image-classifier` — E4 FP16 (`data/artifacts/quant/e4/image-classifier/classifier.e4.fp16.onnx`)

## DECIMER 2.8.1 API (what it does to improve/choose results)

Observed in `.venv` (`DECIMER 2.8.1`, `DECIMER/decimer.py`, `DECIMER/pre_process.py`):
- **Model selection (`hand_drawn` flag):** `predict_SMILES(image, confidence=False, hand_drawn=False)` routes to `DECIMER_V2` (printed, Zenodo 8300489) or `DECIMER_Hand_drawn` (Zenodo 10781330). This is the primary “strategy” — choosing the model matched to input type. (Our cross-eval showed non-exclusive specialization; hand-drawn generalized to printed but was slightly below printed in-domain on printed eval-1000.)
- **Confidence scores:** optional per-token confidences (`confidence=True` / internal return of token confidences) — can be used for rejection/quality gating, not for changing the prediction itself.
- **Fixed preprocessing pipeline (no user-tunable knobs):** `remove_transparent → increase_contrast → get_bnw_image (BGR2GRAY quirk + contrast 1.8) → get_resize → delete_empty_borders → central_square_image → increase_brightness (1.6) → PNG encode/decode → tf.image.resize(512,512, method="gaussian", antialias=True) → efficientnet.tfkeras.preprocess_input`. Tokenizer/vocab identical mapping across printed/hand-drawn in this setup.
- **No built-in TTA/ensemble/denoising options** exposed in the API.

## Runtime vs DECIMER preprocessing differences (baseline)
- Our runtime (`src/runtime/preprocess.py`) mirrors the logical steps but differs in implementation details: final resize uses **PIL LANCZOS to 512×512** (approximation) vs DECIMER **TF gaussian+antialias** after PNG roundtrip; normalization equivalent to `efficientnet` ImageNet mean/std in this env. Parity was previously validated (TF-fidelity measured in benchmarks).

## Experiment setup
- **Protocol:** same frozen manifests/IDs as main evaluation (paired comparison within split). No TF parity baseline unless noted. No prediction cache reuse for variant isolation (each variant = different preprocessing → different predictions).
- **Splits:** screening on `pilot` (n=300) per dossier; promising variants can be confirmed on `eval` (n=1000).
- **Script:** `scripts/benchmark_preprocess_experiment.py`
- **Outputs:** `data/benchmark/experiments/preprocess/<split>/...json` (gitignored)

## Preprocessing variants tested

### OCSR (`ocsr-printed` FP32, `ocsr-handdrawn` FP32)
| Variant | Description |
|---|---|
| `runtime_default` | Current runtime pipeline (PIL LANCZOS final resize) |
| `decimer_tf_ref` | DECIMER 2.8.1 reference (`DECIMER.pre_process.decode_image`: TF gaussian resize + efn preprocess) |
| `no_binarize` | Skip B&W conversion/contrast-enhance step (keep grayscale) |
| `no_contrast_boost` | Skip initial contrast stretching |
| `no_brightness_boost` | Skip final brightness boost |
| `autocontrast` | PIL autocontrast (cutoff 1) before B&W |
| `invert` | Invert grayscale after B&W (dark bg / light strokes cases) |
| `sharpen_unsharp` | Unsharp mask (radius=2, percent=150) after B&W |
| `denoise_median3` | Median filter (3x3) after B&W |
| `stronger_contrast` | Extra contrast enhance (2.2) after B&W |
| `no_border_crop` | Skip empty-border cropping |

### Classifier (`image-classifier` E4 FP16)
| Variant | Description |
|---|---|
| `default` | Current `_classifier_preprocess_np` (RGBA->white, grayscale->RGB, aspect resize+pad white, 224) |
| `autocontrast` | Autocontrast before resize/pad |
| `sharpen` | Unsharp mask after resize/pad |
| `invert` | Invert after resize/pad |
| `higher_contrast` | Contrast enhance 1.5 |
| `equalize` | Histogram equalize |

## Results

### Screening (pilot, n=300) — complete
Pilot is a **filter, not a decision** (SE ≈ 2.7 pp at n=300). Promote to `eval-1000` only variants with **≥3 pp** improvement signal or fidelity checks (`decimer_tf_ref`). Treat as exploratory (multiple comparisons: 12 variants × 3 models; consider Bonferroni-style caution or confirm on eval-1000). Results under `data/benchmark/experiments/preprocess/pilot300/`.

| Model | Variant (split) | n | Primary metric | Δ vs runtime/default | valid | incomplete | p50 ms | Notes |
|---|---|---:|---:|---:|---:|---:|---:|---|
| `ocsr-printed` FP32 | `runtime_default` (pilot) | 300 | 187/263 (0.7110) canonical | (ref) | 0.967 | 0.033 | 1870 | baseline runtime preprocess |
| `ocsr-printed` FP32 | `decimer_tf_ref` (pilot) | 300 | 193/267 (0.7228) canonical | **+1.18 pp** | 0.977 | 0.023 | 1966 | DECIMER 2.8.1 TF gaussian+efn path; more comparables/valid (fidelity signal) |
| `ocsr-printed` FP32 | `autocontrast` (pilot) | 300 | 187/264 (0.7083) canonical | -0.27 pp | 0.963 | 0.037 | 1946 | no clear gain |
| `ocsr-printed` FP32 | `sharpen_unsharp` (pilot) | 300 | 189/263 (0.7186) canonical | +0.76 pp | 0.973 | 0.027 | 1942 | small positive signal |
| `ocsr-printed` FP32 | `no_border_crop` (pilot) | 300 | 182/265 (0.6868) canonical | **-2.42 pp** | 0.953 | 0.047 | 3932 | worse accuracy and much slower |
| `ocsr-handdrawn` FP32 | `runtime_default` (pilot) | 300 | 210/292 (0.7192) canonical | (ref) | 0.997 | 0.003 | 3058 | baseline runtime preprocess (first run includes warm-up) |
| `ocsr-handdrawn` FP32 | `decimer_tf_ref` (pilot) | 300 | 207/292 (0.7089) canonical | **-1.03 pp** | 1.000 | 0.000 | 2856 | opposite direction vs printed (+1.18) — no consistent fidelity gap |
| `ocsr-handdrawn` FP32 | `autocontrast` (pilot) | 300 | 205/292 (0.7021) canonical | -1.71 pp | 0.997 | 0.003 | 2013 | negative signal |
| `ocsr-handdrawn` FP32 | `sharpen_unsharp` (pilot) | 300 | 203/291 (0.6976) canonical | -2.16 pp | 0.997 | 0.003 | 2059 | negative here (positive on printed) |
| `ocsr-handdrawn` FP32 | `no_border_crop` (pilot) | 300 | 211/289 (0.7301) canonical | **+1.09 pp** | 0.997 | 0.003 | 2050 | opposite of printed (-2.42) — domain-specific |
| `ocsr-handdrawn` FP32 | `invert` (pilot) | 300 | 0/289 (0.0000) canonical | **-71.9 pp** | 0.653 | 0.347 | 1071 | catastrophic (validity collapses; model outputs garbage on inverted input) |
| `ocsr-handdrawn` FP32 | `no_binarize` (pilot) | 300 | 206/293 (0.7031) canonical | -1.61 pp | 0.987 | 0.013 | 1023 | negative signal |
| `image-classifier` E4 FP16 | all 6 variants | 1000 | see classifier eval-1000 table below | — | — | — | — | run directly on eval-1000 (cheap) |

### Eval confirmations (n=1000)

#### Priority 1 — fidelity: `decimer_tf_ref` vs `runtime_default` (OCSR, eval-1000)
Baseline = official `runtime_default` runs (`printed.fp32.eval1000.json`, `handdrawn.fp32.eval1000.json`); both runs cover the same frozen eval-1000 IDs.

**Denominators (read carefully — three different views):**
1. **Comparable-basis rate** (official convention): matches / items where the run produced a valid parseable SMILES. Each run has its **own denominator** (printed 872 vs 884; hand-drawn 974 vs 974), so Δ on this basis mixes recognition and parseability.
2. **McNemar test:** exact paired test on the **intersection of items comparable in both runs** (printed n=867, hand-drawn n=969) — the only set where discordances are well-defined. The 18 vs 6 (printed) and 14 vs 16 (hand-drawn) counts refer to this common set, **not** to the 872/884 denominators in the table above.
3. **Production rate** (non-parseable = failure, denominator 1000): the end-to-end view a deployed system sees.

| Model | Variant | comparable-basis rate | Δ (comparable) | production rate (of 1000) | Δ (production) | valid | incomplete | McNemar (common set) | p |
|---|---|---:|---:|---:|---:|---:|---:|---|---:|
| `ocsr-printed` FP32 | `runtime_default` (baseline) | 614/872 (0.7041) | — | 614/1000 (0.614) | — | 969 | 31 | — | — |
| `ocsr-printed` FP32 | `decimer_tf_ref` | 627/884 (0.7093) | +0.51 pp | 627/1000 (**0.627**) | **+1.3 pp** | 977 | 23 | 18 vs 6 favor TF (n=867) | **0.023** |
| `ocsr-handdrawn` FP32 | `runtime_default` (baseline) | 711/974 (0.7300) | — | 711/1000 (0.711) | — | 999 | 1 | — | — |
| `ocsr-handdrawn` FP32 | `decimer_tf_ref` | 708/974 (0.7269) | −0.31 pp | 708/1000 (0.708) | −0.3 pp | 1000 | 0 | 14 vs 16 (n=969) | 0.856 |

**Verdict (fidelity question answered):**
- **The TF path's advantage is completeness, not recognition.** On printed, the comparable-basis gain is only +0.51 pp but the **production-rate gain is +1.3 pp** (0.614 → 0.627) — roughly double — precisely because the TF path parses more images (23 vs 31 incomplete). On hand-drawn the production effect is **0** (−0.3 pp, p=0.86).
- **No systematic runtime fidelity gap.** The pilot's printed +1.18 pp shrank to +0.51 pp comparable-basis (direction consistent, McNemar p=0.023 nominal on the common set — but small in magnitude, and not significant under a Bonferroni-style correction across this experiment's comparisons). Hand-drawn shows no difference (pilot −1.03 was noise).
- **Decision: keep PIL LANCZOS — correct as-is.** The gain is small, domain-specific (0 on hand-drawn), driven by parse completeness rather than recognition quality, and adopting it would add a TF dependency to the lightweight runtime. It alters no selection or ≥70%-floor conclusion. (If printed completeness matters in the product, a cheaper lever than TF is a retry/fallback on incomplete outputs.) Record as experiment finding only; `src/runtime/preprocess.py` stands as validated.

#### Classifier eval-1000
- Ran all 6 variants directly on eval-1000 (cheap) — see classifier table below.

#### OCSR variants
- No variant met the ≥3 pp promotion threshold on pilot; none promoted as accuracy variants. Only the fidelity check above ran on eval-1000.

### Classifier eval-1000 (E4 FP16, n=1000, all 6 variants)
Direct on eval-1000 (cheap). Baseline `default` matches selected reference (≈0.7679 binary accuracy, non-reaction). Binary over 754 non-reaction items (reactions excluded/reported separately per protocol), total rows 1000.

| Variant | Binary acc (non-reaction) | Δ vs default | TP | TN | FP | FN | p50 ms | Notes |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `default` (ref) | **0.7679** (579/754) | — | 319 | 260 | 1 | 174 | 201 | baseline (high precision, misses structures) |
| `autocontrast` | **0.7798** (588/754) | +1.19 pp | 328 | 260 | 1 | 165 | 262 | small gain; slower |
| `sharpen` | **0.7825** (590/754) | +1.46 pp | 329 | 261 | 0 | 164 | 208 | small gain |
| `invert` | **0.6194** (467/754) | **-14.85 pp** | 206 | 261 | 0 | 287 | 200 | harmful (many missed structures) |
| `higher_contrast` | **0.7905** (596/754) | **+2.26 pp** | 335 | 261 | 0 | 158 | 202 | best in this sweep (fewer FN, FP stays ~0) |
| `equalize` | **0.6260** (472/754) | **-14.19 pp** | 211 | 261 | 0 | 282 | 200 | harmful |

> **Matrix orientation (to avoid misreads):** the JSON field is `matrix_rows_expected_cols_predicted = [[TP, FN], [FP, TN]]` with `labels = ["structure", "no-structure"]` (positive class = **structure**), built by `_confusion()` in `src/toolchain/benchmark_run.py`. So for `default`, `[[319, 174], [1, 260]]` means **TP=319, FN=174, FP=1, TN=260** — the 174 are *missed structures* (FN), **not** false positives. The baseline already had FP=1; `higher_contrast` moves FP 1→0 and FN 174→158 (+16 TP, +1 TN = +17 correct → +2.25 pp, exactly the reported Δ). Cross-check: (319+260)/754 = 0.7679 and (335+261)/754 = 0.7905, both matching the reported accuracies.

## Interpretation plan
1. **Fidelity first:** compare `decimer_tf_ref` vs `runtime_default` (causal interpretation: TF gaussian+efn preprocess vs PIL LANCZOS). A consistent gap suggests runtime preprocessing fidelity gap.
2. **Domain/model-specific:** optimal preprocessing may differ per model/domain (printed vs hand-drawn vs classifier); do not assume transfer.
3. **Interactions:** one-ablation-at-a-time; promising single changes may need combination testing.
4. **Product relevance:** `invert` / `no_binarize` especially relevant for photos/scans or non-white backgrounds in chemistry-draw.

## Observations
- **`decimer_tf_ref` fidelity — RESOLVED on eval-1000:** comparable-basis printed +0.51 pp (McNemar p=0.023 on the common set, small), hand-drawn −0.31 pp (p=0.86, none). **On production rate (non-parseable = failure, /1000): printed +1.3 pp (0.614 → 0.627), hand-drawn 0 (−0.3 pp)** — i.e. the TF path's advantage is **completeness, not recognition** (23 vs 31 incomplete on printed). **No systematic runtime fidelity gap; keep PIL LANCZOS, no runtime change** (see denominators note above: table rate, McNemar, and production rate use different denominators). The pilot's +1.18 pp was an overestimate at n=300.
- **`no_border_crop`:** -2.42 pp and ~2.1× slower on printed pilot — harmful here.
- **`sharpen_unsharp`:** +0.76 pp on printed pilot (small, below promotion threshold alone).
- **`autocontrast`:** neutral/slightly negative on printed pilot.
- **Classifier (eval-1000, complete):** `higher_contrast` **+2.26 pp** (0.7905), `sharpen` +1.46 pp, `autocontrast` +1.19 pp vs default (0.7679). Gains mainly reduce false negatives (missed structures) while false positives remain near-zero in this set (e.g. higher_contrast: TP 335, FN 158, FP 0, TN 261 vs default TP 319, FN 174, FP 1, TN 260). `invert` and `equalize` are strongly harmful (~-14 pp). Promising but not yet promoted to official selection — validate on real/product inputs and confirm stability (eval-1000 is decisive here given low cost, but domain shift possible).
- **Hand-drawn OCSR (pilot, complete):** `runtime_default` ref 0.7192. **`decimer_tf_ref` is -1.03 pp (0.7089)** — opposite direction to printed (+1.18 pp), so the two pilot signals together are consistent with **noise (SE ~2.7 pp) rather than a systematic runtime fidelity gap**; any runtime preprocess change would need eval-1000 confirmation per model and must not be assumed to transfer. **Preprocessing winners differ by domain:** `no_border_crop` is harmful on printed (-2.42 pp) but positive on hand-drawn (+1.09 pp); `sharpen_unsharp` positive on printed, negative on hand-drawn. This confirms the per-domain reporting rule (no assumed transfer).
- **`invert` on hand-drawn is catastrophic:** 0/289 canonical, validity drops to 0.653 (34.7% incomplete) — the hand-drawn model cannot parse inverted input at all. Combined with the classifier result (-14.85 pp), **`invert` is harmful in every model measured**; reject it outright (relevant for the chemistry-draw product: never auto-invert scans/photos).
- **`no_binarize` on hand-drawn:** -1.61 pp (0.7031) — the binarization step helps slightly on this domain too (printed not yet run for this variant).
- **No variant met the ≥3 pp promotion threshold on either OCSR domain.** Nothing from this pilot moves to eval-1000 as an accuracy "variant"; the only eval-1000 OCSR run still warranted is the fidelity check `decimer_tf_ref` vs `runtime_default` (causal question, per plan).

## Caveats
- Exploratory; not included in official report. Latency includes preprocessing differences but measured on shared CPU environment.
- Preprocessing changes can help some image types and hurt others; improvements on frozen curated sets may not generalize to chemistry-draw real inputs.
- `decimer_tf_ref` requires TensorFlow/DECIMER in env (toolchain context), not the lightweight runtime path.
