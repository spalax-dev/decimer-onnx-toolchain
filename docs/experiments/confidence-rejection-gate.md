# Experiment: confidence-based rejection gate (3 selected models)

> **Scope:** exploratory experiment (like Imago / preprocessing). **Not part of the official report** (`docs/benchmark-report.md`).
> **Question:** does prediction confidence identify errors well enough that chemistry-draw can **reject** low-confidence outputs ("ask for another image") instead of returning a wrong SMILES?

## Why this (vs. more preprocessing variants)
- The DECIMER API exposes exactly two levers: **model routing** (`hand_drawn`, discarded for now — single-model decision pending) and **`confidence=True`** (per-token confidence). The preprocessing experiment showed no variant clears the promotion bar; confidence gating is orthogonal and potentially more valuable for the product.
- Selective prediction (abstention) trades coverage for accuracy: if confidence is discriminative, a gate lifts effective accuracy above the 70% floor on retained items.

## Confidence sources (no runtime change needed)
- **OCSR (printed/hand-drawn FP32):** the ONNX runtime already returns `Prediction.confidences` — per-token decoder probabilities from `greedy_decode` (`src/runtime/decode.py`, `conf = max prob at each step`). Summary stats per item: **mean, min, geometric mean** token confidence.
- **Classifier (E4 FP16):** sigmoid `score` from the single logit; ranking confidence = **|logit − logit(CLASSIFIER_THRESHOLD)|** (margin from the decision boundary, in logit space — the threshold is 0.000089, so raw-score margin would be dominated by one class).

## Protocol
- Frozen manifests, **`eval` split (n=1000)**, seed `20260308` — same IDs as the official benchmarks.
- **Production view:** non-parseable / non-comparable OCSR outputs count as **failures** (denominator 1000); comparable-basis shown for reference. Classifier over the binary subset (754 non-reaction items).
- **Risk–coverage curve:** sort items by confidence (descending), keep the top coverage ∈ {100, 95, 90, 85, 80, 75, 70, 60, 50}%, report accuracy on the kept set.
- **Discriminative power:** rank-based **AUC of confidence for predicting correctness** (Mann–Whitney; 0.5 = useless, higher = gate works). Reported for mean/min/gmean (OCSR) and margin (classifier).
- **Script:** `scripts/benchmark_confidence_gate.py`
- **Outputs:** `data/benchmark/experiments/confidence/<split>/<model>.<split>.json` (gitignored)

## Results

### Classifier (E4 FP16, eval-1000, binary subset n=754)
- Confidence = `|logit − logit(0.000089)|` margin. **AUC(conf→correct) = 0.603** — weak (target ≥0.7).
- Risk–coverage (production basis):

| coverage | kept | rejected | accuracy |
|---:|---:|---:|---:|
| 100% | 754 | 0 | 0.7679 |
| 90% | 679 | 75 | 0.7982 |
| 80% | 603 | 151 | 0.8325 |
| **75%** | 566 | 188 | **0.8410** (peak) |
| 70% | 528 | 226 | 0.8390 |
| 50% | 377 | 377 | 0.8090 (declining) |

- **Verdict: gate not usable for the classifier.** AUC 0.603 < 0.7 and the curve *declines* below ~75% coverage — the most "confident" items contain errors. Cause: the dominant error mode (146/174 FN = `several_molecules` predicted `no-structure`) is **systematic** — those predictions look identical to correct `no-structure` TN in score space (both have huge margin), so confidence cannot see them. The few confident-wrong FNs at the very top of the margin ranking drag accuracy down at low coverage.

### ocsr-printed FP32 (eval-1000, n=1000, production basis)
- Confidence = mean per-token decoder probability. **AUC = 0.938** — excellent.

| coverage | kept | rejected | production accuracy | min conf kept |
|---:|---:|---:|---:|---:|
| 100% | 1000 | 0 | 0.6140 | 0.657 |
| 95% | 950 | 50 | 0.6453 | 0.853 |
| 90% | 900 | 100 | 0.6789 | 0.891 |
| **85%** | 850 | 150 | **0.7165** | **0.928** |
| 80% | 800 | 200 | 0.7575 | 0.945 |
| 75% | 750 | 250 | 0.8013 | 0.961 |
| 70% | 700 | 300 | 0.8429 | 0.971 |
| 50% | 500 | 500 | 0.9380 | 0.986 |

### ocsr-handdrawn FP32 (eval-1000, n=1000, production basis)
- Confidence = mean per-token decoder probability. **AUC = 0.857** — good.

| coverage | kept | rejected | production accuracy | min conf kept |
|---:|---:|---:|---:|---:|
| 100% | 1000 | 0 | 0.7110 | 0.663 |
| 95% | 950 | 50 | 0.7421 | 0.840 |
| 90% | 900 | 100 | 0.7722 | 0.881 |
| **85%** | 850 | 150 | **0.7965** | **0.901** |
| 80% | 800 | 200 | 0.8200 | 0.915 |
| 75% | 750 | 250 | 0.8453 | 0.927 |
| 70% | 700 | 300 | 0.8600 | 0.936 |
| 50% | 500 | 500 | 0.9300 | 0.962 |

## Verdict
1. **OCSR gates are strong and product-ready in principle:** AUC 0.938 (printed) / 0.857 (hand-drawn). Rejecting the least-confident **15%** lifts production accuracy from **0.614 → 0.7165** (printed, above the 70% floor on retained items) and **0.711 → 0.7965** (hand-drawn). At 80% coverage: 0.758 / 0.820.
2. **Deployable rule (absolute thresholds, not dataset quantiles):** reject when mean token confidence < **0.928** (printed) or < **0.901** (hand-drawn) for the ~85% operating point; the full min-conf column gives the curve for other operating points.
3. **Classifier gate rejected: AUC 0.603; dominant error (`several_molecules` → `no-structure`, 146/174 FN) is systematically confident; needs data/model fix, not rejection.** Confidence cannot see errors the model does not perceive as doubtful — do not retry gating for the classifier without fixing the underlying multi-molecule data gap.
4. **Not a benchmark promotion:** the gate changes no official number; it is a product mechanism ("confident → return SMILES; unsure → ask for another image/crop"). Confidence is free at inference time (computed during greedy decoding anyway).

### The rejected set is real signal, not an artifact
Arithmetic at the 85% operating point (from the run JSONs):
- **printed:** 614 total hits; 850 kept with 609 hits (0.7165) → the **150 rejected items contain only 5 hits = 3.3% accuracy** in the confidence floor. Confidence isolates almost all failures.
- **hand-drawn:** 711 total hits; 850 kept with 677 hits (0.7965) → the **150 rejected contain 34 hits = 22.7% accuracy** — less dramatic, still well below the 0.711 base.

Product translation: the gate does not make the model *better*; it converts **"sometimes wrong" into "sometimes I don't know"** — which a user can act on. A 15% rejection with a clear message is far better than 15% silently wrong SMILES.

## Operating point is a product decision, not a result
The 85% point above is *an* operating point, not *the* answer. Choose it with the product team, trading rejection rate against retained accuracy (both models, production basis):

| Coverage | Rejection rate | printed acc (kept) | hand-drawn acc (kept) |
|---:|---:|---:|---:|
| 85% | 15% | 0.7165 | 0.7965 |
| 80% | 20% | 0.7575 | 0.8200 |
| 70% | 30% | 0.8429 | 0.8600 |

For chemistry-draw, 15% rejection with an actionable message is likely acceptable; 80% (20% rejection) gives more comfortable retained accuracy on printed. Document the chosen point as a product decision with its trade-off, not as a benchmark outcome.

## Rejected items need a UX path (design implication, not benchmark)
The gate is not "reject and done" — it is "reject and guide". Without an actionable message, rejection reads as product failure. Example copy: *"I couldn't read this image confidently — try a sharper photo, better lighting, or crop to a single molecule."* This is product design, but it is a hard prerequisite for using the gate.

## Thresholds are in-domain — re-calibrate on real inputs before shipping
The absolute thresholds (0.928 printed / 0.901 hand-drawn) are calibrated on the frozen in-domain `eval-1000` used for model selection. Real chemistry-draw photos/scans may have a shifted confidence distribution. **Do not hardcode them**: re-calibrate on real inputs before shipping, and treat the eval-based values as starting points only.

## Product implications (chemistry-draw v1)
```
image → classifier (no confidence gate — rejected above)
  ├── other              → "no structure detected" (0.9962, reliable)
  ├── one_molecule       → OCSR (selected ckpt) + confidence gate
  │                        ├── conf ≥ threshold → return SMILES
  │                        └── conf < threshold → "couldn't read confidently; try another image"
  ├── several_molecules  → "found several molecules; crop to one" (do NOT run OCSR — classifier
  │                        accuracy here is 0.416 and OCSR is trained on single structures)
  └── reactions          → "not supported yet"
```
The OCSR confidence gate is the only component of this experiment series with **measured, free, product-ready value**; it requires no model change and no extra inference cost.

## Interpretation plan
1. **Gate usability:** a usable gate needs AUC ≫ 0.5 (target ≥0.7) *and* a meaningful accuracy lift at modest rejection (e.g. reject 10–20% → +5 pp or more on production rate). If AUC ≈ 0.5–0.6, confidence is not usable for rejection regardless of the curve.
2. **Production basis primary:** the gate's job is to catch bad outputs, including unparseable ones — non-comparable counts as failure.
3. **Per-model, per-domain:** do not assume the gate transfers between printed/hand-drawn/classifier.
4. **Not a promotion:** even a working gate is a product mechanism (reject + re-ask), not a benchmark variant; it does not change official numbers.

## Caveats
- Confidence from greedy decoding is known to be overconfident (softmax saturation); the min-token statistic may behave differently from the mean.
- Risk–coverage on a frozen curated set may not transfer to chemistry-draw photos/scans.
- Latency unaffected (confidence is computed anyway during decoding); gate is free at inference time.

## DECIMER API note (confidence)
`predict_SMILES(..., confidence=True)` returns per-token confidences in the TF reference pipeline — the same signal this experiment measures on the ONNX runtime. No other quality levers exist in the API (no TTA, ensemble, or preprocessing knobs).
