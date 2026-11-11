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

### Classifier (E4 FP16, eval-1000, binary subset)
*(pending run)*

### ocsr-printed FP32 (eval-1000)
*(pending run)*

### ocsr-handdrawn FP32 (eval-1000)
*(pending run)*

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
