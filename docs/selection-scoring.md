# Qualification matrix and per-variant scoring (artifact selection)

> **Synthesis/argumentation** document for the selection of the **three distinct models** in this project (`ocsr-printed`, `ocsr-handdrawn`, `image-classifier`), evaluated in independent dossiers. It complements (does not replace) raw evidence in `docs/benchmark-report.md` and records in `data/artifacts/selection/*.selection.json`. Weights reflect project priorities; per-criterion scores are derived from measured results with explicit rules.

## 1. Methodology

- **Final score:** **0–100** scale per variant, computed as weighted sum: `Σ (weight_i × score_i) / 100`, where each criterion is scored **0–100** and weights sum to **100%** within each dossier.
- **Evidence source:** frozen splits (seed `20260308`, same IDs), dossier primary metric, reports in `data/benchmark/{eval-1000,pilot,remainder}/*.json`. Latencies are p50 in ms (CPU/ORT); where there was contention, qualitative comparison/solo measurements are prioritized (accuracy is unaffected).
- **Floor rule (gate):** variants with primary metric **<70%** on frozen `eval` are not acceptable. In OCSR, **robustness** is also required (a borderline eval result is insufficient if it falls on `remainder`/held-out). A floor/robustness failure forces the “Quality/accuracy” score to 0–30 and limits eligibility, even if the weighted total is high on other axes.

### Normalization used (criterion → 0–100 score)
| Criterion | Scoring (0–100) |
|---|---|
| **Primary quality/accuracy** (gate, **determinant**) | ≥75%: 90–100; 70–74.9%: 70–89 (by margin and stability); <70% or not robust (borderline eval but <70% held-out): 0–40. 100 = best observed in the dossier while maintaining robustness. **This criterion is determinant**: failing floor/robustness disqualifies the variant for selection. |
| **Robustness / generalization** | 100: meets floor on eval and holds (no relevant drop on held-out when applicable); 50: borderline without robust evidence; 0–30: falls below floor or collapses. |
| **Fidelity (TF/parity)** | Approx. linear scale over dossier fidelity metric (`tf_match` for OCSR, TF parity for classifier): 100≈best/ref, 0≈severe failure. |
| **Latency** | 100 = fastest (p50) among viable options in the dossier; relative scale (best = 100, worst ≈ 0–30). Penalizes variants slower than reference. |
| **Size / compression** | 100 = smallest size (most useful compression); 0 = largest size; weighted by dossier objective. |
| **Runtime viability** | 100 = runs correctly on CPU/ORT; 0 = not viable (runtime error). |

> **Note:** scores are a **transparent rubric** for comparing measured trade-offs, not an “official” model metric. The dominant acceptance criterion is the ≥70% floor + robustness, reflected with a **determinant weight** on accuracy/quality.

---

## 2. Dossier `ocsr-printed` (printed)

**Objective:** meet robust ≥70% canonical accuracy; compress only if it does not degrade the floor.

| Criterion | Weight | Rationale for weight |
|---|---:|---|
| **Canonical quality/accuracy (gate, determinant)** | **60%** | Primary metric and mandatory floor; determinant eligibility factor. |
| Robustness (eval→held-out) | 15% | Avoids selecting borderline/non-generalizable variants. |
| TF fidelity (`tf_match`) | 10% | Preserve baseline behavior. |
| Latency (p50) | 5% | Relevant, but secondary to correctness. |
| Size (compression) | 5% | Desirable if it does not break the floor. |
| Runtime viability | 5% | Must run on CPU/ORT. |

| Variant | Size | Eval canonical | Held-out canonical | `tf_match` (eval) | p50 (ms) | Quality (60) | Robustness (15) | Fidelity (10) | Latency (5) | Size (5) | Viab. (5) | **Total** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **FP32 (selected)** | 301.2 MB | **0.7041** | — (stable ref.) | 0.820 | 2317 | 78 | 90 | 90 | 80 | 20 | 100 | **79.3** |
| FP32enc+e1sdec | 264.2 MB | 0.7000 (borderline) | **0.6870** ❌ | 0.812 | 2813 | 30 | 30 | 88 | 65 | 35 | 100 | **41.3** |
| selCenc+e1sdec | 123.0 MB | 0.703* (pilot) | **0.6944** ❌ | ~0.797* | 3080* | 28 | 30 | 85 | 60 | 75 | 100 | **41.6** |
| selMidenc+e1sdec | 240.1 MB | 0.701* (pilot) | **0.6880** ❌ | ~0.803* | 2325* | 28 | 30 | 86 | 78 | 45 | 100 | **41.1** |
| E1 dynamic INT8 all | 81.3 MB | **0.6776** ❌ | — | 0.757 | 3816 | 20 | 40 | 70 | 45 | 90 | 100 | **36.8** |
| E3 hybrid (static enc) | 84.4 MB | **0.4854** ❌ | — | 0.389 | 3060 | 0 | 0 | 20 | 55 | 88 | 100 | **14.2** |
| E4 FP16 (OCSR) | — | — | — | — | — | 0 | 0 | 0 | 0 | 60 | 0 | **3.0** |

\* pilot signal (n=300), used for screening; robustness decision is taken with `remainder`/eval.

**Rationale:** with **accuracy as the determinant criterion (60%)**, FP32 is the only variant that passes the **robust ≥70% canonical** gate. Compressed/mixed alternatives either fail the floor on eval (E1/E3) or are borderline and **do not hold on held-out** (dropping ~1–3 pp below 70%), so they are not eligible despite better size/compression. E4 FP16 is not viable on CPU ORT. → **Selection: FP32**.

---

## 3. Dossier `ocsr-handdrawn` (hand-drawn)

**Objective:** meet ≥70% canonical accuracy; among options that pass, prioritize quality/fidelity and latency for default use. (Model trained for hand-drawn according to upstream; cross-eval shows good generalization to printed but not strictly better than the printed model in-domain — non-exclusive specialization.)

| Criterion | Weight | Rationale for weight |
|---|---:|---|
| **Canonical quality/accuracy (gate, determinant)** | **60%** | Primary metric and mandatory floor; determinant eligibility factor. |
| TF fidelity (`tf_match`) | 15% | Important for preserving behavior. |
| Latency (p50) | 10% | Relevant for user experience. |
| Robustness | 5% | Variant stability. |
| Size (compression) | 5% | Desirable, secondary if there is a strong trade-off. |
| Runtime viability | 5% | Must run on CPU/ORT. |

| Variant | Size | Eval canonical | `tf_match` | p50 / p95 (ms) | Quality (60) | Fidelity (15) | Latency (10) | Robustness (5) | Size (5) | Viab. (5) | **Total** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **FP32 (selected)** | 301.2 MB | **0.7300** | **0.912** | **1115 / 2156** | 88 | 100 | 100 | 90 | 20 | 100 | **88.3** |
| E1 dynamic INT8 all | 81.3 MB | **0.7149** (−1.51 pp) | **0.858** (−5.4 pp) | **2860 / 3951** (~2.6×) | 78 | 82 | 35 | 85 | 90 | 100 | **76.4** |
| E3 hybrid (static enc) | 84.4 MB | **0.0000** ❌ | 0.000 | 17705 / 27366 | 0 | 0 | 0 | 0 | 88 | 100 | **9.4** |

**Rationale:** with **accuracy as the determinant criterion (60%)**, E1 **does meet** the floor (≥70%) but is below FP32 in canonical accuracy (−1.51 pp) and also degrades fidelity and latency (~2.6× slower; dynamic quantization does not guarantee CPU speedup). **FP32** offers higher quality, fidelity, and latency with margin above the floor, so it is the default choice for `chemistry_draw`; E1 remains a recorded valid alternative if size is prioritized over speed/fidelity. → **Selection: FP32**.

---

## 4. Dossier `image-classifier` (classifier)

**Objective:** meet ≥70% binary accuracy and **improve compression** while maintaining fidelity.

| Criterion | Weight | Rationale for weight |
|---|---:|---|
| **Quality (binary accuracy, gate, determinant)** | **55%** | Primary metric and mandatory floor; determinant eligibility factor. |
| Size / compression | 25% | Explicit objective (reduce size). |
| TF fidelity/parity | 10% | Preserve baseline behavior. |
| Latency | 5% | Relevant, without degradation. |
| Runtime viability | 5% | Must run correctly. |

| Variant | Size | Binary accuracy (eval) | TF parity | Latency | Quality (55) | Size (25) | Fidelity (10) | Latency (5) | Viab. (5) | **Total** |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|
| **E4 FP16 (selected)** | **7.8 MB** (−49%) | **0.7679** | **0.998** | ~≈ FP32 CPU | 90 | 100 | 99 | 85 | 100 | **93.7** |
| FP32 (reference) | 15.4 MB | 0.7666 | 1.000 (ref) | ref | 90 | 40 | 100 | 85 | 100 | **78.8** |
| E1s selective (no Conv) | ~15.4 MB | 0.7666 | high | — | 90 | 40 | 98 | 85 | 100 | **78.6** |
| E2 static QDQ (holdout) | 4.8 MB | **0.5345** ❌ | — | — | 10 | 100 | 40 | 90 | 100 | **44.0** |
| E1 dynamic INT8 all | 4.3 MB | **0.4085** ❌ | — | — | 0 | 100 | 20 | 90 | 100 | **36.5** |

**Rationale:** with **accuracy as the determinant criterion (55%)**, E4 FP16 is the **only** variant that meets ≥70% and also achieves substantial compression (−49%) with very high TF parity, without penalizing accuracy. E1-all/E2 fall well below the floor due to aggressive Conv quantization; E1s preserves accuracy but does not compress. → **Selection: E4 FP16**.

---

## 5. Selection summary

| Dossier | Selected | Score | Main reason |
|---|---|---:|---|
| `ocsr-printed` | **FP32** | **79.3** | Only variant robustly ≥70% canonical (accuracy determinant); compressed alternatives do not hold the floor on held-out. |
| `ocsr-handdrawn` | **FP32** | **88.3** | Best quality/fidelity/latency among those meeting ≥70%; E1 valid on accuracy but with trade-offs. |
| `image-classifier` | **E4 FP16** | **93.7** | Meets ≥70% (determinant) and maximizes compression with high fidelity (best objective balance). |

*Scores synthesize evidence measured under the described protocol; **accuracy has determinant weight** (gate ≥70% + robustness) and any change in weights/criteria would shift the ranking explicitly in the table.*
