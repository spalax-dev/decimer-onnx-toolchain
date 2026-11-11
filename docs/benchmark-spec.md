# Benchmark specification — DECIMER → ONNX (three models / three dossiers)

> Design/protocol document. This project works with **three distinct models** (different sources/weights, tokenizers, tasks, and domains), each with its **own independent evaluation and selection dossier**: `ocsr-printed`, `ocsr-handdrawn`, and `image-classifier`. **This is not a single benchmark** and there is no ranking across models; each dossier has its own suite, metrics, acceptance criteria, and ONNX variant decision. Only infrastructure is shared: harness, `labels.jsonl` format, export/benchmark/reporting.

**Domain note (observed evidence):** `ocsr-handdrawn` comes from the upstream model trained for hand-drawn structures (Zenodo 10781330). In 2×2 cross-evaluation (FP32) it generalized well to printed but is not strictly better in both domains: on printed `eval-1000` it scored **0.694** canonical (628/905) vs **0.704** for `ocsr-printed` in-domain (614/872), while on hand-drawn pilot it scored **0.719** vs printed dropping to **0.677** (and hand-drawn in-domain eval is **0.730**). Therefore, these are distinct models with **non-exclusive specialization** (hand-drawn generalizes to printed; printed is weaker on hand-drawn), and neither is uniformly superior across both domains in this evaluation.

**Status:** formalized (2026-11-08, per-model/per-purpose reframing).  
**Operational sources:** `docs/benchmark-report.md` (neutral auto-generated results), `data/manifests/*.json`, `data/artifacts/selection/*` (per-dossier/model decisions, separate from the report).

---

## 1. Principle: three qualification dossiers

Each model is validated against **its** suite, with **its** metric, **its** acceptance criterion, and **its** decision. Each data source belongs to **exactly one dossier**.

| Dossier | Model | Suite(s) | Question | Decision it closes |
|---|---|---|---|---|
| [Printed](benchmark-dossier-printed.md) | `ocsr-printed` | CLEF + JPO (n=1441) | Does the ONNX variant reproduce TF and predict correct SMILES on printed structures? | Variant (E1/E3/FP32) + calibration policy |
| [Hand-drawn](benchmark-dossier-handdrawn.md) | `ocsr-handdrawn` | HDM (n=5088) | Does the ONNX variant reproduce TF and predict correct SMILES on hand-drawn structures? | Variant + calibration policy (model trained for hand-drawn; non-exclusive specialization per cross-eval) |
| [Classifier](benchmark-dossier-classifier.md) | `image-classifier` | Zenodo 13378718 (split defined at import) | Does it classify structure/no-structure correctly under the defined mapping? | ONNX export + wiring + variant |
| Colored-background stress (no dossier doc; report §5.C) | `ocsr-printed` + `ocsr-handdrawn` | αExtractor subset (n=200) | How robust are the selected OCSR models on deliberately hard images (colored backgrounds, low quality, hand-drawn)? **Robustness reporting only — no selection decision.** | None: **`floor_applies=false`** (≥70% floor exempt by design) |

**Not applicable:** a 2×2 cross-domain matrix as a requirement (there is no router sending the same image to both OCSR models). Cross-domain cells remain, if anything, an **optional robustness check**, not part of the dossier. There is **no** single comparative or pooled table across dossiers/domains.

## 2. Sources (each in exactly one dossier)

| Source ID | Dossier | n / scope | License / provenance | Status | Manifest / labels |
|---|---|---:|---|---|---|
| `hf-clef` | Printed | 992 | Keylab/OCSR-Benchmarks, observed commit `2fdcdc59`; HF **mixed-see-sources**; verify original CLEF 2012 terms | **IN** | `data/manifests/benchmark-hf-clef.import.json`, `data/benchmark/hf-clef/labels.jsonl` |
| `hf-jpo` | Printed | 449 | same aggregate; verify original JPO/ChemInfty/OSRA terms | **IN** | `data/manifests/benchmark-hf-jpo.import.json`, `data/benchmark/hf-jpo/labels.jsonl` |
| `handdrawn-hdm` | Hand-drawn | 5088 | Zenodo 7617107, CC-BY-4.0, verified checksums | **IN** | `data/manifests/benchmark-handdrawn-hdm.fetch.json`, `data/benchmark/handdrawn-hdm/labels.jsonl` |
| `classifier-13378718` | Classifier | per extracted split | Zenodo 13378718, zip md5 `e7e47f2aa4af4b79874a19382cec06da` (Zenodo API); record indicates MIT, mixed image origins (verify) | **IN** (requires local extraction + ONNX export) | importer `--source classifier-13378718` |
| `hf-colored` | Colored stress (both OCSR models) | 200 | `hheiden/Colored_Background_OCSR_benchmark` (HF, rows API), observed commit `cf73bddf`; **MIT** per HF dataset card; derived from αExtractor test data — verify original αExtractor terms | **IN** (stress/robustness, floor-exempt) | `data/manifests/benchmark-hf-colored.import.json`, `data/benchmark/hf-colored/labels.jsonl` |
| Zenodo **6670746** | — | 126.6 GB, single file | CC-BY-4.0; no exposed separate test set | **OUT OF SCOPE (formalized)** — not downloaded or used | — |
| `decimer-smoke` | Printed (optional sanity) | 4 | DECIMER-Image_Transformer `@9d06861`, MIT, SHA-256 | **IN optional**, outside main report | `data/manifests/benchmark-decimer-smoke.fetch.json` |

## 3. `n` rule (standardization)

- **`n` is standardized per dossier and per frozen ID set.** Never "normalize everything to 1441" across dossiers.
- **Within a dossier:** all compared variants see the **same IDs** (paired comparison). Freeze iteration subsets with **seed + IDs in manifest**; the full dossier set is the final result.
  - **Printed:** frozen pool n=1400 (CLEF+JPO, seed `20260308`) → `eval=1000`, `pilot=300`, `remainder=400` (`data/manifests/frozen/printed.frozen.json`).
  - **Hand-drawn:** frozen n=1400 (HDM, seed `20260308`) → `eval=1000`, `pilot=300`, `remainder=400` (`data/manifests/frozen/handdrawn.frozen.json`).
  - **Classifier:** frozen n=4000 stratified (seed `20260308`) → `eval=1000`, `pilot=300` (`data/manifests/frozen/classifier.frozen.json`). `reactions` excluded from binary and reported separately.
  - **Colored stress:** frozen n=200 (all items, seed `20260308`) → `eval=frozen=200`, `pilot=50` (`data/manifests/frozen/colored.frozen.json`, `floor_applies=false`). Smaller than the standard eval by design (whole source set); reported in §5.C without the ≥70% floor.

## 4. Shared contracts (infrastructure)

### 4.1 Paired measurement protocol (per variant)
Each variant is measured on the **same frozen set / same IDs** (pilot ⊂ eval ⊂ pool; never mix sets across variants). For each run record:
- **Task accuracy (primary):** metric defined by the dossier. **OCSR:** **RDKit canonical SMILES accuracy** (`canonical_match`, **primary**, integrated) + `exact_match` string (secondary); fidelity `tf_match`/`tf_canonical_match` ONNX↔TF. **Classifier:** binary accuracy (primary). **Absolute floor: ≥70%** on the primary metric (frozen eval; **stress dossiers exempt per `floor_applies=false`**). Report **absolute Δ in pp vs FP32** (same metric, same IDs).
- **Latency:** **p50 and p95** on the **same hardware** and **same execution provider** (e.g. ORT `CPUExecutionProvider`), recorded. Do not report only the mean; the tail (p95) matters.
- **Disk size:** MB of model artifacts (encoder+decoder or equivalent), per variant.
- **Throughput:** if batches are processed, report batches/s or images/s with batch size `[if applicable]`.
- **Variant comparison (paired):** **McNemar** over discordant pairs + Δ pp with **bootstrap CI**; report `n` + CI.
- **Cache:** predictions by `(model_key, variant_key, image_id)` to avoid recomputation (`data/cache/predictions/`, gitignored).

**Measurement note:** dynamic INT8 quantization **is not always faster** (runtime activation quantization overhead can offset/eliminate gains). Latency is measured, not assumed.

### 4.2 Labels / metrics by block
- **OCSR (printed/hand-drawn):** **RDKit `canonical_match` (primary, chemical quality)** on comparable pairs + `exact_match` string (secondary), `tf_match`/`tf_canonical_match` (ONNX↔TF fidelity), `valid`/`incomplete`, latency p50/p95, size, throughput if applicable.
- **Classifier:** **binary accuracy (primary)** over 3 mapped classes (`one/several_molecule→structure`, `other/rest→no-structure`), **confusion matrix** `[[TN,FP],[FN,TP]]`, with **`reactions` excluded from binary** and reported separately (`ambiguous_excluded`). ONNX↔baseline/TF parity fidelity.
- **Absolute floor (mandatory, all standard dossiers — printed, hand-drawn, classifier):** **accuracy ≥ 70%** on the dossier primary metric, on the **frozen `eval` split** (same IDs). OCSR = `canonical_match_rate` (RDKit, comparable pairs). Classifier = binary accuracy (non-reaction). Variants < 70% **are not accepted**. **Exemption:** stress dossiers with `floor_applies=false` in their frozen manifest (currently the colored-background set) are **reported without the floor** and do not gate selection — the exemption is explicit in the manifest, never implicit.
- **Additional criteria (resolved by evidence):** for compressed variants, the ≥70% floor must hold **robustly** (not just borderline on `eval`; validated against `remainder`/held-out when applicable). `Δ vs FP32` and p95 are recorded; selection prioritizes meeting the floor and fidelity/quality (and compression when it does not break the floor). Paired McNemar/bootstrap CI remain future tooling improvements (the report records Δpp and paired `n`).

### 4.3 Preprocessing (recorded per model)
Record input, resize/normalization/channels, TF baseline vs ONNX runtime path used in `tf_match`, version. **FP32+TF** control before concluding about quantization (separates export/preprocess from quantization effects).

## 5. Reporting

- Neutral report auto-generated by `dtx report` (`docs/benchmark-report.md`) from `data/benchmark/**/*.json` and manifests: includes **all results** per dossier/split (eval/pilot/remainder), including non-selected variants; Imago excluded (separate experiment in `docs/experiments/`).
- Paired comparison (same IDs) within dossier, Δpp vs FP32, primary/secondary metrics, p50/p95, sizes. Sections per dossier/model and historical sources when applicable.

## 6. Decisions (consolidated)

| Dossier | Selected variant | Evidence (primary metric, frozen eval) | Note |
|---|---|---|---|
| Printed `ocsr-printed` | **FP32** | 0.7041 canonical (614/872, n=1000) ≥70% | No tested compressed/mixed variant (E1/E3, enc/dec mixes, e1s, block-wise partitions) was robustly ≥70% on held-out. |
| Hand-drawn `ocsr-handdrawn` | **FP32** | 0.7300 canonical (576/789, n=1000) ≥70% | Meets floor with better quality/fidelity; E1 met accuracy on eval but with latency/fidelity tradeoff. |
| Classifier `image-classifier` | **E4 FP16** | 0.7679 binary accuracy (n=1000, non-reaction) ≥70% | TF parity 0.998, 7.78 MB (-49%). INT8 Conv (E1-all/E2) <70%; E1s no meaningful compression. |

**Weighted qualification matrix (1–100):** see [`docs/selection-scoring.md`](selection-scoring.md) for criteria, per-dossier weights, per-variant scores, and selection rationale.

Selected artifacts consolidated in `chemistry-draw-ocsr/` (ignored in this repo) for publication to HF/`chemistry_draw`.

## 7. Current status and protocol conformance (reality)

### Executed according to protocol (core)
- **Frozen sets and pairing:** frozen manifests (seed `20260308`) and `eval`/`pilot`/`remainder` splits per dossier (`data/manifests/frozen/*.frozen.json`); comparisons on **same IDs** within the same split (`--frozen-manifest ... --split ...`).
- **Metrics:** OCSR with **RDKit `canonical_match`** (primary, comparable pairs), `exact_match` (secondary), and `tf_match`/`tf_canonical_match` fidelity vs TF; classifier with **binary accuracy** (non-reaction), confusion matrix, TF parity, and `reactions` reported separately.
- **Selection and recording:** **≥70%** floor (primary metric, frozen eval) with robustness criterion (`remainder`/held-out validation when applicable). **All** results (including non-selected) in `data/benchmark/**/*.json` and in auto-generated `docs/benchmark-report.md` (`dtx report`); Imago excluded from report and documented in `docs/experiments/`.
- **Latency/cache:** `p50/p95`, `n_measured`, and `cache_hits` reported; cache by `(model_key, variant_key, image_id)`. FP32+TF references measured and Δpp vs FP32 recorded.

### Deviations / nuances from the ideal
- **Latency and CPU contention:** some runs (pilots/remainders) executed with other jobs in parallel, which inflated `p50/p95` and makes them **not comparable** as latency; **accuracy is unaffected**. Solo measurements (e.g. evals/remainders without contention) are the representative ones for latency comparison.
- **Paired formal statistics not automated:** **McNemar and bootstrap CI** over discordances are not computed automatically; rates, Δpp, and paired `n` are reported. (Pending tooling improvement.)
- **Historical evidence:** historical runs predating the frozen protocol coexist (e.g. CLEF/JPO 1441, ad hoc HDM samples) labeled as such; consolidated decisions are based on frozen splits.
- **Fine-grained criteria:** additional thresholds (Δpp/p95 as a single hard global numeric gate) were resolved by evidence/robustness and metric recording, not as a single pre-fixed global threshold beyond the ≥70% floor.

### Execution incidents (mitigated)
- **Transient ONNX load retries:** `NoSuchFile` failures when starting some runs (e.g. `sel*` variants) despite artifacts existing; retried and completed runs were recorded.
- **Viability/runtime:** OCSR FP16 (E4) not viable on CPU ORT (decoder attention broadcast error) → discarded. E3 (static QDQ encoder) caused severe degradation in OCSR → rejected.
- **Imago:** intermittent native crashes mitigated with a per-image subprocess runner (`scripts/benchmark_imago_resilient.py`); results only in `docs/experiments/` (outside the report).
- **Quantization warnings:** ONNX Runtime pre-processing warnings during dynamic quantization; did not prevent artifact generation/evaluation.

### Pending (tooling/engineering improvements)
- Automated McNemar/bootstrap CI; fine-grained p95 gates if fixed; canonical toolchain packaging (`selection.json` unified/`default`, NOTICE/bundle manifest); M1/M6 tasks (check/CI, probe, hardening).
