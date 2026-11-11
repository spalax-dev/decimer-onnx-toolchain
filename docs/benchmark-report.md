# DECIMER → ONNX benchmark report

> Document auto-generated from JSON reports/manifests in `data/`. It does not include decisions or judgments about results; it reports measurements, definitions, and acceptance criteria defined for evaluation.

**Generated (UTC):** `2026-11-11T01:37:19.855660+00:00`  
**Generator:** `toolchain.report.generate_report` / `dtx report` (alias: `decimer-export report`)  
**Relevant branch/commits:** `main` (workspace fetch/export/runtime/quantization/benchmark).

---

## 1. Glossary

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
| **McNemar** | Paired test over discordances (A correct/B wrong vs inverse) on the same set. | — | `[pending implementation in report/harness]`. |

## 2. Acceptance criteria (defined for evaluation)

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
| **Reproducibility** | Source/pin/manifest/checksums, frozen IDs (seed), hardware/provider/versions. |

## 3. Data sources and provenance (facts)

| Suite / use | Target model | Source | Identifier / pin | License / provenance (per manifest/source) | Local artifacts |
|---|---|---|---|---|---|
| Printed golden CLEF | `ocsr-printed` | `Keylab/OCSR-Benchmarks` (HF, rows API) | observed dataset commit `2fdcdc59`; config `CLEF`, split `test`, 992 | HF aggregate **mixed-see-sources**; verify original CLEF 2012 terms | `data/benchmark/hf-clef/labels.jsonl`; `data/manifests/benchmark-hf-clef.import.json` |
| Printed golden JPO | `ocsr-printed` | `Keylab/OCSR-Benchmarks` (HF, rows API) | config `JPO`, split `test`, 449 | HF aggregate **mixed-see-sources**; verify original JPO/ChemInfty/OSRA terms | `data/benchmark/hf-jpo/labels.jsonl`; `data/manifests/benchmark-hf-jpo.import.json` |
| Hand-drawn HDM | `ocsr-handdrawn` | Zenodo DECIMER Hand-drawn molecule images | record `7617107`, DOI 10.5281/zenodo.7617107, 5088 | CC-BY-4.0 | `data/benchmark/handdrawn-hdm/labels.jsonl`; `data/manifests/benchmark-handdrawn-hdm.fetch.json` |
| Classifier (prepared) | `image-classifier` | Zenodo 13378718 | record `13378718`, zip md5 `e7e47f2aa4af4b79874a19382cec06da` (Zenodo API) | Record indicates MIT; mixed image origins | `decimer-export import-benchmark --source classifier-13378718` (requires local extraction) |
| Classifier 6670746 | `image-classifier` | Zenodo 6670746 | single file 126.6 GB, md5 `9e847917a75ac77d7fbe1f50b2818a5b` | CC-BY-4.0 | Not downloaded. |

**Classifier mapping (13378718, binary):** `one_molecule`→`structure`, `several_molecules`→`structure`, `other`→`no-structure` (included in binary matrix); `reactions`→excluded from binary, separate metric `ambiguous_excluded`.

## 4. Machine characteristics (automatic collection)

Automatically collected at `2026-11-11T01:37:18.460427+00:00` using `platform`, `lscpu`, and `free`.

| Field | Value |
|---|---|
| Hostname | `fedora` |
| OS | `Linux 7.2.15-300.fc45.x86_64` |
| Kernel/OS version | `#1 SMP PREEMPT_DYNAMIC Sun Nov  1 16:11:16 UTC 2026` |
| Architecture | `x86_64` |
| Python | `3.11.15` |
| CPU model | `Intel(R) Core(TM) i5-8365U CPU @ 1.60GHz` |
| Sockets / cores per socket / threads per core | `1` / `4` / `2` |
| Logical CPUs | `8` |
| CPU max/min frequency (MHz) | `4100.0000` / `400.0000` |
| Caches L1d/L1i/L2/L3 | `128 KiB (4 instances)` / `128 KiB (4 instances)` / `1 MiB (4 instances)` / `6 MiB (1 instance)` |
| Relevant CPU flags | `fma sse4_2 avx avx2` |
| Total RAM | 15.42 GiB (16555548672 bytes) |
| Available RAM (at collection) | 3.14 GiB |

### Package versions (execution environment)

| Package | Version |
|---|---|
| `Pillow` | `13.0.0` |
| `decimer` | `2.8.1` |
| `numpy` | `1.26.4` |
| `onnx` | `1.16.2` |
| `onnxruntime` | `1.26.0` |
| `rdkit` | `2026.9.1` |
| `sympy` | `1.14.0` |
| `tensorflow` | `2.15.1` |
| `tensorflow-cpu` | `2.15.1` |
| `tf2onnx` | `1.16.1` |
| `typer` | `0.28.0` |

## 5. Results

This report covers **three distinct models** evaluated in independent dossiers: `ocsr-printed`, `ocsr-handdrawn`, and `image-classifier` (no ranking across models).

Protocol: variants compared on **same IDs** when applicable (paired, pilot ⊂ eval ⊂ frozen pool). Latency **p50/p95** (same hardware/provider), not mean only; dynamic INT8 may not be faster. **Δ vs FP32** = difference in pp over common IDs (same metric). `—` = not measured/not available.

> **Execution note (reality):** some runs (pilots/remainders) were executed with CPU contention, so their `p50/p95` may not be comparable (accuracy is unaffected); prefer solo measurements for latency comparison. McNemar/bootstrap CI are not automated in this report (rates/Δpp/`n` are reported). There were retries due to transient ONNX load failures and runtime/quantization limitations documented in results/selections. See glossary (§1) and `docs/benchmark-spec.md` (§7) for details.

> **Domain/specialization (under evaluation):** `ocsr-printed` and `ocsr-handdrawn` are distinct models (different weights/tokenizers/origins) but **not exclusive specialists** based on observed cross-evaluation evidence (e.g. 2×2 pilot FP32: handdrawn 0.712 on printed vs 0.711 printed in-domain; 0.719 on hand-drawn, while printed drops to 0.677 on hand-drawn). Additional tests (cross eval-1000 and 1:1 printed+hand-drawn mix) are in progress/pending consolidation to confirm differences.

### 5.E `ocsr-printed` — **split `eval` (gate, n=1000)** (frozen n=1400, eval=1000, pilot=300, seed=20260308, source=`data/manifests/frozen/printed.frozen.json`)

| Variant | n | **RDKit canonical (primary, comp.)** | exact string | tf_match string (+ tf_canonical) | Δcanonical pp | Δtf pp | Δexact pp | valid | incomplete | p50 ms | p95 ms | size MB | Report |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| E3 hybrid (static-enc QDQ + dyn-dec) | 1000 | **432/890 (0.4854)** ❌<70% | 8 (0.0080) | 389/1000 (0.3890) / can 429/873 (0.491) | -21.87 pp | -43.10 pp | -0.40 pp | 964 (0.9640) | 36 | 3060.41 | 10825.61 | 84.4 | `data/benchmark/eval-1000/printed.e3.eval1000.json` |
| FP32 (ref) | 1000 | **614/872 (0.7041)** ✅≥70% | 12 (0.0120) | 820/1000 (0.8200) / can 766/878 (0.872) | (ref) | (ref) | (ref) | 969 (0.9690) | 31 | 2316.82 | 5457.46 | 301.2 | `data/benchmark/eval-1000/printed.fp32.eval1000.json` |
| FP32enc + E1dec (dyn) | 1000 | **612/878 (0.6970)** ❌<70% | 12 (0.0120) | 800/1000 (0.8000) / can 757/883 (0.857) | -0.71 pp | -2.00 pp | +0.00 pp | 974 (0.9740) | 26 | 2596.01 | 7434.31 | 228.4 | `data/benchmark/eval-1000/printed.fp32enc-e1dec.eval1000.json` |
| FP32enc + e1sdec (no-attn, mixed dec) | 1000 | **611/873 (0.6999)** ❌<70% | 12 (0.0120) | 812/1000 (0.8120) / can 762/878 (0.868) | -0.42 pp | -0.80 pp | +0.00 pp | 971 (0.9710) | 29 | 2812.58 | 12724.75 | 264.2 | `data/benchmark/eval-1000/printed.fp32enc-e1sdec.eval1000.json` |
| E1 dynamic INT8 all | 1000 | **597/881 (0.6776)** ❌<70% | 12 (0.0120) | 757/1000 (0.7570) / can 721/881 (0.818) | -2.65 pp | -6.30 pp | +0.00 pp | 976 (0.9760) | 24 | 3815.98 | 7302.61 | 81.3 | `data/benchmark/eval-1000/printed.int8-dyn-all.eval1000.json` |

### 5.P `ocsr-printed` — **split `pilot` (signal/selection, n=300)** (frozen n=1400, eval=1000, pilot=300, seed=20260308, source=`data/manifests/frozen/printed.frozen.json`)

| Variant | n | **RDKit canonical (primary, comp.)** | exact string | tf_match string (+ tf_canonical) | Δcanonical pp | Δtf pp | Δexact pp | valid | incomplete | p50 ms | p95 ms | size MB | Report |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| E1enc (dyn) + FP32dec | 300 | **182/267 (0.6816)** ❌<70% | 1 (0.0033) | 238/300 (0.7933) / can 224/264 (0.848) | -2.94 pp | -3.33 pp | +0.00 pp | 294 (0.9800) | 6 | 7891.59 | 16833.27 | 154.1 | `data/benchmark/pilot/printed.e1enc-fp32dec.pilot300.json` |
| E1s selective dyn (decoder without attention) | 300 | **182/267 (0.6816)** ❌<70% | 1 (0.0033) | 231/300 (0.7700) / can 219/263 (0.833) | -2.94 pp | -5.67 pp | +0.00 pp | 293 (0.9767) | 7 | 3622.71 | 6244.48 | 117.1 | `data/benchmark/pilot/printed.e1s-sel-noattn.pilot300.json` |
| FP32 (ref) | 300 | **195/288 (0.6771)** ❌<70% | 105 (0.3500) | 266/300 (0.8867) / can 266/287 (0.927) | -3.39 pp | +6.00 pp (agg) | +34.67 pp (agg) | 299 (0.9967) | 1 | 980.80 | 1943.49 | 301.2 | `data/benchmark/pilot/printed.fp32.cross-on-handdrawn.pilot300.json` |
| FP32 (ref) | 300 | **187/263 (0.7110)** ✅≥70% | 1 (0.0033) | 248/300 (0.8267) / can 230/263 (0.875) | (ref) | (ref) | (ref) | 290 (0.9667) | 10 | — (cache, sin latencia fresca) | — | 301.2 | `data/benchmark/pilot/printed.fp32.pilot300.json` |
| FP32enc + E1dec + reduce_range | 300 | **186/264 (0.7045)** ✅≥70% | 1 (0.0033) | 243/300 (0.8100) / can 230/264 (0.871) | -0.65 pp | -1.67 pp | +0.00 pp | 291 (0.9700) | 9 | 5320.71 | 14961.17 | 228.4 | `data/benchmark/pilot/printed.fp32enc-e1dec-rr.pilot300.json` |
| FP32enc + E1dec (dyn) | 300 | **187/264 (0.7083)** ✅≥70% | 1 (0.0033) | 242/300 (0.8067) / can 230/264 (0.871) | -0.27 pp | -2.00 pp | +0.00 pp | 292 (0.9733) | 8 | 3858.88 | 11189.69 | 228.4 | `data/benchmark/pilot/printed.fp32enc-e1dec.pilot300.json` |
| FP32enc + e1sdec (no-attn) + reduce_range | 300 | **186/263 (0.7072)** ✅≥70% | 1 (0.0033) | 245/300 (0.8167) / can 229/263 (0.871) | -0.38 pp | -1.00 pp | +0.00 pp | 291 (0.9700) | 9 | 5510.78 | 16259.84 | 264.2 | `data/benchmark/pilot/printed.fp32enc-e1sdec-rr.pilot300.json` |
| FP32enc + e1sdec (no-attn, mixed dec) | 300 | **185/262 (0.7061)** ✅≥70% | 1 (0.0033) | 249/300 (0.8300) / can 232/262 (0.885) | -0.49 pp | +0.33 pp | +0.00 pp | 290 (0.9667) | 10 | 3058.73 | 7290.85 | 264.2 | `data/benchmark/pilot/printed.fp32enc-e1sdec.pilot300.json` |
| E1 dynamic INT8 all | 300 | **182/266 (0.6842)** ❌<70% | 1 (0.0033) | 230/300 (0.7667) / can 219/262 (0.836) | -2.68 pp | -6.00 pp | +0.00 pp | 292 (0.9733) | 8 | — (cache, sin latencia fresca) | — | 81.3 | `data/benchmark/pilot/printed.int8-dyn-all.pilot300.json` |
| selAenc (stem+0..5+50..56 FP32) + e1sdec | 300 | **186/265 (0.7019)** ✅≥70% | 1 (0.0033) | 237/300 (0.7900) / can 226/264 (0.856) | -0.91 pp | -3.67 pp | +0.00 pp | 294 (0.9800) | 6 | 6587.35 | 10923.83 | 175.9 | `data/benchmark/pilot/printed.selAenc-e1sdec.pilot300.json` |
| selBenc (blocks 40..56 FP32) + e1sdec | 300 | **182/267 (0.6816)** ❌<70% | 1 (0.0033) | 228/300 (0.7600) / can 217/264 (0.822) | -2.94 pp | -6.67 pp | +0.00 pp | 294 (0.9800) | 6 | 7187.43 | 17653.01 | 215.2 | `data/benchmark/pilot/printed.selBenc-e1sdec.pilot300.json` |
| selCenc (stem+0..15 FP32) + e1sdec | 300 | **185/263 (0.7034)** ✅≥70% | 1 (0.0033) | 239/300 (0.7967) / can 227/262 (0.866) | -0.76 pp | -3.00 pp | +0.00 pp | 292 (0.9733) | 8 | 3080.11 | 5490.57 | 123.0 | `data/benchmark/pilot/printed.selCenc-e1sdec.pilot300.json` |
| selDenc-e1sdec | 300 | **185/263 (0.7034)** ✅≥70% | 1 (0.0033) | 239/300 (0.7967) / can 227/262 (0.866) | -0.76 pp | -3.00 pp | +0.00 pp | 293 (0.9767) | 7 | 7030.32 | 32944.69 | 181.3 | `data/benchmark/pilot/printed.selDenc-e1sdec.pilot300.json` |
| selEenc-e1sdec | 300 | **185/262 (0.7061)** ✅≥70% | 1 (0.0033) | 241/300 (0.8033) / can 229/262 (0.874) | -0.49 pp | -2.33 pp | +0.00 pp | 291 (0.9700) | 9 | 4032.80 | 22111.39 | 201.3 | `data/benchmark/pilot/printed.selEenc-e1sdec.pilot300.json` |
| selFenc-e1sdec | 300 | **186/264 (0.7045)** ✅≥70% | 1 (0.0033) | 243/300 (0.8100) / can 230/263 (0.875) | -0.65 pp | -1.67 pp | +0.00 pp | 291 (0.9700) | 9 | 4142.59 | 13840.12 | 127.1 | `data/benchmark/pilot/printed.selFenc-e1sdec.pilot300.json` |
| selGenc-e1sdec | 300 | **184/264 (0.6970)** ❌<70% | 1 (0.0033) | 244/300 (0.8133) / can 230/263 (0.875) | -1.41 pp | -1.33 pp | +0.00 pp | 291 (0.9700) | 9 | 3557.65 | 15299.97 | 133.8 | `data/benchmark/pilot/printed.selGenc-e1sdec.pilot300.json` |
| selMidenc-e1sdec | 300 | **185/264 (0.7008)** ✅≥70% | 1 (0.0033) | 241/300 (0.8033) / can 228/263 (0.867) | -1.03 pp | -2.33 pp | +0.00 pp | 293 (0.9767) | 7 | 2324.51 | 5022.42 | 240.1 | `data/benchmark/pilot/printed.selMidenc-e1sdec.pilot300.json` |

### 5.E `ocsr-handdrawn` — **split `eval` (gate, n=1000)** (frozen n=1400, eval=1000, pilot=300, seed=20260308, source=`data/manifests/frozen/handdrawn.frozen.json`)

| Variant | n | **RDKit canonical (primary, comp.)** | exact string | tf_match string (+ tf_canonical) | Δcanonical pp | Δtf pp | Δexact pp | valid | incomplete | p50 ms | p95 ms | size MB | Report |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| E3 hybrid (static-enc QDQ + dyn-dec) | 1000 | **0/550 (0.0000)** ❌<70% | 0 (0.0000) | 0/1000 (0.0000) / can 0/536 (0.000) | -73.00 pp | -91.20 pp | -38.10 pp | 551 (0.5510) | 449 | 17705.23 | 27366.41 | 84.4 | `data/benchmark/eval-1000/handdrawn.e3.eval1000.json` |
| FP32 (ref) | 1000 | **711/974 (0.7300)** ✅≥70% | 381 (0.3810) | 912/1000 (0.9120) / can 900/969 (0.929) | (ref) | (ref) | (ref) | 999 (0.9990) | 1 | 1114.95 | 2155.76 | 301.2 | `data/benchmark/eval-1000/handdrawn.fp32.eval1000.json` |
| E1 dynamic INT8 all | 1000 | **692/968 (0.7149)** ✅≥70% | 368 (0.3680) | 858/1000 (0.8580) / can 859/958 (0.897) | -1.51 pp | -5.40 pp | -1.30 pp | 999 (0.9990) | 1 | 2859.57 | 3951.42 | 81.3 | `data/benchmark/eval-1000/handdrawn.int8-dyn-all.eval1000.json` |

### 5.P `ocsr-handdrawn` — **split `pilot` (signal/selection, n=300)** (frozen n=1400, eval=1000, pilot=300, seed=20260308, source=`data/manifests/frozen/handdrawn.frozen.json`)

| Variant | n | **RDKit canonical (primary, comp.)** | exact string | tf_match string (+ tf_canonical) | Δcanonical pp | Δtf pp | Δexact pp | valid | incomplete | p50 ms | p95 ms | size MB | Report |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| E1s selective dyn (decoder without attention) | 300 | **207/292 (0.7089)** ✅≥70% | 118 (0.3933) | 263/300 (0.8767) / can 260/289 (0.900) | -1.03 pp | -5.33 pp | -1.00 pp | 299 (0.9967) | 1 | 2860.85 | 3963.42 | 117.1 | `data/benchmark/pilot/handdrawn.e1s-sel-noattn.pilot300.json` |
| FP32 (ref) | 300 | **198/278 (0.7122)** ✅≥70% | 1 (0.0033) | 252/300 (0.8400) / can 244/279 (0.875) | -0.69 pp | -9.00 pp (agg) | -40.00 pp (agg) | 299 (0.9967) | 1 | 2046.58 | 5223.96 | 301.2 | `data/benchmark/pilot/handdrawn.fp32.cross-on-printed.pilot300.json` |
| FP32 (ref) | 300 | **210/292 (0.7192)** ✅≥70% | 121 (0.4033) | 279/300 (0.9300) / can 273/291 (0.938) | (ref) | (ref) | (ref) | 299 (0.9967) | 1 | — (cache, sin latencia fresca) | — | 301.2 | `data/benchmark/pilot/handdrawn.fp32.pilot300.json` |
| E1 dynamic INT8 all | 300 | **204/293 (0.6962)** ❌<70% | 116 (0.3867) | 266/300 (0.8867) / can 263/290 (0.907) | -2.29 pp | -4.33 pp | -1.67 pp | 299 (0.9967) | 1 | — (cache, sin latencia fresca) | — | 81.3 | `data/benchmark/pilot/handdrawn.int8-dyn-all.pilot300.json` |

### 5.1 `ocsr-printed` — golden CLEF (Keylab/OCSR-Benchmarks, n=992, historical evidence)

| Variant | n | **RDKit canonical (primary, comp.)** | exact string | tf_match string (+ tf_canonical) | Δcanonical pp | Δtf pp | Δexact pp | valid | incomplete | p50 ms | p95 ms | size MB (enc+dec) | Report |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| E1 dynamic INT8 (+TF) | 992 | — | 7 (0.0071) | 838/992 (0.8448) | — | — | — | 991 (0.9990) | 1 | 5871.20 | 10230.40 | 81.3 | `data/benchmark/hf-clef/report.e1.tf.json` |
| E3 static-enc + dyn-dec (tf_match por join a TF de E1) | 992 | — | 6 (0.0060) | 444/992 (0.4476) [join] | — | — | — | 990 (0.9980) | 2 | 2861.30 | 4919.40 | 84.4 | `data/benchmark/hf-clef/report.e3.tfjoined.json` |

### 5.2 `ocsr-printed` — golden JPO (Keylab/OCSR-Benchmarks, n=449, historical evidence)

| Variant | n | **RDKit canonical (primary, comp.)** | exact string | tf_match string (+ tf_canonical) | Δcanonical pp | Δtf pp | Δexact pp | valid | incomplete | p50 ms | p95 ms | size MB (enc+dec) | Report |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| E1 dynamic INT8 (+TF) | 449 | — | 9 (0.0200) | 264/449 (0.5880) | — | — | — | 412 (0.9176) | 37 | 6149.83 | 17156.27 | 81.3 | `data/benchmark/hf-jpo/report.e1.tf.json` |
| E3 static-enc + dyn-dec (tf_match por join a TF de E1) | 449 | — | 3 (0.0067) | 123/449 (0.2739) [join] | — | — | — | 393 (0.8753) | 56 | 2767.13 | 26985.81 | 84.4 | `data/benchmark/hf-jpo/report.e3.tfjoined.json` |

### 5.3 `ocsr-handdrawn` — HDM Zenodo 7617107 (ad hoc sample `--limit 500`, not a formal decision)

| Variant | n | **RDKit canonical (primary, comp.)** | exact string | tf_match string (+ tf_canonical) | Δcanonical pp | Δtf pp | Δexact pp | valid | incomplete | p50 ms | p95 ms | size MB (enc+dec) | Report |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| FP32 (+TF) | 500 | — | 198 (0.3960) | 471/500 (0.9420) | (ref) | (ref) | (ref) | 499 (0.9980) | 1 | 1988.25 | 4504.17 | 301.2 | `data/benchmark/handdrawn-hdm/report.fp32.tf.limit500.json` |
| E1 dynamic INT8 (+TF) | 500 | — | 195 (0.3900) | 448/500 (0.8960) | — | -4.60 pp | -0.60 pp | 499 (0.9980) | 1 | 6239.98 | 9709.75 | 81.3 | `data/benchmark/handdrawn-hdm/report.e1.tf.limit500.json` |
| E3 static-enc + dyn-dec (+TF, calib 32 HDM) | 500 | — | 0 (0.0000) | 0/500 (0.0000) | — | -94.20 pp | -39.60 pp | 297 (0.5940) | 203 | 25397.42 | 31685.82 | 84.4 | `data/benchmark/handdrawn-hdm/report.e3.tf.limit500.json` |

### 5.4 `ocsr-handdrawn` — full HDM (5088)

| Variant | n | Status | Expected report |
|---|---:|---|---|
| E1 dynamic INT8 (+TF) | 5088 | Pending / not available at generation time | `data/benchmark/handdrawn-hdm/report.e1.tf.full5088.json` |

### 5.5 `image-classifier` (Zenodo 13378718, congelado) — eval-1000

Frozen: n=4000, eval=1000, pilot=300, seed=20260308, classes {'one_molecule': 1000, 'other': 1000, 'reactions': 1000, 'several_molecules': 1000}. Binary over `one/several_molecule→structure`, `other→no-structure` (threshold `0.000089`); `reactions` excluded from binary and reported separately. Goal: **accuracy ≥70%** while improving compression. **All** results for this suite are recorded (including non-selected).

| Variant (split) | total rows | Binary acc (non-reaction) | Matrix `[[TN,FP],[FN,TP]]` | Δacc vs FP32 | TF parity (incl. reactions) | reactions (separate) | p50 ms | p95 ms | Report |
|---|---:|---:|---|---:|---:|---:|---:|---:|---|
| E1s selective dyn (no Conv) (`eval-1000`) | 1000 | 0.7666 (n=754) ✅≥70% | `[[318, 175], [1, 260]]` | +0.00 pp | 1000/1000 (1.0000) | 246 → struct 173 / no-struct 73 | 36.51 | 167.87 | `data/benchmark/eval-1000/classifier.e1s-sel.eval1000.json` |
| E2 static QDQ (holdout calib) (`eval-1000`) | 1000 | 0.5345 (n=754) ❌<70% | `[[142, 351], [0, 261]]` | -23.21 pp | 667/1000 (0.6670) | 246 → struct 19 / no-struct 227 | 63.35 | 232.49 | `data/benchmark/eval-1000/classifier.e2-static-holdout.eval1000.json` |
| E4 FP16 (selected, ≥50% compression, acc ≥70%) (`eval-1000`) | 1000 | 0.7679 (n=754) ✅≥70% | `[[319, 174], [1, 260]]` | +0.13 pp | 998/1000 (0.9980) | 246 → struct 174 / no-struct 72 | 49.20 | 206.48 | `data/benchmark/eval-1000/classifier.e4-fp16.eval1000.json` |
| FP32 (ref) (`eval-1000`) | 1000 | 0.7666 (n=754) ✅≥70% | `[[318, 175], [1, 260]]` | (ref) | 1000/1000 (1.0000) | 246 → struct 173 / no-struct 73 | 46.16 | 205.17 | `data/benchmark/eval-1000/classifier.fp32.eval1000.json` |
| E1 dynamic INT8 all (`eval-1000`) | 1000 | 0.4085 (n=754) ❌<70% | `[[47, 446], [0, 261]]` | -35.81 pp | 559/1000 (0.5590) | 246 → struct 4 / no-struct 242 | 161.42 | 307.14 | `data/benchmark/eval-1000/classifier.int8-dyn-all.eval1000.json` |

### 5.5 `image-classifier` (Zenodo 13378718, congelado) — pilot

| Variant (split) | total rows | Binary acc (non-reaction) | Matrix `[[TN,FP],[FN,TP]]` | Δacc vs FP32 | TF parity (incl. reactions) | reactions (separate) | p50 ms | p95 ms | Report |
|---|---:|---:|---|---:|---:|---:|---:|---:|---|
| FP32 (ref) (`pilot`) | 300 | 0.7361 (n=216) ✅≥70% | `[[90, 56], [1, 69]]` | (ref) | 300/300 (1.0000) | 84 → struct 55 / no-struct 29 | — (cache, sin latencia fresca) | — | `data/benchmark/pilot/classifier.fp32.pilot300.json` |
| E1 dynamic INT8 all (`pilot`) | 300 | 0.3796 (n=216) ❌<70% | `[[12, 134], [0, 70]]` | -35.65 pp | 166/300 (0.5533) | 84 → struct 0 / no-struct 84 | 321.64 | 572.67 | `data/benchmark/pilot/classifier.int8-dyn-all.pilot300.json` |

> **Decision (classifier, for now):** `E4 FP16` meets the goal (acc ≈ FP32, ≥70%; **-49% size**; high TF parity). INT8 on Conv (E1-all/E2 static holdout) drops below 70%; E1s preserves accuracy but does not compress.

| Suite | Status | Note |
|---|---|---|
| Zenodo 6670746 | OUT OF SCOPE | Do not download/use (126.6 GB). |
| Zenodo 13378718 | All runs recorded in this report | 4→binary mapped; `reactions` separate. Imago excluded (curiosity experiment, `docs/experiments/`). |

## 7. Reference files

- Fetch/import manifests (tracked): `data/manifests/benchmark-*.json`.
- Detailed reports (gitignored, generated locally): `data/benchmark/**/*.json`.
- Printed golden summary (tracked): `data/artifacts/selection/ocsr-printed.hf-golden.json`.
- Printed selection (tracked, may contain interpretation): `data/artifacts/selection/ocsr-printed.selection.json`.

## 6. Measurement limitations (methodological facts)

- `exact_match` (string) compares `predicted == expected` **without RDKit canonicalization**. `canonical_match` (RDKit) compares canonical SMILES when both parse (integrated in OCSR benchmark; may be absent in historical reports). Equivalent representation differences may count as non-exact in `exact_match`.
- `tf_match` measures **ONNX↔TF** fidelity according to the harness baseline/pipeline; it is not canonical chemical correctness against the label.
- `valid`/`incomplete` are heuristics without RDKit (approximation).
- Latencies come from the execution environment/harness (machine/provider/load/threads/versions). Report **p50/p95**; do not use mean only. `cache_hits` invalidate fresh latency. Dynamic INT8 can be slower than FP32 on CPU.
- Δ in pp computed over the same metric and IDs when paired data is available; if pairing is missing, an aggregate may be indicated.
- HF CLEF/JPO subsets: aggregate `mixed-see-sources`; verify original terms before redistribution.
- Static quantization (E3/QDQ) strongly depends on the calibration set (size/distribution/holdout); results do not generalize without a controlled protocol.
- Classifier: binary only over 3 mapped classes; `reactions` excluded and reported separately (`ambiguous_excluded`).

## 8. Pending items at generation time

- `ocsr-handdrawn` full HDM n=5088, E1+TF: `data/benchmark/handdrawn-hdm/report.e1.tf.full5088.json` (not found at generation time).
