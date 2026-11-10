# TODO — DECIMER → ONNX toolchain

## Today (bootstrap)
- [x] Save plan locally (`docs/plan.md`, gitignored)
- [x] Initialize uv workspace + package skeletons
- [x] Add dependencies and lock
- [x] Implement model download/fetch (first pipeline stage): printed OCSR, hand-drawn OCSR, image classifier (pinned checksums/commit)
- [x] Create placeholder modules for remaining pipeline stages (commented)
- [x] Add PlantUML pipeline diagram + SVG in README
- [x] Add concise README and TODO

## Models in scope
- [x] DECIMER v2 / printed OCSR (`ocsr-printed`, Zenodo 8300489)
- [x] Hand-drawn specialist OCSR (`ocsr-handdrawn`, Zenodo 10781330)
- [x] DECIMER Image Classifier (`image-classifier`, Iagea/DECIMER-Image-Classifier @ pinned commit)

## M0 — Feasibility spike (in progress)
- [x] Fetch weights (printed/hand-drawn cached; classifier available) and inspect SavedModel
- [x] Confirm `m.DECIMER.encoder` + `m.DECIMER.transformer` callable from SavedModel (Path A viable)
- [x] Freeze tokenizer to `vocab.json` (156 tokens, ids preserved)
- [x] Export FP32 `encoder.onnx` + `decoder_step.onnx` (tf2onnx, opset 17)
- [x] Component parity smoke: encoder maxabs ~7e-7, decoder probs maxabs ~1e-7
- [x] Greedy loop token parity vs TF full SavedModel on preprocessed sample (**exact match**)
- [x] Clean decoder ONNX interface to `(tokens, enc_out)` (pos-encoding inputs removed)
- [x] Token parity on 4 samples incl. real structures: daidzein selftest (`C1=CC(=CC=C1C2=COC3=CC(=CC=C3C2=O)O)O`) and caffeine PubChem (**exact SMILES parity**)
- [ ] INT16 QDQ smoke test (opset 21 vs `com.microsoft`)
- [ ] Record rough latency and decide go/no-go

## M4 — Quantization & selection (in progress)
- [x] E1 dynamic INT8 (no calibration): 55+31MB, **exact SMILES parity** on fixtures — safe default
- [x] E2 static INT8 QDQ: 58+29MB, **failed parity** (garbage/incomplete) on fixtures with 2-image calibration — rejected
- [x] E3 hybrid (static encoder QDQ + dynamic decoder): 58+31MB, exact parity on fixtures but **rejected** after broader/frozen evaluation (severe fidelity regression in printed OCSR)
- [x] E4 FP16 (keep_io_types): 105+55MB, **decoder runtime broadcast error on CPU** (attention) — rejected on CPU
- [x] Evaluation snapshots + per-dossier selections (`data/artifacts/selection/*.selection.json`) based on frozen benchmarks; HF artifacts consolidated in `chemistry-draw-ocsr/`

## M1 — Skeleton & reproducibility
- [ ] `scripts/check.sh`, lint/type/test config, CI skeleton
- [x] Freeze tokenizer to `vocab.json` and audit vocab/embedding/off-by-one
- [ ] `probe` command validates environment + cached weights (pending tooling)

## M2–M6
- [~] FP32 export hardening + numeric audit (core export done; extended audit pending)
- [~] Runtime (exact preprocessing, greedy decode, API/CLI, selftest) (core runtime/API used; hardening/selftest pending)
- [~] Quantization E1–E4 (+E5 rescue) and auto-selection (`selection.json`) (candidates evaluated, per-dossier selections recorded; auto-selection tooling pending)
- [~] Benchmarks vs TF baseline, optional KV-cache, packaging (benchmarks+TF parity done; HF artifacts consolidated; canonical bundle packaging pending)
- [~] Docs/model card, CI drift checks, v1.0.0 RC (spec/report + HF README done; CI/drift/RC pending)
