# decimer-onnx-toolchain

Reproducible pipeline to convert [DECIMER](https://github.com/Kohulan/DECIMER-Image_Transformer) (optical chemical structure recognition) into **optimized ONNX** artifacts for fast, self-contained inference (no TensorFlow at runtime).

> Status: bootstrap. Model fetch covers **printed OCSR (DECIMER v2)**, **hand-drawn specialist OCSR**, and the **DECIMER Image Classifier**. The download/fetch stage is implemented; export/quantize/runtime stages are scaffolded.

## Pipeline

![DECIMER to ONNX pipeline](docs/pipeline.svg)

1. **Fetch** — download pinned model weights (Zenodo/GitHub raw, pinned checksums/commit) with verification.
2. **Probe/extract** — load the SavedModel or rebuild encoder/decoder from checkpoints.
3. **FP32 export** — export `encoder.onnx` + 1-step `decoder.onnx` (tf2onnx); greedy loop stays in Python.
4. **Quantize & select** — build candidates (E1 dynamic INT8, E2 static INT8, E3 hybrid, E4 FP16), measure token parity/latency/size, pick the best that passes gates.
5. **Validate** — preprocessing parity, token parity vs TensorFlow, golden-set chemistry checks.
6. **Bundle** — manifest, model card, NOTICE, `selection.json`, and a `default/` artifact.

See [`TODO.md`](TODO.md) for progress.

## Quick start (bootstrap)

Requires [uv](https://docs.astral.sh/uv/) and Python 3.11.

```bash
uv sync --all-packages
uv run decimer-export models                        # list available model keys
uv run decimer-export fetch --model ocsr-printed    # DECIMER v2 (printed)
uv run decimer-export fetch --model ocsr-handdrawn  # hand-drawn specialist
uv run decimer-export fetch --model image-classifier
```

Downloaded weights are cached under `data/weights/`  and verified against pinned checksums.

## Models & sources

| Fetch key | What it is | Source | License / notes |
|---|---|---|---|
| `ocsr-printed` | DECIMER v2 printed OCSR (inference, `models.zip`) | [Zenodo 8300489 — DECIMER V2 Models](https://zenodo.org/records/8300489) | CC-BY-4.0 (Rajan et al.) |
| `ocsr-handdrawn` | DECIMER v2 hand-drawn specialist OCSR | [Zenodo 10781330 — DECIMER V2 Hand drawn Model](https://zenodo.org/records/10781330) | CC-BY-4.0 |
| `image-classifier` | DECIMER Image Classifier (EfficientNetB0, structure vs non-structure) | [Iagea/DECIMER-Image-Classifier](https://github.com/Iagea/DECIMER-Image-Classifier) (pinned commit) | MIT (upstream code); trained weights fetched from upstream repo at a pinned commit |



## Layout

```
packages/decimer-onnx/          # runtime (no TensorFlow) — scaffolded
packages/decimer-onnx-export/   # export/quantize toolchain — fetch implemented
configs/                        # model & quantization configs
docs/                           # pipeline, decisions (plan.md is local/gitignored)
data/                           # weights, calibration, validation (gitignored)
tests/ scripts/
```

## License & attribution

Code: **Apache-2.0** (see [`LICENSE`](LICENSE)). DECIMER OCSR weights: **CC-BY-4.0** (Zenodo [8300489](https://zenodo.org/records/8300489) printed, [10781330](https://zenodo.org/records/10781330) hand-drawn, Rajan et al.). DECIMER Image Classifier: upstream [Iagea/DECIMER-Image-Classifier](https://github.com/Iagea/DECIMER-Image-Classifier) (MIT code; trained weights fetched from the upstream repo at a pinned commit). Redistributed/quantized weights must keep required attribution; see `NOTICE` (to be added with first bundle).
