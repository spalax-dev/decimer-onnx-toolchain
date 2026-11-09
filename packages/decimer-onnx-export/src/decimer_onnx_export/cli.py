# Pipeline: export CLI (all export-side stages). Commands grow over milestones:
# fetch (M1, implemented), probe, export, quantize, evaluate, select, validate, bundle.
from __future__ import annotations

import json
from pathlib import Path

import typer

from decimer_onnx_export.fetch import available_models, fetch_model

app = typer.Typer(help="DECIMER export/quantize toolchain.", no_args_is_help=True)


@app.command("fetch")
def fetch_cmd(
    model: str = typer.Option("ocsr-printed", help=f"Model key: {', '.join(available_models())}"),
    data_dir: Path = typer.Option(Path("data/weights"), help="Cache directory for weights"),
    force: bool = typer.Option(False, help="Re-download and re-extract even if cached"),
    keep_archives: bool = typer.Option(True, help="Keep downloaded zip archives"),
) -> None:
    """Stage 1: download pinned DECIMER weights from Zenodo and verify checksums."""
    fetch_model(model, data_dir=data_dir, force=force, keep_archives=keep_archives)


@app.command()
def models() -> None:
    """List available model keys."""
    for key in available_models():
        typer.echo(key)


@app.command()
def probe(
    model: str = typer.Option("ocsr-printed", help=f"Model key: {', '.join(available_models())}"),
    data_dir: Path = typer.Option(Path("data/weights"), help="Cache directory for weights"),
) -> None:
    """Stage 2 (M0): inspect fetched weights / decide extraction path."""
    from decimer_onnx_export.source import format_probe, probe_model

    typer.echo(format_probe(probe_model(model, data_dir=data_dir)))


@app.command("export")
def export_cmd(
    model: str = typer.Option("ocsr-printed", help=f"Model key: {', '.join(available_models())}"),
    data_dir: Path = typer.Option(Path("data/weights"), help="Cache directory for weights"),
    out_dir: Path = typer.Option(Path("data/artifacts/fp32"), help="Output directory"),
    opset: int = typer.Option(17, help="ONNX opset"),
) -> None:
    """Stage 3 (M0/M2): export FP32 encoder + decoder-step ONNX graphs."""
    from decimer_onnx_export.to_onnx import export_fp32_onnx

    model_dir = data_dir / model / "DECIMER_model"
    if not model_dir.exists():
        # try common alternative layout
        candidates = list((data_dir / model).rglob("saved_model.pb"))
        if not candidates:
            raise typer.BadParameter(f"no saved_model.pb under {data_dir / model}; run fetch first")
        model_dir = candidates[0].parent
    res = export_fp32_onnx(model_dir, out_dir / model, opset=opset)
    typer.echo(f"encoder: {res.encoder_path}")
    typer.echo(f"decoder: {res.decoder_path}")
    if res.errors:
        for e in res.errors:
            typer.echo(f"error: {e}", err=True)
        raise typer.Exit(1)
    typer.echo(f"meta: {res.metadata.get('meta_path') if res.metadata else ''}")

@app.command("quantize")
def quantize_cmd(
    model: str = typer.Option("ocsr-printed", help="Model key"),
    fp32_dir: Path = typer.Option(Path("data/artifacts/fp32"), help="FP32 artifacts root"),
    out_dir: Path = typer.Option(Path("data/artifacts/quant"), help="Quantized artifacts root"),
    variant: str = typer.Option("e1", help="Variant: e1, e2, e3, e4, all"),
    calibration_dir: Path = typer.Option(None, help="Optional calibration image directory"),
) -> None:
    """Stage 4 (M4): build quantization candidates."""
    from decimer_onnx_export.quantize import quantize_e1_dynamic, quantize_e2_static, quantize_e3_hybrid, quantize_e4_fp16

    fp = fp32_dir / model
    variants = [variant] if variant != "all" else ["e1", "e2", "e3", "e4"]
    for v in variants:
        if v == "e1":
            r = quantize_e1_dynamic(fp, out_dir / "e1" / model)
        elif v == "e2":
            r = quantize_e2_static(fp, out_dir / "e2" / model, calibration_dir)
        elif v == "e3":
            r = quantize_e3_hybrid(fp, out_dir / "e3" / model, calibration_dir)
        elif v == "e4":
            r = quantize_e4_fp16(fp, out_dir / "e4" / model)
        else:
            raise typer.BadParameter(f"unknown variant {v}")
        typer.echo(f"{r.variant}: enc={r.encoder_path} dec={r.decoder_path} err={r.errors}")


@app.command("fetch-benchmark")
def fetch_benchmark_cmd(
    suite: str = typer.Option(..., help="Benchmark suite: decimer-smoke, handdrawn-hdm"),
    data_dir: Path = typer.Option(Path("data/benchmark"), help="Benchmark data directory"),
    force: bool = typer.Option(False, help="Re-download/extract"),
) -> None:
    """Download pinned benchmark datasets with checksum verification."""
    from decimer_onnx_export.benchmark_fetch import fetch_benchmark

    m = fetch_benchmark(suite, data_dir=data_dir, force=force)
    typer.echo(f"labels: {m.get('labels_path') or m.get('labels_count')}")


@app.command("import-benchmark")
def import_benchmark_cmd(
    source: str = typer.Option(..., help="Source: hf-clef, hf-jpo, hf-uob, hf-uspto, classifier-13378718"),
    data_dir: Path = typer.Option(Path("data/benchmark"), help="Benchmark data directory"),
    root: Path = typer.Option(None, help="Local root (for classifier-13378718 extracted zip)"),
    split: str = typer.Option("classified", help="Classifier split: classified, for_model/test, ..."),
    model: str = typer.Option("ocsr-printed", help="Target model key for labels"),
    limit: int = typer.Option(None, help="Max rows (for HF import/testing)"),
    force: bool = typer.Option(False, help="Re-download images"),
) -> None:
    """Import external benchmark data (HF rows API or local 13378718) into local manifest format."""
    from decimer_onnx_export.benchmark_import import import_classifier_13378718, import_hf_ocsr

    if source.startswith("hf-"):
        config = source[3:].upper()
        m = import_hf_ocsr(config, data_dir=data_dir, model=model, force=force, limit=limit)
    elif source == "classifier-13378718":
        if root is None:
            raise typer.BadParameter("--root is required (extracted dataset_for_image_classifier.zip)")
        m = import_classifier_13378718(root, data_dir=data_dir, split=split, force=force)
    else:
        raise typer.BadParameter(f"unknown source {source}")
    typer.echo(f"imported: {m.get('labels_path')} ({m.get('labels_count')})")


@app.command("benchmark")
def benchmark_cmd(
    labels: Path = typer.Option(None, "--labels", help="labels.jsonl path"),
    frozen_manifest: Path = typer.Option(None, "--frozen-manifest", help="Frozen manifest JSON (dossier)"),
    split: str = typer.Option("eval", help="Frozen split: pilot, eval, frozen"),
    task: str = typer.Option("ocsr", help="Task: ocsr, classifier"),
    encoder: Path = typer.Option(None, help="ONNX encoder path (OCSR)"),
    decoder: Path = typer.Option(None, help="ONNX decoder path (OCSR)"),
    vocab: Path = typer.Option(Path("data/artifacts/tokenizer_SMILES.vocab.json"), help="Vocab JSON"),
    tf_model_dir: Path = typer.Option(None, help="Optional TF SavedModel dir for parity baseline"),
    model_key: str = typer.Option(None, help="Model key for prediction cache, e.g. ocsr-printed"),
    variant_key: str = typer.Option(None, help="Variant key for prediction cache, e.g. fp32, e1-dyn, e3"),
    cache_dir: Path = typer.Option(None, help="Prediction cache dir (default: data/cache/predictions)"),
    no_cache: bool = typer.Option(False, help="Disable prediction cache"),
    out: Path = typer.Option(None, help="Write JSON report here"),
    limit: int = typer.Option(None, help="Max rows (applied after split)"),
) -> None:
    """Stage 5: run benchmark/validation over a labeled suite or frozen split."""
    from decimer_onnx_export.benchmark_run import benchmark_classifier, benchmark_ocsr, make_onnx_ocsr_predict_fn, make_tf_ocsr_predict_fn
    from decimer_onnx_export.frozen import PredictionCache, load_frozen, resolve_frozen_rows

    rows = None
    labels_path = labels
    if frozen_manifest:
        frozen = load_frozen(frozen_manifest)
        rows, labels_path = resolve_frozen_rows(frozen, split=split)
        model_key = model_key or frozen.get("model")
        typer.echo(f"frozen: {frozen_manifest} split={split} n={len(rows)} model={model_key}")
    elif labels is None:
        raise typer.BadParameter("--labels or --frozen-manifest required")
    elif not labels.exists():
        raise typer.BadParameter(f"labels not found: {labels}")

    cache = None if no_cache else PredictionCache(cache_dir)
    if task == "ocsr":
        if encoder is None or decoder is None:
            raise typer.BadParameter("--encoder and --decoder required for ocsr")
        predict = make_onnx_ocsr_predict_fn(encoder, decoder, vocab)
        tf_pred = make_tf_ocsr_predict_fn(tf_model_dir, vocab) if tf_model_dir else None
        report = benchmark_ocsr(labels_path, rows=rows, predict_fn=predict, tf_predict_fn=tf_pred, limit=limit,
                                cache=cache, model_key=model_key, variant_key=variant_key)
    elif task == "classifier":
        raise typer.BadParameter("classifier benchmark requires --predict integration (export classifier ONNX first); use benchmark_run.benchmark_classifier with a predict_fn")
    else:
        raise typer.BadParameter(f"unknown task {task}")
    if frozen_manifest:
        report["frozen_manifest"] = str(frozen_manifest); report["split"] = split
    text = json.dumps(report, indent=2, ensure_ascii=False)
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n", encoding="utf-8")
        typer.echo(f"wrote {out}")
    typer.echo(f"count={report.get('count')} exact={report.get('exact_match')} exact_rate={report.get('exact_match_rate'):.3f} tf_match={report.get('tf_match')} cache_hits={report.get('cache_hits')}")
    if report.get("latency_ms") and report["latency_ms"].get("p50") is not None:
        typer.echo(f"latency_ms p50={report['latency_ms']['p50']:.1f} p95={report['latency_ms']['p95']:.1f} n_measured={report['latency_ms'].get('n_measured')}")


@app.command("validate")
def validate_cmd(
    labels: Path = typer.Option(..., "--labels", exists=True),
    encoder: Path = typer.Option(..., exists=True),
    decoder: Path = typer.Option(..., exists=True),
    vocab: Path = typer.Option(Path("data/artifacts/tokenizer_SMILES.vocab.json"), exists=True),
    tf_model_dir: Path = typer.Option(None, help="TF baseline for parity"),
    out: Path = typer.Option(None),
    limit: int = typer.Option(None),
) -> None:
    """Stage 5 alias: OCSR parity/accuracy validation."""
    benchmark_cmd(labels=labels, task="ocsr", encoder=encoder, decoder=decoder, vocab=vocab,
                  tf_model_dir=tf_model_dir, out=out, limit=limit)


@app.command("report")
def report_cmd(
    out: Path = typer.Option(None, help="Output path (default: docs/benchmark-report.md)"),
) -> None:
    """Generate benchmark report (glossary, acceptance criteria, machine info, results) from data/."""
    from decimer_onnx_export.report import generate_report

    targets = [out] if out else None
    generate_report(targets=targets)
    for p in ([out] if out else [Path("docs/benchmark-report.md")]):
        typer.echo(f"wrote {p}")


if __name__ == "__main__":
    app()
