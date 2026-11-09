# Pipeline: export CLI (all export-side stages). Commands grow over milestones:
# fetch (M1, implemented), probe, export, quantize, evaluate, select, validate, bundle.
from __future__ import annotations

import json
from pathlib import Path

import typer

from toolchain.fetch import available_models, fetch_model

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
    from toolchain.source import format_probe, probe_model

    typer.echo(format_probe(probe_model(model, data_dir=data_dir)))


@app.command("export")
def export_cmd(
    model: str = typer.Option("ocsr-printed", help=f"Model key: {', '.join(available_models())}"),
    data_dir: Path = typer.Option(Path("data/weights"), help="Cache directory for weights"),
    out_dir: Path = typer.Option(Path("data/artifacts/fp32"), help="Output directory"),
    opset: int = typer.Option(17, help="ONNX opset"),
) -> None:
    """Stage 3 (M0/M2): export FP32 ONNX graphs."""
    from toolchain.to_onnx import export_classifier_fp32_onnx, export_fp32_onnx

    model_dir = data_dir / model / "DECIMER_model"
    if not model_dir.exists():
        candidates = list((data_dir / model).rglob("saved_model.pb"))
        if not candidates:
            raise typer.BadParameter(f"no saved_model.pb under {data_dir / model}; run fetch first")
        model_dir = candidates[0].parent
    if model == "image-classifier":
        meta = export_classifier_fp32_onnx(model_dir, out_dir / model, opset=opset)
        typer.echo(f"classifier: {meta.get('output_path')}")
        if meta.get("errors"):
            for e in meta["errors"]:
                typer.echo(f"error: {e}", err=True)
            raise typer.Exit(1)
        typer.echo(f"meta: {meta.get('meta_path')}")
        return
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
    variant: str = typer.Option("e1", help="Variant: e1, e1s, e2, e3, e4, all"),
    calibration_dir: Path = typer.Option(None, help="Optional calibration image directory"),
) -> None:
    """Stage 4 (M4): build quantization candidates."""
    from toolchain.quantize import quantize_e1_dynamic, quantize_e1s_selective, quantize_e2_static, quantize_e3_hybrid, quantize_e4_fp16

    fp = fp32_dir / model
    variants = [variant] if variant != "all" else ["e1", "e1s", "e2", "e3", "e4"]
    for v in variants:
        if v == "e1":
            r = quantize_e1_dynamic(fp, out_dir / "e1" / model)
        elif v in ("e1s", "e1-selective", "selective"):
            r = quantize_e1s_selective(fp, out_dir / "e1s" / model)
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
    from toolchain.benchmark_fetch import fetch_benchmark

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
    from toolchain.benchmark_import import import_classifier_13378718, import_hf_ocsr

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
    predictor: str = typer.Option(None, help="External OCSR predictor (bypasses encoder/decoder/TF): imago"),
    encoder: Path = typer.Option(None, help="ONNX encoder path (OCSR), or classifier ONNX model path"),
    decoder: Path = typer.Option(None, help="ONNX decoder path (OCSR)"),
    classifier_onnx: Path = typer.Option(None, "--classifier-onnx", help="Classifier ONNX model path"),
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
    from toolchain.benchmark_run import (
        benchmark_classifier, benchmark_ocsr,
        make_imago_predict_fn,
        make_onnx_classifier_predict_fn, make_onnx_ocsr_predict_fn,
        make_tf_classifier_predict_fn, make_tf_ocsr_predict_fn,
    )
    from toolchain.frozen import PredictionCache, load_frozen, resolve_frozen_rows

    rows = None
    labels_path = labels
    if frozen_manifest:
        from toolchain.benchmark_data import ensure_benchmark_data
        frozen = load_frozen(frozen_manifest)
        # auto-detect/decompress curated benchmark data if missing
        ensure_benchmark_data(frozen.get("model") or frozen.get("dossier") or "", registry=None)
        rows, labels_path = resolve_frozen_rows(frozen, split=split)
        model_key = model_key or frozen.get("model")
        typer.echo(f"frozen: {frozen_manifest} split={split} n={len(rows)} model={model_key}")
    elif labels is None:
        raise typer.BadParameter("--labels or --frozen-manifest required")
    elif not labels.exists():
        raise typer.BadParameter(f"labels not found: {labels}")

    cache = None if no_cache else PredictionCache(cache_dir)
    if task == "ocsr":
        if predictor:
            if predictor.lower() != "imago":
                raise typer.BadParameter(f"unknown external predictor: {predictor} (supported: imago)")
            model_key = model_key or "external-imago"
            variant_key = variant_key or "imago-2.0.0rc1"
            typer.echo(f"external predictor: imago (model_key={model_key} variant_key={variant_key})")
            predict = make_imago_predict_fn()
            report = benchmark_ocsr(labels_path, rows=rows, predict_fn=predict, tf_predict_fn=None, limit=limit,
                                    cache=cache, model_key=model_key, variant_key=variant_key)
        else:
            if encoder is None or decoder is None:
                raise typer.BadParameter("--encoder and --decoder required for ocsr (or use --predictor imago)")
            predict = make_onnx_ocsr_predict_fn(encoder, decoder, vocab)
            tf_pred = make_tf_ocsr_predict_fn(tf_model_dir, vocab) if tf_model_dir else None
            report = benchmark_ocsr(labels_path, rows=rows, predict_fn=predict, tf_predict_fn=tf_pred, limit=limit,
                                    cache=cache, model_key=model_key, variant_key=variant_key)
    elif task == "classifier":
        onnx_path = classifier_onnx or encoder
        if onnx_path is None:
            raise typer.BadParameter("--classifier-onnx (or --encoder) required for classifier")
        predict = make_onnx_classifier_predict_fn(onnx_path)
        tf_pred = make_tf_classifier_predict_fn(tf_model_dir) if tf_model_dir else None
        report = benchmark_classifier(labels_path, rows=rows, predict_fn=predict, limit=limit,
                                      cache=cache, model_key=model_key, variant_key=variant_key)
        # add tf parity summary if we have tf predictions? classifier harness doesn't take tf_pred; compute simple parity
        if tf_model_dir:
            # run tf pass for parity via cache-aware wrapper
            tf_fn = make_tf_classifier_predict_fn(tf_model_dir)
            from toolchain.benchmark_run import _cached_call, _image_id, _resolve_image
            agree = tot = 0
            for row in (rows or []):
                img = _resolve_image(row, labels_path); iid = _image_id(row)
                tfp, _, _ = _cached_call(cache, model_key, "tf-baseline", iid, tf_fn, img, time_it=False)
                # find matching detail
                det = next((d for d in report["details"] if d.get("id")==iid), None)
                if det:
                    tot += 1; agree += int(det.get("predicted")== (tfp.get("label") if isinstance(tfp,dict) else tfp))
            report["tf_parity"] = {"agree": agree, "total": tot, "rate": (agree/tot if tot else None)}
    else:
        raise typer.BadParameter(f"unknown task {task}")
    if frozen_manifest:
        report["frozen_manifest"] = str(frozen_manifest); report["split"] = split
    text = json.dumps(report, indent=2, ensure_ascii=False)
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n", encoding="utf-8")
        typer.echo(f"wrote {out}")
    if task == "classifier":
        cm = report.get("binary_confusion_matrix", {})
        typer.echo(f"count={report.get('total_rows')} binary_acc={cm.get('accuracy')} confusion={cm.get('matrix_rows_expected_cols_predicted')} tf_parity={report.get('tf_parity')} cache_hits={report.get('cache_hits')} reactions={report.get('reactions_separate_metric',{}).get('count')}")
    else:
        can = report.get("canonical_match"); ccomp = report.get("canonical_comparable"); canr = report.get("canonical_match_rate")
        can_s = f"canonical_rdkit(primary)={can}/{ccomp} ({canr:.3f}) " if ccomp else "canonical_rdkit(primary)=n/a "
        tcm = report.get("tf_canonical_match"); tcc = report.get("tf_canonical_comparable"); tcr = report.get("tf_canonical_match_rate")
        tfc_s = f" tf_canonical={tcm}/{tcc} ({tcr:.3f})" if tcc else ""
        typer.echo(f"count={report.get('count')} {can_s}exact_string={report.get('exact_match')} ({report.get('exact_match_rate'):.3f}) tf_match_string={report.get('tf_match')}{tfc_s} cache_hits={report.get('cache_hits')}")
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
    from toolchain.report import generate_report

    targets = [out] if out else None
    generate_report(targets=targets)
    for p in ([out] if out else [Path("docs/benchmark-report.md")]):
        typer.echo(f"wrote {p}")


@app.command("benchmark-data")
def benchmark_data_cmd(
    action: str = typer.Argument("status", help="status | ensure | ensure-all"),
    benchmark_id: str = typer.Option(None, "--id", help="Benchmark id/dossier/model (for ensure)"),
) -> None:
    """Manage curated benchmark datasets (auto-detect compressed archives, decompress to fixed locations)."""
    from toolchain.benchmark_data import load_registry, get_entry, is_available, ensure_benchmark_data, ROOT

    reg = load_registry()
    if action == "status":
        for b in reg.get("benchmarks", []):
            ok = is_available(b)
            arch = ROOT / b["curated_archive"]
            typer.echo(f"{b['id']}: available={ok} archive={arch.exists()} -> {b.get('decompress_to')} n={b.get('frozen_n')} sha256={'set' if b.get('sha256') else 'empty'}")
            typer.echo(f"  source: {b['technical_sheet'].get('source_url')}")
        return
    if action == "ensure":
        if not benchmark_id:
            raise typer.BadParameter("--id required for ensure")
        e = ensure_benchmark_data(benchmark_id)
        if not e:
            raise typer.BadParameter(f"unknown benchmark id {benchmark_id}")
        typer.echo(f"ensured {e['id']} available={is_available(e)}")
        return
    if action == "ensure-all":
        for b in reg.get("benchmarks", []):
            ensure_benchmark_data(b["id"])
            typer.echo(f"{b['id']}: available={is_available(b)}")
        return
    raise typer.BadParameter(f"unknown action {action}")


@app.command("pipeline")
@app.command("menu")
def pipeline_cmd() -> None:
    """Interactive menu to choose and run pipeline stages."""
    from toolchain.pipeline import menu_loop

    menu_loop()


if __name__ == "__main__":
    app()
