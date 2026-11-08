# Pipeline: export CLI (all export-side stages). Commands grow over milestones:
# fetch (M1, implemented), probe, export, quantize, evaluate, select, validate, bundle.
from __future__ import annotations

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


@app.command()
def quantize() -> None:
    """Stage 4 (M4): build quantization candidates E1-E4 and select. Not implemented yet."""
    typer.echo("quantize: scaffolded (M4).")


@app.command()
def validate() -> None:
    """Stage 5 (M3/M4): run parity and golden-set validation. Not implemented yet."""
    typer.echo("validate: scaffolded (M3/M4).")




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

if __name__ == "__main__":
    app()

