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
def probe() -> None:
    """Stage 2 (M0): inspect fetched weights / decide extraction path. Not implemented yet."""
    typer.echo("probe: scaffolded (M0).")


@app.command("export")
def export_cmd() -> None:
    """Stage 3 (M2): export FP32 encoder/decoder ONNX graphs. Not implemented yet."""
    typer.echo("export: scaffolded (M2).")


@app.command()
def quantize() -> None:
    """Stage 4 (M4): build quantization candidates E1-E4 and select. Not implemented yet."""
    typer.echo("quantize: scaffolded (M4).")


@app.command()
def validate() -> None:
    """Stage 5 (M3/M4): run parity and golden-set validation. Not implemented yet."""
    typer.echo("validate: scaffolded (M3/M4).")


if __name__ == "__main__":
    app()
