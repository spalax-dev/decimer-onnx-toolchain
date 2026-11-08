"""Pipeline: runtime CLI (M3). Commands: predict, selftest."""
from __future__ import annotations

from pathlib import Path

import typer

app = typer.Typer(help="Run DECIMER ONNX inference (runtime, no TF).", no_args_is_help=True)


def _load(encoder: Path, decoder: Path, vocab: Path):
    from decimer_onnx.api import DecimerONNX

    return DecimerONNX(encoder, decoder, vocab)


@app.command()
def predict(
    image: Path = typer.Argument(..., help="Image path"),
    encoder: Path = typer.Option(..., "--encoder", exists=True),
    decoder: Path = typer.Option(..., "--decoder", exists=True),
    vocab: Path = typer.Option(..., "--vocab", exists=True),
) -> None:
    """Predict SMILES from an image."""
    model = _load(encoder, decoder, vocab)
    pred = model.predict(image)
    typer.echo(pred.smiles)


@app.command()
def selftest(
    image: Path = typer.Argument(..., help="Reference image"),
    expected: str = typer.Argument(..., help="Expected SMILES"),
    encoder: Path = typer.Option(..., "--encoder", exists=True),
    decoder: Path = typer.Option(..., "--decoder", exists=True),
    vocab: Path = typer.Option(..., "--vocab", exists=True),
) -> None:
    """Run an exact-match selftest."""
    model = _load(encoder, decoder, vocab)
    pred = model.predict(image)
    if pred.smiles != expected:
        typer.echo(f"FAIL: got {pred.smiles!r}, expected {expected!r}", err=True)
        raise typer.Exit(1)
    typer.echo(f"OK: {pred.smiles}")


if __name__ == "__main__":
    app()
