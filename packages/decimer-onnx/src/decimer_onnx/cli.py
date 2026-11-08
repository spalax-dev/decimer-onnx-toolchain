# Pipeline: runtime CLI (M3). Commands: predict, bench, selftest, info.
# Default variant comes from dist/default or an explicit --variant/--model path.
import typer

app = typer.Typer(help="Run DECIMER ONNX inference (runtime).", no_args_is_help=True)


@app.command()
def info() -> None:
    """Pipeline: runtime (M3). Show installed runtime and selected artifact info."""
    typer.echo("decimer-onnx runtime scaffolded. Implemented in M3.")


if __name__ == "__main__":
    app()
