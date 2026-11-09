"""Interactive CLI to choose and run DECIMER→ONNX pipeline stages."""
from __future__ import annotations

import shlex
import subprocess
import sys
from pathlib import Path

import typer

app = typer.Typer(help="Interactive DECIMER→ONNX pipeline.", invoke_without_command=True, no_args_is_help=False)

STAGES = [
    ("fetch-weights", "Download/verify model weights (DECIMER, classifier)", "dtx fetch --model {model}"),
    ("probe", "Inspect fetched weights / extraction path", "dtx probe --model {model}"),
    ("export-fp32", "Export FP32 ONNX graphs", "dtx export --model {model}"),
    ("quantize", "Build quantization variants (E1/E2/E3/E4)", "dtx quantize --model {model} --variant {variant}"),
    ("benchmark-data", "Curated benchmarks: status/auto-decompress frozen n sets (from config)", "dtx benchmark-data {action}"),
    ("fetch-benchmark", "Download pinned benchmark dataset (raw)", "dtx fetch-benchmark --suite {suite}"),
    ("import-benchmark", "Import external benchmark (HF CLEF/JPO or classifier-13378718)", "dtx import-benchmark --source {source}"),
    ("benchmark", "Run benchmark/validation (auto-decompresses curated if needed)", "dtx benchmark {args}"),
    ("validate", "OCSR parity/accuracy validation alias", "dtx validate {args}"),
    ("report", "Generate docs/benchmark-report.md from data/", "dtx report"),
    ("predict", "Runtime ONNX prediction (no TF)", "dtx runtime predict {args}"),
    ("selftest", "Runtime exact-match selftest (no TF)", "dtx runtime selftest {args}"),
    ("shell", "Run a custom dtx command", "{cmd}"),
    ("exit", "Exit", ""),
]


def _prompt(label: str, default: str | None = None) -> str:
    if default is None:
        return typer.prompt(label).strip()
    return typer.prompt(label, default=default).strip()


def _run(cmd: str) -> int:
    typer.echo(f"\n$ {cmd}\n")
    try:
        return subprocess.call(cmd, shell=True)
    except KeyboardInterrupt:
        typer.echo("\n(interrupted)")
        return 130


def _stage_fetch_weights():
    model = _prompt("model (ocsr-printed, ocsr-handdrawn, image-classifier)", "ocsr-printed")
    return _run(f"uv run dtx fetch --model {shlex.quote(model)}")


def _stage_probe():
    model = _prompt("model", "ocsr-printed")
    return _run(f"uv run dtx probe --model {shlex.quote(model)}")


def _stage_export():
    model = _prompt("model", "ocsr-printed")
    return _run(f"uv run dtx export --model {shlex.quote(model)}")


def _stage_quantize():
    model = _prompt("model", "ocsr-printed")
    variant = _prompt("variant (e1, e2, e3, e4, all)", "e1")
    extra = _prompt("extra args (e.g. --calibration-dir ...), or empty", "")
    return _run(f"uv run dtx quantize --model {shlex.quote(model)} --variant {shlex.quote(variant)} {extra}".strip())


def _stage_fetch_benchmark():
    suite = _prompt("suite (decimer-smoke, handdrawn-hdm)", "handdrawn-hdm")
    return _run(f"uv run dtx fetch-benchmark --suite {shlex.quote(suite)}")


def _stage_import_benchmark():
    source = _prompt("source (hf-clef, hf-jpo, classifier-13378718)", "hf-clef")
    extra = ""
    if source == "classifier-13378718":
        root = _prompt("root (extracted dataset path)", "data/benchmark/classifier-13378718/extracted/dataset_for_image_classifier")
        split = _prompt("split", "classified")
        extra = f"--root {shlex.quote(root)} --split {shlex.quote(split)}"
    else:
        limit = _prompt("limit (optional, empty for all)", "")
        if limit:
            extra = f"--limit {shlex.quote(limit)}"
    return _run(f"uv run dtx import-benchmark --source {shlex.quote(source)} {extra}".strip())


def _stage_benchmark():
    typer.echo("Examples: --task classifier --frozen-manifest data/manifests/frozen/classifier.frozen.json --split pilot --model-key image-classifier --variant-key fp32 --classifier-onnx <path> --tf-model-dir <dir> --out <json>")
    typer.echo("          --frozen-manifest data/manifests/frozen/printed.frozen.json --split pilot --model-key ocsr-printed --variant-key fp32 --encoder <enc> --decoder <dec> --vocab <vocab> --tf-model-dir <dir> --out <json>")
    args = _prompt("benchmark args (paste flags after 'benchmark')", "")
    return _run(f"uv run dtx benchmark {args}".strip())


def _stage_validate():
    args = _prompt("validate args (--labels --encoder --decoder --vocab [--tf-model-dir] [--out])", "")
    return _run(f"uv run dtx validate {args}".strip())


def _stage_benchmark_data():
    action = _prompt("action (status, ensure, ensure-all)", "status")
    extra = ""
    if action == "ensure":
        bid = _prompt("benchmark id (ocsr-printed, ocsr-handdrawn, image-classifier)", "image-classifier")
        extra = f"--id {shlex.quote(bid)}"
    return _run(f"uv run dtx benchmark-data {shlex.quote(action)} {extra}".strip())


def _stage_report():
    return _run("uv run dtx report")


def _stage_predict():
    args = _prompt("predict args (IMAGE --encoder --decoder --vocab)", "")
    return _run(f"uv run dtx runtime predict {args}".strip())


def _stage_selftest():
    args = _prompt("selftest args (IMAGE EXPECTED_SMILES --encoder --decoder --vocab)", "")
    return _run(f"uv run dtx runtime selftest {args}".strip())


def _stage_shell():
    cmd = _prompt("command", "uv run dtx --help")
    return _run(cmd)


HANDLERS = {
    "fetch-weights": _stage_fetch_weights,
    "probe": _stage_probe,
    "export-fp32": _stage_export,
    "quantize": _stage_quantize,
    "benchmark-data": _stage_benchmark_data,
    "fetch-benchmark": _stage_fetch_benchmark,
    "import-benchmark": _stage_import_benchmark,
    "benchmark": _stage_benchmark,
    "validate": _stage_validate,
    "report": _stage_report,
    "predict": _stage_predict,
    "selftest": _stage_selftest,
    "shell": _stage_shell,
}


def menu_loop() -> None:
    typer.echo("\nDECIMER → ONNX pipeline (interactive)\n")
    while True:
        typer.echo("Stages:")
        for i, (key, desc, _) in enumerate(STAGES, 1):
            typer.echo(f"  {i:2d}. {key:<18} - {desc}")
        choice = _prompt("\nChoose stage number (or q)", "q")
        if choice.lower() in {"q", "quit", "exit"}:
            typer.echo("bye")
            return
        try:
            idx = int(choice) - 1
            key = STAGES[idx][0]
        except Exception:
            typer.echo("Invalid choice.")
            continue
        if key == "exit":
            typer.echo("bye")
            return
        handler = HANDLERS.get(key)
        if not handler:
            typer.echo("No handler.")
            continue
        rc = handler()
        typer.echo(f"\n[exit code {rc}]\n")


@app.callback(invoke_without_command=True)
def main(ctx: typer.Context):
    """Run the interactive pipeline menu."""
    if ctx.invoked_subcommand is None:
        menu_loop()


@app.command("menu")
def menu_cmd():
    """Show interactive stage menu."""
    menu_loop()


def main_entry() -> None:
    app()


if __name__ == "__main__":
    main_entry()
