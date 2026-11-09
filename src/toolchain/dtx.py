"""Unified CLI (dtx): choose and run any pipeline stage (runtime + toolchain)."""
from __future__ import annotations

import typer

from toolchain import cli as toolchain_cli
from runtime import cli as runtime_cli

app = typer.Typer(
    help="Unified DECIMER→ONNX CLI: runtime inference + export/quantize/benchmark/report pipeline.",
    no_args_is_help=False,
    invoke_without_command=True,
)

# Runtime inference stages
app.add_typer(runtime_cli.app, name="runtime", help="ONNX runtime inference (no TF): predict, selftest.")

# Register toolchain/pipeline stage commands at top level
for cmd in toolchain_cli.app.registered_commands:
    app.registered_commands.append(cmd)
# Include toolchain sub-typers if any (none currently, but safe)
for group in toolchain_cli.app.registered_groups:
    app.registered_groups.append(group)


@app.callback(invoke_without_command=True)
def _main(ctx: typer.Context):
    """Show interactive menu when no subcommand is given."""
    if ctx.invoked_subcommand is None:
        from toolchain.pipeline import menu_loop

        menu_loop()


def main() -> None:
    app()


if __name__ == "__main__":
    main()
