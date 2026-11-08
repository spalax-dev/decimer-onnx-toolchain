#!/usr/bin/env bash
# Pipeline: developer/CI gate. Runs fast checks (lint, type where configured, unit tests).
# Heavy parity/export tests are run separately (M1+).
set -euo pipefail
uv run ruff check .
uv run pytest -q -m "not slow"
