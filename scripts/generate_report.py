#!/usr/bin/env python3
"""Convenience wrapper: generate docs/benchmark-report.md from data/ (same as `dtx report`)."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from toolchain.report import generate_report  # noqa: E402


if __name__ == "__main__":
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    targets = [out] if out else None
    generate_report(targets=targets)
    print(f"wrote {out or 'docs/benchmark-report.md'}")
