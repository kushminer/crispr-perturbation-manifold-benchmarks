#!/usr/bin/env python3
"""Validate and execute the maintained tutorial notebook."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

NOTEBOOK_DIR = Path(__file__).parent
NOTEBOOK = NOTEBOOK_DIR / "tutorial_end_to_end_results.ipynb"
KERNEL_NAME = "codex-nih-research"
KERNEL_DISPLAY_NAME = "Python (codex-nih-research)"


def ensure_kernel() -> None:
    subprocess.run(
        [
            sys.executable,
            "-m",
            "ipykernel",
            "install",
            "--user",
            "--name",
            KERNEL_NAME,
            "--display-name",
            KERNEL_DISPLAY_NAME,
        ],
        check=True,
        capture_output=True,
        text=True,
    )


def validate_structure(path: Path) -> None:
    notebook = json.loads(path.read_text())
    if "cells" not in notebook or not notebook["cells"]:
        raise ValueError(f"Notebook has no cells: {path}")


def execute_notebook(path: Path) -> None:
    ensure_kernel()
    with tempfile.TemporaryDirectory() as tmpdir:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "jupyter",
                "nbconvert",
                "--to",
                "notebook",
                "--execute",
                "--ExecutePreprocessor.timeout=300",
                f"--ExecutePreprocessor.kernel_name={KERNEL_NAME}",
                "--output",
                path.name,
                "--output-dir",
                tmpdir,
                str(path),
            ],
            check=True,
            cwd=NOTEBOOK_DIR,
        )


def main() -> int:
    validate_structure(NOTEBOOK)
    execute_notebook(NOTEBOOK)
    print(f"PASS: {NOTEBOOK.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
