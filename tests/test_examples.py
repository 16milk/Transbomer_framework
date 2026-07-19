"""Smoke tests for the user-facing stage examples."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parents[1]


def test_examples_run_from_repository_root() -> None:
    examples = [
        "tensor_basics.py",
        "linear_regression.py",
        "binary_classification.py",
        "multiclass_classification.py",
        "sequence_classification.py",
        "rag_demo.py",
    ]
    for example in examples:
        completed = subprocess.run(
            [sys.executable, str(ROOT / "examples" / example)],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        assert completed.stdout.strip(), example
