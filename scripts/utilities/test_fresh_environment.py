#!/usr/bin/env python3
"""Smoke checks for a fresh project environment."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def check_dependencies() -> bool:
    """Verify the core Python dependencies import cleanly."""
    print("Checking Python dependencies...")

    required = [
        ("numpy", "np"),
        ("pandas", "pd"),
        ("scipy", None),
        ("sklearn", None),
        ("anndata", "ad"),
        ("yaml", None),
        ("matplotlib", None),
        ("seaborn", None),
    ]
    optional = ["torch", "umap"]

    ok = True
    for module_name, alias in required:
        try:
            module = __import__(module_name)
            version = getattr(module, "__version__", "unknown")
            label = alias or module_name
            print(f"  PASS {label} ({version})")
        except Exception as exc:
            print(f"  FAIL {module_name}: {exc}")
            ok = False

    for module_name in optional:
        try:
            module = __import__(module_name)
            version = getattr(module, "__version__", "unknown")
            print(f"  WARN optional {module_name} available ({version})")
        except Exception as exc:
            print(f"  WARN optional {module_name} missing: {exc}")

    return ok


def check_framework_imports() -> bool:
    """Verify the maintained framework modules import."""
    print("\nChecking framework imports...")
    try:
        import embeddings  # noqa: F401
        from embeddings.registry import list_embeddings
        from functional_class.functional_class import run_class_holdout  # noqa: F401
        from shared.io import load_annotations, load_expression_dataset  # noqa: F401
        from shared.linear_model import fit_linear_model, solve_y_axb  # noqa: F401
        from shared.metrics import compute_metrics  # noqa: F401
        from shared.validation import validate_annotation_quality  # noqa: F401

        registered = list(list_embeddings())
        print(f"  PASS registered embeddings: {', '.join(registered)}")
        return True
    except Exception as exc:
        print(f"  FAIL framework import: {exc}")
        return False


def check_basic_math() -> bool:
    """Run a lightweight linear-model and metrics smoke test."""
    print("\nChecking core linear-model math...")
    try:
        import numpy as np
        from shared.linear_model import solve_y_axb
        from shared.metrics import compute_metrics

        rng = np.random.default_rng(1)
        y = rng.normal(size=(8, 6))
        a = rng.normal(size=(8, 3))
        b = rng.normal(size=(3, 6))

        result = solve_y_axb(y, a, b, ridge_penalty=0.1)
        if result["K"].shape != (3, 3):
            raise ValueError(f"unexpected K shape: {result['K'].shape}")

        metrics = compute_metrics(y[:, 0], y[:, 0] + 0.01)
        if "pearson_r" not in metrics or "l2" not in metrics:
            raise ValueError("metrics output missing expected keys")

        print("  PASS solve_y_axb and compute_metrics")
        return True
    except Exception as exc:
        print(f"  FAIL core math: {exc}")
        return False


def check_entrypoints() -> bool:
    """Verify the documented entrypoint files exist."""
    print("\nChecking documented entrypoints...")
    expected = [
        REPO_ROOT / "scripts" / "demo" / "run_end_to_end_results_demo.py",
        REPO_ROOT / "scripts" / "execution" / "run_single_cell_baselines.sh",
        REPO_ROOT / "scripts" / "execution" / "run_single_cell_lsft.sh",
        REPO_ROOT / "scripts" / "execution" / "run_single_cell_logo.sh",
        REPO_ROOT / "tutorials" / "tutorial_end_to_end_results.ipynb",
    ]

    ok = True
    for path in expected:
        if path.exists():
            print(f"  PASS {path.relative_to(REPO_ROOT)}")
        else:
            print(f"  FAIL missing {path.relative_to(REPO_ROOT)}")
            ok = False
    return ok


def main() -> int:
    print("=" * 60)
    print("Fresh Environment Smoke Check")
    print("=" * 60)

    checks = [
        ("Dependencies", check_dependencies()),
        ("Framework Imports", check_framework_imports()),
        ("Core Math", check_basic_math()),
        ("Entry Points", check_entrypoints()),
    ]

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    failed = False
    for name, passed in checks:
        status = "PASS" if passed else "FAIL"
        print(f"{status} {name}")
        failed = failed or not passed

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
