#!/usr/bin/env python3
"""
Master script to run all Manifold Law diagnostic tests.

This script orchestrates running all 5 epics of the Manifold Law Diagnostic Suite.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List

from goal_2_baselines.baseline_types import BaselineType

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
LOGGER = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Run all Manifold Law diagnostic tests"
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        required=True,
        choices=["adamson", "replogle_k562_essential", "replogle_rpe1_essential"],
        help="Dataset name",
    )
    parser.add_argument(
        "--baseline_types",
        type=str,
        nargs="+",
        default=["lpm_selftrained"],
        help="Baseline types to test",
    )
    parser.add_argument(
        "--output_base",
        type=Path,
        default=Path("results/manifold_law_diagnostics"),
        help="Base output directory",
    )
    parser.add_argument(
        "--skip_epics",
        type=int,
        nargs="+",
        help="Epic numbers to skip (1-5)",
    )
    
    args = parser.parse_args()
    
    LOGGER.info("=" * 70)
    LOGGER.info("Manifold Law Diagnostic Suite")
    LOGGER.info("=" * 70)
    LOGGER.info(f"Dataset: {args.dataset_name}")
    LOGGER.info(f"Baselines: {args.baseline_types}")
    LOGGER.info(f"Output: {args.output_base}")
    LOGGER.info("")
    
    skip_epics = set(args.skip_epics) if args.skip_epics else set()
    
    # Dataset paths (would need to be configured based on your setup)
    # This is a placeholder - actual paths would come from configuration
    LOGGER.warning("Note: This is a placeholder script. Individual epic modules")
    LOGGER.warning("should be run separately until full integration is complete.")
    
    for baseline_type_str in args.baseline_types:
        baseline_type = BaselineType(baseline_type_str)
        LOGGER.info(f"\nProcessing baseline: {baseline_type.value}")
        
        # Epic 1: Curvature Sweep
        if 1 not in skip_epics:
            LOGGER.info("  Epic 1: Curvature Sweep - Use curvature_sweep.py module")
        
        # Epic 2: Mechanism Ablation
        if 2 not in skip_epics:
            LOGGER.info("  Epic 2: Mechanism Ablation - Use mechanism_ablation.py module")
        
        # Epic 3: Noise Injection
        if 3 not in skip_epics:
            LOGGER.info("  Epic 3: Noise Injection - Use noise_injection.py module")
        
        # Epic 4: Direction Flip
        if 4 not in skip_epics:
            LOGGER.info("  Epic 4: Direction Flip - Use direction_flip_probe.py module")
        
        # Epic 5: Tangent Alignment
        if 5 not in skip_epics:
            LOGGER.info("  Epic 5: Tangent Alignment - Use tangent_alignment.py module")
    
    LOGGER.info("\n" + "=" * 70)
    LOGGER.info("Note: Run individual epic modules for full execution.")
    LOGGER.info("See docs/manifold_law_diagnostics.md for usage instructions.")
    LOGGER.info("=" * 70)


if __name__ == "__main__":
    main()

