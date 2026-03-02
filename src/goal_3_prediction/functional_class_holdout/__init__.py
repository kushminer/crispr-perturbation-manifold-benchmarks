"""
Functional Class Holdout (LOGO) module.

This module implements LOGO evaluation that isolates specific functional classes
(e.g., Transcription genes) as the test set and trains on all other classes.
"""

from .logo import run_logo_evaluation, LogoResult
from .compare_baselines import compare_baselines

__all__ = ["run_logo_evaluation", "LogoResult", "compare_baselines"]

