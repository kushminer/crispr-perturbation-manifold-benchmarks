"""
Lipschitz constant estimation utilities.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


def estimate_lipschitz_constant(
    noise_levels: np.ndarray,
    r_values: np.ndarray,
) -> float:
    """
    Estimate Lipschitz constant from noise sensitivity curve.
    
    L ≈ max |Δprediction| / σ
    
    Args:
        noise_levels: Array of noise levels (σ values)
        r_values: Array of Pearson r values at each noise level
        
    Returns:
        Estimated Lipschitz constant
    """
    if len(noise_levels) < 2:
        return np.nan
    
    # Compute sensitivity (drop in r per unit noise)
    r_drop = r_values[0] - r_values  # Drop from baseline (no noise)
    sensitivity = np.abs(np.diff(r_drop)) / np.diff(noise_levels)
    
    # Lipschitz constant is maximum sensitivity
    L = np.max(sensitivity) if len(sensitivity) > 0 else np.nan
    return float(L)

