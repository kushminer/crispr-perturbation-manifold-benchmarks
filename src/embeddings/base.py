from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class EmbeddingLoadRequest:
    """Generic request describing how to load an embedding."""

    source_path: Path
    subset_path: Optional[Path] = None
    config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EmbeddingResult:
    """
    Container for embedding matrices.

    Attributes
    ----------
    values:
        Embedding matrix with shape (embedding_dim, n_items).
    item_labels:
        Ordered labels corresponding to each column in `values`.
    dim_labels:
        Optional names for each embedding dimension.
    metadata:
        Additional loader-specific metadata (hashes, parameters, etc.).
    """

    values: np.ndarray
    item_labels: List[str]
    dim_labels: Optional[List[str]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dataframe(self):
        """Return a pandas DataFrame view of the embedding."""
        import pandas as pd

        df = pd.DataFrame(self.values, columns=self.item_labels)
        if self.dim_labels and len(self.dim_labels) == self.values.shape[0]:
            df.insert(0, "dimension", self.dim_labels)
        return df

