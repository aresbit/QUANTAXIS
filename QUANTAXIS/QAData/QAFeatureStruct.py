"""Feature data structures for factor research.

Provides FeatureFrame for storing and querying factor/feature matrices.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class FeatureFrame:
    """Container for feature/alpha factor data with metadata."""

    data: pd.DataFrame
    feature_names: list[str] = field(default_factory=list)
    target_name: str = ""
    frequency: str = "day"

    def __post_init__(self) -> None:
        if not self.feature_names:
            self.feature_names = [
                col for col in self.data.columns
                if col not in {"datetime", "symbol", "date"}
            ]

    @property
    def feature_matrix(self) -> np.ndarray:
        return self.data[self.feature_names].to_numpy(dtype=float)

    def select(self, **kwargs: slice | list) -> "FeatureFrame":
        """Select subset of data by criteria."""
        mask = pd.Series(True, index=self.data.index)
        for col, value in kwargs.items():
            if col in self.data.columns:
                if isinstance(value, (list, tuple)):
                    mask &= self.data[col].isin(value)
                else:
                    mask &= self.data[col] == value
        return FeatureFrame(
            data=self.data[mask].copy(),
            feature_names=self.feature_names,
            target_name=self.target_name,
            frequency=self.frequency,
        )
