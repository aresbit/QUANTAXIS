from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from QUANTAXIS.backtest.features import chan_feature_frame


@dataclass(slots=True)
class StrategyConfig:
    sequence_length: int = 32
    hidden_dim: int = 8
    fractal_window: int = 5
    trade_size: int = 100
    buy_threshold: float = 0.03
    sell_threshold: float = -0.03
    state_decay: float = 0.82
    attention_temperature: float = 4.0
    allow_short: bool = False


class RecursiveQTransformerStrategy:
    """
    A deterministic research strategy inspired by recurrent state updates and
    Chan-theory market structure features.

    This is not a proprietary Jane Street model. It is a local research scaffold
    that mixes:
    1. recursive hidden-state updates
    2. attention-like weighting on recent bars
    3. Chan-style fractal and stroke features
    """

    FEATURE_COLUMNS = [
        "return_1",
        "return_5",
        "return_10",
        "range_ratio",
        "volume_z",
        "trend_gap",
        "top_fractal",
        "bottom_fractal",
        "chan_bias",
        "stroke_strength",
        "volatility",
    ]

    def __init__(self, config: StrategyConfig | None = None) -> None:
        self.config = config or StrategyConfig()
        self._hidden = np.zeros(self.config.hidden_dim, dtype=float)
        self._projection = self._build_projection(len(self.FEATURE_COLUMNS), self.config.hidden_dim)
        self._readout = np.linspace(-0.35, 0.35, self.config.hidden_dim)

    @staticmethod
    def _build_projection(feature_dim: int, hidden_dim: int) -> np.ndarray:
        grid = np.arange(feature_dim * hidden_dim, dtype=float).reshape(feature_dim, hidden_dim)
        return np.sin(grid / (feature_dim + hidden_dim + 1.0)) * 0.6

    def reset(self) -> None:
        self._hidden.fill(0.0)

    def _attention_weights(self, seq: np.ndarray) -> np.ndarray:
        momentum = seq[:, 0] + 0.7 * seq[:, 5] + 0.5 * seq[:, 8]
        scaled = np.clip(momentum * self.config.attention_temperature, -20.0, 20.0)
        exp_scores = np.exp(scaled - scaled.max())
        weights = exp_scores / exp_scores.sum()
        return weights

    def _update_hidden(self, context: np.ndarray) -> np.ndarray:
        proposal = np.tanh(context @ self._projection)
        self._hidden = self.config.state_decay * self._hidden + (1.0 - self.config.state_decay) * proposal
        return self._hidden

    def on_bar(self, data: pd.DataFrame) -> float:
        feature_frame = chan_feature_frame(data, fractal_window=self.config.fractal_window)
        feature_values = feature_frame[self.FEATURE_COLUMNS].tail(self.config.sequence_length).to_numpy(dtype=float)
        if len(feature_values) < 4:
            return 0.0

        weights = self._attention_weights(feature_values)
        context = np.average(feature_values, axis=0, weights=weights)
        hidden = self._update_hidden(context)

        structure_bias = context[8] * 0.08 + (context[7] - context[6]) * 0.12
        momentum_bias = context[0] * 0.8 + context[1] * 0.5 + context[5] * 0.6
        hidden_bias = float(np.tanh(hidden @ self._readout))
        signal = np.tanh(momentum_bias + structure_bias + hidden_bias)
        return float(signal)
