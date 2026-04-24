from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import time

import numpy as np
import pandas as pd

from QUANTAXIS.backtest.features import chan_feature_frame


@dataclass(slots=True)
class StrategyConfig:
    sequence_length: int = 32
    hidden_dim: int = 8
    fractal_window: int = 5
    trade_size: int = 100
    buy_threshold: float = 0.02
    sell_threshold: float = -0.03
    state_decay: float = 0.82
    attention_temperature: float = 4.0
    rank_temperature: float = 6.0
    max_position_weight: float = 0.18
    rebalance_buffer: float = 0.02
    gross_exposure: float = 0.95
    hold_threshold: float = -0.01
    min_holding_bars: int = 6
    fft_event_threshold: float = 0.50
    fft_regime_threshold: float = 0.35
    require_event_for_entry: bool = True
    per_group_limit: int = 1
    symbol_groups: dict[str, str] = field(default_factory=dict)
    market_breadth_threshold: float = 0.35
    market_event_threshold: float = 0.02
    market_volume_threshold: float = 0.0
    market_regime_score_threshold: float = 0.45
    group_activation_threshold: float = 0.02
    min_target_weight: float = 0.06
    min_signal_target_scale: float = 0.45
    exit_on_regime_off: bool = True
    morning_confirm_bars: int = 2
    afternoon_confirm_bars: int = 2
    morning_event_gate_threshold: float = 0.12
    afternoon_event_gate_threshold: float = 0.10
    allow_short: bool = False
    trade_windows: tuple[tuple[time, time], ...] = (
        (time(9, 50), time(10, 50)),
        (time(13, 50), time(14, 30)),
    )


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
        "fractal_type",
        "fractal_density",
        "chan_bias",
        "stroke_direction",
        "stroke_strength",
        "segment_direction",
        "segment_strength",
        "pivot_active",
        "pivot_width",
        "pivot_leg_phase",
        "pivot_leg_direction",
        "pivot_leg_strength",
        "buy_point_1",
        "buy_point_2",
        "buy_point_3",
        "sell_point_1",
        "sell_point_2",
        "sell_point_3",
        "buy_divergence_score",
        "sell_divergence_score",
        "trade_point_score",
        "volatility",
        "fft_high_band_ratio",
        "fft_peak_frequency",
        "fft_spectral_entropy",
        "fft_burst",
        "fft_regime_shift",
        "intraday_vwap_gap",
        "opening_range_break",
        "opening_range_distance",
        "session_momentum",
        # Alpha101 factors (base set)
        "alpha101_05",
        "alpha101_11",
        "alpha101_25",
        "alpha101_41",
        "alpha101_42",
        "alpha101_101",
        # Extended Alpha101 factors
        "alpha101_04",
        "alpha101_07",
        "alpha101_12",
        "alpha101_20",
        "alpha101_28",
        "alpha101_51",
        "alpha101_54",
        "alpha101_60",
        # Market microstructure
        "vol_rank_20",
        "gap_open",
        "gap_high",
        # Combined physical factor
        "alpha101_physical",
    ]

    def __init__(self, config: StrategyConfig | None = None) -> None:
        self.config = config or StrategyConfig()
        self._hidden = np.zeros(self.config.hidden_dim, dtype=float)
        self._projection = self._build_projection(len(self.FEATURE_COLUMNS), self.config.hidden_dim)
        self._readout = np.linspace(-0.35, 0.35, self.config.hidden_dim)
        self._feature_index = {name: idx for idx, name in enumerate(self.FEATURE_COLUMNS)}

    def clone(self) -> "RecursiveQTransformerStrategy":
        return type(self)(config=replace(self.config))

    @staticmethod
    def _build_projection(feature_dim: int, hidden_dim: int) -> np.ndarray:
        grid = np.arange(feature_dim * hidden_dim, dtype=float).reshape(feature_dim, hidden_dim)
        return np.sin(grid / (feature_dim + hidden_dim + 1.0)) * 0.6

    def reset(self) -> None:
        self._hidden.fill(0.0)

    def _attention_weights(self, seq: np.ndarray) -> np.ndarray:
        idx = self._feature_index
        momentum = (
            seq[:, idx["return_1"]]
            + 0.7 * seq[:, idx["trend_gap"]]
            + 0.5 * seq[:, idx["fractal_type"]]
            + 0.3 * seq[:, idx["trade_point_score"]]
        )
        momentum = momentum + 0.8 * seq[:, idx["fft_burst"]] + 0.4 * seq[:, idx["fft_regime_shift"]]
        scaled = np.clip(momentum * self.config.attention_temperature, -20.0, 20.0)
        exp_scores = np.exp(scaled - scaled.max())
        weights = exp_scores / exp_scores.sum()
        return weights

    def _update_hidden(self, context: np.ndarray) -> np.ndarray:
        proposal = np.tanh(context @ self._projection)
        self._hidden = self.config.state_decay * self._hidden + (1.0 - self.config.state_decay) * proposal
        return self._hidden

    def _f(self, context: np.ndarray, name: str) -> float:
        return float(context[self._feature_index[name]])

    @staticmethod
    def _empty_score() -> dict[str, float | str]:
        return {
            "signal": 0.0,
            "base_signal": 0.0,
            "event_signal": 0.0,
            "event_gate": 0.0,
            "event_active": 0.0,
            "session_confirmed": 0.0,
            "session_label": "none",
            "volume_score": 0.0,
            "momentum_score": 0.0,
            "structure_score": 0.0,
            "fractal_score": 0.0,
            "buy_sell_score": 0.0,
            "pivot_leg_score": 0.0,
            "trade_point_score": 0.0,
            "fft_score": 0.0,
            "alpha_score": 0.0,
            "alpha_physical": 0.0,
            "alpha_body": 0.0,
            "alpha_reversal": 0.0,
            "hidden_score": 0.0,
            "risk_penalty": 0.0,
        }

    def score_frame(self, data: pd.DataFrame) -> pd.DataFrame:
        feature_frame = chan_feature_frame(data, fractal_window=self.config.fractal_window).reset_index(drop=True)
        self.reset()
        rows: list[dict[str, float | str]] = []
        feature_matrix = feature_frame[self.FEATURE_COLUMNS].to_numpy(dtype=float)
        for idx in range(len(feature_frame)):
            if idx + 1 < 4:
                rows.append(self._empty_score())
                continue
            start = max(0, idx + 1 - self.config.sequence_length)
            seq = feature_matrix[start : idx + 1]
            weights = self._attention_weights(seq)
            context = np.average(seq, axis=0, weights=weights)
            hidden = self._update_hidden(context)
            latest = feature_frame.iloc[idx]
            prev = feature_frame.iloc[idx - 1] if idx >= 1 else latest

            momentum_score = (
                self._f(context, "return_1") * 0.42
                + self._f(context, "return_5") * 0.28
                + self._f(context, "return_10") * 0.12
                + self._f(context, "trend_gap") * 0.30
                + self._f(context, "session_momentum") * 0.12
            )
            structure_score = (
                self._f(context, "chan_bias") * 0.16
                + self._f(context, "stroke_direction") * 0.09
                + self._f(context, "stroke_strength") * 0.08
                + self._f(context, "segment_direction") * 0.15
                + self._f(context, "segment_strength") * 0.08
                + self._f(context, "pivot_active") * 0.05
                - self._f(context, "pivot_width") * 0.08
            )
            fractal_score = (
                (self._f(context, "bottom_fractal") - self._f(context, "top_fractal")) * 0.18
                + self._f(context, "fractal_type") * 0.12
                + self._f(context, "fractal_density") * 0.03
            )
            pivot_leg_score = (
                self._f(context, "pivot_leg_phase") * 0.16
                + self._f(context, "pivot_leg_direction") * 0.10
                + self._f(context, "pivot_leg_strength") * 0.18
            )
            trade_point_score = self._f(context, "trade_point_score")
            buy_sell_score = (
                (self._f(context, "buy_point_1") - self._f(context, "sell_point_1")) * 0.20
                + (self._f(context, "buy_point_2") - self._f(context, "sell_point_2")) * 0.35
                + (self._f(context, "buy_point_3") - self._f(context, "sell_point_3")) * 0.45
                + (self._f(context, "buy_divergence_score") - self._f(context, "sell_divergence_score")) * 0.45
                + trade_point_score * 0.25
            )
            fft_score = (
                self._f(context, "fft_burst") * 0.22
                + self._f(context, "fft_high_band_ratio") * 0.10
                - self._f(context, "fft_spectral_entropy") * 0.08
                + self._f(context, "fft_regime_shift") * 0.16
            )
            alpha_score = (
                self._f(context, "alpha101_05") * 0.08
                + self._f(context, "alpha101_11") * 0.06
                + self._f(context, "alpha101_25") * 0.14
                + self._f(context, "alpha101_41") * 0.05
                + self._f(context, "alpha101_42") * 0.06
                + self._f(context, "alpha101_101") * 0.06
                + self._f(context, "alpha101_04") * 0.04
                + self._f(context, "alpha101_07") * 0.05
                + self._f(context, "alpha101_12") * 0.04
                + self._f(context, "alpha101_20") * 0.05
                + self._f(context, "alpha101_28") * 0.03
                + self._f(context, "alpha101_51") * 0.03
                + self._f(context, "alpha101_54") * 0.04
                + self._f(context, "alpha101_60") * 0.04
                + self._f(context, "vol_rank_20") * 0.03
                + self._f(context, "gap_open") * 0.04
                + self._f(context, "gap_high") * 0.04
                + self._f(context, "alpha101_physical") * 0.16
            )
            hidden_score = float(np.tanh(hidden @ self._readout))
            risk_penalty = max(self._f(context, "volatility"), 0.0) * 0.28 + max(self._f(context, "volume_z"), 0.0) * 0.04
            current_time = pd.Timestamp(latest["datetime"]).time() if "datetime" in latest else time(9, 50)
            is_morning = current_time <= time(10, 50)
            session_label = "morning" if is_morning else "afternoon"
            fft_burst = float(latest["fft_burst"])
            fft_regime_shift = float(latest["fft_regime_shift"])
            fractal_impulse = float(latest["bottom_fractal"] - latest["top_fractal"])
            chan_impulse = float(latest["chan_bias"] - prev["chan_bias"])
            stroke_impulse = float(latest["stroke_strength"])
            segment_impulse = float(latest["segment_direction"] * latest["segment_strength"])
            pivot_bias = float(latest["pivot_active"] * np.tanh(latest["pivot_width"] * 50.0))
            buy_sell_impulse = float(
                latest["buy_point_1"] + latest["buy_point_2"] + latest["buy_point_3"]
                - latest["sell_point_1"] - latest["sell_point_2"] - latest["sell_point_3"]
            )
            vwap_gap = float(latest["intraday_vwap_gap"])
            opening_break = float(latest["opening_range_break"])
            opening_distance = float(latest["opening_range_distance"])
            session_momentum = float(latest["session_momentum"])
            alpha_physical = float(latest["alpha101_physical"])
            alpha_body = float(latest["alpha101_101"])
            alpha_reversal = float(latest["alpha101_42"])
            event_pressure = max(fft_burst - self.config.fft_event_threshold, 0.0) + max(
                fft_regime_shift - self.config.fft_regime_threshold,
                0.0,
            )
            if is_morning:
                event_alignment = (
                    fractal_impulse * 0.75
                    + chan_impulse * 0.25
                    + np.tanh(stroke_impulse) * 0.15
                    + np.tanh(segment_impulse) * 0.20
                    - pivot_bias * 0.10
                    + buy_sell_impulse * 0.20
                    + opening_break * 0.8
                    + np.tanh(opening_distance * 20.0) * 0.35
                    + np.tanh(alpha_body * 2.5) * 0.30
                )
                confirm_gate = self.config.morning_event_gate_threshold
                confirm_bars = self.config.morning_confirm_bars
            else:
                event_alignment = (
                    fractal_impulse * 0.55
                    + chan_impulse * 0.20
                    + np.tanh(stroke_impulse) * 0.15
                    + np.tanh(segment_impulse) * 0.20
                    - pivot_bias * 0.10
                    + buy_sell_impulse * 0.18
                    + np.tanh(vwap_gap * 30.0) * 0.55
                    + np.tanh(session_momentum * 12.0) * 0.25
                    + np.tanh(alpha_reversal * 5.0) * 0.25
                )
                confirm_gate = self.config.afternoon_event_gate_threshold
                confirm_bars = self.config.afternoon_confirm_bars
            event_gate = float(np.tanh(event_pressure * (1.0 + abs(event_alignment))))
            recent_gate = feature_frame["fft_burst"].iloc[max(0, idx + 1 - confirm_bars) : idx + 1].gt(self.config.fft_event_threshold).sum()
            session_confirmed = 1.0 if event_gate > confirm_gate and recent_gate >= confirm_bars else 0.0
            event_active = 1.0 if session_confirmed > 0 else 0.0
            base_signal = float(
                np.tanh(
                    momentum_score
                    + structure_score
                    + fractal_score
                    + pivot_leg_score * 0.45
                    + buy_sell_score * 0.35
                    + fft_score
                    + alpha_score
                    + hidden_score
                    - risk_penalty
                )
            )
            event_signal = float(np.tanh(base_signal * 0.30 + alpha_physical * 0.50 + event_alignment * 0.85) * event_gate)
            signal = event_signal if self.config.require_event_for_entry else float(np.tanh(base_signal * 0.7 + event_signal * 0.8))
            rows.append(
                {
                    "signal": float(signal),
                    "base_signal": base_signal,
                    "event_signal": event_signal,
                    "event_gate": event_gate,
                    "event_active": event_active,
                    "session_confirmed": session_confirmed,
                    "session_label": session_label,
                    "volume_score": float(latest["volume_z"]),
                    "momentum_score": float(momentum_score),
                    "structure_score": float(structure_score),
                    "fractal_score": float(fractal_score),
                    "pivot_leg_score": float(pivot_leg_score),
                    "buy_sell_score": float(buy_sell_score),
                    "trade_point_score": float(trade_point_score),
                    "fft_score": float(fft_score),
                    "alpha_score": float(alpha_score),
                    "alpha_physical": float(alpha_physical),
                    "alpha_body": float(alpha_body),
                    "alpha_reversal": float(alpha_reversal),
                    "hidden_score": float(hidden_score),
                    "risk_penalty": float(risk_penalty),
                }
            )
        return pd.DataFrame(rows)

    def score_components(self, data: pd.DataFrame) -> dict[str, float]:
        return dict(self.score_frame(data).iloc[-1])

    def on_bar(self, data: pd.DataFrame) -> float:
        return self.score_components(data)["signal"]
