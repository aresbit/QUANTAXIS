from __future__ import annotations

import numpy as np
import pandas as pd


def rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    mean = series.rolling(window, min_periods=max(5, window // 4)).mean()
    std = series.rolling(window, min_periods=max(5, window // 4)).std().replace(0, np.nan)
    return ((series - mean) / std).fillna(0.0)


def chan_feature_frame(data: pd.DataFrame, fractal_window: int = 5) -> pd.DataFrame:
    frame = data.copy()
    close = frame["close"].astype(float)
    high = frame["high"].astype(float)
    low = frame["low"].astype(float)
    volume = frame["volume"].astype(float).replace(0, np.nan)

    frame["return_1"] = close.pct_change().fillna(0.0)
    frame["return_5"] = close.pct_change(5).fillna(0.0)
    frame["return_10"] = close.pct_change(10).fillna(0.0)
    frame["range_ratio"] = ((high - low) / close.replace(0, np.nan)).fillna(0.0)
    frame["volume_z"] = rolling_zscore(np.log(volume).replace([np.inf, -np.inf], np.nan).fillna(0.0), 20)
    frame["trend_gap"] = ((close.ewm(span=8).mean() / close.ewm(span=21).mean()) - 1.0).fillna(0.0)

    local_high = high.rolling(fractal_window, center=True, min_periods=fractal_window).max()
    local_low = low.rolling(fractal_window, center=True, min_periods=fractal_window).min()
    frame["top_fractal"] = (high == local_high).astype(float).fillna(0.0)
    frame["bottom_fractal"] = (low == local_low).astype(float).fillna(0.0)
    frame["chan_bias"] = (
        frame["bottom_fractal"].rolling(fractal_window, min_periods=1).sum()
        - frame["top_fractal"].rolling(fractal_window, min_periods=1).sum()
    )
    frame["stroke_strength"] = rolling_zscore(close.diff().abs().rolling(3, min_periods=1).sum(), 20)
    frame["volatility"] = close.pct_change().rolling(20, min_periods=5).std().fillna(0.0)
    return frame.fillna(0.0)
