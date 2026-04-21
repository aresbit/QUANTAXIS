from __future__ import annotations

import numpy as np
import pandas as pd


def rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    mean = series.rolling(window, min_periods=max(5, window // 4)).mean()
    std = series.rolling(window, min_periods=max(5, window // 4)).std().replace(0, np.nan)
    return ((series - mean) / std).fillna(0.0)


def _fft_high_band_ratio(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    if arr.size < 8:
        return 0.0
    centered = arr - arr.mean()
    spectrum = np.abs(np.fft.rfft(centered)) ** 2
    if spectrum.size <= 2:
        return 0.0
    spectrum = spectrum[1:]
    total = float(spectrum.sum())
    if total <= 0:
        return 0.0
    split = max(1, int(np.ceil(spectrum.size * 0.6)))
    return float(spectrum[split:].sum() / total)


def _fft_peak_frequency(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    if arr.size < 8:
        return 0.0
    centered = arr - arr.mean()
    spectrum = np.abs(np.fft.rfft(centered)) ** 2
    if spectrum.size <= 2:
        return 0.0
    spectrum = spectrum[1:]
    peak = int(np.argmax(spectrum))
    return float((peak + 1) / max(spectrum.size, 1))


def _fft_spectral_entropy(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    if arr.size < 8:
        return 0.0
    centered = arr - arr.mean()
    spectrum = np.abs(np.fft.rfft(centered)) ** 2
    if spectrum.size <= 2:
        return 0.0
    power = spectrum[1:]
    total = float(power.sum())
    if total <= 0:
        return 0.0
    probs = np.clip(power / total, 1e-12, None)
    entropy = -np.sum(probs * np.log(probs))
    return float(entropy / np.log(probs.size))


def rolling_percentile(series: pd.Series, window: int) -> pd.Series:
    min_periods = max(5, window // 4)
    return series.rolling(window, min_periods=min_periods).apply(
        lambda arr: float(pd.Series(arr).rank(pct=True).iloc[-1]),
        raw=False,
    ).fillna(0.5)


def chan_feature_frame(data: pd.DataFrame, fractal_window: int = 5) -> pd.DataFrame:
    frame = data.copy()
    if "datetime" in frame.columns:
        frame["datetime"] = pd.to_datetime(frame["datetime"])
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
    frame["adv20"] = volume.rolling(20, min_periods=5).mean().fillna(0.0)

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
    fft_window = max(16, fractal_window * 4)
    returns = close.pct_change().fillna(0.0)
    frame["fft_high_band_ratio"] = returns.rolling(fft_window, min_periods=8).apply(_fft_high_band_ratio, raw=True).fillna(0.0)
    frame["fft_peak_frequency"] = returns.rolling(fft_window, min_periods=8).apply(_fft_peak_frequency, raw=True).fillna(0.0)
    frame["fft_spectral_entropy"] = returns.rolling(fft_window, min_periods=8).apply(_fft_spectral_entropy, raw=True).fillna(0.0)
    frame["fft_burst"] = rolling_zscore(frame["fft_high_band_ratio"] * (1.0 + frame["range_ratio"]), 20)
    frame["fft_regime_shift"] = rolling_zscore(
        frame["fft_peak_frequency"].diff().abs() + frame["fft_spectral_entropy"].diff().abs(),
        20,
    )
    price_span = (high - low).replace(0.0, np.nan)
    if "datetime" in frame.columns:
        session_day = frame["datetime"].dt.strftime("%Y-%m-%d")
        typical_price = (high + low + close) / 3.0
        turnover = (typical_price * volume.fillna(0.0)).groupby(session_day).cumsum()
        cumulative_volume = volume.fillna(0.0).groupby(session_day).cumsum().replace(0.0, np.nan)
        vwap = (turnover / cumulative_volume).replace([np.inf, -np.inf], np.nan)
        frame["intraday_vwap_gap"] = ((close / vwap) - 1.0).fillna(0.0)

        session_index = frame.groupby(session_day).cumcount()
        opening_high = high.where(session_index < 4).groupby(session_day).transform("max")
        opening_low = low.where(session_index < 4).groupby(session_day).transform("min")
        frame["opening_range_break"] = (
            ((close > opening_high) & (session_index >= 4)).astype(float)
            - ((close < opening_low) & (session_index >= 4)).astype(float)
        ).fillna(0.0)
        frame["opening_range_distance"] = (
            (close - opening_high.fillna(close)) / close.replace(0, np.nan)
        ).fillna(0.0)
        frame["session_momentum"] = frame["return_1"].groupby(session_day).cumsum().fillna(0.0)
    else:
        vwap = ((high + low + close) / 3.0).rolling(5, min_periods=1).mean()
        frame["intraday_vwap_gap"] = 0.0
        frame["opening_range_break"] = 0.0
        frame["opening_range_distance"] = 0.0
        frame["session_momentum"] = 0.0
    frame["vwap"] = pd.Series(vwap, index=frame.index).fillna(((high + low + close) / 3.0))

    alpha5_raw = (open := frame["open"].astype(float)) - frame["vwap"].rolling(10, min_periods=3).mean()
    alpha5_rank = rolling_percentile(alpha5_raw, 20)
    alpha5_gap_rank = rolling_percentile((close - frame["vwap"]).abs(), 20)
    frame["alpha101_05"] = (alpha5_rank * (1.0 - alpha5_gap_rank) * 2.0 - 1.0).fillna(0.0)

    vwap_gap = frame["vwap"] - close
    alpha11_raw = (
        rolling_percentile(vwap_gap.rolling(3, min_periods=1).max(), 20)
        + rolling_percentile(vwap_gap.rolling(3, min_periods=1).min(), 20)
    ) * rolling_percentile(volume.diff(3).fillna(0.0), 20)
    frame["alpha101_11"] = (alpha11_raw - 1.0).fillna(0.0)

    alpha25_raw = (-frame["return_1"] * frame["adv20"].replace(0.0, np.nan) * frame["vwap"] * (high - close)).replace(
        [np.inf, -np.inf],
        np.nan,
    )
    frame["alpha101_25"] = rolling_zscore(alpha25_raw.fillna(0.0), 20)

    frame["alpha101_41"] = (((high * low).clip(lower=0.0) ** 0.5) - frame["vwap"]) / close.replace(0.0, np.nan)
    frame["alpha101_41"] = frame["alpha101_41"].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    alpha42_denom = (frame["vwap"] + close).abs().replace(0.0, np.nan)
    frame["alpha101_42"] = ((frame["vwap"] - close) / alpha42_denom).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    frame["alpha101_101"] = ((close - open) / (price_span + 1e-3)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    frame["alpha101_physical"] = (
        frame["alpha101_05"] * 0.18
        + frame["alpha101_11"] * 0.16
        + frame["alpha101_25"] * 0.24
        + rolling_zscore(frame["alpha101_41"], 20) * 0.10
        + rolling_zscore(frame["alpha101_42"], 20) * 0.12
        + rolling_zscore(frame["alpha101_101"], 20) * 0.20
    ).fillna(0.0)
    return frame.fillna(0.0)
