from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True, slots=True)
class ChanFractal:
    index: int
    kind: int
    price: float


@dataclass(frozen=True, slots=True)
class ChanStroke:
    start: int
    end: int
    direction: int
    high: float
    low: float
    strength: float


@dataclass(frozen=True, slots=True)
class ChanSegment:
    start: int
    end: int
    direction: int
    high: float
    low: float
    strength: float


@dataclass(frozen=True, slots=True)
class ChanPivot:
    start: int
    end: int
    high: float
    low: float
    width: float


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


def _chan_object_projection(frame: pd.DataFrame, fractal_window: int) -> pd.DataFrame:
    high = frame["high"].astype(float).to_numpy()
    low = frame["low"].astype(float).to_numpy()
    close = frame["close"].astype(float).to_numpy()
    top = frame["top_fractal"].astype(float).to_numpy()
    bottom = frame["bottom_fractal"].astype(float).to_numpy()
    n = len(frame)
    min_gap = max(2, fractal_window // 2)

    fractals: list[ChanFractal] = []
    for idx in range(n):
        if bottom[idx] > 0:
            fractals.append(ChanFractal(idx, 1, float(low[idx])))
        if top[idx] > 0:
            fractals.append(ChanFractal(idx, -1, float(high[idx])))

    confirmed: list[ChanFractal] = []
    for fractal in fractals:
        if not confirmed:
            confirmed.append(fractal)
            continue
        last = confirmed[-1]
        if fractal.kind == last.kind:
            more_extreme = (fractal.kind == 1 and fractal.price < last.price) or (fractal.kind == -1 and fractal.price > last.price)
            if more_extreme:
                confirmed[-1] = fractal
            continue
        if fractal.index - last.index < min_gap:
            continue
        confirmed.append(fractal)

    strokes: list[ChanStroke] = []
    for left, right in zip(confirmed, confirmed[1:]):
        direction = 1 if left.kind == 1 and right.kind == -1 else -1
        start, end = sorted((left.index, right.index))
        span_high = float(np.max(high[start : end + 1]))
        span_low = float(np.min(low[start : end + 1]))
        base = max(abs(close[start]), 1e-9)
        strength = abs(right.price - left.price) / base
        strokes.append(ChanStroke(start, end, direction, span_high, span_low, float(strength)))

    segments: list[ChanSegment] = []
    for idx in range(2, len(strokes)):
        window = strokes[idx - 2 : idx + 1]
        direction = window[-1].direction
        start = window[0].start
        end = window[-1].end
        span_high = max(item.high for item in window)
        span_low = min(item.low for item in window)
        strength = sum(item.strength for item in window) / 3.0
        segments.append(ChanSegment(start, end, direction, span_high, span_low, float(strength)))

    pivots: list[ChanPivot] = []
    for idx in range(2, len(strokes)):
        window = strokes[idx - 2 : idx + 1]
        overlap_high = min(item.high for item in window)
        overlap_low = max(item.low for item in window)
        if overlap_high <= overlap_low:
            continue
        width = (overlap_high - overlap_low) / max(close[window[-1].end], 1e-9)
        pivots.append(ChanPivot(window[0].start, window[-1].end, float(overlap_high), float(overlap_low), float(width)))

    out = pd.DataFrame(index=frame.index)
    for column in [
        "fractal_type",
        "fractal_density",
        "stroke_direction",
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
    ]:
        out[column] = 0.0

    for fractal in confirmed:
        out.iat[fractal.index, out.columns.get_loc("fractal_type")] = float(fractal.kind)
    out["fractal_density"] = (frame["top_fractal"] + frame["bottom_fractal"]).rolling(fractal_window * 2, min_periods=1).sum()

    for stroke in strokes:
        out.loc[frame.index[stroke.start : stroke.end + 1], "stroke_direction"] = stroke.direction
        out.loc[frame.index[stroke.end], "pivot_leg_direction"] = stroke.direction
        out.loc[frame.index[stroke.end], "pivot_leg_strength"] = stroke.strength

    for segment in segments:
        out.loc[frame.index[segment.start : segment.end + 1], "segment_direction"] = segment.direction
        out.loc[frame.index[segment.start : segment.end + 1], "segment_strength"] = segment.strength

    for pivot in pivots:
        out.loc[frame.index[pivot.start : pivot.end + 1], "pivot_active"] = 1.0
        out.loc[frame.index[pivot.start : pivot.end + 1], "pivot_width"] = pivot.width

    prev_same_dir_strength: dict[int, float] = {1: 0.0, -1: 0.0}
    for stroke in strokes:
        idx = stroke.end
        previous_pivots = [pivot for pivot in pivots if pivot.end < idx]
        if not previous_pivots:
            continue
        last_pivot = previous_pivots[-1]
        phase = 1.0 if stroke.direction > 0 else -1.0
        out.loc[frame.index[idx], "pivot_leg_phase"] = phase
        prev_strength = prev_same_dir_strength.get(stroke.direction, 0.0)
        divergence = max(prev_strength - stroke.strength, 0.0) / max(prev_strength, 1e-9) if prev_strength > 0 else 0.0
        prev_same_dir_strength[stroke.direction] = stroke.strength
        if stroke.direction > 0:
            if stroke.high > last_pivot.high:
                out.loc[frame.index[idx], "buy_point_1"] = 1.0
                out.loc[frame.index[idx], "buy_divergence_score"] = divergence
            if low[idx] <= last_pivot.high and close[idx] > last_pivot.high:
                out.loc[frame.index[idx], "buy_point_2"] = 1.0
            if low[idx] > last_pivot.high and stroke.strength > 0:
                out.loc[frame.index[idx], "buy_point_3"] = 1.0
        else:
            if stroke.low < last_pivot.low:
                out.loc[frame.index[idx], "sell_point_1"] = 1.0
                out.loc[frame.index[idx], "sell_divergence_score"] = divergence
            if high[idx] >= last_pivot.low and close[idx] < last_pivot.low:
                out.loc[frame.index[idx], "sell_point_2"] = 1.0
            if high[idx] < last_pivot.low and stroke.strength > 0:
                out.loc[frame.index[idx], "sell_point_3"] = 1.0

    out["trade_point_score"] = (
        out["buy_point_1"] * 0.35
        + out["buy_point_2"] * 0.55
        + out["buy_point_3"] * 0.65
        - out["sell_point_1"] * 0.35
        - out["sell_point_2"] * 0.55
        - out["sell_point_3"] * 0.65
        + out["buy_divergence_score"] * 0.45
        - out["sell_divergence_score"] * 0.45
    )
    return out.fillna(0.0)


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
    chan_projection = _chan_object_projection(frame, fractal_window=fractal_window)
    for column in chan_projection.columns:
        frame[column] = chan_projection[column]
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
