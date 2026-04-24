from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import hashlib
from pathlib import Path

import pandas as pd

from QUANTAXIS.ashare.quotes import DEFAULT_TDX_SERVERS, detect_market


PYTDX_KLINE_TYPES = {
    "1min": 8,
    "5min": 0,
    "15min": 1,
    "30min": 2,
    "60min": 3,
    "day": 9,
}

AKSHARE_MINUTE_PERIODS = {
    "1min": "1",
    "5min": "5",
    "15min": "15",
    "30min": "30",
    "60min": "60",
}


DEFAULT_CACHE_DIR = Path("outputs/data_cache")


def _cache_key(*parts: object) -> str:
    raw = "|".join(str(part) for part in parts)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def _cache_path(
    cache_dir: str | Path,
    source: str,
    frequency: str,
    symbol: str,
    start_date: str,
    end_date: str,
    adjust: str,
) -> Path:
    safe_adjust = adjust or "none"
    name = f"{symbol}_{start_date}_{end_date}_{safe_adjust}_{_cache_key(source, frequency, symbol, start_date, end_date, adjust)}.csv"
    return Path(cache_dir) / source / frequency / name


def _read_cached_frame(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return load_multi_ohlcv_csv(path)


def _write_cached_frame(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def validate_ohlcv_frame(frame: pd.DataFrame, context: str = "data") -> pd.DataFrame:
    """Validate and clean OHLCV data. Returns cleaned frame or raises."""
    required = {"datetime", "open", "high", "low", "close", "volume"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"[{context}] missing required columns: {sorted(missing)}")

    frame = frame.copy()
    frame["datetime"] = pd.to_datetime(frame["datetime"], errors="coerce")
    for column in ["open", "high", "low", "close", "volume"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    # Drop rows with NaN in core fields
    before_drop = len(frame)
    frame = frame.dropna(subset=["datetime", "open", "high", "low", "close", "volume"])
    after_drop = len(frame)
    if after_drop < before_drop:
        print(f"[{context}] dropped {before_drop - after_drop} rows with NaN values")

    # Price sanity checks
    invalid_ohlc = (
        (frame["high"] < frame["low"])
        | (frame["close"] > frame["high"])
        | (frame["close"] < frame["low"])
        | (frame["open"] > frame["high"])
        | (frame["open"] < frame["low"])
    )
    bad = invalid_ohlc.sum()
    if bad > 0:
        print(f"[{context}] found {bad} rows with invalid OHLC relationship; clamping")
        frame.loc[invalid_ohlc, "high"] = frame.loc[invalid_ohlc, ["open", "high", "low", "close"]].max(axis=1)
        frame.loc[invalid_ohlc, "low"] = frame.loc[invalid_ohlc, ["open", "high", "low", "close"]].min(axis=1)

    # Detect limit-up/limit-down (≈ 20% for GEM/STAR, 10% for main)
    frame["prev_close"] = frame["close"].shift(1)
    frame["daily_return"] = (frame["close"] / frame["prev_close"] - 1.0).fillna(0.0)
    extreme_moves = frame["daily_return"].abs() > 0.22
    if extreme_moves.sum() > 0:
        print(f"[{context}] warning: {extreme_moves.sum()} bars with >22% daily move (possible bad adjustment)")

    # Detect suspicious volume spikes (>20x median)
    median_vol = frame["volume"].median()
    if median_vol > 0:
        vol_spikes = frame["volume"] > median_vol * 20
        if vol_spikes.sum() > 0:
            print(f"[{context}] warning: {vol_spikes.sum()} bars with volume >20x median")

    # Detect zero-volume bars
    zero_vol = frame["volume"] == 0
    if zero_vol.sum() > 0:
        print(f"[{context}] warning: {zero_vol.sum()} zero-volume bars (possible suspension)")

    return frame.reset_index(drop=True)


def load_ohlcv_csv(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    return validate_ohlcv_frame(frame, context=f"csv:{path}")


def load_multi_ohlcv_csv(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"datetime", "symbol", "open", "high", "low", "close", "volume"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"missing required columns: {sorted(missing)}")
    frame["datetime"] = pd.to_datetime(frame["datetime"])
    frame["symbol"] = frame["symbol"].astype(str)
    for column in ["open", "high", "low", "close", "volume"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame.dropna(subset=["datetime", "symbol", "open", "high", "low", "close", "volume"]).reset_index(drop=True)


def _normalize_ohlcv_frame(frame: pd.DataFrame, symbol: str) -> pd.DataFrame:
    normalized = frame.copy()
    normalized["datetime"] = pd.to_datetime(normalized["datetime"])
    for column in ["open", "high", "low", "close", "volume"]:
        normalized[column] = pd.to_numeric(normalized[column], errors="coerce")
    normalized["symbol"] = symbol
    return normalized.dropna(subset=["datetime", "open", "high", "low", "close", "volume"]).reset_index(drop=True)


def _fetch_ashare_daily_akshare(
    symbol: str,
    start_date: str,
    end_date: str,
    adjust: str = "",
) -> pd.DataFrame:
    try:
        import akshare as ak
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "akshare is required for historical A-share backtest data. Install it with `uv sync --extra research`."
        ) from exc

    raw = ak.stock_zh_a_hist(
        symbol=symbol,
        period="daily",
        start_date=start_date.replace("-", ""),
        end_date=end_date.replace("-", ""),
        adjust=adjust,
    )
    renamed = raw.rename(
        columns={
            "日期": "datetime",
            "开盘": "open",
            "收盘": "close",
            "最高": "high",
            "最低": "low",
            "成交量": "volume",
            "成交额": "amount",
            "振幅": "amplitude",
            "涨跌幅": "pct_change",
            "涨跌额": "change",
            "换手率": "turnover",
        }
    )
    if renamed.empty:
        raise RuntimeError(f"no history returned for {symbol} between {start_date} and {end_date}")

    return _normalize_ohlcv_frame(renamed, symbol)


def _fetch_ashare_daily_pytdx(
    symbol: str,
    start_date: str,
    end_date: str,
    servers: tuple[tuple[str, int], ...] = DEFAULT_TDX_SERVERS,
) -> pd.DataFrame:
    try:
        from pytdx.hq import TdxHq_API
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "pytdx is required for TDX historical A-share data. Install it with `uv sync`."
        ) from exc

    market = detect_market(symbol)
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    last_error: Exception | None = None

    for host, port in servers:
        api = TdxHq_API()
        try:
            if not api.connect(host, port, time_out=2):
                raise RuntimeError(f"failed to connect {host}:{port}")
            chunks: list[pd.DataFrame] = []
            offset = 0
            while True:
                batch = api.get_security_bars(9, market, symbol, offset, 800)
                if not batch:
                    break
                frame = pd.DataFrame(batch)
                chunks.append(frame)
                if len(batch) < 800:
                    break
                offset += 800
            if not chunks:
                raise RuntimeError(f"empty history returned from {host}:{port} for {symbol}")

            raw = pd.concat(chunks, ignore_index=True)
            raw["datetime"] = pd.to_datetime(raw["datetime"])
            raw = raw[(raw["datetime"] >= start) & (raw["datetime"] <= end)]
            if raw.empty:
                raise RuntimeError(f"no TDX history returned for {symbol} between {start_date} and {end_date}")
            renamed = raw.rename(columns={"vol": "volume"})
            return _normalize_ohlcv_frame(renamed.loc[:, ["datetime", "open", "high", "low", "close", "volume"]], symbol)
        except Exception as exc:  # pragma: no cover - network paths vary
            last_error = exc
        finally:
            try:
                api.disconnect()
            except Exception:
                pass

    if last_error is None:
        raise RuntimeError(f"no TDX server configured for {symbol}")
    raise RuntimeError(f"failed to fetch TDX history for {symbol}: {last_error}") from last_error


def _fetch_ashare_bars_pytdx(
    symbol: str,
    start_date: str,
    end_date: str,
    frequency: str,
    servers: tuple[tuple[str, int], ...] = DEFAULT_TDX_SERVERS,
) -> pd.DataFrame:
    try:
        from pytdx.hq import TdxHq_API
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "pytdx is required for TDX historical A-share data. Install it with `uv sync`."
        ) from exc

    if frequency not in PYTDX_KLINE_TYPES:
        raise ValueError(f"unsupported pytdx frequency: {frequency}")

    market = detect_market(symbol)
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    category = PYTDX_KLINE_TYPES[frequency]
    last_error: Exception | None = None

    for host, port in servers:
        api = TdxHq_API()
        try:
            if not api.connect(host, port, time_out=2):
                raise RuntimeError(f"failed to connect {host}:{port}")
            chunks: list[pd.DataFrame] = []
            offset = 0
            while True:
                batch = api.get_security_bars(category, market, symbol, offset, 800)
                if not batch:
                    break
                frame = pd.DataFrame(batch)
                chunks.append(frame)
                if len(batch) < 800:
                    break
                offset += 800
            if not chunks:
                raise RuntimeError(f"empty history returned from {host}:{port} for {symbol}")

            raw = pd.concat(chunks, ignore_index=True)
            raw["datetime"] = pd.to_datetime(raw["datetime"])
            raw = raw[(raw["datetime"] >= start) & (raw["datetime"] <= end)]
            if raw.empty:
                raise RuntimeError(f"no TDX history returned for {symbol} between {start_date} and {end_date}")
            renamed = raw.rename(columns={"vol": "volume"})
            return _normalize_ohlcv_frame(renamed.loc[:, ["datetime", "open", "high", "low", "close", "volume"]], symbol)
        except Exception as exc:  # pragma: no cover - network paths vary
            last_error = exc
        finally:
            try:
                api.disconnect()
            except Exception:
                pass

    if last_error is None:
        raise RuntimeError(f"no TDX server configured for {symbol}")
    raise RuntimeError(f"failed to fetch TDX history for {symbol}: {last_error}") from last_error


def _fetch_ashare_bars_akshare(
    symbol: str,
    start_date: str,
    end_date: str,
    frequency: str,
    adjust: str = "",
) -> pd.DataFrame:
    try:
        import akshare as ak
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "akshare is required for historical A-share minute data. Install it with `uv sync --extra research`."
        ) from exc

    if frequency not in AKSHARE_MINUTE_PERIODS:
        raise ValueError(f"unsupported akshare minute frequency: {frequency}")

    raw = ak.stock_zh_a_hist_min_em(
        symbol=symbol,
        start_date=f"{start_date} 09:30:00",
        end_date=f"{end_date} 15:00:00",
        period=AKSHARE_MINUTE_PERIODS[frequency],
        adjust=adjust,
    )
    renamed = raw.rename(
        columns={
            "时间": "datetime",
            "开盘": "open",
            "收盘": "close",
            "最高": "high",
            "最低": "low",
            "成交量": "volume",
            "成交额": "amount",
            "最新价": "close",
        }
    )
    if renamed.empty:
        raise RuntimeError(f"no akshare minute history returned for {symbol} between {start_date} and {end_date}")

    columns = ["datetime", "open", "high", "low", "close", "volume"]
    if "amount" in renamed.columns:
        columns.append("amount")
    return _normalize_ohlcv_frame(renamed.loc[:, [column for column in columns if column in renamed.columns]], symbol)


def fetch_ashare_daily(
    symbol: str,
    start_date: str,
    end_date: str,
    adjust: str = "",
    source: str = "auto",
) -> pd.DataFrame:
    if source == "akshare":
        return _fetch_ashare_daily_akshare(symbol, start_date, end_date, adjust=adjust)
    if source == "pytdx":
        return _fetch_ashare_daily_pytdx(symbol, start_date, end_date)
    if source != "auto":
        raise ValueError(f"unsupported A-share history source: {source}")

    try:
        return _fetch_ashare_daily_akshare(symbol, start_date, end_date, adjust=adjust)
    except Exception:
        return _fetch_ashare_daily_pytdx(symbol, start_date, end_date)


def fetch_ashare_bars(
    symbol: str,
    start_date: str,
    end_date: str,
    frequency: str = "day",
    adjust: str = "",
    source: str = "auto",
    cache_dir: str | Path | None = DEFAULT_CACHE_DIR,
    refresh_cache: bool = False,
) -> pd.DataFrame:
    path = _cache_path(cache_dir, source, frequency, symbol, start_date, end_date, adjust) if cache_dir else None
    if path is not None and not refresh_cache:
        cached = _read_cached_frame(path)
        if cached is not None:
            return cached

    if frequency == "day":
        frame = fetch_ashare_daily(symbol, start_date, end_date, adjust=adjust, source=source)
    else:
        if source == "akshare":
            frame = _fetch_ashare_bars_akshare(symbol, start_date, end_date, frequency=frequency, adjust=adjust)
        elif source == "pytdx":
            frame = _fetch_ashare_bars_pytdx(symbol, start_date, end_date, frequency=frequency)
        elif source == "auto":
            try:
                frame = _fetch_ashare_bars_pytdx(symbol, start_date, end_date, frequency=frequency)
            except Exception:
                frame = _fetch_ashare_bars_akshare(symbol, start_date, end_date, frequency=frequency, adjust=adjust)
        else:
            raise ValueError("minute-level A-share backtest supports `pytdx`, `akshare`, or `auto` sources")

    if path is not None:
        _write_cached_frame(path, frame)
    return frame


def fetch_ashare_portfolio_bars(
    symbols: list[str],
    start_date: str,
    end_date: str,
    frequency: str = "day",
    adjust: str = "",
    source: str = "auto",
    cache_dir: str | Path | None = DEFAULT_CACHE_DIR,
    refresh_cache: bool = False,
) -> pd.DataFrame:
    if not symbols:
        raise ValueError("symbols list is empty")
    worker_count = min(len(symbols), 8)
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        frames = list(
            executor.map(
                lambda symbol: fetch_ashare_bars(
                    symbol,
                    start_date,
                    end_date,
                    frequency=frequency,
                    adjust=adjust,
                    source=source,
                    cache_dir=cache_dir,
                    refresh_cache=refresh_cache,
                ),
                symbols,
            )
        )
    return pd.concat(frames, ignore_index=True).sort_values(["datetime", "symbol"]).reset_index(drop=True)
