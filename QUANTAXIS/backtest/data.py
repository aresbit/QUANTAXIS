from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
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


def load_ohlcv_csv(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"datetime", "open", "high", "low", "close", "volume"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"missing required columns: {sorted(missing)}")
    frame["datetime"] = pd.to_datetime(frame["datetime"])
    for column in ["open", "high", "low", "close", "volume"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame.dropna(subset=["datetime", "open", "high", "low", "close", "volume"]).reset_index(drop=True)


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
) -> pd.DataFrame:
    if frequency == "day":
        return fetch_ashare_daily(symbol, start_date, end_date, adjust=adjust, source=source)
    if source not in {"auto", "pytdx"}:
        raise ValueError("minute-level A-share backtest currently supports only `pytdx` or `auto` sources")
    return _fetch_ashare_bars_pytdx(symbol, start_date, end_date, frequency=frequency)


def fetch_ashare_portfolio_bars(
    symbols: list[str],
    start_date: str,
    end_date: str,
    frequency: str = "day",
    adjust: str = "",
    source: str = "auto",
) -> pd.DataFrame:
    if not symbols:
        raise ValueError("symbols list is empty")
    worker_count = min(len(symbols), 8)
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        frames = list(
            executor.map(
                lambda symbol: fetch_ashare_bars(symbol, start_date, end_date, frequency=frequency, adjust=adjust, source=source),
                symbols,
            )
        )
    return pd.concat(frames, ignore_index=True).sort_values(["datetime", "symbol"]).reset_index(drop=True)
