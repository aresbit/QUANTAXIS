from __future__ import annotations

from pathlib import Path

import pandas as pd


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


def fetch_ashare_daily(
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

    normalized = renamed.copy()
    normalized["datetime"] = pd.to_datetime(normalized["datetime"])
    for column in ["open", "high", "low", "close", "volume"]:
        normalized[column] = pd.to_numeric(normalized[column], errors="coerce")
    normalized["symbol"] = symbol
    return normalized.dropna(subset=["datetime", "open", "high", "low", "close", "volume"]).reset_index(drop=True)
