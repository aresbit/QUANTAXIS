from __future__ import annotations

from pathlib import Path

import pandas as pd


def _extract_market_frame(strategy) -> pd.DataFrame:
    market_data = getattr(strategy, "_market_data", [])
    if isinstance(market_data, list):
        if not market_data:
            return pd.DataFrame(columns=["datetime", "code", "open", "high", "low", "close", "volume"])
        if isinstance(market_data[0], pd.Series):
            frame = pd.DataFrame(market_data)
            frame.index = pd.MultiIndex.from_tuples(frame.index, names=["datetime", "code"])
            frame = frame.reset_index()
        else:
            frame = pd.concat(market_data, axis=0, sort=False).reset_index()
    elif isinstance(market_data, pd.DataFrame):
        frame = market_data.reset_index()
    else:
        return pd.DataFrame(columns=["datetime", "code", "open", "high", "low", "close", "volume"])
    if "level_0" in frame.columns and "datetime" not in frame.columns:
        frame = frame.rename(columns={"level_0": "datetime"})
    if "level_1" in frame.columns and "code" not in frame.columns:
        frame = frame.rename(columns={"level_1": "code"})
    if "datetime" not in frame.columns or "code" not in frame.columns:
        raise ValueError("legacy strategy market data is missing datetime/code columns")
    frame["datetime"] = pd.to_datetime(frame["datetime"])
    frame["code"] = frame["code"].astype(str)
    columns = ["datetime", "code", "open", "high", "low", "close", "volume"]
    for column in columns:
        if column not in frame.columns:
            frame[column] = None
    return frame.loc[:, columns].sort_values(["datetime", "code"]).reset_index(drop=True)


def _extract_signal_frame(strategy) -> pd.DataFrame:
    signal_rows: list[dict[str, object]] = []
    for payload in getattr(strategy, "_signal", []):
        row: dict[str, object] = {}
        dt = None
        for name, meta in payload.items():
            dt = meta.get("datetime", dt)
            row[name] = meta.get("value")
            row[f"{name}__format"] = meta.get("format")
        if row:
            row["datetime"] = dt
            signal_rows.append(row)
    if not signal_rows:
        return pd.DataFrame(columns=["datetime"])
    frame = pd.DataFrame(signal_rows)
    frame["datetime"] = pd.to_datetime(frame["datetime"])
    deduped = frame.sort_values("datetime").groupby("datetime", as_index=False).last()
    format_cols = [column for column in deduped.columns if column.endswith("__format")]
    return deduped.drop(columns=format_cols, errors="ignore")


def _extract_account_frame(strategy) -> pd.DataFrame:
    snapshots = getattr(strategy, "_account_snapshots", [])
    if not snapshots:
        return pd.DataFrame(columns=["datetime", "balance", "available", "close"])
    frame = pd.DataFrame(snapshots)
    frame["datetime"] = pd.to_datetime(frame["datetime"])
    return frame.sort_values("datetime").groupby("datetime", as_index=False).last()


def save_legacy_strategy_figure(strategy, output_path: str | Path, title: str) -> str:
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "matplotlib is required for legacy strategy visualization. Install it with `uv sync --extra research`."
        ) from exc

    market = _extract_market_frame(strategy)
    account = _extract_account_frame(strategy)
    signals = _extract_signal_frame(strategy)

    if market.empty:
        raise ValueError("legacy strategy market data is empty")

    price_codes = market["code"].dropna().unique().tolist()
    account = account.copy()
    if not account.empty:
        account["drawdown"] = account["balance"] / account["balance"].cummax() - 1.0

    fig, axes = plt.subplots(4, 1, figsize=(16, 12), sharex=True, constrained_layout=True)
    fig.suptitle(title)

    for code in price_codes:
        subset = market[market["code"] == code]
        axes[0].plot(subset["datetime"], subset["close"], linewidth=1.25, label=code)
    axes[0].set_ylabel("Price")
    axes[0].grid(alpha=0.25)
    if price_codes:
        axes[0].legend(loc="upper left", ncol=min(len(price_codes), 4))

    signal_columns = [column for column in signals.columns if column != "datetime"]
    if signal_columns:
        for column in signal_columns:
            axes[1].plot(signals["datetime"], signals[column], linewidth=1.15, label=column)
        axes[1].legend(loc="upper left", ncol=min(len(signal_columns), 4))
    axes[1].axhline(0.0, color="black", linewidth=0.8, alpha=0.6)
    axes[1].set_ylabel("Signal")
    axes[1].grid(alpha=0.25)

    if not account.empty:
        axes[2].plot(account["datetime"], account["balance"], label="balance", color="#2ca02c", linewidth=1.35)
        axes[2].plot(account["datetime"], account["available"], label="available", color="#1f77b4", linewidth=1.05, alpha=0.85)
        axes[2].legend(loc="upper left")
    axes[2].set_ylabel("Equity")
    axes[2].grid(alpha=0.25)

    if not account.empty:
        axes[3].fill_between(account["datetime"], account["drawdown"], 0.0, color="#d62728", alpha=0.35)
    axes[3].set_ylabel("Drawdown")
    axes[3].grid(alpha=0.25)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    plt.close(fig)
    return str(output)
