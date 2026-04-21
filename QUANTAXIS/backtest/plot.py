from __future__ import annotations

from pathlib import Path

import pandas as pd


def save_backtest_figure(
    equity_curve: list[dict[str, float | str]],
    trades_log: list[dict[str, float | str]],
    output_path: str | Path,
    title: str,
) -> str:
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "matplotlib is required for backtest visualization. Install it with `uv sync --extra research`."
        ) from exc

    curve = pd.DataFrame(equity_curve).copy()
    if curve.empty:
        raise ValueError("equity curve is empty")
    curve["datetime"] = pd.to_datetime(curve["datetime"])
    curve["drawdown"] = curve["equity"] / curve["equity"].cummax() - 1.0
    trades = pd.DataFrame(trades_log).copy() if trades_log else pd.DataFrame(columns=["datetime", "side", "price"])
    if not trades.empty:
        trades["datetime"] = pd.to_datetime(trades["datetime"])

    fig, axes = plt.subplots(4, 1, figsize=(16, 12), sharex=True, constrained_layout=True)
    fig.suptitle(title)

    axes[0].plot(curve["datetime"], curve["close"], label="close", color="#1f77b4", linewidth=1.3)
    if not trades.empty:
        buys = trades[trades["side"] == "buy"]
        sells = trades[trades["side"] == "sell"]
        axes[0].scatter(buys["datetime"], buys["price"], label="buy", color="#2ca02c", marker="^", s=55)
        axes[0].scatter(sells["datetime"], sells["price"], label="sell", color="#d62728", marker="v", s=55)
    axes[0].set_ylabel("Price")
    axes[0].legend(loc="upper left")
    axes[0].grid(alpha=0.25)

    axes[1].plot(curve["datetime"], curve["signal"], label="signal", color="#ff7f0e", linewidth=1.2)
    axes[1].axhline(0.0, color="black", linewidth=0.8, alpha=0.6)
    axes[1].set_ylabel("Signal")
    axes[1].grid(alpha=0.25)

    axes[2].plot(curve["datetime"], curve["equity"], label="equity", color="#2ca02c", linewidth=1.4)
    if "active_positions" in curve.columns:
        ax_positions = axes[2].twinx()
        ax_positions.plot(curve["datetime"], curve["active_positions"], label="active_positions", color="#9467bd", linewidth=1.0, alpha=0.7)
        ax_positions.set_ylabel("Positions")
    axes[2].set_ylabel("Equity")
    axes[2].grid(alpha=0.25)

    axes[3].fill_between(curve["datetime"], curve["drawdown"], 0.0, color="#d62728", alpha=0.35)
    axes[3].set_ylabel("Drawdown")
    axes[3].grid(alpha=0.25)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    plt.close(fig)
    return str(output)
