from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(slots=True)
class MarketContext:
    """Per-bar market microstructure state for a single symbol."""

    symbol: str
    datetime: pd.Timestamp
    open_price: float
    high_price: float
    low_price: float
    close_price: float
    volume: float
    prev_close: float | None = None
    is_suspended: bool = False
    limit_up: float | None = None
    limit_down: float | None = None

    @property
    def price_range(self) -> float:
        return self.high_price - self.low_price if self.high_price > self.low_price else 0.0

    @property
    def effective_limit_up(self) -> float:
        if self.limit_up is not None:
            return self.limit_up
        if self.prev_close is not None and self.prev_close > 0:
            return round(self.prev_close * 1.1, 2)
        return self.high_price * 1.1

    @property
    def effective_limit_down(self) -> float:
        if self.limit_down is not None:
            return self.limit_down
        if self.prev_close is not None and self.prev_close > 0:
            return round(self.prev_close * 0.9, 2)
        return self.low_price * 0.9


def detect_suspension(
    row: pd.Series,
    prev_row: pd.Series | None = None,
    zero_volume_threshold: int = 1,
) -> bool:
    """Heuristic suspension detection from bar data."""
    if row.get("volume", 1) == 0:
        return True
    if prev_row is not None:
        price_unchanged = row["close"] == prev_row["close"] and row["open"] == prev_row["open"]
        vol_unchanged = row["volume"] == prev_row["volume"]
        if price_unchanged and vol_unchanged and row["volume"] == 0:
            return True
    return False


def compute_limit_prices(prev_close: float, market: str = "main") -> tuple[float, float]:
    """Compute limit-up/limit-down prices for A-shares.

    Args:
        prev_close: Previous close price.
        market: "main" (主板/中小板, ±10%), "gem" (创业板, ±20%),
                "star" (科创板, ±20%), "bse" (北交所, ±30%).
    """
    if prev_close <= 0:
        return 0.0, 0.0
    ratios = {"main": 0.1, "gem": 0.2, "star": 0.2, "bse": 0.3}
    ratio = ratios.get(market, 0.1)
    limit_up = round(prev_close * (1.0 + ratio), 2)
    limit_down = round(prev_close * (1.0 - ratio), 2)
    return limit_up, limit_down


def infer_market_segment(symbol: str) -> str:
    """Infer A-share market segment for limit rules."""
    s = symbol.strip()
    if s.startswith(("688", "689")):
        return "star"
    if s.startswith(("300", "301")):
        return "gem"
    if s.startswith(("8", "43", "83", "87", "88")):
        return "bse"
    return "main"


def apply_slippage(
    price: float,
    side: str,
    slippage_model: str = "fixed",
    slippage_value: float = 0.0,
    volatility: float = 0.0,
    volume: float = 1.0,
    trade_size: float = 100.0,
) -> float:
    """Apply slippage to a theoretical fill price.

    Models:
        - fixed:   price +/- slippage_value
        - percent: price * (1 +/- slippage_value)
        - impact:  price * (1 +/- slippage_value * (trade_size/volume)**0.6)
    """
    side = side.lower()
    sign = 1.0 if side == "buy" else -1.0
    if slippage_model == "fixed":
        delta = slippage_value
    elif slippage_model == "percent":
        delta = price * slippage_value
    elif slippage_model == "impact":
        if volume <= 0:
            vol_ratio = 1.0
        else:
            vol_ratio = min(trade_size / volume, 1.0)
        delta = price * slippage_value * (vol_ratio ** 0.6)
    else:
        delta = 0.0
    # Volatility jitter (std ~ vol*price/sqrt(252))
    if volatility > 0:
        jitter = np.random.normal(0.0, volatility * price / 15.87)
        delta += abs(jitter) * sign
    return round(price + sign * delta, 3)


def can_trade(
    ctx: MarketContext,
    side: str,
    slippage_model: str = "fixed",
    slippage_value: float = 0.0,
) -> tuple[bool, float, str]:
    """Determine if a trade can execute and at what price.

    Returns (ok, fill_price, reason).
    """
    if ctx.is_suspended:
        return False, 0.0, "suspended"

    side = side.lower()
    base_price = ctx.close_price
    fill_price = apply_slippage(
        base_price, side, slippage_model=slippage_model, slippage_value=slippage_value
    )

    # Limit-up/down constraint: cannot buy at limit-up, cannot sell at limit-down
    if side == "buy" and fill_price >= ctx.effective_limit_up:
        return False, 0.0, "limit_up"
    if side == "sell" and fill_price <= ctx.effective_limit_down:
        return False, 0.0, "limit_down"

    return True, fill_price, "ok"


def build_market_contexts(
    snapshot: pd.DataFrame,
    prev_snapshot: pd.DataFrame | None,
    symbol_groups: dict[str, str] | None = None,
) -> dict[str, MarketContext]:
    """Build MarketContext for each symbol in a bar snapshot."""
    contexts: dict[str, MarketContext] = {}
    symbol_col = "symbol" if "symbol" in snapshot.columns else "code"
    for symbol, row in snapshot.groupby(symbol_col).last().iterrows():
        symbol = str(symbol)
        prev_row = None
        if prev_snapshot is not None and symbol in prev_snapshot.groupby(symbol_col).groups:
            prev_row = prev_snapshot.groupby(symbol_col).last().loc[symbol]
        prev_close = float(prev_row["close"]) if prev_row is not None else float(row.get("open", row["close"]))
        market_seg = infer_market_segment(symbol)
        limit_up, limit_down = compute_limit_prices(prev_close, market_seg)
        is_suspended = detect_suspension(row, prev_row)
        contexts[symbol] = MarketContext(
            symbol=symbol,
            datetime=pd.Timestamp(row.name if isinstance(row.name, pd.Timestamp) else row.get("datetime")),
            open_price=float(row["open"]),
            high_price=float(row["high"]),
            low_price=float(row["low"]),
            close_price=float(row["close"]),
            volume=float(row["volume"]),
            prev_close=prev_close,
            is_suspended=is_suspended,
            limit_up=limit_up,
            limit_down=limit_down,
        )
    return contexts
