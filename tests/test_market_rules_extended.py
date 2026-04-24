"""Extended tests for market_rules: slippage, limits, context building."""

import pandas as pd
import numpy as np

from QUANTAXIS.backtest.market_rules import (
    MarketContext,
    apply_slippage,
    build_market_contexts,
    can_trade,
    compute_limit_prices,
    detect_suspension,
    infer_market_segment,
)


def _make_snapshot(symbol: str = "000001", n: int = 1) -> pd.DataFrame:
    return pd.DataFrame({
        "symbol": [symbol] * n,
        "open": [10.0] * n,
        "high": [10.5] * n,
        "low": [9.8] * n,
        "close": [10.2] * n,
        "volume": [10000] * n,
    })


def test_infer_market_segment():
    assert infer_market_segment("000001") == "main"
    assert infer_market_segment("600001") == "main"
    assert infer_market_segment("688001") == "star"
    assert infer_market_segment("300001") == "gem"
    assert infer_market_segment("833333") == "bse"


def test_compute_limit_prices():
    up, down = compute_limit_prices(10.0, "main")
    assert up == 11.0
    assert down == 9.0

    up, down = compute_limit_prices(10.0, "gem")
    assert up == 12.0
    assert down == 8.0

    up, down = compute_limit_prices(10.0, "star")
    assert up == 12.0

    up, down = compute_limit_prices(10.0, "bse")
    assert up == 13.0

    up, down = compute_limit_prices(0.0)
    assert up == 0.0
    assert down == 0.0


def test_detect_suspension():
    row = pd.Series({"volume": 0, "close": 10.0, "open": 10.0})
    assert detect_suspension(row) is True

    row = pd.Series({"volume": 1000, "close": 10.0, "open": 10.0})
    assert detect_suspension(row) is False


def test_apply_slippage_fixed():
    buy_price = apply_slippage(10.0, "buy", "fixed", 0.02)
    assert buy_price == 10.02

    sell_price = apply_slippage(10.0, "sell", "fixed", 0.02)
    assert sell_price == 9.98


def test_apply_slippage_percent():
    buy_price = apply_slippage(10.0, "buy", "percent", 0.001)
    assert buy_price == 10.01

    sell_price = apply_slippage(10.0, "sell", "percent", 0.001)
    assert sell_price == 9.99


def test_apply_slippage_impact():
    buy_price = apply_slippage(10.0, "buy", "impact", 0.01, volume=10000, trade_size=1000)
    assert buy_price > 10.0
    assert buy_price < 10.05


def test_can_trade_normal():
    ctx = MarketContext(
        symbol="000001", datetime=pd.Timestamp("2024-01-01"),
        open_price=10.0, high_price=10.5, low_price=9.8, close_price=10.2,
        volume=10000, prev_close=10.0,
    )
    ok, price, reason = can_trade(ctx, "buy")
    assert ok
    assert price > 0
    assert reason == "ok"


def test_can_trade_suspended():
    ctx = MarketContext(
        symbol="000001", datetime=pd.Timestamp("2024-01-01"),
        open_price=10.0, high_price=10.5, low_price=9.8, close_price=10.2,
        volume=10000, is_suspended=True,
    )
    ok, price, reason = can_trade(ctx, "buy")
    assert not ok
    assert reason == "suspended"


def test_can_trade_limit_up():
    ctx = MarketContext(
        symbol="000001", datetime=pd.Timestamp("2024-01-01"),
        open_price=10.0, high_price=10.5, low_price=9.8, close_price=10.2,
        volume=10000, prev_close=10.0, limit_up=10.1,
    )
    ok, price, reason = can_trade(ctx, "buy")
    assert not ok
    assert reason == "limit_up"


def test_can_trade_limit_down():
    ctx = MarketContext(
        symbol="000001", datetime=pd.Timestamp("2024-01-01"),
        open_price=10.0, high_price=10.5, low_price=9.8, close_price=10.2,
        volume=10000, prev_close=10.0, limit_down=10.3,
    )
    ok, price, reason = can_trade(ctx, "sell")
    assert not ok
    assert reason == "limit_down"


def test_build_market_contexts():
    snapshot = _make_snapshot()
    contexts = build_market_contexts(snapshot)
    assert "000001" in contexts
    ctx = contexts["000001"]
    assert ctx.close_price == 10.2
    assert ctx.effective_limit_up == 11.0
    assert ctx.effective_limit_down == 9.0


def test_build_market_contexts_with_prev():
    prev = _make_snapshot("000001")
    snapshot = _make_snapshot("000001")
    snapshot["close"] = 10.5
    contexts = build_market_contexts(snapshot, prev)
    ctx = contexts["000001"]
    assert ctx.prev_close == 10.2


def test_market_context_price_range():
    ctx = MarketContext(
        symbol="000001", datetime=pd.Timestamp("2024-01-01"),
        open_price=10.0, high_price=10.5, low_price=9.8, close_price=10.2,
        volume=10000,
    )
    assert ctx.price_range == 0.7


def test_market_context_effective_limits_fallback():
    ctx = MarketContext(
        symbol="000001", datetime=pd.Timestamp("2024-01-01"),
        open_price=10.0, high_price=10.5, low_price=9.8, close_price=10.2,
        volume=10000,
    )
    # When no prev_close or explicit limit, fallback is used
    assert ctx.effective_limit_up > 0
    assert ctx.effective_limit_down > 0
