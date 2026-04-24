import pandas as pd
import pytest

from QUANTAXIS.backtest.market_rules import (
    MarketContext,
    apply_slippage,
    can_trade,
    compute_limit_prices,
    detect_suspension,
    infer_market_segment,
)


def test_detect_suspension_zero_volume():
    row = pd.Series({"close": 10.0, "open": 10.0, "volume": 0})
    assert detect_suspension(row) is True


def test_detect_suspension_normal():
    row = pd.Series({"close": 10.0, "open": 10.0, "volume": 1000})
    assert detect_suspension(row) is False


def test_compute_limit_main():
    up, down = compute_limit_prices(10.0, "main")
    assert up == 11.0
    assert down == 9.0


def test_compute_limit_gem():
    up, down = compute_limit_prices(10.0, "gem")
    assert up == 12.0
    assert down == 8.0


def test_infer_market_segment():
    assert infer_market_segment("688001") == "star"
    assert infer_market_segment("300001") == "gem"
    assert infer_market_segment("600001") == "main"
    assert infer_market_segment("000001") == "main"


def test_apply_slippage_buy():
    price = apply_slippage(10.0, "buy", slippage_model="fixed", slippage_value=0.05)
    assert price == 10.05


def test_apply_slippage_sell():
    price = apply_slippage(10.0, "sell", slippage_model="fixed", slippage_value=0.05)
    assert price == 9.95


def test_can_trade_suspended():
    ctx = MarketContext(
        symbol="000001",
        datetime=pd.Timestamp("2024-01-01"),
        open_price=10.0,
        high_price=10.0,
        low_price=10.0,
        close_price=10.0,
        volume=0.0,
        is_suspended=True,
    )
    ok, _, reason = can_trade(ctx, "buy")
    assert not ok
    assert reason == "suspended"


def test_can_trade_limit_up():
    ctx = MarketContext(
        symbol="000001",
        datetime=pd.Timestamp("2024-01-01"),
        open_price=10.0,
        high_price=11.0,
        low_price=10.0,
        close_price=11.0,
        volume=1000.0,
        prev_close=10.0,
        limit_up=11.0,
    )
    ok, _, reason = can_trade(ctx, "buy")
    assert not ok
    assert reason == "limit_up"
