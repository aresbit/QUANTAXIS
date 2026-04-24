from QUANTAXIS.ashare.account import AccountSnapshot, Position


def test_position_market_value():
    pos = Position(symbol="000001", quantity=100, avg_price=10.0)
    assert pos.market_value(12.0) == 1200.0


def test_equity_without_prices():
    acc = AccountSnapshot(cash=100_000)
    acc.positions["000001"] = Position(symbol="000001", quantity=100, avg_price=10.0)
    assert acc.equity() == 101_000.0


def test_equity_with_last_prices():
    acc = AccountSnapshot(cash=100_000)
    acc.positions["000001"] = Position(symbol="000001", quantity=100, avg_price=10.0)
    assert acc.equity({"000001": 12.0}) == 101_200.0
