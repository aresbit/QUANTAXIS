import numpy as np
import pandas as pd

from QUANTAXIS.backtest.risk import (
    RiskChecker,
    RiskConfig,
    RiskState,
    compute_cvar,
    compute_sortino,
    compute_var,
    generate_risk_report,
)


def test_risk_state_update():
    state = RiskState(peak_equity=100_000)
    state.update_equity(105_000)
    assert state.peak_equity == 105_000
    state.update_equity(90_000)
    assert state.current_drawdown == (105_000 - 90_000) / 105_000


def test_risk_checker_drawdown_halt():
    checker = RiskChecker(RiskConfig(max_drawdown_pct=0.10))
    checker.reset(100_000)
    checker.state.update_equity(85_000)
    ok, reason = checker.check_portfolio("2024-01-01", 85_000, {}, {})
    assert not ok
    assert reason == "max_drawdown"
    assert checker.state.halted


def test_risk_checker_order_size():
    checker = RiskChecker(RiskConfig(max_single_order_value=50_000))
    checker.reset(100_000)
    ok, reason = checker.check_order(
        "2024-01-01", "000001", "buy", 6000, 10.0, 100_000, 0, 6000
    )
    assert not ok
    assert reason == "max_single_order_value"


def test_compute_var():
    returns = pd.Series(np.random.normal(0.001, 0.02, 500))
    var_val = compute_var(returns, confidence=0.95)
    assert var_val > 0


def test_compute_sortino():
    returns = pd.Series([0.01, -0.02, 0.005, 0.015, -0.01])
    s = compute_sortino(returns, bars_per_year=252)
    assert isinstance(s, float)


def test_generate_risk_report():
    equity = [100_000 + i * 100 for i in range(100)]
    curve = [{"datetime": f"2020-01-{i+1:02d}", "equity": e} for i, e in enumerate(equity)]
    report = generate_risk_report(curve)
    assert "sharpe" in report
    assert "max_drawdown" in report
    assert "var_95" in report
