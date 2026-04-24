from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


@dataclass(slots=True)
class RiskConfig:
    """Risk control parameters."""

    # Pre-trade limits
    max_position_weight: float = 0.20
    max_gross_exposure: float = 1.0
    max_net_exposure: float = 1.0
    max_single_order_value: float = 1_000_000.0
    max_daily_trades: int = 100

    # Position risk
    max_drawdown_pct: float = 0.15
    daily_loss_limit_pct: float = 0.05
    single_trade_loss_limit_pct: float = 0.03
    max_sector_concentration: float = 0.40
    max_correlated_positions: int = 3

    # Volatility controls
    max_portfolio_volatility: float = 0.30
    var_confidence: float = 0.95
    var_horizon: int = 1

    # Operational
    enforce_t1: bool = True
    enforce_limit_price: bool = True
    enforce_suspension_skip: bool = True


@dataclass(slots=True)
class RiskState:
    """Mutable risk tracking state during a backtest."""

    peak_equity: float = 0.0
    daily_start_equity: float = 0.0
    daily_trades: int = 0
    daily_pnl: float = 0.0
    current_drawdown: float = 0.0
    trade_pnls: list[float] = field(default_factory=list)
    violation_log: list[dict[str, Any]] = field(default_factory=list)
    halted: bool = False

    def reset_daily(self, equity: float) -> None:
        self.daily_start_equity = equity
        self.daily_trades = 0
        self.daily_pnl = 0.0

    def record_trade(self, pnl: float) -> None:
        self.daily_trades += 1
        self.daily_pnl += pnl
        self.trade_pnls.append(pnl)

    def record_violation(self, dt: str, rule: str, detail: str) -> None:
        self.violation_log.append({"datetime": dt, "rule": rule, "detail": detail})

    def update_equity(self, equity: float) -> None:
        if equity > self.peak_equity:
            self.peak_equity = equity
        self.current_drawdown = (self.peak_equity - equity) / self.peak_equity if self.peak_equity > 0 else 0.0


class RiskChecker:
    """Pre-trade and position risk checker."""

    def __init__(self, config: RiskConfig | None = None) -> None:
        self.config = config or RiskConfig()
        self.state = RiskState()

    def reset(self, initial_equity: float) -> None:
        self.state = RiskState(peak_equity=initial_equity, daily_start_equity=initial_equity)

    def check_order(
        self,
        dt: str,
        symbol: str,
        side: str,
        size: int,
        price: float,
        equity: float,
        current_shares: int,
        target_shares: int,
    ) -> tuple[bool, str]:
        """Pre-trade risk check. Returns (allowed, reason)."""
        if self.state.halted:
            return False, "risk_halted"

        side = side.lower()
        order_value = price * size

        # Single order size limit
        if order_value > self.config.max_single_order_value:
            self.state.record_violation(dt, "max_single_order_value", f"{symbol} {side} value={order_value:.0f}")
            return False, "max_single_order_value"

        # Daily trade count limit
        if self.config.max_daily_trades > 0 and self.state.daily_trades >= self.config.max_daily_trades:
            self.state.record_violation(dt, "max_daily_trades", f"daily_trades={self.state.daily_trades}")
            return False, "max_daily_trades"

        # Position weight limit
        if side == "buy" and equity > 0:
            new_weight = (target_shares * price) / equity
            if new_weight > self.config.max_position_weight:
                self.state.record_violation(
                    dt, "max_position_weight", f"{symbol} weight={new_weight:.2%}"
                )
                return False, "max_position_weight"

        return True, "ok"

    def check_portfolio(
        self,
        dt: str,
        equity: float,
        shares: dict[str, int],
        prices: dict[str, float],
    ) -> tuple[bool, str]:
        """Post-rebalance portfolio-level risk check."""
        if self.state.halted:
            return False, "risk_halted"

        # Drawdown halt
        if self.config.max_drawdown_pct > 0 and self.state.current_drawdown >= self.config.max_drawdown_pct:
            self.state.record_violation(
                dt, "max_drawdown", f"drawdown={self.state.current_drawdown:.2%}"
            )
            self.state.halted = True
            return False, "max_drawdown"

        # Daily loss limit
        if self.config.daily_loss_limit_pct > 0 and self.state.daily_start_equity > 0:
            daily_return = (equity - self.state.daily_start_equity) / self.state.daily_start_equity
            if daily_return <= -self.config.daily_loss_limit_pct:
                self.state.record_violation(
                    dt, "daily_loss_limit", f"daily_return={daily_return:.2%}"
                )
                self.state.halted = True
                return False, "daily_loss_limit"

        # Gross exposure limit
        if equity > 0:
            gross = sum(shares[s] * prices.get(s, 0.0) for s in shares) / equity
            if gross > self.config.max_gross_exposure:
                self.state.record_violation(dt, "max_gross_exposure", f"gross={gross:.2%}")
                return False, "max_gross_exposure"

        return True, "ok"


def compute_var(returns: pd.Series, confidence: float = 0.95) -> float:
    """Historical Value at Risk (positive number = loss magnitude)."""
    clean = returns.dropna()
    if clean.empty:
        return 0.0
    return float(-clean.quantile(1.0 - confidence))


def compute_cvar(returns: pd.Series, confidence: float = 0.95) -> float:
    """Conditional VaR (expected shortfall)."""
    clean = returns.dropna()
    if clean.empty:
        return 0.0
    var_threshold = -compute_var(clean, confidence)
    tail = clean[clean <= var_threshold]
    if tail.empty:
        return 0.0
    return float(-tail.mean())


def compute_sortino(returns: pd.Series, bars_per_year: int = 252, target: float = 0.0) -> float:
    """Sortino ratio using downside deviation."""
    clean = returns.dropna()
    if clean.empty:
        return 0.0
    downside = clean[clean < target]
    if downside.empty or downside.std() == 0:
        return 0.0
    return float(((clean.mean() - target) / downside.std(ddof=0)) * (bars_per_year ** 0.5))


def compute_beta(returns: pd.Series, benchmark: pd.Series) -> float:
    """Beta against a benchmark return series."""
    aligned = pd.concat([returns, benchmark], axis=1).dropna()
    if aligned.empty or len(aligned) < 2:
        return 0.0
    cov = aligned.cov().iloc[0, 1]
    bench_var = aligned.iloc[:, 1].var(ddof=0)
    if bench_var == 0:
        return 0.0
    return float(cov / bench_var)


def generate_risk_report(equity_curve: list[dict[str, Any]], bars_per_year: int = 252) -> dict[str, Any]:
    """Generate a post-backtest risk report from equity curve."""
    if not equity_curve:
        return {}
    df = pd.DataFrame(equity_curve)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime")
    returns = df["equity"].pct_change().fillna(0.0)
    equity = df["equity"]

    peak = equity.cummax()
    drawdown = equity / peak - 1.0

    # Rolling metrics
    rolling_sharpe_252 = (returns.rolling(252, min_periods=60).mean()
                          / returns.rolling(252, min_periods=60).std().replace(0, np.nan)
                          * np.sqrt(252))

    report = {
        "total_return": float(equity.iloc[-1] / equity.iloc[0] - 1.0),
        "annual_return": float((equity.iloc[-1] / equity.iloc[0]) ** (bars_per_year / len(df)) - 1.0) if len(df) > 1 else 0.0,
        "max_drawdown": float(drawdown.min()),
        "max_drawdown_duration_days": int(
            (drawdown.groupby((drawdown == 0).cumsum()).cumcount().max())
        ),
        "volatility_annual": float(returns.std() * (bars_per_year ** 0.5)),
        "sharpe": float((returns.mean() / returns.std()) * (bars_per_year ** 0.5)) if returns.std() != 0 else 0.0,
        "sortino": compute_sortino(returns, bars_per_year),
        "var_95": compute_var(returns),
        "cvar_95": compute_cvar(returns),
        "skewness": float(returns.skew()) if not returns.empty else 0.0,
        "kurtosis": float(returns.kurtosis()) if not returns.empty else 0.0,
        "win_rate": float((returns > 0).mean()) if not returns.empty else 0.0,
        "profit_factor": (
            float(returns[returns > 0].sum() / abs(returns[returns < 0].sum()))
            if returns[returns < 0].sum() != 0 else float("inf")
        ),
        "rolling_sharpe_mean": float(rolling_sharpe_252.mean()),
        "rolling_sharpe_std": float(rolling_sharpe_252.std()),
        "rolling_sharpe_min": float(rolling_sharpe_252.min()),
        "calmar": float(
            (equity.iloc[-1] / equity.iloc[0] - 1.0) / abs(drawdown.min())
        ) if drawdown.min() < 0 else 0.0,
        # Consecutive losing days
        "max_consecutive_losses": int(
            (returns <= 0).astype(int).groupby((returns > 0).cumsum()).cumsum().max()
        ) if not returns.empty else 0,
        # Recovery factor
        "recovery_factor": float(
            (equity.iloc[-1] - equity.iloc[0]) / abs(equity[drawdown == drawdown.min()].iloc[0] - peak[drawdown == drawdown.min()].iloc[0])
        ) if drawdown.min() < 0 and not equity[drawdown == drawdown.min()].empty else float("inf"),
    }
    return report


# ── Stress Testing ──────────────────────────────────────────────────────────


def stress_test_scenarios(
    returns: pd.Series,
) -> dict[str, dict[str, float]]:
    """Run standard stress test scenarios on a return series."""
    scenarios = {
        "2008_financial_crisis": ("2008-07-01", "2009-03-01"),
        "2015_china_meltdown": ("2015-06-01", "2016-01-01"),
        "2020_covid_crash": ("2020-02-01", "2020-04-01"),
        "2022_china_downturn": ("2022-01-01", "2022-10-01"),
    }
    results: dict[str, dict[str, float]] = {}
    for name, (start, end) in scenarios.items():
        mask = (returns.index >= start) & (returns.index <= end)
        scenario_returns = returns[mask]
        if scenario_returns.empty:
            continue
        cumulative = float((1 + scenario_returns).prod() - 1)
        max_dd = float((1 + scenario_returns).cumprod().div(
            (1 + scenario_returns).cumprod().cummax()).min() - 1)
        results[name] = {
            "cumulative_return": round(cumulative, 6),
            "max_drawdown": round(max_dd, 6),
            "volatility": round(float(scenario_returns.std() * np.sqrt(252)), 6),
            "n_bars": len(scenario_returns),
        }
    return results


def monte_carlo_var(
    returns: pd.Series,
    n_simulations: int = 10000,
    horizon: int = 21,
    confidence: float = 0.95,
    seed: int = 42,
) -> dict[str, float]:
    """
    Monte Carlo VaR estimation.

    Simulates portfolio returns using bootstrap with replacement,
    then computes VaR and CVaR at the specified horizon.
    """
    clean = returns.dropna()
    if len(clean) < 10:
        return {"mc_var_95": 0.0, "mc_cvar_95": 0.0, "mc_expected_return": 0.0}

    rng = np.random.default_rng(seed)
    simulated_final = np.zeros(n_simulations)

    for i in range(n_simulations):
        sampled = rng.choice(clean.values, size=horizon, replace=True)
        simulated_final[i] = np.prod(1 + sampled) - 1

    var = float(-np.percentile(simulated_final, (1 - confidence) * 100))
    cvar = float(-simulated_final[simulated_final <= -var].mean()) if any(simulated_final <= -var) else var
    expected = float(np.mean(simulated_final))

    return {
        "mc_var_95": round(var, 6),
        "mc_cvar_95": round(cvar, 6),
        "mc_expected_return": round(expected, 6),
    }


# ── Factor Exposure / Attribution ──────────────────────────────────────────


def factor_exposure_analysis(
    portfolio_returns: pd.Series,
    factor_returns: pd.DataFrame,
) -> dict[str, Any]:
    """
    Multi-factor exposure analysis using OLS regression.

    Args:
        portfolio_returns: Series of portfolio daily returns.
        factor_returns: DataFrame of factor returns (columns = factors).

    Returns:
        dict with exposures (betas), t-stats, r-squared, adj r-squared.
    """
    aligned = pd.concat([portfolio_returns, factor_returns], axis=1).dropna()
    if aligned.empty:
        return {}

    y = aligned.iloc[:, 0].values
    X = aligned.iloc[:, 1:].values
    n, k = X.shape

    if n < k + 2:
        return {}

    # OLS with intercept
    X_with_const = np.column_stack([np.ones(n), X])
    try:
        beta = np.linalg.lstsq(X_with_const, y, rcond=None)[0]
    except np.linalg.LinAlgError:
        return {}

    # Statistics
    residuals = y - X_with_const @ beta
    mse = np.sum(residuals ** 2) / (n - k - 1)
    var_beta = mse * np.linalg.pinv(X_with_const.T @ X_with_const).diagonal()
    se = np.sqrt(np.abs(var_beta))
    t_stats = beta / np.where(se > 0, se, 1.0)

    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r_squared = 1.0 - ss_res / max(ss_tot, 1e-10)
    adj_r_squared = 1.0 - (1.0 - r_squared) * (n - 1) / max(n - k - 1, 1)

    factor_names = ["alpha"] + list(factor_returns.columns)
    return {
        "exposures": dict(zip(factor_names, np.round(beta, 6).tolist())),
        "t_stats": dict(zip(factor_names, np.round(t_stats, 4).tolist())),
        "r_squared": round(float(r_squared), 6),
        "adj_r_squared": round(float(adj_r_squared), 6),
        "n_observations": n,
    }


def correlation_risk_matrix(
    returns: pd.DataFrame,
    lookback: int = 252,
) -> dict[str, Any]:
    """
    Analyze correlation matrix for concentration risk.

    Args:
        returns: Asset return DataFrame (columns = symbols).
        lookback: Rolling window for correlation estimation.

    Returns:
        dict with average correlation, condition number, cluster info.
    """
    if returns.shape[1] < 2:
        return {"avg_correlation": 0.0, "condition_number": 0.0}

    corr = returns.iloc[-lookback:].corr().to_numpy(dtype=float)
    corr = np.clip(corr, -1.0, 1.0)
    np.fill_diagonal(corr, 1.0)

    # Average pairwise correlation (off-diagonal)
    n = corr.shape[0]
    mask = ~np.eye(n, dtype=bool)
    avg_corr = float(corr[mask].mean())

    # Condition number of correlation matrix (multicollinearity)
    try:
        eigvals = np.linalg.eigvalsh(corr)
        cond = float(eigvals.max() / max(eigvals.min(), 1e-10))
    except np.linalg.LinAlgError:
        cond = 1.0

    # Pseudo-inverse condition number
    try:
        inv_corr = np.linalg.pinv(corr + np.eye(n) * 1e-6)
    except np.linalg.LinAlgError:
        inv_corr = np.eye(n)

    return {
        "avg_correlation": round(avg_corr, 4),
        "max_correlation": round(float(corr[mask].max()), 4),
        "min_correlation": round(float(corr[mask].min()), 4),
        "condition_number": round(float(min(cond, 1e6)), 2),
        "n_assets": n,
    }


def sensitivity_analysis(
    returns: pd.Series,
    factor: str = "market",
    benchmark_returns: pd.Series | None = None,
) -> dict[str, float]:
    """
    Compute strategy sensitivity metrics.

    Returns:
        dict with beta, alpha, upside/downside capture, tail dependence.
    """
    if benchmark_returns is None:
        return {"beta": 0.0, "alpha": 0.0}

    aligned = pd.concat([returns, benchmark_returns], axis=1).dropna()
    if len(aligned) < 20:
        return {"beta": 0.0, "alpha": 0.0}

    strat = aligned.iloc[:, 0]
    bench = aligned.iloc[:, 1]

    # Beta
    cov = float(strat.cov(bench))
    var_bench = float(bench.var(ddof=0))
    beta = cov / max(var_bench, 1e-10)

    # Alpha (annualized)
    alpha = float((strat.mean() - beta * bench.mean()) * 252)

    # Up/Down capture
    up_days = bench > 0
    down_days = bench < 0
    up_capture = float(strat[up_days].mean() / max(bench[up_days].mean(), 1e-10)) if up_days.any() else 1.0
    down_capture = float(strat[down_days].mean() / max(bench[down_days].mean(), 1e-10)) if down_days.any() else 1.0

    # Correlation
    correlation = float(strat.corr(bench))

    # Up/Down percentage
    up_pct = float((strat > 0).mean())

    return {
        "beta": round(beta, 4),
        "alpha": round(alpha, 6),
        "correlation": round(correlation, 4),
        "up_capture": round(up_capture, 4),
        "down_capture": round(down_capture, 4),
        "up_percentage": round(up_pct, 4),
    }
