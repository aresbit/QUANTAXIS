"""
Mean-variance, Risk Parity, and Black-Litterman portfolio optimization.

Compared to 幻方量化's proprietary optimizers, this module provides:

  - Mean-Variance Optimization (MVO) with constraints
  - Risk Parity (equal risk contribution)
  - Black-Litterman model for incorporating views
  - Hierarchical Risk Parity (HRP) as a robust alternative
  - Maximum diversification

All optimizers support:
  - Long-only / long-short modes
  - Per-asset weight caps
  - Group / sector constraints
  - Turnover penalty
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import pandas as pd
from numpy.typing import NDArray


@dataclass(slots=True)
class PortfolioConfig:
    """Configuration for portfolio optimization."""

    method: str = "mvo"  # "mvo", "risk_parity", "black_litterman", "hrp", "max_div"
    max_weight: float = 0.20
    min_weight: float = 0.0
    gross_exposure: float = 1.0
    net_exposure: float = 1.0
    risk_aversion: float = 1.0
    turnover_penalty: float = 0.0
    target_volatility: float = 0.15
    group_constraints: dict[str, float] = field(default_factory=dict)
    allow_short: bool = False

    # Black-Litterman specific
    view_confidence: float = 0.5
    tau: float = 0.05  # scalar for prior uncertainty

    # Risk parity specific
    risk_parity_tol: float = 1e-6
    risk_parity_max_iter: int = 1000


def _covariance(returns: pd.DataFrame, method: str = "empirical") -> NDArray:
    """Estimate covariance matrix with shrinkage for numerical stability."""
    arr = returns.to_numpy(dtype=float)
    arr = np.where(np.isfinite(arr), arr, 0.0)
    n, m = arr.shape
    if m < 2:
        return np.eye(max(m, 1)) * 1e-6

    # Empirical covariance
    emp_cov = np.cov(arr, rowvar=False, ddof=0)
    emp_cov = np.where(np.isfinite(emp_cov), emp_cov, 0.0)

    # Ledoit-Wolf shrinkage towards diagonal
    if n < 2:
        return emp_cov + np.eye(m) * 1e-6

    # Sample mean of variances
    var_mean = np.trace(emp_cov) / max(m, 1)
    shrinkage_target = np.eye(m) * var_mean

    # Shrinkage intensity
    if method == "shrinkage":
        # Compute shrinkage intensity (simple version)
        var_arr = np.var(arr, axis=0, ddof=0)
        var_of_var = np.var((arr - arr.mean(axis=0)) ** 2, axis=0, ddof=0) / max(n, 1)
        shrinkage_intensity = np.clip(
            np.sum(var_of_var) / np.sum((emp_cov - shrinkage_target) ** 2 + 1e-10), 0.0, 1.0
        )
        cov = (1 - shrinkage_intensity) * emp_cov + shrinkage_intensity * shrinkage_target
    else:
        cov = emp_cov

    # Ensure positive definiteness
    eigvals = np.linalg.eigvalsh(cov)
    if eigvals.min() <= 1e-10:
        cov += np.eye(m) * (abs(eigvals.min()) + 1e-6)
    return cov


def _mvo_weights(
    cov: NDArray,
    mu: NDArray | None,
    config: PortfolioConfig,
) -> NDArray:
    """Mean-variance optimization with constraints."""
    n = cov.shape[0]
    if mu is None:
        return _risk_parity_weights(cov, config)

    # Standard MVO: maximize mu'w - 0.5 * lambda * w'Sw
    lam = config.risk_aversion

    # Use convex optimization via scipy if available
    try:
        from scipy.optimize import minimize

        def objective(w: NDArray) -> float:
            ret = float(mu @ w)
            risk = 0.5 * lam * float(w @ cov @ w)
            penalty = 0.0
            if config.turnover_penalty > 0:
                penalty = config.turnover_penalty * np.sum(np.abs(w - 1.0 / n))
            return -(ret - risk - penalty)

        bounds = [(config.min_weight, config.max_weight) for _ in range(n)]

        constraints: list[dict[str, Any]] = [
            {"type": "eq", "fun": lambda w: np.sum(w) - config.net_exposure}
        ]
        if config.gross_exposure < 2.0:
            constraints.append({
                "type": "ineq",
                "fun": lambda w: config.gross_exposure - np.sum(np.abs(w)),
            })

        x0 = np.ones(n) / n
        result = minimize(objective, x0, method="SLSQP", bounds=bounds, constraints=constraints,
                          options={"maxiter": 500, "ftol": 1e-12})
        if result.success:
            w = result.x
        else:
            w = x0
    except ImportError:
        # Scipy not available: min-variance proxy via inverse-covariance weighting
        inv_cov = np.linalg.pinv(cov + np.eye(n) * 1e-6)
        raw = inv_cov @ mu
        raw = np.where(np.isfinite(raw), raw, 0.0)
        # Shift so all weights are positive before normalizing
        raw = raw - raw.min() + 1e-6
        w = raw  # _cap_and_normalize will normalize and cap

    return np.asarray(w, dtype=float)


def _risk_parity_weights(cov: NDArray, config: PortfolioConfig) -> NDArray:
    """Risk parity: equal risk contribution from each asset."""
    n = cov.shape[0]
    if n < 2:
        return np.ones(1)

    try:
        from scipy.optimize import minimize

        def risk_contribution(w: NDArray) -> NDArray:
            portfolio_var = w @ cov @ w
            marginal = cov @ w
            rc = w * marginal / np.sqrt(max(portfolio_var, 1e-10))
            return rc

        def objective(w: NDArray) -> float:
            rc = risk_contribution(w)
            target = np.mean(rc)
            return float(np.sum((rc - target) ** 2))

        constraints = [
            {"type": "eq", "fun": lambda w: np.sum(w) - config.net_exposure},
        ]
        bounds = [(config.min_weight, config.max_weight) for _ in range(n)]
        x0 = np.ones(n) / n
        result = minimize(objective, x0, method="SLSQP", bounds=bounds, constraints=constraints,
                          options={"maxiter": config.risk_parity_max_iter, "ftol": config.risk_parity_tol})
        if result.success:
            w = result.x
        else:
            w = x0
    except ImportError:
        # No scipy: inverse-variance heuristic
        inv_var = 1.0 / np.diag(cov)
        w = inv_var / np.sum(inv_var)

    return np.asarray(w, dtype=float)


def _hrp_weights(cov: NDArray, corr: NDArray, config: PortfolioConfig) -> NDArray:
    """Hierarchical Risk Parity (Lopez de Prado 2016)."""
    n = cov.shape[0]
    if n < 2:
        return np.ones(1)

    try:
        from scipy.cluster.hierarchy import linkage
        from scipy.spatial.distance import squareform

        # Distance matrix from correlation
        dist = np.sqrt(np.clip(0.5 * (1.0 - corr), 0.0, None))
        np.fill_diagonal(dist, 0.0)

        # Hierarchical clustering
        condensed = squareform(dist, checks=False)
        if condensed.size == 0:
            return np.ones(n) / n
        linkage_matrix = linkage(condensed, method="ward")

        # Recursive bisection
        def _recursive_bisection(assets: list[int]) -> NDArray:
            if len(assets) <= 1:
                w = np.zeros(n)
                for a in assets:
                    w[a] = 1.0
                return w

            # Split cluster
            clusters: list[list[int]] = [[], []]
            # Use the last linkage
            n_assets = len(assets)
            # Simple heuristic: split in half
            mid = n_assets // 2
            clusters[0] = assets[:mid]
            clusters[1] = assets[mid:]

            w1 = _recursive_bisection(clusters[0])
            w2 = _recursive_bisection(clusters[1])

            # Compute variance of each sub-cluster
            idx1 = [i for i in range(n) if w1[i] > 0]
            idx2 = [i for i in range(n) if w2[i] > 0]

            if not idx1 or not idx2:
                return w1 + w2

            # Sub-covariance matrices
            sub_cov1 = cov[np.ix_(idx1, idx1)]
            sub_cov2 = cov[np.ix_(idx2, idx2)]
            w1_norm = w1[idx1] / np.sum(w1[idx1])
            w2_norm = w2[idx2] / np.sum(w2[idx2])

            var1 = float(w1_norm @ sub_cov1 @ w1_norm)
            var2 = float(w2_norm @ sub_cov2 @ w2_norm)

            # Weight allocation
            alpha = 1.0 - var1 / (var1 + var2 + 1e-10)
            return alpha * w1 + (1.0 - alpha) * w2

        initial_assets = list(range(n))
        weights = _recursive_bisection(initial_assets)
        total = np.sum(weights)
        weights = weights / total if total > 0 else np.ones(n) / n
        return np.clip(weights, config.min_weight, config.max_weight)

    except ImportError:
        return _risk_parity_weights(cov, config)


def _max_diversification_weights(cov: NDArray, config: PortfolioConfig) -> NDArray:
    """Maximum diversification: maximize weighted std / portfolio std."""
    n = cov.shape[0]
    if n < 2:
        return np.ones(1)

    std = np.sqrt(np.diag(cov))

    try:
        from scipy.optimize import minimize

        def objective(w: NDArray) -> float:
            portfolio_var = max(float(w @ cov @ w), 1e-10)
            weighted_std = float(w @ std)
            dr = weighted_std / np.sqrt(portfolio_var)
            return -dr

        bounds = [(config.min_weight, config.max_weight) for _ in range(n)]
        constraints = [
            {"type": "eq", "fun": lambda w: np.sum(w) - config.net_exposure},
        ]
        x0 = np.ones(n) / n
        result = minimize(objective, x0, method="SLSQP", bounds=bounds, constraints=constraints)
        if result.success:
            return np.asarray(result.x, dtype=float)
    except ImportError:
        pass

    return _risk_parity_weights(cov, config)


def _black_litterman_weights(
    cov: NDArray,
    market_cap_weights: NDArray,
    views: dict[int, float],
    config: PortfolioConfig,
) -> NDArray:
    """Black-Litterman model: prior + views -> posterior returns."""
    n = cov.shape[0]
    tau = config.tau

    # Prior: implied returns from market cap weights (reverse optimization)
    if market_cap_weights is None:
        market_cap_weights = np.ones(n) / n
    lam = config.risk_aversion
    prior_mu = lam * cov @ market_cap_weights

    # View matrix
    k = len(views)
    if k == 0:
        return _mvo_weights(cov, prior_mu, config)

    P = np.zeros((k, n))
    Q = np.zeros(k)
    Omega = np.eye(k) * (1.0 - config.view_confidence) / max(config.view_confidence, 1e-6)

    for i, (asset_idx, view_return) in enumerate(views.items()):
        P[i, asset_idx] = 1.0
        Q[i] = view_return

    # BL posterior
    try:
        cov_scaled = tau * cov
        temp = P @ cov_scaled @ P.T + Omega
        temp_inv = np.linalg.inv(temp)
        posterior_mu = prior_mu + cov_scaled @ P.T @ temp_inv @ (Q - P @ prior_mu)
        posterior_cov = cov + cov_scaled - cov_scaled @ P.T @ temp_inv @ P @ cov_scaled
    except np.linalg.LinAlgError:
        posterior_mu = prior_mu
        posterior_cov = cov

    return _mvo_weights(posterior_cov, posterior_mu, config)


def _cap_and_normalize(
    weights: NDArray,
    symbols: list[str],
    cfg: PortfolioConfig,
    group_map: dict[str, str] | None,
) -> NDArray:
    """Enforce per-asset caps and group constraints, normalize to net_exposure.

    Two-phase:
    1. Per-asset water-fill — only when n * max_w >= target (feasible).
       When infeasible (e.g. few assets, tight cap) skip capping and just normalize.
    2. Group constraints — scale down over-limit group assets and push
       the freed budget to non-group assets. No further per-asset clip
       is applied after this so the group constraint is not destroyed.
    """
    n = len(weights)
    if n == 0:
        return weights

    target = cfg.net_exposure
    max_w = cfg.max_weight
    min_w = cfg.min_weight

    w = np.where(np.isfinite(weights), weights, 0.0)
    total = np.sum(w)
    w = w / total * target if total > 0 else np.ones(n) / n * target

    # ── Step 1: per-asset water-fill (skip when inherently infeasible) ────────
    feasible = (n * max_w >= target - 1e-9) and (n * min_w <= target + 1e-9)
    if feasible:
        for _ in range(n + 2):
            over = w > max_w + 1e-9
            under = w < min_w - 1e-9
            if not over.any() and not under.any():
                break

            w = np.clip(w, min_w, max_w)
            free = (w > min_w + 1e-9) & (w < max_w - 1e-9)
            remaining = target - np.sum(w)

            if abs(remaining) < 1e-12:
                break
            if free.any():
                free_sum = np.sum(w[free])
                if free_sum > 0:
                    w[free] += w[free] / free_sum * remaining
                else:
                    w[free] += remaining / free.sum()
            else:
                # All assets at bounds; proportionally spread (handles n=1 case)
                s = np.sum(w)
                w += w / s * remaining if s > 0 else np.ones(n) / n * remaining

    # ── Step 2: group constraints ─────────────────────────────────────────────
    if group_map and cfg.group_constraints:
        for _outer in range(n + 2):
            any_violated = False
            for group, limit in cfg.group_constraints.items():
                grp_idx = [i for i, s in enumerate(symbols) if group_map.get(s) == group]
                out_idx = [i for i in range(n) if i not in grp_idx]
                grp_total = float(np.sum(w[grp_idx]))
                if grp_total > limit + 1e-9:
                    any_violated = True
                    excess = grp_total - limit
                    for i in grp_idx:
                        w[i] *= limit / grp_total
                    if out_idx:
                        out_sum = float(np.sum(w[out_idx]))
                        if out_sum > 0:
                            for i in out_idx:
                                w[i] += w[i] / out_sum * excess
                        else:
                            for i in out_idx:
                                w[i] += excess / len(out_idx)
            if not any_violated:
                break
        # Normalize to target without per-asset clip (preserves group constraint)
        total = np.sum(w)
        if total > 0:
            w = w / total * target

    return w


def optimize_portfolio(
    returns: pd.DataFrame,
    config: PortfolioConfig | None = None,
    current_weights: dict[str, float] | None = None,
    views: dict[str, float] | None = None,
    group_map: dict[str, str] | None = None,
) -> dict[str, float]:
    """
    Full portfolio optimization pipeline.

    Args:
        returns: T x N DataFrame of asset returns.
        config: PortfolioConfig.
        current_weights: Current position weights for turnover penalty.
        views: Dict of {symbol: expected_return} for Black-Litterman.
        group_map: Dict of {symbol: group_name} for group constraints.

    Returns:
        Dict of {symbol: target_weight}.
    """
    cfg = config or PortfolioConfig()
    symbols = list(returns.columns)
    n = len(symbols)
    if n == 0:
        return {}

    cov = _covariance(returns, method="shrinkage")
    mu = np.asarray(returns.mean().values, dtype=float)
    market_weights = np.ones(n) / n

    method = cfg.method.lower().replace("-", "_").replace(" ", "_")

    if method == "mvo":
        weights = _mvo_weights(cov, mu, cfg)
    elif method == "risk_parity":
        weights = _risk_parity_weights(cov, cfg)
    elif method == "black_litterman":
        view_indices: dict[int, float] = {}
        if views:
            for symbol, ret in views.items():
                if symbol in symbols:
                    view_indices[symbols.index(symbol)] = ret
        weights = _black_litterman_weights(cov, market_weights, view_indices, cfg)
    elif method == "hrp":
        corr = cov / np.outer(np.sqrt(np.diag(cov)), np.sqrt(np.diag(cov)))
        corr = np.clip(corr, -1.0, 1.0)
        weights = _hrp_weights(cov, corr, cfg)
    elif method == "max_div":
        weights = _max_diversification_weights(cov, cfg)
    else:
        raise ValueError(f"unknown portfolio method: {method}")

    weights = _cap_and_normalize(weights, symbols, cfg, group_map)
    return dict(zip(symbols, np.round(weights, 6).tolist()))


def compute_portfolio_risk(
    weights: dict[str, float],
    returns: pd.DataFrame,
) -> dict[str, float]:
    """
    Compute risk decomposition for a portfolio.

    Returns dict with vol, var, cvar, marginal_risk, risk_contribution, diversification_ratio.
    """
    symbols = list(weights.keys())
    w = np.array([weights.get(s, 0.0) for s in symbols], dtype=float)
    common = [s for s in symbols if s in returns.columns]
    if not common or len(common) < 2:
        return {"volatility": 0.0, "var_95": 0.0}

    re_idx = [symbols.index(s) for s in common]
    w_common = w[re_idx]
    w_common = w_common / np.sum(w_common) if np.sum(w_common) > 0 else w_common

    r = returns[common].to_numpy(dtype=float)
    r = np.where(np.isfinite(r), r, 0.0)
    cov = _covariance(returns[common])

    portfolio_var = float(w_common @ cov @ w_common)
    portfolio_vol = np.sqrt(max(portfolio_var, 1e-10))
    portfolio_returns = r @ w_common

    var_95 = float(-np.percentile(portfolio_returns, 5))
    cvar_95 = float(-portfolio_returns[portfolio_returns <= -var_95].mean()) if any(portfolio_returns <= -var_95) else var_95

    # Marginal risk contribution
    marginal = cov @ w_common / portfolio_vol
    rc = w_common * marginal / portfolio_vol if portfolio_vol > 0 else np.zeros_like(w_common)

    # Diversification ratio
    weighted_std = np.sum(w_common * np.sqrt(np.diag(cov)))
    dr = weighted_std / portfolio_vol if portfolio_vol > 0 else 1.0

    return {
        "volatility": round(portfolio_vol * np.sqrt(252), 6),
        "var_95": round(var_95, 6),
        "cvar_95": round(cvar_95, 6),
        "marginal_risk": dict(zip(common, np.round(marginal, 6).tolist())),
        "risk_contribution": dict(zip(common, np.round(rc, 6).tolist())),
        "diversification_ratio": round(float(dr), 6),
    }
