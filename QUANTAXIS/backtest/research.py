"""
Research utilities for systematic quant strategy development.

Provides:
  - Factor IC (Information Coefficient) analysis: Rank IC, Normal IC, IC decay
  - Factor combination optimization: maximize ICIR
  - Walk-forward cross-validation
  - Parameter grid search
  - Experiment tracking

Compared to 幻方量化's research platform, this is a local Python equivalent
for the open-source stack.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd


# ── Factor IC Analysis ──────────────────────────────────────────────────────


@dataclass(slots=True)
class FactorICReport:
    factor: str
    rank_ic_mean: float
    rank_ic_std: float
    rank_ic_sharpe: float
    normal_ic_mean: float
    normal_ic_std: float
    normal_ic_sharpe: float
    ic_decay: list[float]  # IC at lag 1,2,3,5,10
    q1_return: float  # top quintile forward return
    q5_return: float  # bottom quintile forward return
    spread_return: float  # Q1 - Q5
    turnover: float  # factor rank turnover
    p_value: float


def compute_rank_ic(
    factor: pd.Series,
    forward_returns: pd.Series,
) -> float:
    """Spearman rank correlation between factor value and forward return."""
    clean = pd.concat([factor, forward_returns], axis=1).dropna()
    if len(clean) < 10:
        return 0.0
    return float(clean.iloc[:, 0].corr(clean.iloc[:, 1], method="spearman"))


def compute_normal_ic(
    factor: pd.Series,
    forward_returns: pd.Series,
) -> float:
    """Pearson correlation between factor value and forward return."""
    clean = pd.concat([factor, forward_returns], axis=1).dropna()
    if len(clean) < 10:
        return 0.0
    return float(clean.iloc[:, 0].corr(clean.iloc[:, 1], method="pearson"))


def factor_ic_analysis(
    factor_values: pd.Series,
    forward_returns: pd.Series,
    factor_name: str = "factor",
    n_lags: int = 5,
) -> FactorICReport:
    """
    Full factor IC analysis.

    Args:
        factor_values: Series of factor values indexed by time/symbol.
        forward_returns: Series of forward returns.
        factor_name: Name for reporting.
        n_lags: Number of lags for IC decay.

    Returns:
        FactorICReport with all metrics.
    """
    clean = pd.concat([factor_values, forward_returns], axis=1).dropna()
    clean.columns = ["factor", "forward_return"]

    if len(clean) < 20:
        return FactorICReport(
            factor=factor_name,
            rank_ic_mean=0.0,
            rank_ic_std=0.0,
            rank_ic_sharpe=0.0,
            normal_ic_mean=0.0,
            normal_ic_std=0.0,
            normal_ic_sharpe=0.0,
            ic_decay=[0.0] * n_lags,
            q1_return=0.0,
            q5_return=0.0,
            spread_return=0.0,
            turnover=1.0,
            p_value=1.0,
        )

    # Rank IC
    rank_ic = clean.groupby(level=0).apply(
        lambda g: compute_rank_ic(g["factor"], g["forward_return"])
    ) if isinstance(clean.index, pd.MultiIndex) else pd.Series(
        [compute_rank_ic(clean["factor"], clean["forward_return"])]
    )

    # Normal IC
    normal_ic = clean.groupby(level=0).apply(
        lambda g: compute_normal_ic(g["factor"], g["forward_return"])
    ) if isinstance(clean.index, pd.MultiIndex) else pd.Series(
        [compute_normal_ic(clean["factor"], clean["forward_return"])]
    )

    # Quintile analysis
    if isinstance(clean.index, pd.MultiIndex):
        # Cross-sectional: rank within each time period
        clean["rank"] = clean.groupby(level=0)["factor"].rank(pct=True)
        clean["quintile"] = pd.qcut(clean["rank"], 5, labels=False, duplicates="drop")
    else:
        clean["quintile"] = pd.qcut(clean["factor"], 5, labels=False, duplicates="drop")

    quintile_returns = clean.groupby("quintile")["forward_return"].mean()
    q1 = float(quintile_returns.get(4, 0.0))  # top quintile
    q5 = float(quintile_returns.get(0, 0.0))  # bottom quintile

    # IC decay
    decay: list[float] = []
    for lag in [1, 2, 3, 5, 10]:
        if lag >= len(clean):
            decay.append(0.0)
            continue
        decay.append(float(
            clean["factor"].shift(lag).corr(clean["forward_return"], method="spearman")
        ))

    # Factor turnover (rank correlation stability)
    if isinstance(clean.index, pd.MultiIndex):
        dates = clean.index.get_level_values(0).unique()
        if len(dates) > 1:
            rank_changes = []
            for i in range(1, len(dates)):
                prev = clean.loc[dates[i - 1]]["factor"].rank()
                curr = clean.loc[dates[i]]["factor"].rank()
                common = prev.index.intersection(curr.index)
                if len(common) > 1:
                    rank_changes.append(prev[common].corr(curr[common], method="spearman"))
            turnover = 1.0 - np.mean(rank_changes) if rank_changes else 1.0
        else:
            turnover = 1.0
    else:
        turnover = 1.0

    return FactorICReport(
        factor=factor_name,
        rank_ic_mean=float(np.mean(rank_ic)) if len(rank_ic) > 0 else 0.0,
        rank_ic_std=float(np.std(rank_ic)) if len(rank_ic) > 0 else 0.0,
        rank_ic_sharpe=float(np.mean(rank_ic) / max(np.std(rank_ic), 1e-6)) if len(rank_ic) > 1 else 0.0,
        normal_ic_mean=float(np.mean(normal_ic)) if len(normal_ic) > 0 else 0.0,
        normal_ic_std=float(np.std(normal_ic)) if len(normal_ic) > 0 else 0.0,
        normal_ic_sharpe=float(np.mean(normal_ic) / max(np.std(normal_ic), 1e-6)) if len(normal_ic) > 1 else 0.0,
        ic_decay=decay,
        q1_return=q1,
        q5_return=q5,
        spread_return=q1 - q5,
        turnover=float(turnover),
        p_value=float(np.mean(rank_ic) / max(np.std(rank_ic) / np.sqrt(max(len(rank_ic), 1)), 1e-6)) if len(rank_ic) > 1 else 1.0,
    )


def combine_factors_icir(
    factor_dict: dict[str, pd.DataFrame],
    forward_returns: pd.Series,
    top_n: int = 10,
) -> dict[str, Any]:
    """
    Combine multiple factors using ICIR (IC/IC_std) weighted average.

    Args:
        factor_dict: {factor_name: DataFrame with columns [date, symbol, value]}
        forward_returns: Series with multi-index (date, symbol)
        top_n: Only use top N factors by ICIR

    Returns:
        dict with combined_factor, weights, individual_performance
    """
    performances: list[tuple[str, float, pd.Series]] = []

    for name, frame in factor_dict.items():
        # Align factor with forward returns
        if "value" not in frame.columns:
            continue
        ic = compute_rank_ic(frame["value"], forward_returns)
        performances.append((name, ic, frame["value"]))

    # Sort by IC and pick top N
    performances.sort(key=lambda x: abs(x[1]), reverse=True)
    top = performances[:top_n]

    if not top:
        return {"combined": pd.Series(dtype=float), "weights": {}, "performance": []}

    # Weight by IC
    total_ic = sum(abs(ic) for _, ic, _ in top)
    if total_ic <= 0:
        return {"combined": pd.Series(dtype=float), "weights": {}, "performance": []}

    weights = {name: abs(ic) / total_ic for name, ic, _ in top}
    combined = sum(weight * values * np.sign(ic)
                   for (name, ic, values), weight in zip(top, weights.values()))

    return {
        "combined": combined,
        "weights": weights,
        "performance": [
            {"factor": name, "ic": ic, "weight": weights[name]}
            for name, ic, _ in top
        ],
    }


# ── Walk-forward Analysis ───────────────────────────────────────────────────


@dataclass(slots=True)
class WalkForwardResult:
    window: int
    step: int
    folds: list[dict[str, Any]]
    oos_sharpe_mean: float
    oos_sharpe_std: float
    oos_return_mean: float
    parameter_stability: dict[str, float]  # param_name -> std across folds


def walk_forward(
    data: pd.DataFrame,
    strategy_builder: Callable[[pd.DataFrame], Any],
    backtest_fn: Callable[..., Any],
    window: int = 252,
    step: int = 63,
    min_train: int = 126,
    date_col: str = "datetime",
    **backtest_kwargs: Any,
) -> WalkForwardResult:
    """
    Walk-forward backtest: train on expanding/rolling window, test on out-of-sample.

    Args:
        data: Full OHLCV dataset.
        strategy_builder: Callable(data) -> strategy instance.
        backtest_fn: run_backtest function.
        window: Training window size (bars).
        step: Test period size (bars).
        min_train: Minimum training bars.
        date_col: Name of datetime column.
        **backtest_kwargs: Additional kwargs to backtest_fn.

    Returns:
        WalkForwardResult with fold-by-fold results.
    """
    df = data.copy().sort_values(date_col).reset_index(drop=True)
    n = len(df)
    folds: list[dict[str, Any]] = []
    all_params: list[dict[str, float]] = []

    for start in range(min_train, n - step, step):
        train_end = start + window if start + window < n else n - step
        train = df.iloc[max(0, train_end - window):train_end]
        test = df.iloc[train_end:min(train_end + step, n)]

        if len(train) < min_train or len(test) < step // 2:
            break

        strategy = strategy_builder(train)
        result = backtest_fn(test, strategy, **backtest_kwargs)

        fold_result = {
            "train_start": str(train[date_col].iloc[0]),
            "train_end": str(train[date_col].iloc[-1]),
            "test_start": str(test[date_col].iloc[0]),
            "test_end": str(test[date_col].iloc[-1]),
            "sharpe": result.sharpe,
            "return": result.total_return,
            "max_dd": result.max_drawdown,
            "trades": result.trades,
        }
        folds.append(fold_result)
        all_params.append(asdict(strategy.config) if hasattr(strategy, "config") else {})

    if not folds:
        return WalkForwardResult(
            window=window, step=step, folds=[],
            oos_sharpe_mean=0.0, oos_sharpe_std=0.0,
            oos_return_mean=0.0, parameter_stability={},
        )

    sharpe_values = [f["sharpe"] for f in folds]
    return_values = [f["return"] for f in folds]

    param_stability: dict[str, float] = {}
    if all_params:
        for key in all_params[0]:
            values = [p.get(key, 0.0) for p in all_params if isinstance(p.get(key), (int, float))]
            param_stability[key] = float(np.std(values)) if len(values) > 1 else 0.0

    return WalkForwardResult(
        window=window,
        step=step,
        folds=folds,
        oos_sharpe_mean=float(np.mean(sharpe_values)),
        oos_sharpe_std=float(np.std(sharpe_values)),
        oos_return_mean=float(np.mean(return_values)),
        parameter_stability=param_stability,
    )


# ── Parameter Grid Search ────────────────────────────────────────────────────


@dataclass(slots=True)
class GridSearchResult:
    best_params: dict[str, Any]
    best_score: float
    all_results: list[dict[str, Any]]
    param_importance: dict[str, float]


def grid_search(
    data: pd.DataFrame,
    strategy_class: type,
    param_grid: dict[str, list[Any]],
    backtest_fn: Callable[..., Any],
    scoring: str = "sharpe",
    **backtest_kwargs: Any,
) -> GridSearchResult:
    """
    Exhaustive grid search over strategy parameters.

    Args:
        data: OHLCV DataFrame.
        strategy_class: Strategy class (takes config as first arg).
        param_grid: {param_name: [values]}.
        backtest_fn: run_backtest function.
        scoring: Metric to maximize. Can be "sharpe", "return", "calmar", "sortino".
        **backtest_kwargs: Passed to backtest_fn.

    Returns:
        GridSearchResult with best params and full results.
    """
    import itertools

    keys = list(param_grid.keys())
    values = list(param_grid.values())
    all_results: list[dict[str, Any]] = []

    for combo in itertools.product(*values):
        params = dict(zip(keys, combo))
        # Build config with overrides
        from QUANTAXIS.backtest.strategy import StrategyConfig
        config = StrategyConfig(**{k: v for k, v in params.items() if hasattr(StrategyConfig, k)})
        strategy = strategy_class(config)
        result = backtest_fn(data, strategy, **backtest_kwargs)

        score_map = {
            "sharpe": result.sharpe,
            "return": result.total_return,
            "calmar": result.calmar,
        }
        score = score_map.get(scoring, result.sharpe)

        entry: dict[str, Any] = {"params": params, "score": score}
        entry.update(asdict(result))
        all_results.append(entry)

    if not all_results:
        return GridSearchResult(best_params={}, best_score=0.0, all_results=[], param_importance={})

    # Find best
    best = max(all_results, key=lambda x: x["score"])

    # Compute parameter importance (score variance attributable to each param)
    param_importance: dict[str, float] = {}
    for key in keys:
        grouped = {}
        for entry in all_results:
            param_val = str(entry["params"].get(key, "none"))
            grouped.setdefault(param_val, []).append(entry["score"])
        # Variance of group means
        group_means = [np.mean(scores) for scores in grouped.values()]
        param_importance[key] = float(np.std(group_means)) if len(group_means) > 1 else 0.0

    total = sum(param_importance.values())
    if total > 0:
        param_importance = {k: v / total for k, v in param_importance.items()}

    return GridSearchResult(
        best_params=best["params"],
        best_score=float(best["score"]),
        all_results=all_results,
        param_importance=param_importance,
    )


# ── Experiment Tracking ────────────────────────────────────────────────────


@dataclass
class Experiment:
    name: str
    timestamp: str = ""
    parameters: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)
    description: str = ""


_experiment_dir: Path | None = None


def set_experiment_dir(path: str | Path) -> None:
    """Set directory for experiment logging."""
    global _experiment_dir
    _experiment_dir = Path(path)
    _experiment_dir.mkdir(parents=True, exist_ok=True)


def log_experiment(
    name: str,
    parameters: dict[str, Any] | None = None,
    metrics: dict[str, float] | None = None,
    tags: list[str] | None = None,
    description: str = "",
) -> Experiment:
    """Log an experiment to disk."""
    exp = Experiment(
        name=name,
        timestamp=datetime.now().isoformat(timespec="seconds"),
        parameters=parameters or {},
        metrics=metrics or {},
        tags=tags or [],
        description=description,
    )

    if _experiment_dir is not None:
        safe_name = name.replace("/", "_").replace(" ", "_")
        path = _experiment_dir / f"{safe_name}_{exp.timestamp}.json"
        with open(path, "w") as f:
            json.dump(asdict(exp), f, indent=2, ensure_ascii=False)

    return exp


def list_experiments() -> list[Experiment]:
    """List all logged experiments."""
    if _experiment_dir is None or not _experiment_dir.exists():
        return []
    experiments: list[Experiment] = []
    for path in sorted(_experiment_dir.glob("*.json")):
        try:
            with open(path) as f:
                data = json.load(f)
                experiments.append(Experiment(**data))
        except (json.JSONDecodeError, KeyError):
            continue
    return experiments
