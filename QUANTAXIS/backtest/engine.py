from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, replace
from datetime import datetime

import numpy as np
import pandas as pd

from QUANTAXIS.backtest.market_rules import (
    MarketContext,
    build_market_contexts,
    can_trade,
    infer_market_segment,
)
from QUANTAXIS.backtest.portfolio import PortfolioConfig, compute_portfolio_risk, optimize_portfolio
from QUANTAXIS.backtest.risk import RiskChecker, RiskConfig, generate_risk_report
from QUANTAXIS.backtest.strategy import RecursiveQTransformerStrategy

logger = logging.getLogger("quantaxis.backtest")


@dataclass(slots=True)
class BacktestResult:
    bars: int
    trades: int
    rejected: int
    final_equity: float
    total_return: float
    annual_return: float
    max_drawdown: float
    sharpe: float
    calmar: float
    equity_curve: list[dict[str, float | str]]
    trades_log: list[dict[str, float | str]]
    rejected_log: list[dict[str, float | str]]

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def _annualize(total_return: float, bars: int, bars_per_year: int) -> float:
    if bars <= 1:
        return 0.0
    gross = 1.0 + total_return
    if gross <= 0:
        return -1.0
    return gross ** (bars_per_year / bars) - 1.0


def _max_drawdown(equity_series: pd.Series) -> float:
    if equity_series.empty:
        return 0.0
    peak = equity_series.cummax()
    drawdown = equity_series / peak - 1.0
    return float(drawdown.min())


def _sharpe(returns: pd.Series, bars_per_year: int) -> float:
    clean = returns.dropna()
    if clean.empty or clean.std() == 0:
        return 0.0
    return float((clean.mean() / clean.std()) * (bars_per_year ** 0.5))


def _calmar(annual_return: float, max_drawdown: float) -> float:
    if max_drawdown >= 0 or max_drawdown == 0:
        return 0.0
    return float(annual_return / abs(max_drawdown))


def _softmax(values: list[float], temperature: float) -> list[float]:
    if not values:
        return []
    scaled = pd.Series(values, dtype=float) * max(temperature, 1e-6)
    scaled = (scaled - scaled.max()).clip(lower=-50.0, upper=50.0)
    expv = scaled.map(np.exp)
    total = float(expv.sum())
    if total <= 0:
        return [0.0 for _ in values]
    return [float(item / total) for item in expv]


def _cap_and_normalize(weights: dict[str, float], cap: float, gross_exposure: float) -> dict[str, float]:
    if not weights:
        return {}
    bounded = {symbol: min(max(weight, 0.0), cap) for symbol, weight in weights.items()}
    total = sum(bounded.values())
    if total <= 0:
        return {symbol: 0.0 for symbol in bounded}
    target_gross = min(max(gross_exposure, 0.0), 1.0)
    scale = min(target_gross / total, 1.0)
    return {symbol: value * scale for symbol, value in bounded.items()}


def _zscore_map(values: dict[str, float]) -> dict[str, float]:
    if not values:
        return {}
    series = pd.Series(values, dtype=float)
    std = float(series.std(ddof=0))
    if std <= 1e-9:
        return {key: 0.0 for key in values}
    centered = (series - float(series.mean())) / std
    return {str(key): float(value) for key, value in centered.items()}


def _is_in_trade_window(ts: pd.Timestamp | datetime, windows: tuple[tuple[object, object], ...]) -> bool:
    if not windows:
        return True
    current = ts.time()
    return any(start <= current <= end for start, end in windows)


def _compute_market_regime(
    base_signals: dict[str, float],
    event_signals: dict[str, float],
    event_active: dict[str, bool],
    volume_scores: dict[str, float],
    config,
) -> tuple[bool, float]:
    if not base_signals:
        return False, 0.0
    breadth = sum(1 for value in base_signals.values() if value > 0) / len(base_signals)
    avg_event = float(pd.Series(event_signals, dtype=float).clip(lower=0.0).mean())
    avg_volume = float(pd.Series(volume_scores, dtype=float).clip(lower=0.0).mean())
    active_ratio = sum(1 for value in event_active.values() if value) / len(event_active)
    regime_score = (
        max(breadth - config.market_breadth_threshold, 0.0) * 0.7
        + max(avg_event - config.market_event_threshold, 0.0) * 0.9
        + max(avg_volume - config.market_volume_threshold, 0.0) * 0.5
        + active_ratio * 0.15
    )
    regime_on = (
        breadth >= config.market_breadth_threshold
        and avg_event >= config.market_event_threshold
        and avg_volume >= config.market_volume_threshold
        and active_ratio >= 0.2
        and regime_score >= config.market_regime_score_threshold
    )
    return regime_on, float(regime_score)


# ── Market Impact Models ─────────────────────────────────────────────────────

@dataclass(slots=True)
class ImpactConfig:
    """Market impact parameters."""

    model: str = "none"  # "square_root", "linear", "almgren_chriss", "none"
    permanent_impact: float = 0.0001  # permanent impact coefficient
    temporary_impact: float = 0.0002  # temporary impact coefficient
    # Empirical square-root impact: I = impact_coeff * sigma * sqrt(Q/V)
    impact_coefficient: float = 0.1
    # Almgren-Chriss parameters
    ac_eta: float = 0.1
    ac_gamma: float = 0.05
    # Minimum participation rate for viability check
    max_participation: float = 0.15


def estimate_market_impact(
    trade_value: float,
    volume: float,
    price: float,
    volatility: float,
    config: ImpactConfig | None = None,
) -> float:
    """Estimate market impact cost (in price units) for a trade.

    Returns the per-share impact cost to add to the base price (as a spread).
    """
    cfg = config or ImpactConfig()

    if cfg.model == "none" or volume <= 0 or trade_value <= 0:
        return 0.0

    participation = min(trade_value / max(volume * price, 1.0), 1.0)
    sqrt_qv = np.sqrt(participation / max(cfg.max_participation, 0.01))

    if cfg.model == "square_root":
        impact = cfg.impact_coefficient * volatility * sqrt_qv
    elif cfg.model == "linear":
        impact = cfg.impact_coefficient * volatility * participation
    elif cfg.model == "almgren_chriss":
        # Permanent + temporary impact
        sigma = volatility * price
        perm = cfg.ac_gamma * sigma
        temp = cfg.ac_eta * sigma * np.sqrt(participation / 0.01)
        impact = (perm + temp) / price
    else:
        impact = 0.0

    return float(np.clip(impact * max(cfg.max_participation, 0.01) * price, 0.0, price * 0.05))


def _walk_forward_impact_adjustment(
    bar_count: int,
    total_bars: int,
    impact_config: ImpactConfig,
) -> float:
    """Decay impact for small orders as the algo walks the book."""
    if total_bars <= 1:
        return impact_config.temporary_impact * 0.5
    # Over multiple bars, average impact decreases
    return impact_config.temporary_impact * (0.5 + 0.5 / max(bar_count, 1))


# ── Main Backtest ────────────────────────────────────────────────────────────


def run_backtest(
    data: pd.DataFrame,
    strategy: RecursiveQTransformerStrategy,
    initial_cash: float = 1_000_000,
    commission_rate: float = 0.0003,
    stamp_duty_rate: float = 0.001,
    slippage_model: str = "fixed",
    slippage_value: float = 0.0,
    bars_per_year: int = 252,
    portfolio_size: int = 3,
    impact_config: ImpactConfig | None = None,
    portfolio_config: PortfolioConfig | None = None,
    execution_mode: str = "research",
) -> BacktestResult:
    if execution_mode not in {"research", "paper", "paper_strict"}:
        raise ValueError("execution_mode must be one of: research, paper, paper_strict")
    required = {"datetime", "open", "high", "low", "close", "volume"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"missing required columns: {sorted(missing)}")
    if portfolio_size < 1:
        raise ValueError("portfolio_size must be at least 1")

    df = data.copy().reset_index(drop=True)
    symbol_column = "symbol" if "symbol" in df.columns else "code" if "code" in df.columns else None
    if symbol_column is None:
        df["symbol"] = "SINGLE"
        symbol_column = "symbol"
    df[symbol_column] = df[symbol_column].astype(str)
    df["datetime"] = pd.to_datetime(df["datetime"])

    if execution_mode in {"paper", "paper_strict"}:
        if impact_config is None:
            impact_config = ImpactConfig(model="square_root", impact_coefficient=0.03)
        if slippage_model == "fixed" and slippage_value == 0.0:
            slippage_model = "percent"
            slippage_value = 0.0005
    elif impact_config is None:
        impact_config = ImpactConfig(model="none")

    logger.info(
        "Backtest start | symbols=%d cash=%.0f portfolio_size=%d slippage=%s/%.4f impact=%s",
        df[symbol_column].nunique(), initial_cash, portfolio_size,
        slippage_model, slippage_value, impact_config.model,
    )

    cash = float(initial_cash)
    shares: dict[str, int] = {symbol: 0 for symbol in df[symbol_column].unique().tolist()}
    holding_bars: dict[str, int] = {symbol: 0 for symbol in shares}
    # T+1 tracking: symbol -> list of (buy_dt, quantity) still locked
    t1_locks: dict[str, list[tuple[pd.Timestamp, int]]] = {symbol: [] for symbol in shares}
    strategy_state: dict[str, RecursiveQTransformerStrategy] = {
        symbol: RecursiveQTransformerStrategy(replace(strategy.config))
        for symbol in shares
    }
    equity_curve: list[dict[str, float | str]] = []
    trades_log: list[dict[str, float | str]] = []
    rejected_log: list[dict[str, float | str]] = []
    # Risk accumulator
    daily_pnl: dict[str, float] = {}

    symbol_frames = {
        symbol: frame.reset_index(drop=True)
        for symbol, frame in df.groupby(symbol_column, sort=False)
    }
    score_frames = {
        symbol: strategy_state[symbol].score_frame(frame).reset_index(drop=True)
        for symbol, frame in symbol_frames.items()
    }
    symbol_indices = {symbol: -1 for symbol in symbol_frames}
    last_prices: dict[str, float] = {}
    grouped = df.groupby("datetime", sort=True)
    prev_snapshot: pd.DataFrame | None = None
    for dt, snapshot in grouped:
        use_execution_rules = execution_mode in {"paper", "paper_strict"}

        # Clear expired T+1 locks (previous trading day purchases are now sellable).
        # Research mode does not apply T+1 so alpha tests remain comparable.
        current_dt = pd.Timestamp(dt)
        if use_execution_rules:
            for symbol in t1_locks:
                t1_locks[symbol] = [
                    (buy_dt, qty) for buy_dt, qty in t1_locks[symbol] if buy_dt >= current_dt
                ]
        trading_open = _is_in_trade_window(pd.Timestamp(dt), strategy.config.trade_windows)
        signals: dict[str, float] = {}
        base_signals: dict[str, float] = {}
        event_signals: dict[str, float] = {}
        event_active: dict[str, bool] = {}
        volume_scores: dict[str, float] = {}
        alpha_scores: dict[str, float] = {}
        alpha_physical_scores: dict[str, float] = {}
        buy_sell_scores: dict[str, float] = {}
        pivot_leg_scores: dict[str, float] = {}
        trade_point_scores: dict[str, float] = {}
        groups: dict[str, str] = {}
        prices: dict[str, float] = {}
        for symbol, row in snapshot.groupby(symbol_column):
            symbol_indices[symbol] += len(row)
            row_now = row.iloc[-1]
            score = score_frames[symbol].iloc[symbol_indices[symbol]].to_dict()
            signals[symbol] = float(score["signal"])
            base_signals[symbol] = float(score.get("base_signal", score["signal"]))
            event_signals[symbol] = float(score.get("event_signal", score["signal"]))
            event_active[symbol] = bool(score.get("event_active", 1.0))
            volume_scores[symbol] = float(score.get("volume_score", 0.0))
            alpha_scores[symbol] = float(score.get("alpha_score", 0.0))
            alpha_physical_scores[symbol] = float(score.get("alpha_physical", 0.0))
            buy_sell_scores[symbol] = float(score.get("buy_sell_score", 0.0))
            pivot_leg_scores[symbol] = float(score.get("pivot_leg_score", 0.0))
            trade_point_scores[symbol] = float(score.get("trade_point_score", 0.0))
            groups[symbol] = strategy.config.symbol_groups.get(symbol, "default")
            prices[symbol] = float(row_now["close"])
        last_prices.update(prices)

        regime_on, regime_score = _compute_market_regime(
            base_signals=base_signals,
            event_signals=event_signals,
            event_active=event_active,
            volume_scores=volume_scores,
            config=strategy.config,
        )
        rank_scores: dict[str, float] = {}
        group_scores: dict[str, float] = {}
        group_active: dict[str, bool] = {}
        signal_cross_section = _zscore_map(signals)
        base_cross_section = _zscore_map(base_signals)
        event_cross_section = _zscore_map(event_signals)
        alpha_cross_section = _zscore_map(alpha_scores)
        alpha_physical_cross = _zscore_map(alpha_physical_scores)
        buy_sell_cross = _zscore_map(buy_sell_scores)
        pivot_leg_cross = _zscore_map(pivot_leg_scores)
        trade_point_cross = _zscore_map(trade_point_scores)
        for group in sorted(set(groups.values())):
            members = [symbol for symbol, member_group in groups.items() if member_group == group]
            group_signals = {symbol: event_signals[symbol] for symbol in members}
            group_zscores = _zscore_map(group_signals)
            group_alpha_zscores = _zscore_map({symbol: alpha_scores[symbol] for symbol in members})
            group_alpha_physical = _zscore_map({symbol: alpha_physical_scores[symbol] for symbol in members})
            group_buy_sell = _zscore_map({symbol: buy_sell_scores[symbol] for symbol in members})
            group_pivot_leg = _zscore_map({symbol: pivot_leg_scores[symbol] for symbol in members})
            group_trade_point = _zscore_map({symbol: trade_point_scores[symbol] for symbol in members})
            event_mean = float(pd.Series(group_signals, dtype=float).clip(lower=0.0).mean()) if group_signals else 0.0
            base_mean = float(pd.Series({symbol: base_signals[symbol] for symbol in members}, dtype=float).mean()) if members else 0.0
            alpha_mean = float(pd.Series({symbol: alpha_scores[symbol] for symbol in members}, dtype=float).clip(lower=0.0).mean()) if members else 0.0
            structure_mean = float(pd.Series({symbol: buy_sell_scores[symbol] + pivot_leg_scores[symbol] + trade_point_scores[symbol] for symbol in members}, dtype=float).clip(lower=0.0).mean()) if members else 0.0
            group_score = event_mean * 0.45 + max(base_mean, 0.0) * 0.18 + alpha_mean * 0.27 + structure_mean * 0.25
            group_scores[group] = group_score
            group_active[group] = regime_on and group_score >= strategy.config.group_activation_threshold
            for symbol in members:
                event_intensity = max(event_signals[symbol], 0.0) + max(base_signals[symbol], 0.0) * 0.25
                alpha_cross = (
                    group_alpha_zscores.get(symbol, 0.0) * 0.40
                    + group_alpha_physical.get(symbol, 0.0) * 0.25
                    + alpha_cross_section.get(symbol, 0.0) * 0.15
                    + alpha_physical_cross.get(symbol, 0.0) * 0.10
                )
                structure_cross = (
                    group_buy_sell.get(symbol, 0.0) * 0.08
                    + group_pivot_leg.get(symbol, 0.0) * 0.06
                    + group_trade_point.get(symbol, 0.0) * 0.08
                    + buy_sell_cross.get(symbol, 0.0) * 0.06
                    + pivot_leg_cross.get(symbol, 0.0) * 0.04
                    + trade_point_cross.get(symbol, 0.0) * 0.06
                )
                rank_scores[symbol] = (
                    signal_cross_section.get(symbol, 0.0) * 0.40
                    + event_cross_section.get(symbol, 0.0) * 0.30
                    + base_cross_section.get(symbol, 0.0) * 0.15
                    + group_zscores.get(symbol, 0.0) * 0.15
                    + base_signals[symbol] * 0.10
                    + event_intensity * 0.15
                    + alpha_cross
                    + structure_cross
                )

        ranked = sorted(rank_scores.items(), key=lambda item: item[1], reverse=True)
        held_candidates = [
            (symbol, rank_score)
            for symbol, rank_score in ranked
            if shares[symbol] > 0
            and (
                (regime_on and base_signals[symbol] > strategy.config.hold_threshold)
                or (not strategy.config.exit_on_regime_off and base_signals[symbol] > strategy.config.hold_threshold)
            )
        ]
        fresh_candidates = [
            (symbol, rank_score)
            for symbol, rank_score in ranked
            if shares[symbol] <= 0
            and regime_on
            and group_active.get(groups[symbol], False)
            and rank_score > strategy.config.buy_threshold
            and (event_active[symbol] or not strategy.config.require_event_for_entry)
        ]
        selected: list[tuple[str, float]] = []
        group_counts: dict[str, int] = {}
        for item in held_candidates + fresh_candidates:
            if item[0] in {symbol for symbol, _ in selected}:
                continue
            group = groups[item[0]]
            if shares[item[0]] <= 0 and group_counts.get(group, 0) >= strategy.config.per_group_limit:
                continue
            selected.append(item)
            group_counts[group] = group_counts.get(group, 0) + 1
            if len(selected) >= portfolio_size:
                break
        long_candidates = selected
        # ── Portfolio Optimization ──────────────────────────────────────
        # Keep the historical softmax allocator as the default. Portfolio
        # optimizers are opt-in because signal deltas are not price returns and
        # can over-allocate risk if treated as an asset return history.
        opt_returns: dict[str, pd.Series] = {}
        if portfolio_config is not None:
            for symbol in list(dict(long_candidates).keys()):
                idx = symbol_indices.get(symbol, -1)
                if idx >= 5:
                    history = symbol_frames[symbol]["close"].astype(float).iloc[max(0, idx - 251) : idx + 1]
                    opt_returns[symbol] = history.pct_change().fillna(0.0).reset_index(drop=True)

        if portfolio_config is not None and opt_returns and len(opt_returns) >= 2:
            opt_returns_df = pd.DataFrame(opt_returns)
            opt_cfg = replace(
                portfolio_config,
                max_weight=min(portfolio_config.max_weight, strategy.config.max_position_weight),
                gross_exposure=min(
                    portfolio_config.gross_exposure,
                    strategy.config.gross_exposure if regime_on else min(strategy.config.gross_exposure, 0.2),
                ),
            )
            # Use signal scores as views for Black-Litterman
            opt_views = {
                symbol: max(base_signals[symbol], 0.0) * 0.5 + max(alpha_scores[symbol], 0.0) * 0.3
                for symbol in list(dict(long_candidates).keys())
            }
            target_weights = optimize_portfolio(
                opt_returns_df,
                config=opt_cfg,
                views=opt_views,
                group_map=groups,
            )
        else:
            # Fallback: use original softmax
            long_weights_raw = _softmax(
                [signal - strategy.config.buy_threshold for _, signal in long_candidates],
                strategy.config.rank_temperature,
            )
            raw_target_weights = {}
            for (symbol, rank_score), weight in zip(long_candidates, long_weights_raw):
                if shares[symbol] > 0 and not regime_on:
                    raw_target_weights[symbol] = min(weight, strategy.config.min_target_weight)
                    continue
                event_strength = max(event_signals[symbol], 0.0) + max(regime_score, 0.0) * 0.25
                group_strength = max(group_scores.get(groups[symbol], 0.0), 0.0)
                signal_scale = strategy.config.min_signal_target_scale if rank_score > strategy.config.buy_threshold else 0.0
                scaled_weight = weight * min(1.0, max(event_strength + group_strength, signal_scale))
                raw_target_weights[symbol] = max(scaled_weight, 0.0)
            target_weights = _cap_and_normalize(
                raw_target_weights,
                cap=strategy.config.max_position_weight,
                gross_exposure=strategy.config.gross_exposure if regime_on else min(strategy.config.gross_exposure, 0.2),
            )

        mark_prices = {symbol: prices.get(symbol, last_prices.get(symbol, 0.0)) for symbol in shares}
        equity_before = cash + sum(shares[symbol] * mark_prices[symbol] for symbol in shares)
        target_shares = {}
        for symbol in shares:
            if symbol not in prices:
                target_shares[symbol] = shares[symbol]
                continue
            weight = target_weights.get(symbol, 0.0)
            if not regime_on and strategy.config.exit_on_regime_off:
                weight = 0.0
            target_value = equity_before * weight
            raw_shares = int(target_value / prices[symbol] / 100) * 100 if prices[symbol] > 0 else 0
            current_value = shares[symbol] * prices[symbol]
            if (
                regime_on
                and shares[symbol] > 0
                and holding_bars[symbol] < strategy.config.min_holding_bars
                and base_signals[symbol] > strategy.config.hold_threshold
            ):
                raw_shares = max(raw_shares, shares[symbol])
            if equity_before > 0 and abs(target_value - current_value) / equity_before < strategy.config.rebalance_buffer:
                raw_shares = shares[symbol]
            target_shares[symbol] = max(raw_shares, 0)

        impact_cfg = impact_config
        market_contexts = build_market_contexts(snapshot, prev_snapshot) if use_execution_rules else {}
        prev_snapshot = snapshot.copy()

        if trading_open:
            # Prepare impact estimates for each symbol
            impact_costs: dict[str, float] = {}
            for symbol in shares:
                if symbol not in prices:
                    continue
                vol = float(snapshot[snapshot[symbol_column] == symbol]["volume"].iloc[-1]) if symbol in snapshot[symbol_column].values else 0
                ctx = market_contexts.get(symbol)
                if ctx is not None and vol > 0:
                    impact = estimate_market_impact(
                        trade_value=abs(target_shares.get(symbol, 0) - shares[symbol]) * prices[symbol],
                        volume=vol,
                        price=prices[symbol],
                        volatility=float(score_frames[symbol]["volatility"].iloc[-1]) if "volatility" in score_frames[symbol].columns else 0.01,
                        config=impact_cfg,
                    )
                    impact_costs[symbol] = impact
            # ---- SELL phase ----
            for symbol, current_shares in shares.items():
                if symbol not in prices:
                    continue
                desired_shares = target_shares[symbol]
                delta = desired_shares - current_shares
                if delta >= 0:
                    continue
                sell_size = abs(delta)

                if use_execution_rules:
                    # T+1 check: cannot sell shares bought today
                    t1_locked = sum(qty for buy_dt, qty in t1_locks.get(symbol, []) if buy_dt == pd.Timestamp(dt))
                    sellable = max(current_shares - t1_locked, 0)
                    sell_size = min(sell_size, sellable)
                    if sell_size <= 0:
                        rejected_log.append(
                            {
                                "datetime": str(dt),
                                "symbol": symbol,
                                "side": "sell",
                                "size": abs(delta),
                                "reason": "t1_lock",
                            }
                        )
                        continue

                ctx = market_contexts.get(symbol)
                price = prices[symbol]
                impact = impact_costs.get(symbol, 0.0)
                signal = rank_scores[symbol]
                if use_execution_rules and ctx is not None:
                    ok, fill_price, reason = can_trade(ctx, "sell", slippage_model=slippage_model, slippage_value=slippage_value)
                    if not ok:
                        rejected_log.append(
                            {
                                "datetime": str(dt),
                                "symbol": symbol,
                                "side": "sell",
                                "size": sell_size,
                                "reason": reason,
                            }
                        )
                        continue
                    # Apply market impact (sell: price reduced by impact)
                    fill_price = fill_price * (1.0 - impact / max(price, 1e-6))
                    price = fill_price

                gross = price * sell_size
                fees = gross * (commission_rate + stamp_duty_rate)
                cash += gross - fees
                shares[symbol] -= sell_size
                if shares[symbol] <= 0:
                    holding_bars[symbol] = 0
                trades_log.append(
                    {
                        "datetime": str(dt),
                        "symbol": symbol,
                        "side": "sell",
                        "price": price,
                        "size": sell_size,
                        "signal": signal,
                        "fees": round(fees, 2),
                        "target_weight": round(target_weights.get(symbol, 0.0), 6),
                    }
                )

            # ---- BUY phase ----
            for symbol, current_shares in shares.items():
                if symbol not in prices:
                    continue
                desired_shares = target_shares[symbol]
                delta = desired_shares - current_shares
                if delta <= 0:
                    continue

                ctx = market_contexts.get(symbol)
                price = prices[symbol]
                impact = impact_costs.get(symbol, 0.0)
                signal = rank_scores[symbol]
                if use_execution_rules and ctx is not None:
                    ok, fill_price, reason = can_trade(ctx, "buy", slippage_model=slippage_model, slippage_value=slippage_value)
                    if not ok:
                        rejected_log.append(
                            {
                                "datetime": str(dt),
                                "symbol": symbol,
                                "side": "buy",
                                "size": delta,
                                "reason": reason,
                            }
                        )
                        continue
                    # Apply market impact (buy: price increased by impact)
                    fill_price = fill_price * (1.0 + impact / max(price, 1e-6))
                    price = fill_price

                gross = price * delta
                fees = gross * commission_rate
                if cash >= gross + fees:
                    cash -= gross + fees
                    shares[symbol] += delta
                    if use_execution_rules:
                        t1_locks[symbol].append((pd.Timestamp(dt), delta))
                    trades_log.append(
                        {
                            "datetime": str(dt),
                            "symbol": symbol,
                            "side": "buy",
                            "price": price,
                            "size": delta,
                            "signal": signal,
                            "fees": round(fees, 2),
                            "target_weight": round(target_weights.get(symbol, 0.0), 6),
                        }
                    )

        for symbol, share_count in shares.items():
            holding_bars[symbol] = holding_bars[symbol] + 1 if share_count > 0 else 0

        mark_to_market = cash + sum(shares[symbol] * mark_prices[symbol] for symbol in shares)
        avg_signal = sum(signals.values()) / len(signals) if signals else 0.0
        avg_rank_signal = sum(rank_scores.values()) / len(rank_scores) if rank_scores else 0.0
        avg_price = sum(prices.values()) / len(prices) if prices else 0.0
        equity_curve.append(
            {
                "datetime": str(dt),
                "close": round(avg_price, 4),
                "signal": round(avg_signal, 6),
                "rank_signal": round(avg_rank_signal, 6),
                "regime_on": 1 if regime_on else 0,
                "regime_score": round(regime_score, 6),
                "position": sum(1 for value in shares.values() if value > 0),
                "equity": round(mark_to_market, 2),
                "active_positions": sum(1 for value in shares.values() if value > 0),
                "trade_window_open": 1 if trading_open else 0,
                "gross_exposure": round(sum(shares[symbol] * mark_prices[symbol] for symbol in shares) / mark_to_market, 6) if mark_to_market else 0.0,
            }
        )

    equity_df = pd.DataFrame(equity_curve)
    returns = equity_df["equity"].pct_change().fillna(0.0)
    final_equity = float(equity_df["equity"].iloc[-1]) if not equity_df.empty else initial_cash
    total_return = final_equity / initial_cash - 1.0
    mdd = _max_drawdown(equity_df["equity"])
    ann_ret = _annualize(total_return, len(equity_df), bars_per_year)

    logger.info(
        "Backtest complete | return=%.2f%% ann=%.2f%% sharpe=%.2f maxdd=%.2f%% trades=%d/%d",
        total_return * 100, ann_ret * 100,
        _sharpe(returns, bars_per_year), mdd * 100,
        len(trades_log), len(rejected_log),
    )

    return BacktestResult(
        bars=len(df),
        trades=len(trades_log),
        rejected=len(rejected_log),
        final_equity=round(final_equity, 2),
        total_return=round(total_return, 6),
        annual_return=round(ann_ret, 6),
        max_drawdown=round(mdd, 6),
        sharpe=round(_sharpe(returns, bars_per_year), 6),
        calmar=round(_calmar(ann_ret, mdd), 6),
        equity_curve=equity_curve,
        trades_log=trades_log,
        rejected_log=rejected_log,
    )
