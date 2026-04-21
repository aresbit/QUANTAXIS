from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from datetime import datetime

import numpy as np
import pandas as pd

from QUANTAXIS.backtest.strategy import RecursiveQTransformerStrategy


@dataclass(slots=True)
class BacktestResult:
    bars: int
    trades: int
    final_equity: float
    total_return: float
    annual_return: float
    max_drawdown: float
    sharpe: float
    equity_curve: list[dict[str, float | str]]
    trades_log: list[dict[str, float | str]]

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


def run_backtest(
    data: pd.DataFrame,
    strategy: RecursiveQTransformerStrategy,
    initial_cash: float = 1_000_000,
    commission_rate: float = 0.0003,
    stamp_duty_rate: float = 0.001,
    bars_per_year: int = 252,
    portfolio_size: int = 3,
) -> BacktestResult:
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

    cash = float(initial_cash)
    shares: dict[str, int] = {symbol: 0 for symbol in df[symbol_column].unique().tolist()}
    holding_bars: dict[str, int] = {symbol: 0 for symbol in shares}
    strategy_state: dict[str, RecursiveQTransformerStrategy] = {
        symbol: RecursiveQTransformerStrategy(replace(strategy.config))
        for symbol in shares
    }
    equity_curve: list[dict[str, float | str]] = []
    trades_log: list[dict[str, float | str]] = []

    symbol_frames = {
        symbol: frame.reset_index(drop=True)
        for symbol, frame in df.groupby(symbol_column, sort=False)
    }
    score_frames = {
        symbol: strategy_state[symbol].score_frame(frame).reset_index(drop=True)
        for symbol, frame in symbol_frames.items()
    }
    symbol_indices = {symbol: -1 for symbol in symbol_frames}
    grouped = df.groupby("datetime", sort=True)
    for dt, snapshot in grouped:
        trading_open = _is_in_trade_window(pd.Timestamp(dt), strategy.config.trade_windows)
        signals: dict[str, float] = {}
        base_signals: dict[str, float] = {}
        event_signals: dict[str, float] = {}
        event_active: dict[str, bool] = {}
        volume_scores: dict[str, float] = {}
        alpha_scores: dict[str, float] = {}
        alpha_physical_scores: dict[str, float] = {}
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
            groups[symbol] = strategy.config.symbol_groups.get(symbol, "default")
            prices[symbol] = float(row_now["close"])

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
        alpha_cross_section = _zscore_map(alpha_scores)
        alpha_physical_cross = _zscore_map(alpha_physical_scores)
        for group in sorted(set(groups.values())):
            members = [symbol for symbol, member_group in groups.items() if member_group == group]
            group_signals = {symbol: event_signals[symbol] for symbol in members}
            group_zscores = _zscore_map(group_signals)
            group_alpha_zscores = _zscore_map({symbol: alpha_scores[symbol] for symbol in members})
            group_alpha_physical = _zscore_map({symbol: alpha_physical_scores[symbol] for symbol in members})
            event_mean = float(pd.Series(group_signals, dtype=float).clip(lower=0.0).mean()) if group_signals else 0.0
            base_mean = float(pd.Series({symbol: base_signals[symbol] for symbol in members}, dtype=float).mean()) if members else 0.0
            alpha_mean = float(pd.Series({symbol: alpha_scores[symbol] for symbol in members}, dtype=float).clip(lower=0.0).mean()) if members else 0.0
            group_score = event_mean * 0.55 + max(base_mean, 0.0) * 0.20 + alpha_mean * 0.35
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
                rank_scores[symbol] = (
                    group_zscores.get(symbol, 0.0) * 0.35
                    + base_signals[symbol] * 0.15
                    + event_intensity * 0.20
                    + alpha_cross
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
        long_weights_raw = _softmax([signal - strategy.config.buy_threshold for _, signal in long_candidates], strategy.config.rank_temperature)
        raw_target_weights = {}
        for (symbol, _), weight in zip(long_candidates, long_weights_raw):
            if shares[symbol] > 0 and not regime_on:
                raw_target_weights[symbol] = min(weight, strategy.config.min_target_weight)
                continue
            event_strength = max(event_signals[symbol], 0.0) + max(regime_score, 0.0) * 0.25
            group_strength = max(group_scores.get(groups[symbol], 0.0), 0.0)
            scaled_weight = weight * min(1.0, event_strength + group_strength)
            raw_target_weights[symbol] = max(scaled_weight, 0.0)
        target_weights = _cap_and_normalize(
            raw_target_weights,
            cap=strategy.config.max_position_weight,
            gross_exposure=strategy.config.gross_exposure if regime_on else min(strategy.config.gross_exposure, 0.2),
        )

        equity_before = cash + sum(shares[symbol] * prices[symbol] for symbol in shares)
        target_shares = {}
        for symbol in shares:
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

        if trading_open:
            for symbol, current_shares in shares.items():
                price = prices[symbol]
                signal = rank_scores[symbol]
                desired_shares = target_shares[symbol]
                delta = desired_shares - current_shares
                if delta == 0:
                    continue
                if delta < 0:
                    sell_size = abs(delta)
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

            for symbol, current_shares in shares.items():
                price = prices[symbol]
                signal = rank_scores[symbol]
                desired_shares = target_shares[symbol]
                delta = desired_shares - current_shares
                if delta <= 0:
                    continue
                gross = price * delta
                fees = gross * commission_rate
                if cash >= gross + fees:
                    cash -= gross + fees
                    shares[symbol] += delta
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

        mark_to_market = cash + sum(shares[symbol] * prices[symbol] for symbol in shares)
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
                "gross_exposure": round(sum(shares[symbol] * prices[symbol] for symbol in shares) / mark_to_market, 6) if mark_to_market else 0.0,
            }
        )

    equity_df = pd.DataFrame(equity_curve)
    returns = equity_df["equity"].pct_change().fillna(0.0)
    final_equity = float(equity_df["equity"].iloc[-1]) if not equity_df.empty else initial_cash
    total_return = final_equity / initial_cash - 1.0
    return BacktestResult(
        bars=len(df),
        trades=len(trades_log),
        final_equity=round(final_equity, 2),
        total_return=round(total_return, 6),
        annual_return=round(_annualize(total_return, len(equity_df), bars_per_year), 6),
        max_drawdown=round(_max_drawdown(equity_df["equity"]), 6),
        sharpe=round(_sharpe(returns, bars_per_year), 6),
        equity_curve=equity_curve,
        trades_log=trades_log,
    )
