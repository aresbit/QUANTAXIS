from __future__ import annotations

from dataclasses import asdict, dataclass

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


def run_backtest(
    data: pd.DataFrame,
    strategy: RecursiveQTransformerStrategy,
    initial_cash: float = 1_000_000,
    commission_rate: float = 0.0003,
    stamp_duty_rate: float = 0.001,
    bars_per_year: int = 252,
) -> BacktestResult:
    required = {"datetime", "open", "high", "low", "close", "volume"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"missing required columns: {sorted(missing)}")

    df = data.copy().reset_index(drop=True)
    cash = float(initial_cash)
    position = 0
    equity_curve: list[dict[str, float | str]] = []
    trades_log: list[dict[str, float | str]] = []

    for index, row in df.iterrows():
        window = df.iloc[: index + 1]
        signal = strategy.on_bar(window)
        price = float(row["close"])
        desired_position = 1 if signal > strategy.config.buy_threshold else 0
        if strategy.config.allow_short and signal < strategy.config.sell_threshold:
            desired_position = -1

        if desired_position != position:
            if position == 1:
                gross = price * strategy.config.trade_size
                fees = gross * (commission_rate + stamp_duty_rate)
                cash += gross - fees
                trades_log.append(
                    {
                        "datetime": str(row["datetime"]),
                        "side": "sell",
                        "price": price,
                        "size": strategy.config.trade_size,
                        "signal": signal,
                        "fees": round(fees, 2),
                    }
                )
            elif position == -1:
                gross = price * strategy.config.trade_size
                fees = gross * commission_rate
                cash -= gross + fees
                trades_log.append(
                    {
                        "datetime": str(row["datetime"]),
                        "side": "cover",
                        "price": price,
                        "size": strategy.config.trade_size,
                        "signal": signal,
                        "fees": round(fees, 2),
                    }
                )

            if desired_position == 1:
                gross = price * strategy.config.trade_size
                fees = gross * commission_rate
                if cash >= gross + fees:
                    cash -= gross + fees
                    position = 1
                    trades_log.append(
                        {
                            "datetime": str(row["datetime"]),
                            "side": "buy",
                            "price": price,
                            "size": strategy.config.trade_size,
                            "signal": signal,
                            "fees": round(fees, 2),
                        }
                    )
                else:
                    position = 0
            elif desired_position == -1:
                gross = price * strategy.config.trade_size
                fees = gross * commission_rate
                cash += gross - fees
                position = -1
                trades_log.append(
                    {
                        "datetime": str(row["datetime"]),
                        "side": "short",
                        "price": price,
                        "size": strategy.config.trade_size,
                        "signal": signal,
                        "fees": round(fees, 2),
                    }
                )
            else:
                position = 0

        mark_to_market = cash + position * strategy.config.trade_size * price
        equity_curve.append(
            {
                "datetime": str(row["datetime"]),
                "close": price,
                "signal": round(signal, 6),
                "position": position,
                "equity": round(mark_to_market, 2),
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
        annual_return=round(_annualize(total_return, len(df), bars_per_year), 6),
        max_drawdown=round(_max_drawdown(equity_df["equity"]), 6),
        sharpe=round(_sharpe(returns, bars_per_year), 6),
        equity_curve=equity_curve,
        trades_log=trades_log,
    )
