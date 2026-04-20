from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any

from QUANTAXIS.ashare.account import AccountSnapshot, Position
from QUANTAXIS.ashare.quotes import Quote


@dataclass(slots=True)
class Order:
    symbol: str
    side: str
    amount: int
    price: float | None = None


@dataclass(slots=True)
class ExecutionReport:
    broker: str
    symbol: str
    side: str
    amount: int
    requested_price: float | None
    fill_price: float
    cost: float
    fees: float
    status: str
    raw: dict[str, Any]
    timestamp: str

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


class PaperBroker:
    """Immediate-fill broker for local dry-runs and strategy debugging."""

    def __init__(
        self,
        initial_cash: float = 1_000_000,
        commission_rate: float = 0.0003,
        min_commission: float = 5.0,
        stamp_duty_rate: float = 0.001,
        slippage: float = 0.0,
    ) -> None:
        self.commission_rate = commission_rate
        self.min_commission = min_commission
        self.stamp_duty_rate = stamp_duty_rate
        self.slippage = slippage
        self.account = AccountSnapshot(cash=round(initial_cash, 2))

    def submit(self, order: Order, quote: Quote) -> ExecutionReport:
        side = order.side.lower()
        if side not in {"buy", "sell"}:
            raise ValueError(f"unsupported side: {order.side}")
        if order.amount <= 0 or order.amount % 100 != 0:
            raise ValueError("A-share order amount must be a positive multiple of 100")

        base_price = quote.last_price if order.price is None else order.price
        signed_slippage = self.slippage if side == "buy" else -self.slippage
        fill_price = round(base_price + signed_slippage, 3)
        gross = round(fill_price * order.amount, 2)
        commission = max(round(gross * self.commission_rate, 2), self.min_commission)
        stamp_duty = round(gross * self.stamp_duty_rate, 2) if side == "sell" else 0.0
        fees = round(commission + stamp_duty, 2)
        total_cash_delta = gross + fees if side == "buy" else -(gross - fees)
        position = self.account.positions.get(order.symbol, Position(symbol=order.symbol))

        if side == "buy":
            if self.account.cash < total_cash_delta:
                raise RuntimeError(
                    f"insufficient cash: need {total_cash_delta:.2f}, have {self.account.cash:.2f}"
                )
            total_cost = position.quantity * position.avg_price + gross
            position.quantity += order.amount
            position.avg_price = round(total_cost / position.quantity, 4)
            self.account.cash = round(self.account.cash - total_cash_delta, 2)
        else:
            if position.quantity < order.amount:
                raise RuntimeError(
                    f"insufficient position: need {order.amount}, have {position.quantity}"
                )
            position.quantity -= order.amount
            if position.quantity == 0:
                position.avg_price = 0.0
            self.account.cash = round(self.account.cash - total_cash_delta, 2)

        if position.quantity == 0:
            self.account.positions.pop(order.symbol, None)
        else:
            self.account.positions[order.symbol] = position

        return ExecutionReport(
            broker="paper",
            symbol=order.symbol,
            side=side,
            amount=order.amount,
            requested_price=order.price,
            fill_price=fill_price,
            cost=gross,
            fees=fees,
            status="filled",
            raw={"account": self.account.as_dict()},
            timestamp=datetime.now().isoformat(timespec="seconds"),
        )


class EasyTraderBroker:
    """Thin adapter around easytrader for live A-share execution."""

    def __init__(self, client: str, prepare: dict[str, Any] | None = None) -> None:
        try:
            import easytrader
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "easytrader is required for live broker execution. Install it with `pip install easytrader`."
            ) from exc

        self.user = easytrader.use(client)
        prepare = prepare or {}
        if prepare:
            self.user.prepare(**prepare)
        self.client = client

    def submit(self, order: Order, quote: Quote) -> ExecutionReport:
        side = order.side.lower()
        if side not in {"buy", "sell"}:
            raise ValueError(f"unsupported side: {order.side}")

        price = quote.last_price if order.price is None else order.price
        action = getattr(self.user, side)
        raw = action(order.symbol, price=price, amount=order.amount)
        return ExecutionReport(
            broker=f"easytrader:{self.client}",
            symbol=order.symbol,
            side=side,
            amount=order.amount,
            requested_price=order.price,
            fill_price=price,
            cost=round(price * order.amount, 2),
            fees=0.0,
            status="submitted",
            raw=raw if isinstance(raw, dict) else {"result": raw},
            timestamp=datetime.now().isoformat(timespec="seconds"),
        )
