from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass(slots=True)
class Position:
    symbol: str
    quantity: int = 0
    avg_price: float = 0.0

    def market_value(self, last_price: float) -> float:
        return round(self.quantity * last_price, 2)


@dataclass(slots=True)
class AccountSnapshot:
    cash: float
    positions: dict[str, Position] = field(default_factory=dict)

    @property
    def equity(self) -> float:
        return round(self.cash + sum(pos.quantity * pos.avg_price for pos in self.positions.values()), 2)

    def as_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["positions"] = {symbol: asdict(position) for symbol, position in self.positions.items()}
        payload["equity"] = self.equity
        return payload
