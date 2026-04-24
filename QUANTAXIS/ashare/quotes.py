from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime


DEFAULT_TDX_SERVERS: tuple[tuple[str, int], ...] = (
    ("119.147.212.81", 7709),
    ("119.147.212.83", 7709),
    ("58.67.221.146", 7709),
    ("218.80.248.229", 7709),
    ("47.92.127.181", 7709),
    ("39.108.28.120", 7709),
)

PRICE_FIELDS = {
    "price",
    "last_close",
    "open",
    "high",
    "low",
    "bid1",
    "bid2",
    "bid3",
    "bid4",
    "bid5",
    "ask1",
    "ask2",
    "ask3",
    "ask4",
    "ask5",
}


@dataclass(slots=True)
class Quote:
    symbol: str
    market: int
    last_price: float
    open_price: float
    high_price: float
    low_price: float
    pre_close: float
    bid_price: float
    ask_price: float
    volume: int
    amount: float
    server: str
    timestamp: str

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def make_manual_quote(symbol: str, price: float, source: str = "manual") -> Quote:
    market = detect_market(symbol)
    clean_price = round(float(price), 3)
    return Quote(
        symbol=symbol,
        market=market,
        last_price=clean_price,
        open_price=clean_price,
        high_price=clean_price,
        low_price=clean_price,
        pre_close=clean_price,
        bid_price=clean_price,
        ask_price=clean_price,
        volume=0,
        amount=0.0,
        server=source,
        timestamp=datetime.now().isoformat(timespec="seconds"),
    )


def detect_market(symbol: str) -> int:
    """Detect TDX market code for A-share symbols.

    Rules:
        - Shanghai (market=1): 600*, 601*, 603*, 605*, 688*, 689* (科创),
          900* (B股), 51* (ETF), 50* (国债), 99* (指数)
        - Shenzhen (market=0): 000*, 001*, 002*, 003*, 300* (创业板),
          301* (创业板), 200* (B股), 15* (ETF), 10* (国债), 39* (指数)
        - Beijing (market=0 via fallback): 8*, 43*, 83*, 87*, 88*
    """
    if not symbol:
        return 0
    s = symbol.strip()
    if s.startswith(("6", "9", "5")):
        return 1
    if s.startswith(("0", "1", "2", "3", "15", "10", "39")):
        return 0
    if s.startswith(("8", "43", "83", "87", "88")):
        return 0
    return 0


def _normalize_price(value: object) -> float:
    if value is None:
        return 0.0
    if isinstance(value, int):
        return round(value / 100.0, 3)
    if isinstance(value, float) and abs(value) >= 1000:
        return round(value / 100.0, 3)
    return round(float(value), 3)


class PytdxQuoteClient:
    """Minimal pytdx wrapper for China A-share quotes."""

    def __init__(
        self,
        host: str | None = None,
        port: int = 7709,
        timeout: float = 0.7,
        servers: tuple[tuple[str, int], ...] | None = None,
    ) -> None:
        self.timeout = timeout
        self.servers = ((host, port),) if host else (servers or DEFAULT_TDX_SERVERS)

    def _load_api(self):
        try:
            from pytdx.hq import TdxHq_API
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "pytdx is required for live A-share quotes. Install it with `pip install pytdx`."
            ) from exc
        return TdxHq_API(raise_exception=True)

    def get_quote(self, symbol: str) -> Quote:
        market = detect_market(symbol)
        last_error: Exception | None = None

        for host, port in self.servers:
            api = self._load_api()
            try:
                with api.connect(host, port, time_out=self.timeout):
                    raw = api.get_security_quotes([(market, symbol)])
                    if not raw:
                        raise RuntimeError(f"empty quote returned for {symbol}")
                    row = raw[0]
                    return Quote(
                        symbol=symbol,
                        market=market,
                        last_price=_normalize_price(row.get("price")),
                        open_price=_normalize_price(row.get("open")),
                        high_price=_normalize_price(row.get("high")),
                        low_price=_normalize_price(row.get("low")),
                        pre_close=_normalize_price(row.get("last_close")),
                        bid_price=_normalize_price(row.get("bid1")),
                        ask_price=_normalize_price(row.get("ask1")),
                        volume=int(row.get("vol", 0)),
                        amount=float(row.get("amount", 0)),
                        server=f"{host}:{port}",
                        timestamp=datetime.now().isoformat(timespec="seconds"),
                    )
            except Exception as exc:  # pragma: no cover - network paths vary
                last_error = exc

        if last_error is None:
            raise RuntimeError(f"no TDX server configured for {symbol}")
        raise RuntimeError(f"failed to fetch quote for {symbol}: {last_error}") from last_error
