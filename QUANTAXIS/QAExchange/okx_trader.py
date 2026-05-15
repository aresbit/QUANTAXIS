"""OKX live/paper trading engine.

Usage:
    from QUANTAXIS.QAExchange import OKXTrader, OKXConfig
    from QUANTAXIS.backtest.strategy import RecursiveQTransformerStrategy, StrategyConfig

    cfg = OKXConfig(api_key="...", secret_key="...", passphrase="...", simulated=True)
    strategy = RecursiveQTransformerStrategy(StrategyConfig())
    trader = OKXTrader(config=cfg, strategy=strategy, inst_id="BTC-USDT-SWAP")
    session = trader.start(bar="1H")  # blocks until session.stop()
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable

import pandas as pd

from QUANTAXIS.QAExchange.okx_client import OKXClient, OKXConfig

logger = logging.getLogger("quantaxis.okx.trader")


@dataclass
class LivePosition:
    inst_id: str
    side: str          # "long" / "short" / "net"
    qty: float
    avg_px: float
    unrealized_pnl: float = 0.0
    leverage: int = 1


@dataclass
class LiveOrder:
    cl_ord_id: str
    inst_id: str
    side: str          # "buy" / "sell"
    ord_type: str
    sz: str
    px: str
    status: str = "pending"
    fill_sz: str = "0"
    fill_px: str = "0"
    ts: str = ""


@dataclass
class TradingSession:
    """Returned by OKXTrader.start(); call .stop() to gracefully halt."""
    _stop: threading.Event = field(default_factory=threading.Event)
    _threads: list[threading.Thread] = field(default_factory=list)
    orders: list[LiveOrder] = field(default_factory=list)
    positions: dict[str, LivePosition] = field(default_factory=dict)
    equity_log: list[dict[str, Any]] = field(default_factory=list)

    def stop(self) -> None:
        self._stop.set()
        for t in self._threads:
            t.join(timeout=10)
        logger.info("TradingSession stopped.")

    @property
    def is_running(self) -> bool:
        return not self._stop.is_set()


# ── Risk guard for live trading ───────────────────────────────────────────────

@dataclass
class LiveRiskConfig:
    max_position_usdt: float = 500.0       # single-side max notional
    max_daily_loss_usdt: float = 100.0     # halt if daily loss exceeds this
    max_order_usdt: float = 200.0          # max single order notional
    min_signal_threshold: float = 0.03     # min |signal| to act
    cooldown_bars: int = 3                 # bars to wait after a trade


class _LiveRiskGuard:
    def __init__(self, cfg: LiveRiskConfig) -> None:
        self.cfg = cfg
        self._daily_pnl: float = 0.0
        self._halted: bool = False
        self._cooldown: int = 0

    def tick(self, pnl_delta: float) -> None:
        self._daily_pnl += pnl_delta
        if self._cooldown > 0:
            self._cooldown -= 1
        if self._daily_pnl <= -self.cfg.max_daily_loss_usdt:
            self._halted = True
            logger.warning("Daily loss limit hit (%.2f). Trading halted.", self._daily_pnl)

    def can_trade(self, signal: float, position_notional: float) -> bool:
        if self._halted:
            return False
        if self._cooldown > 0:
            return False
        if abs(signal) < self.cfg.min_signal_threshold:
            return False
        if position_notional >= self.cfg.max_position_usdt:
            return False
        return True

    def on_trade(self) -> None:
        self._cooldown = self.cfg.cooldown_bars

    def reset_daily(self) -> None:
        self._daily_pnl = 0.0
        self._halted = False


# ── Main trader ───────────────────────────────────────────────────────────────

class OKXTrader:
    """Wraps strategy + OKXClient into a 24/7 live trading loop.

    Flow per bar:
      1. Pull latest candles from REST
      2. Run strategy.score_frame() → signal
      3. LiveRiskGuard gate
      4. Place market order via REST
      5. Log position & equity
    """

    def __init__(
        self,
        config: OKXConfig,
        strategy: Any,                      # RecursiveQTransformerStrategy or duck-typed
        inst_id: str = "BTC-USDT-SWAP",
        td_mode: str = "cross",             # cash / cross / isolated
        leverage: int = 3,
        risk_config: LiveRiskConfig | None = None,
        on_order: Callable[[LiveOrder], None] | None = None,
    ) -> None:
        self.client = OKXClient(config)
        self.strategy = strategy
        self.inst_id = inst_id
        self.td_mode = td_mode
        self.leverage = leverage
        self.risk = _LiveRiskGuard(risk_config or LiveRiskConfig())
        self.on_order = on_order
        self._bar_seconds: dict[str, int] = {
            "1m": 60, "3m": 180, "5m": 300, "15m": 900, "30m": 1800,
            "1H": 3600, "2H": 7200, "4H": 14400, "6H": 21600, "12H": 43200,
            "1D": 86400,
        }

    # ── Public ────────────────────────────────────────────────────────────────

    def start(self, bar: str = "1H", lookback: int = 64) -> TradingSession:
        """Start the live trading loop in a daemon thread. Returns TradingSession."""
        session = TradingSession()
        self._set_leverage()
        t = threading.Thread(
            target=self._loop,
            args=(bar, lookback, session),
            daemon=True,
            name=f"okx-trader-{self.inst_id}",
        )
        session._threads.append(t)
        t.start()
        logger.info("OKXTrader started: %s bar=%s leverage=%dx simulated=%s",
                    self.inst_id, bar, self.leverage, self.client.cfg.simulated)
        return session

    def run_once(self, bar: str = "1H", lookback: int = 64) -> dict[str, Any]:
        """Run a single bar evaluation and return the decision dict (no order placed)."""
        candles = self._fetch_candles(bar, lookback)
        if candles is None or len(candles) < 10:
            return {"action": "skip", "reason": "insufficient data"}
        df = self._candles_to_df(candles)
        signal = self._score(df)
        position = self._get_position()
        last_px = float(candles[0]["close"])
        pos_notional = (position.qty * last_px) if position else 0.0
        can = self.risk.can_trade(signal, pos_notional)
        return {
            "inst_id": self.inst_id,
            "bar": bar,
            "last_price": last_px,
            "signal": signal,
            "position_qty": position.qty if position else 0.0,
            "pos_notional": pos_notional,
            "can_trade": can,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    # ── Internal loop ─────────────────────────────────────────────────────────

    def _loop(self, bar: str, lookback: int, session: TradingSession) -> None:
        bar_secs = self._bar_seconds.get(bar, 3600)
        prev_day: str | None = None

        while not session._stop.is_set():
            try:
                now = datetime.now(timezone.utc)
                today = now.strftime("%Y-%m-%d")
                if today != prev_day:
                    self.risk.reset_daily()
                    prev_day = today

                candles = self._fetch_candles(bar, lookback)
                if candles is None or len(candles) < 10:
                    logger.debug("Not enough candles, waiting...")
                    session._stop.wait(timeout=min(bar_secs, 60))
                    continue

                df = self._candles_to_df(candles)
                signal = self._score(df)
                last_px = float(candles[0]["close"])
                position = self._get_position()
                pos_notional = (position.qty * last_px) if position else 0.0

                action = self._decide(signal, position, pos_notional, last_px)
                if action and self.risk.can_trade(signal, pos_notional):
                    order = self._execute(action, last_px, session)
                    if order:
                        self.risk.on_trade()
                        if self.on_order:
                            self.on_order(order)

                equity = self._estimate_equity(last_px, session)
                session.equity_log.append({
                    "ts": now.isoformat(),
                    "price": last_px,
                    "signal": signal,
                    "equity": equity,
                    "pos_notional": pos_notional,
                })
                logger.info("Bar [%s] price=%.4f signal=%.4f equity=%.2f",
                            bar, last_px, signal, equity)

                # sleep until next bar boundary
                elapsed = time.time() % bar_secs
                sleep_secs = bar_secs - elapsed
                session._stop.wait(timeout=max(sleep_secs, 10))

            except Exception as exc:
                logger.exception("Trader loop error: %s", exc)
                session._stop.wait(timeout=30)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _set_leverage(self) -> None:
        if self.td_mode in ("cross", "isolated") and self.leverage > 1:
            try:
                self.client.set_leverage(self.inst_id, self.leverage, self.td_mode)
                logger.info("Leverage set: %dx %s", self.leverage, self.td_mode)
            except Exception as e:
                logger.warning("set_leverage failed: %s", e)

    def _fetch_candles(self, bar: str, limit: int) -> list[dict] | None:
        try:
            return self.client.get_candles(self.inst_id, bar=bar, limit=limit)
        except Exception as e:
            logger.warning("fetch candles error: %s", e)
            return None

    def _candles_to_df(self, candles: list[dict]) -> pd.DataFrame:
        df = pd.DataFrame(candles[::-1])  # oldest first
        df["datetime"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
        df["symbol"] = self.inst_id
        df.rename(columns={"vol": "volume"}, inplace=True)
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = df[col].astype(float)
        return df.reset_index(drop=True)

    def _score(self, df: pd.DataFrame) -> float:
        try:
            scored = self.strategy.score_frame(df)
            if scored.empty or "signal" not in scored.columns:
                return 0.0
            return float(scored["signal"].iloc[-1])
        except Exception as e:
            logger.debug("score_frame error: %s", e)
            return 0.0

    def _get_position(self) -> LivePosition | None:
        try:
            positions = self.client.get_positions()
            for p in positions:
                if p.get("instId") == self.inst_id:
                    return LivePosition(
                        inst_id=self.inst_id,
                        side=p.get("posSide", "net"),
                        qty=float(p.get("pos", "0") or 0),
                        avg_px=float(p.get("avgPx", "0") or 0),
                        unrealized_pnl=float(p.get("upl", "0") or 0),
                        leverage=int(p.get("lever", "1") or 1),
                    )
        except Exception as e:
            logger.debug("get_positions error: %s", e)
        return None

    def _decide(
        self,
        signal: float,
        position: LivePosition | None,
        pos_notional: float,
        last_px: float,
    ) -> dict | None:
        """Return action dict or None."""
        has_long = position is not None and position.qty > 0 and position.side in ("long", "net")
        has_short = position is not None and position.qty < 0 and position.side == "short"
        cfg = self.risk.cfg

        if signal > cfg.min_signal_threshold:
            if has_short:
                return {"side": "buy", "reason": "close_short", "reduce_only": True}
            if not has_long and pos_notional < cfg.max_position_usdt:
                order_usdt = min(cfg.max_order_usdt, cfg.max_position_usdt - pos_notional)
                sz = order_usdt / last_px
                return {"side": "buy", "sz": f"{sz:.6f}", "reason": "open_long"}
        elif signal < -cfg.min_signal_threshold:
            if has_long:
                return {"side": "sell", "reason": "close_long", "reduce_only": True}
            if not has_short and pos_notional < cfg.max_position_usdt:
                order_usdt = min(cfg.max_order_usdt, cfg.max_position_usdt - pos_notional)
                sz = order_usdt / last_px
                return {"side": "sell", "sz": f"{sz:.6f}", "reason": "open_short"}
        return None

    def _execute(self, action: dict, last_px: float, session: TradingSession) -> LiveOrder | None:
        import uuid
        cl_ord_id = f"btcq-{uuid.uuid4().hex[:12]}"
        side = action["side"]
        sz = action.get("sz", "0")
        reduce_only = action.get("reduce_only", False)
        reason = action.get("reason", "")

        if reduce_only:
            pos = self._get_position()
            if pos:
                sz = str(abs(pos.qty))
            else:
                logger.debug("reduce_only but no position; skipping.")
                return None

        logger.info("ORDER side=%s sz=%s reason=%s cl_ord_id=%s", side, sz, reason, cl_ord_id)
        try:
            resp = self.client.place_order(
                inst_id=self.inst_id,
                td_mode=self.td_mode,
                side=side,
                ord_type="market",
                sz=sz,
                cl_ord_id=cl_ord_id,
                reduce_only=reduce_only,
            )
            order = LiveOrder(
                cl_ord_id=cl_ord_id,
                inst_id=self.inst_id,
                side=side,
                ord_type="market",
                sz=sz,
                px=str(last_px),
                status=resp.get("sCode", "unknown"),
                ts=datetime.now(timezone.utc).isoformat(),
            )
            session.orders.append(order)
            logger.info("Order placed: %s -> %s", cl_ord_id, resp)
            return order
        except Exception as e:
            logger.error("place_order failed: %s", e)
            return None

    def _estimate_equity(self, last_px: float, session: TradingSession) -> float:
        try:
            bal = self.client.get_balance()
            details = bal.get("details", [])
            total = sum(float(d.get("eqUsd", "0") or 0) for d in details)
            return total
        except Exception:
            return 0.0
