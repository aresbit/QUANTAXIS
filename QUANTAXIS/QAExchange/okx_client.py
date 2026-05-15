"""OKX V5 REST + WebSocket client.

REST auth: HMAC-SHA256(timestamp + method + path + body, secret).
All private endpoints require api_key, secret_key, passphrase.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable

logger = logging.getLogger("quantaxis.okx")

REST_BASE = "https://www.okx.com"
WS_PUBLIC = "wss://ws.okx.com:8443/ws/v5/public"
WS_PRIVATE = "wss://ws.okx.com:8443/ws/v5/private"

# Simnet (paper trading) endpoints
REST_SIM = "https://www.okx.com"
WS_SIM_PUBLIC = "wss://wspap.okx.com:8443/ws/v5/public?brokerId=9999"
WS_SIM_PRIVATE = "wss://wspap.okx.com:8443/ws/v5/private?brokerId=9999"


@dataclass
class OKXConfig:
    api_key: str = ""
    secret_key: str = ""
    passphrase: str = ""
    simulated: bool = False  # True = paper trading (simnet)
    timeout: int = 10
    max_retries: int = 3


class OKXClient:
    """Thin OKX V5 REST client. Covers market data + private trading."""

    def __init__(self, config: OKXConfig | None = None) -> None:
        self.cfg = config or OKXConfig()
        self._base = REST_SIM if self.cfg.simulated else REST_BASE

    # ── Signature ────────────────────────────────────────────────────────────

    def _sign(self, timestamp: str, method: str, path: str, body: str) -> str:
        msg = timestamp + method.upper() + path + body
        mac = hmac.new(self.cfg.secret_key.encode(), msg.encode(), hashlib.sha256)
        return base64.b64encode(mac.digest()).decode()

    def _headers(self, method: str, path: str, body: str = "") -> dict[str, str]:
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
        headers: dict[str, str] = {
            "Content-Type": "application/json",
            "OK-ACCESS-KEY": self.cfg.api_key,
            "OK-ACCESS-SIGN": self._sign(ts, method, path, body),
            "OK-ACCESS-TIMESTAMP": ts,
            "OK-ACCESS-PASSPHRASE": self.cfg.passphrase,
        }
        if self.cfg.simulated:
            headers["x-simulated-trading"] = "1"
        return headers

    # ── HTTP ─────────────────────────────────────────────────────────────────

    def _request(
        self,
        method: str,
        path: str,
        params: dict | None = None,
        body: dict | None = None,
        auth: bool = False,
    ) -> dict:
        url = self._base + path
        if params:
            url += "?" + urllib.parse.urlencode(params)
        raw_body = json.dumps(body) if body else ""
        headers = self._headers(method, path, raw_body) if auth else {"Content-Type": "application/json"}
        data = raw_body.encode() if raw_body else None
        req = urllib.request.Request(url, data=data, headers=headers, method=method.upper())

        for attempt in range(self.cfg.max_retries):
            try:
                with urllib.request.urlopen(req, timeout=self.cfg.timeout) as resp:
                    result = json.loads(resp.read())
                    if result.get("code") not in ("0", 0):
                        logger.warning("OKX API error %s: %s", result.get("code"), result.get("msg"))
                    return result
            except urllib.error.HTTPError as e:
                body_txt = e.read().decode()
                logger.error("HTTP %s on %s: %s", e.code, path, body_txt)
                if e.code in (429, 500, 502, 503) and attempt < self.cfg.max_retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                return {"code": str(e.code), "msg": body_txt, "data": []}
            except OSError as e:
                logger.error("Network error on %s: %s", path, e)
                if attempt < self.cfg.max_retries - 1:
                    time.sleep(1)
                    continue
                return {"code": "-1", "msg": str(e), "data": []}
        return {"code": "-1", "msg": "max retries exceeded", "data": []}

    # ── Public Market Data ───────────────────────────────────────────────────

    def get_ticker(self, inst_id: str) -> dict:
        r = self._request("GET", "/api/v5/market/ticker", {"instId": inst_id})
        return r.get("data", [{}])[0]

    def get_tickers(self, inst_type: str = "SPOT") -> list[dict]:
        r = self._request("GET", "/api/v5/market/tickers", {"instType": inst_type})
        return r.get("data", [])

    def get_candles(
        self,
        inst_id: str,
        bar: str = "1H",
        limit: int = 100,
        after: str | None = None,
        before: str | None = None,
    ) -> list[dict]:
        params: dict[str, str] = {"instId": inst_id, "bar": bar, "limit": str(min(limit, 300))}
        if after:
            params["after"] = after
        if before:
            params["before"] = before
        r = self._request("GET", "/api/v5/market/candles", params)
        candles = []
        for c in r.get("data", []):
            candles.append({
                "ts": int(c[0]),
                "open": float(c[1]),
                "high": float(c[2]),
                "low": float(c[3]),
                "close": float(c[4]),
                "vol": float(c[5]),
                "volCcy": float(c[6]),
            })
        return candles

    def get_orderbook(self, inst_id: str, sz: int = 20) -> dict:
        r = self._request("GET", "/api/v5/market/books", {"instId": inst_id, "sz": str(sz)})
        return r.get("data", [{}])[0]

    def get_funding_rate(self, inst_id: str) -> dict:
        r = self._request("GET", "/api/v5/public/funding-rate", {"instId": inst_id})
        return r.get("data", [{}])[0]

    def get_instruments(self, inst_type: str = "SPOT", uly: str | None = None) -> list[dict]:
        params: dict[str, str] = {"instType": inst_type}
        if uly:
            params["uly"] = uly
        r = self._request("GET", "/api/v5/public/instruments", params)
        return r.get("data", [])

    # ── Private Account ──────────────────────────────────────────────────────

    def get_balance(self, ccy: str | None = None) -> dict:
        params: dict[str, str] = {}
        if ccy:
            params["ccy"] = ccy
        r = self._request("GET", "/api/v5/account/balance", params, auth=True)
        return r.get("data", [{}])[0]

    def get_positions(self, inst_type: str | None = None) -> list[dict]:
        params: dict[str, str] = {}
        if inst_type:
            params["instType"] = inst_type
        r = self._request("GET", "/api/v5/account/positions", params, auth=True)
        return r.get("data", [])

    def get_account_config(self) -> dict:
        r = self._request("GET", "/api/v5/account/config", auth=True)
        return r.get("data", [{}])[0]

    def set_leverage(self, inst_id: str, lever: int, mgn_mode: str = "cross") -> dict:
        body = {"instId": inst_id, "lever": str(lever), "mgnMode": mgn_mode}
        r = self._request("POST", "/api/v5/account/set-leverage", body=body, auth=True)
        return r.get("data", [{}])[0]

    # ── Private Trading ──────────────────────────────────────────────────────

    def place_order(
        self,
        inst_id: str,
        td_mode: str,        # cash / cross / isolated
        side: str,           # buy / sell
        ord_type: str,       # market / limit / post_only / fok / ioc
        sz: str,
        px: str | None = None,
        cl_ord_id: str | None = None,
        reduce_only: bool = False,
        tag: str = "btcquant",
    ) -> dict:
        body: dict[str, Any] = {
            "instId": inst_id,
            "tdMode": td_mode,
            "side": side,
            "ordType": ord_type,
            "sz": sz,
            "tag": tag,
        }
        if px is not None:
            body["px"] = px
        if cl_ord_id:
            body["clOrdId"] = cl_ord_id
        if reduce_only:
            body["reduceOnly"] = "true"
        r = self._request("POST", "/api/v5/trade/order", body=body, auth=True)
        return r.get("data", [{}])[0]

    def cancel_order(self, inst_id: str, ord_id: str | None = None, cl_ord_id: str | None = None) -> dict:
        body: dict[str, str] = {"instId": inst_id}
        if ord_id:
            body["ordId"] = ord_id
        if cl_ord_id:
            body["clOrdId"] = cl_ord_id
        r = self._request("POST", "/api/v5/trade/cancel-order", body=body, auth=True)
        return r.get("data", [{}])[0]

    def get_order(self, inst_id: str, ord_id: str | None = None, cl_ord_id: str | None = None) -> dict:
        params: dict[str, str] = {"instId": inst_id}
        if ord_id:
            params["ordId"] = ord_id
        if cl_ord_id:
            params["clOrdId"] = cl_ord_id
        r = self._request("GET", "/api/v5/trade/order", params, auth=True)
        return r.get("data", [{}])[0]

    def get_pending_orders(self, inst_id: str | None = None) -> list[dict]:
        params: dict[str, str] = {}
        if inst_id:
            params["instId"] = inst_id
        r = self._request("GET", "/api/v5/trade/orders-pending", params, auth=True)
        return r.get("data", [])

    def get_order_history(self, inst_type: str = "SPOT", limit: int = 100) -> list[dict]:
        r = self._request(
            "GET", "/api/v5/trade/orders-history",
            {"instType": inst_type, "limit": str(limit)},
            auth=True,
        )
        return r.get("data", [])

    def place_algo_order(
        self,
        inst_id: str,
        td_mode: str,
        side: str,
        ord_type: str,   # conditional / oco / trigger / iceberg / twap
        sz: str,
        tp_trigger_px: str | None = None,
        tp_ord_px: str | None = None,
        sl_trigger_px: str | None = None,
        sl_ord_px: str | None = None,
    ) -> dict:
        body: dict[str, Any] = {
            "instId": inst_id,
            "tdMode": td_mode,
            "side": side,
            "ordType": ord_type,
            "sz": sz,
        }
        if tp_trigger_px:
            body["tpTriggerPx"] = tp_trigger_px
        if tp_ord_px:
            body["tpOrdPx"] = tp_ord_px
        if sl_trigger_px:
            body["slTriggerPx"] = sl_trigger_px
        if sl_ord_px:
            body["slOrdPx"] = sl_ord_px
        r = self._request("POST", "/api/v5/trade/order-algo", body=body, auth=True)
        return r.get("data", [{}])[0]

    # ── WebSocket helpers (sync wrapper via thread) ──────────────────────────

    def subscribe_tickers(
        self,
        inst_ids: list[str],
        callback: Callable[[dict], None],
        stop_event: threading.Event | None = None,
    ) -> threading.Thread:
        """Start a background thread subscribing to public ticker channel."""
        return _ws_subscribe(
            url=WS_SIM_PUBLIC if self.cfg.simulated else WS_PUBLIC,
            channels=[{"channel": "tickers", "instId": iid} for iid in inst_ids],
            callback=callback,
            stop_event=stop_event,
        )

    def subscribe_candles(
        self,
        inst_id: str,
        bar: str,
        callback: Callable[[dict], None],
        stop_event: threading.Event | None = None,
    ) -> threading.Thread:
        channel = f"candle{bar}"
        return _ws_subscribe(
            url=WS_SIM_PUBLIC if self.cfg.simulated else WS_PUBLIC,
            channels=[{"channel": channel, "instId": inst_id}],
            callback=callback,
            stop_event=stop_event,
        )

    def subscribe_orders(
        self,
        callback: Callable[[dict], None],
        stop_event: threading.Event | None = None,
    ) -> threading.Thread:
        """Subscribe private order channel (requires valid credentials)."""
        return _ws_subscribe_private(
            url=WS_SIM_PRIVATE if self.cfg.simulated else WS_PRIVATE,
            api_key=self.cfg.api_key,
            secret_key=self.cfg.secret_key,
            passphrase=self.cfg.passphrase,
            channels=[{"channel": "orders", "instType": "ANY"}],
            callback=callback,
            stop_event=stop_event,
            simulated=self.cfg.simulated,
        )


# ── WebSocket internals ───────────────────────────────────────────────────────

def _ws_subscribe(
    url: str,
    channels: list[dict],
    callback: Callable[[dict], None],
    stop_event: threading.Event | None = None,
) -> threading.Thread:
    stop = stop_event or threading.Event()

    def _run() -> None:
        try:
            import websocket  # type: ignore[import]
        except ImportError:
            logger.error("websocket-client not installed. pip install websocket-client")
            return

        def _on_message(ws: Any, message: str) -> None:
            try:
                data = json.loads(message)
                if "data" in data:
                    callback(data)
            except Exception as exc:
                logger.debug("WS parse error: %s", exc)

        def _on_open(ws: Any) -> None:
            ws.send(json.dumps({"op": "subscribe", "args": channels}))

        def _on_error(ws: Any, error: Any) -> None:
            logger.warning("WS error: %s", error)

        while not stop.is_set():
            ws = websocket.WebSocketApp(url, on_message=_on_message, on_open=_on_open, on_error=_on_error)
            ws.run_forever(ping_interval=20, ping_timeout=10)
            if not stop.is_set():
                time.sleep(3)  # reconnect delay

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    return t


def _ws_subscribe_private(
    url: str,
    api_key: str,
    secret_key: str,
    passphrase: str,
    channels: list[dict],
    callback: Callable[[dict], None],
    stop_event: threading.Event | None = None,
    simulated: bool = False,
) -> threading.Thread:
    stop = stop_event or threading.Event()

    def _login_msg() -> str:
        ts = str(int(time.time()))
        msg = ts + "GET" + "/users/self/verify"
        mac = hmac.new(secret_key.encode(), msg.encode(), hashlib.sha256)
        sign = base64.b64encode(mac.digest()).decode()
        args = [{"apiKey": api_key, "passphrase": passphrase, "timestamp": ts, "sign": sign}]
        return json.dumps({"op": "login", "args": args})

    def _run() -> None:
        try:
            import websocket  # type: ignore[import]
        except ImportError:
            logger.error("websocket-client not installed")
            return

        logged_in = threading.Event()

        def _on_message(ws: Any, message: str) -> None:
            try:
                data = json.loads(message)
                if data.get("event") == "login" and data.get("code") == "0":
                    logged_in.set()
                    ws.send(json.dumps({"op": "subscribe", "args": channels}))
                elif "data" in data:
                    callback(data)
            except Exception as exc:
                logger.debug("WS private parse error: %s", exc)

        def _on_open(ws: Any) -> None:
            ws.send(_login_msg())

        def _on_error(ws: Any, error: Any) -> None:
            logger.warning("WS private error: %s", error)

        while not stop.is_set():
            logged_in.clear()
            ws = websocket.WebSocketApp(url, on_message=_on_message, on_open=_on_open, on_error=_on_error)
            ws.run_forever(ping_interval=20, ping_timeout=10)
            if not stop.is_set():
                time.sleep(3)

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    return t
