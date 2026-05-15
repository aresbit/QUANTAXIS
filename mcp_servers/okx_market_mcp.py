#!/usr/bin/env python3
"""OKX public market data MCP server (no API key required).

Tools exposed:
  okx_ticker        - 获取单个或多个交易对的最新行情
  okx_candles       - 获取K线数据 (1m/5m/15m/1H/4H/1D)
  okx_orderbook     - 获取深度数据
  okx_funding_rate  - 获取永续合约资金费率
  okx_instruments   - 获取可交易合约列表
"""

from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request
from typing import Any

OKX_BASE = "https://www.okx.com"
BAR_MAP = {
    "1m": "1m", "3m": "3m", "5m": "5m", "15m": "15m", "30m": "30m",
    "1H": "1H", "2H": "2H", "4H": "4H", "6H": "6H", "12H": "12H",
    "1D": "1D", "1W": "1W",
}


def _get(path: str, params: dict[str, str] | None = None) -> Any:
    url = OKX_BASE + path
    if params:
        qs = "&".join(f"{k}={v}" for k, v in params.items())
        url = f"{url}?{qs}"
    req = urllib.request.Request(url, headers={"User-Agent": "btcquant-mcp/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        return {"code": str(e.code), "msg": str(e.reason), "data": []}


def _tool_okx_ticker(instId: str) -> str:
    """获取交易对行情，instId 支持逗号分隔多个，例如 BTC-USDT,ETH-USDT"""
    ids = [i.strip().upper() for i in instId.split(",") if i.strip()]
    results = []
    for iid in ids:
        r = _get("/api/v5/market/ticker", {"instId": iid})
        if r.get("code") == "0" and r.get("data"):
            d = r["data"][0]
            results.append({
                "instId": d["instId"],
                "last": d["last"],
                "open24h": d["open24h"],
                "high24h": d["high24h"],
                "low24h": d["low24h"],
                "vol24h": d["vol24h"],
                "volCcy24h": d["volCcy24h"],
                "ts": d["ts"],
            })
        else:
            results.append({"instId": iid, "error": r.get("msg", "unknown")})
    return json.dumps(results, ensure_ascii=False)


def _tool_okx_candles(instId: str, bar: str = "1H", limit: str = "100") -> str:
    """获取K线数据。bar: 1m/5m/15m/30m/1H/4H/1D。返回 [ts,o,h,l,c,vol] 列表"""
    bar_key = BAR_MAP.get(bar, "1H")
    lim = min(max(int(limit), 1), 300)
    iid = instId.strip().upper()
    r = _get("/api/v5/market/candles", {"instId": iid, "bar": bar_key, "limit": str(lim)})
    if r.get("code") != "0":
        return json.dumps({"error": r.get("msg", "api error")})
    rows = []
    for candle in r.get("data", []):
        rows.append({
            "ts": candle[0],
            "open": candle[1],
            "high": candle[2],
            "low": candle[3],
            "close": candle[4],
            "vol": candle[5],
            "volCcy": candle[6],
        })
    return json.dumps({"instId": iid, "bar": bar_key, "count": len(rows), "candles": rows}, ensure_ascii=False)


def _tool_okx_orderbook(instId: str, sz: str = "20") -> str:
    """获取深度数据，sz=档位数(1-400)"""
    depth = min(max(int(sz), 1), 400)
    iid = instId.strip().upper()
    r = _get("/api/v5/market/books", {"instId": iid, "sz": str(depth)})
    if r.get("code") != "0":
        return json.dumps({"error": r.get("msg", "api error")})
    data = r.get("data", [{}])[0]
    return json.dumps({
        "instId": iid,
        "ts": data.get("ts"),
        "asks": data.get("asks", [])[:10],
        "bids": data.get("bids", [])[:10],
    }, ensure_ascii=False)


def _tool_okx_funding_rate(instId: str) -> str:
    """获取永续合约当期和预测资金费率"""
    iid = instId.strip().upper()
    r = _get("/api/v5/public/funding-rate", {"instId": iid})
    if r.get("code") != "0":
        return json.dumps({"error": r.get("msg", "api error")})
    data = r.get("data", [{}])[0]
    return json.dumps({
        "instId": data.get("instId"),
        "fundingRate": data.get("fundingRate"),
        "nextFundingRate": data.get("nextFundingRate"),
        "fundingTime": data.get("fundingTime"),
    }, ensure_ascii=False)


def _tool_okx_instruments(instType: str = "SPOT", uly: str = "") -> str:
    """获取可交易产品列表。instType: SPOT/SWAP/FUTURES/OPTION"""
    params: dict[str, str] = {"instType": instType.upper()}
    if uly:
        params["uly"] = uly.upper()
    r = _get("/api/v5/public/instruments", params)
    if r.get("code") != "0":
        return json.dumps({"error": r.get("msg", "api error")})
    items = [
        {"instId": d["instId"], "baseCcy": d.get("baseCcy",""), "quoteCcy": d.get("quoteCcy",""), "state": d.get("state","")}
        for d in r.get("data", [])
        if d.get("state") == "live"
    ]
    return json.dumps({"instType": instType, "count": len(items), "instruments": items[:50]}, ensure_ascii=False)


# ── MCP stdio protocol ────────────────────────────────────────────────────────

TOOLS = [
    {
        "name": "okx_ticker",
        "description": "获取OKX交易对最新行情，支持逗号分隔多个，例如 BTC-USDT,ETH-USDT",
        "inputSchema": {
            "type": "object",
            "properties": {
                "instId": {"type": "string", "description": "交易对ID，多个用逗号分隔"}
            },
            "required": ["instId"],
        },
    },
    {
        "name": "okx_candles",
        "description": "获取OKX K线数据",
        "inputSchema": {
            "type": "object",
            "properties": {
                "instId": {"type": "string", "description": "交易对ID，例如 BTC-USDT"},
                "bar": {"type": "string", "description": "时间粒度: 1m/5m/15m/30m/1H/4H/1D", "default": "1H"},
                "limit": {"type": "string", "description": "数量，最多300", "default": "100"},
            },
            "required": ["instId"],
        },
    },
    {
        "name": "okx_orderbook",
        "description": "获取OKX深度数据",
        "inputSchema": {
            "type": "object",
            "properties": {
                "instId": {"type": "string", "description": "交易对ID"},
                "sz": {"type": "string", "description": "档位数", "default": "20"},
            },
            "required": ["instId"],
        },
    },
    {
        "name": "okx_funding_rate",
        "description": "获取OKX永续合约资金费率",
        "inputSchema": {
            "type": "object",
            "properties": {
                "instId": {"type": "string", "description": "永续合约ID，例如 BTC-USDT-SWAP"}
            },
            "required": ["instId"],
        },
    },
    {
        "name": "okx_instruments",
        "description": "获取OKX可交易产品列表",
        "inputSchema": {
            "type": "object",
            "properties": {
                "instType": {"type": "string", "description": "产品类型: SPOT/SWAP/FUTURES", "default": "SPOT"},
                "uly": {"type": "string", "description": "标的指数，例如 BTC-USDT（可选）", "default": ""},
            },
        },
    },
]

TOOL_HANDLERS = {
    "okx_ticker": _tool_okx_ticker,
    "okx_candles": _tool_okx_candles,
    "okx_orderbook": _tool_okx_orderbook,
    "okx_funding_rate": _tool_okx_funding_rate,
    "okx_instruments": _tool_okx_instruments,
}


def _send(obj: dict) -> None:
    line = json.dumps(obj, ensure_ascii=False)
    sys.stdout.write(line + "\n")
    sys.stdout.flush()


def _handle(request: dict) -> dict | None:
    method = request.get("method", "")
    req_id = request.get("id")

    if method == "initialize":
        return {
            "jsonrpc": "2.0", "id": req_id,
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "okx-market", "version": "1.0.0"},
            },
        }

    if method == "tools/list":
        return {"jsonrpc": "2.0", "id": req_id, "result": {"tools": TOOLS}}

    if method == "tools/call":
        params = request.get("params", {})
        tool_name = params.get("name", "")
        args = params.get("arguments", {})
        handler = TOOL_HANDLERS.get(tool_name)
        if handler is None:
            return {
                "jsonrpc": "2.0", "id": req_id,
                "error": {"code": -32601, "message": f"Tool not found: {tool_name}"},
            }
        try:
            result = handler(**args)
        except Exception as exc:
            result = json.dumps({"error": str(exc)})
        return {
            "jsonrpc": "2.0", "id": req_id,
            "result": {"content": [{"type": "text", "text": result}]},
        }

    if method == "notifications/initialized":
        return None

    return {
        "jsonrpc": "2.0", "id": req_id,
        "error": {"code": -32601, "message": f"Method not found: {method}"},
    }


def main() -> None:
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
        except json.JSONDecodeError:
            continue
        response = _handle(request)
        if response is not None:
            _send(response)


if __name__ == "__main__":
    main()
