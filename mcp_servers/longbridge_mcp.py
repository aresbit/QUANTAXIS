#!/usr/bin/env python3
"""长桥(Longbridge) OpenAPI MCP server.

环境变量（必须）：
  LONGBRIDGE_APP_KEY       - App Key
  LONGBRIDGE_APP_SECRET    - App Secret
  LONGBRIDGE_ACCESS_TOKEN  - Access Token

Tools:
  lb_quote          - 获取股票/ETF/期权实时报价
  lb_candlesticks   - 获取K线数据
  lb_order_book     - 获取深度行情
  lb_submit_order   - 提交交易订单
  lb_cancel_order   - 撤销订单
  lb_account_balance - 查询账户余额
  lb_positions      - 查询持仓
  lb_today_orders   - 查询当日订单
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any


def _require_env(key: str) -> str:
    val = os.environ.get(key, "")
    if not val:
        raise RuntimeError(f"Missing env var: {key}")
    return val


def _quote_ctx():
    from longbridge.openapi import Config, QuoteContext
    cfg = Config.from_env()
    return QuoteContext(cfg)


def _trade_ctx():
    from longbridge.openapi import Config, TradeContext
    cfg = Config.from_env()
    return TradeContext(cfg)


# ── Tool handlers ─────────────────────────────────────────────────────────────

def _tool_lb_quote(symbols: str) -> str:
    """获取实时报价，symbols 为逗号分隔，例如 700.HK,AAPL.US,BTC.CC"""
    sym_list = [s.strip().upper() for s in symbols.split(",") if s.strip()]
    try:
        ctx = _quote_ctx()
        quotes = ctx.quote(sym_list)
        results = []
        for q in quotes:
            results.append({
                "symbol": q.symbol,
                "last_done": str(q.last_done),
                "open": str(q.open),
                "high": str(q.high),
                "low": str(q.low),
                "volume": str(q.volume),
                "turnover": str(q.turnover),
                "timestamp": str(q.timestamp),
            })
        return json.dumps(results, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": str(e)})


def _tool_lb_candlesticks(symbol: str, period: str = "Day", count: str = "100") -> str:
    """获取K线。period: Min1/Min5/Min15/Min30/Min60/Day/Week/Month"""
    from longbridge.openapi import Period
    period_map = {
        "Min1": Period.Min1, "Min5": Period.Min5, "Min15": Period.Min15,
        "Min30": Period.Min30, "Min60": Period.Min60,
        "Day": Period.Day, "Week": Period.Week, "Month": Period.Month,
    }
    p = period_map.get(period, Period.Day)
    n = min(max(int(count), 1), 1000)
    try:
        ctx = _quote_ctx()
        candles = ctx.candlesticks(symbol.upper(), p, n, None)
        rows = []
        for c in candles:
            rows.append({
                "close": str(c.close),
                "open": str(c.open),
                "high": str(c.high),
                "low": str(c.low),
                "volume": str(c.volume),
                "turnover": str(c.turnover),
                "timestamp": str(c.timestamp),
            })
        return json.dumps({"symbol": symbol, "period": period, "count": len(rows), "candles": rows}, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": str(e)})


def _tool_lb_order_book(symbol: str) -> str:
    """获取深度行情（Level 2）"""
    try:
        ctx = _quote_ctx()
        ob = ctx.security_depth(symbol.upper())
        asks = [{"price": str(a.price), "volume": str(a.volume), "order_num": a.order_num} for a in ob.asks[:10]]
        bids = [{"price": str(b.price), "volume": str(b.volume), "order_num": b.order_num} for b in ob.bids[:10]]
        return json.dumps({"symbol": symbol, "asks": asks, "bids": bids}, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": str(e)})


def _tool_lb_submit_order(
    symbol: str,
    order_type: str,
    side: str,
    submitted_quantity: str,
    submitted_price: str = "",
    time_in_force: str = "Day",
    remark: str = "",
) -> str:
    """提交订单。order_type: LO(限价)/MO(市价)/ELO/SLO。side: Buy/Sell。time_in_force: Day/GTC/GTD"""
    from longbridge.openapi import OrderSide, OrderType, TimeInForceType
    from decimal import Decimal

    side_map = {"Buy": OrderSide.Buy, "Sell": OrderSide.Sell}
    type_map = {
        "LO": OrderType.LO, "MO": OrderType.MO,
        "ELO": OrderType.ELO, "SLO": OrderType.SLO,
        "AO": OrderType.AO, "AON": OrderType.AON,
    }
    tif_map = {
        "Day": TimeInForceType.Day,
        "GTC": TimeInForceType.GoodTilCanceled,
        "GTD": TimeInForceType.GoodTilDate,
    }
    try:
        ctx = _trade_ctx()
        px = Decimal(submitted_price) if submitted_price else None
        resp = ctx.submit_order(
            symbol=symbol.upper(),
            order_type=type_map.get(order_type, OrderType.LO),
            side=side_map.get(side, OrderSide.Buy),
            submitted_quantity=Decimal(submitted_quantity),
            time_in_force=tif_map.get(time_in_force, TimeInForceType.Day),
            submitted_price=px,
            remark=remark or None,
        )
        return json.dumps({"order_id": resp.order_id}, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": str(e)})


def _tool_lb_cancel_order(order_id: str) -> str:
    """撤销订单"""
    try:
        ctx = _trade_ctx()
        ctx.cancel_order(order_id)
        return json.dumps({"status": "cancelled", "order_id": order_id})
    except Exception as e:
        return json.dumps({"error": str(e)})


def _tool_lb_account_balance(currency: str = "") -> str:
    """查询账户余额。currency: HKD/USD/CNH（可选）"""
    try:
        ctx = _trade_ctx()
        balances = ctx.account_balance(currency or None)
        result = []
        for b in balances:
            result.append({
                "currency": b.currency,
                "total_cash": str(b.total_cash),
                "max_finance_amount": str(b.max_finance_amount),
                "remaining_finance_amount": str(b.remaining_finance_amount),
                "buying_power": str(b.buying_power) if hasattr(b, "buying_power") else "",
            })
        return json.dumps(result, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": str(e)})


def _tool_lb_positions(symbol: str = "") -> str:
    """查询持仓。symbol 可选过滤"""
    try:
        ctx = _trade_ctx()
        resp = ctx.stock_positions(symbols=[symbol.upper()] if symbol else None)
        result = []
        for ch in resp.channels:
            for pos in ch.positions:
                result.append({
                    "symbol": pos.symbol,
                    "symbol_name": pos.symbol_name,
                    "quantity": str(pos.quantity),
                    "available_quantity": str(pos.available_quantity),
                    "currency": pos.currency,
                    "cost_price": str(pos.cost_price),
                    "market_value": str(pos.market_value),
                    "unrealized_pl": str(pos.unrealized_pl),
                    "unrealized_pl_ratio": str(pos.unrealized_pl_ratio),
                })
        return json.dumps(result, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": str(e)})


def _tool_lb_today_orders(symbol: str = "", status: str = "") -> str:
    """查询当日订单。status 可选: NotReported/ReportRejected/Normal/Wait/Filled/Cancelled"""
    from longbridge.openapi import OrderStatus
    status_map = {
        "NotReported": OrderStatus.NotReported,
        "ReportRejected": OrderStatus.ReportRejected,
        "Normal": OrderStatus.Normal,
        "Wait": OrderStatus.Wait,
        "Filled": OrderStatus.Filled,
        "Cancelled": OrderStatus.Cancelled,
    }
    try:
        ctx = _trade_ctx()
        orders = ctx.today_orders(
            symbol=symbol.upper() or None,
            status=[status_map[status]] if status in status_map else None,
        )
        result = []
        for o in orders:
            result.append({
                "order_id": o.order_id,
                "symbol": o.symbol,
                "side": str(o.side),
                "order_type": str(o.order_type),
                "submitted_quantity": str(o.submitted_quantity),
                "executed_quantity": str(o.executed_quantity),
                "submitted_price": str(o.submitted_price),
                "executed_price": str(o.executed_price) if hasattr(o, "executed_price") else "",
                "status": str(o.status),
                "created_at": str(o.created_at),
            })
        return json.dumps(result, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": str(e)})


# ── MCP protocol ──────────────────────────────────────────────────────────────

TOOLS: list[dict[str, Any]] = [
    {
        "name": "lb_quote",
        "description": "获取长桥股票/ETF/加密货币实时报价，symbols 逗号分隔，例如 700.HK,AAPL.US",
        "inputSchema": {
            "type": "object",
            "properties": {"symbols": {"type": "string", "description": "逗号分隔的证券代码"}},
            "required": ["symbols"],
        },
    },
    {
        "name": "lb_candlesticks",
        "description": "获取长桥K线数据",
        "inputSchema": {
            "type": "object",
            "properties": {
                "symbol": {"type": "string", "description": "证券代码，例如 700.HK"},
                "period": {"type": "string", "description": "周期: Min1/Min5/Min15/Min30/Min60/Day/Week/Month", "default": "Day"},
                "count": {"type": "string", "description": "数量，最多1000", "default": "100"},
            },
            "required": ["symbol"],
        },
    },
    {
        "name": "lb_order_book",
        "description": "获取长桥深度行情（Level 2）",
        "inputSchema": {
            "type": "object",
            "properties": {"symbol": {"type": "string", "description": "证券代码"}},
            "required": ["symbol"],
        },
    },
    {
        "name": "lb_submit_order",
        "description": "通过长桥提交交易订单",
        "inputSchema": {
            "type": "object",
            "properties": {
                "symbol": {"type": "string", "description": "证券代码，例如 700.HK"},
                "order_type": {"type": "string", "description": "订单类型: LO(限价)/MO(市价)/ELO/SLO"},
                "side": {"type": "string", "description": "方向: Buy/Sell"},
                "submitted_quantity": {"type": "string", "description": "委托数量（股/手）"},
                "submitted_price": {"type": "string", "description": "委托价格（市价单可不填）", "default": ""},
                "time_in_force": {"type": "string", "description": "有效期: Day/GTC/GTD", "default": "Day"},
                "remark": {"type": "string", "description": "备注", "default": ""},
            },
            "required": ["symbol", "order_type", "side", "submitted_quantity"],
        },
    },
    {
        "name": "lb_cancel_order",
        "description": "撤销长桥订单",
        "inputSchema": {
            "type": "object",
            "properties": {"order_id": {"type": "string", "description": "订单ID"}},
            "required": ["order_id"],
        },
    },
    {
        "name": "lb_account_balance",
        "description": "查询长桥账户余额",
        "inputSchema": {
            "type": "object",
            "properties": {"currency": {"type": "string", "description": "货币: HKD/USD/CNH（可选）", "default": ""}},
        },
    },
    {
        "name": "lb_positions",
        "description": "查询长桥账户持仓",
        "inputSchema": {
            "type": "object",
            "properties": {"symbol": {"type": "string", "description": "过滤证券代码（可选）", "default": ""}},
        },
    },
    {
        "name": "lb_today_orders",
        "description": "查询长桥当日订单",
        "inputSchema": {
            "type": "object",
            "properties": {
                "symbol": {"type": "string", "description": "过滤证券代码（可选）", "default": ""},
                "status": {"type": "string", "description": "过滤状态: Filled/Cancelled/Normal/Wait（可选）", "default": ""},
            },
        },
    },
]

TOOL_HANDLERS: dict[str, Any] = {
    "lb_quote": _tool_lb_quote,
    "lb_candlesticks": _tool_lb_candlesticks,
    "lb_order_book": _tool_lb_order_book,
    "lb_submit_order": _tool_lb_submit_order,
    "lb_cancel_order": _tool_lb_cancel_order,
    "lb_account_balance": _tool_lb_account_balance,
    "lb_positions": _tool_lb_positions,
    "lb_today_orders": _tool_lb_today_orders,
}


def _send(obj: dict) -> None:
    sys.stdout.write(json.dumps(obj, ensure_ascii=False) + "\n")
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
                "serverInfo": {"name": "longbridge-mcp", "version": "1.0.0"},
            },
        }

    if method == "tools/list":
        return {"jsonrpc": "2.0", "id": req_id, "result": {"tools": TOOLS}}

    if method == "tools/call":
        params = request.get("params", {})
        name = params.get("name", "")
        args = params.get("arguments", {})
        handler = TOOL_HANDLERS.get(name)
        if handler is None:
            return {"jsonrpc": "2.0", "id": req_id, "error": {"code": -32601, "message": f"Tool not found: {name}"}}
        try:
            result = handler(**args)
        except Exception as exc:
            result = json.dumps({"error": str(exc)})
        return {"jsonrpc": "2.0", "id": req_id, "result": {"content": [{"type": "text", "text": result}]}}

    if method == "notifications/initialized":
        return None

    return {"jsonrpc": "2.0", "id": req_id, "error": {"code": -32601, "message": f"Method not found: {method}"}}


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
