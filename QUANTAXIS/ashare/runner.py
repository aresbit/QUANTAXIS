from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from dataclasses import asdict

import yaml

from QUANTAXIS.ashare.broker import EasyTraderBroker, Order, PaperBroker
from QUANTAXIS.ashare.quotes import PytdxQuoteClient, make_manual_quote


def _build_quote_client(config: dict[str, Any]) -> PytdxQuoteClient:
    quote_config = config.get("quote", {})
    provider = quote_config.get("provider", "pytdx")
    if provider != "pytdx":
        raise ValueError(f"unsupported quote provider: {provider}")
    return PytdxQuoteClient(
        host=quote_config.get("host"),
        port=int(quote_config.get("port", 7709)),
        timeout=float(quote_config.get("timeout", 0.7)),
    )


def _build_broker(config: dict[str, Any]):
    broker_config = config.get("broker", {})
    kind = broker_config.get("kind", "paper")
    if kind == "paper":
        return PaperBroker(
            initial_cash=float(broker_config.get("initial_cash", 1_000_000)),
            commission_rate=float(broker_config.get("commission_rate", 0.0003)),
            min_commission=float(broker_config.get("min_commission", 5.0)),
            stamp_duty_rate=float(broker_config.get("stamp_duty_rate", 0.001)),
            slippage=float(broker_config.get("slippage", 0.0)),
        )
    if kind == "easytrader":
        return EasyTraderBroker(
            client=broker_config["client"],
            prepare=broker_config.get("prepare", {}),
        )
    raise ValueError(f"unsupported broker kind: {kind}")


def run_once_from_config(config_path: str | Path) -> dict[str, Any]:
    path = Path(config_path)
    config = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    quote_client = _build_quote_client(config)
    broker = _build_broker(config)

    reports: list[dict[str, Any]] = []
    for entry in config.get("orders", []):
        order = Order(
            symbol=str(entry["symbol"]),
            side=str(entry["side"]),
            amount=int(entry["amount"]),
            price=None if entry.get("price") in (None, "market") else float(entry["price"]),
        )
        quote = (
            make_manual_quote(order.symbol, order.price, source="config")
            if order.price is not None
            else quote_client.get_quote(order.symbol)
        )
        report = broker.submit(order, quote)
        reports.append({"order": asdict(order), "quote": quote.as_dict(), "report": report.as_dict()})

    result: dict[str, Any] = {"reports": reports}
    account = getattr(broker, "account", None)
    if account is not None:
        result["account"] = account.as_dict()
    return result


def dump_run_result(result: dict[str, Any]) -> str:
    return json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True)
