from __future__ import annotations

import argparse
import json

from QUANTAXIS.ashare.broker import Order, PaperBroker
from QUANTAXIS.ashare.quotes import PytdxQuoteClient, make_manual_quote
from QUANTAXIS.ashare.runner import dump_run_result, run_once_from_config


def _add_quote_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser("quote", help="fetch a single A-share quote from pytdx")
    parser.add_argument("symbol")
    parser.add_argument("--host")
    parser.add_argument("--port", type=int, default=7709)
    parser.set_defaults(command="quote")


def _add_paper_order_parser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser], command: str
) -> None:
    parser = subparsers.add_parser(command, help=f"submit a {command.split('-')[1]} order to the paper broker")
    parser.add_argument("symbol")
    parser.add_argument("amount", type=int)
    parser.add_argument("--price", type=float)
    parser.add_argument("--initial-cash", type=float, default=1_000_000)
    parser.add_argument("--host")
    parser.add_argument("--port", type=int, default=7709)
    parser.set_defaults(command=command)


def _add_run_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser("run", help="run a batch of orders from a YAML config")
    parser.add_argument("--config", required=True)
    parser.set_defaults(command="run")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="quantaxis-a")
    subparsers = parser.add_subparsers(dest="command", required=True)
    _add_quote_parser(subparsers)
    _add_paper_order_parser(subparsers, "paper-buy")
    _add_paper_order_parser(subparsers, "paper-sell")
    _add_run_parser(subparsers)
    return parser


def _run_paper_order(args: argparse.Namespace) -> int:
    quote = (
        make_manual_quote(args.symbol, args.price)
        if args.price is not None
        else PytdxQuoteClient(host=args.host, port=args.port).get_quote(args.symbol)
    )
    broker = PaperBroker(initial_cash=args.initial_cash)
    side = "buy" if args.command == "paper-buy" else "sell"
    order = Order(symbol=args.symbol, side=side, amount=args.amount, price=args.price)
    report = broker.submit(order, quote)
    print(
        json.dumps(
            {
                "quote": quote.as_dict(),
                "report": report.as_dict(),
                "account": broker.account.as_dict(),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "quote":
        quote = PytdxQuoteClient(host=args.host, port=args.port).get_quote(args.symbol)
        print(json.dumps(quote.as_dict(), ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    if args.command in {"paper-buy", "paper-sell"}:
        return _run_paper_order(args)
    if args.command == "run":
        print(dump_run_result(run_once_from_config(args.config)))
        return 0
    parser.error(f"unsupported command: {args.command}")
    return 2
