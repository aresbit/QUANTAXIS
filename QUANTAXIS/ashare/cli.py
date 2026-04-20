from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from QUANTAXIS.ashare.broker import Order, PaperBroker
from QUANTAXIS.ashare.quotes import PytdxQuoteClient, make_manual_quote
from QUANTAXIS.ashare.runner import dump_run_result, run_once_from_config
from QUANTAXIS.backtest.data import fetch_ashare_daily, load_ohlcv_csv
from QUANTAXIS.backtest.engine import run_backtest
from QUANTAXIS.backtest.plot import save_backtest_figure
from QUANTAXIS.backtest.strategy import RecursiveQTransformerStrategy, StrategyConfig


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


def _add_backtest_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser("backtest", help="run the recursive quant transformer strategy on local CSV data")
    parser.add_argument("--csv")
    parser.add_argument("--symbol")
    parser.add_argument("--start")
    parser.add_argument("--end")
    parser.add_argument("--adjust", default="")
    parser.add_argument("--sequence-length", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=8)
    parser.add_argument("--fractal-window", type=int, default=5)
    parser.add_argument("--buy-threshold", type=float, default=0.03)
    parser.add_argument("--sell-threshold", type=float, default=-0.03)
    parser.add_argument("--trade-size", type=int, default=100)
    parser.add_argument("--initial-cash", type=float, default=1_000_000)
    parser.add_argument("--allow-short", action="store_true")
    parser.add_argument("--export-equity")
    parser.add_argument("--plot")
    parser.set_defaults(command="backtest")


def _load_backtest_frame(args: argparse.Namespace) -> pd.DataFrame:
    if args.csv:
        return load_ohlcv_csv(args.csv)
    if args.symbol and args.start and args.end:
        return fetch_ashare_daily(args.symbol, args.start, args.end, adjust=args.adjust)
    raise ValueError("backtest requires either --csv or --symbol/--start/--end")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="quantaxis-a")
    subparsers = parser.add_subparsers(dest="command", required=True)
    _add_quote_parser(subparsers)
    _add_paper_order_parser(subparsers, "paper-buy")
    _add_paper_order_parser(subparsers, "paper-sell")
    _add_run_parser(subparsers)
    _add_backtest_parser(subparsers)
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


def _run_backtest(args: argparse.Namespace) -> int:
    frame = _load_backtest_frame(args)
    config = StrategyConfig(
        sequence_length=args.sequence_length,
        hidden_dim=args.hidden_dim,
        fractal_window=args.fractal_window,
        trade_size=args.trade_size,
        buy_threshold=args.buy_threshold,
        sell_threshold=args.sell_threshold,
        allow_short=args.allow_short,
    )
    strategy = RecursiveQTransformerStrategy(config)
    result = run_backtest(frame, strategy, initial_cash=args.initial_cash)
    if args.export_equity:
        export_path = Path(args.export_equity)
        export_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(result.equity_curve).to_csv(export_path, index=False)
    if args.plot:
        title = (args.symbol or Path(args.csv).stem) if args.csv else (args.symbol or "backtest")
        save_backtest_figure(result.equity_curve, result.trades_log, args.plot, title=f"RecursiveQTransformer {title}")
    print(json.dumps(result.as_dict(), ensure_ascii=False, indent=2, sort_keys=True))
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
    if args.command == "backtest":
        return _run_backtest(args)
    parser.error(f"unsupported command: {args.command}")
    return 2
