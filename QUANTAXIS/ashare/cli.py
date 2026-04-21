from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from QUANTAXIS.ashare.broker import Order, PaperBroker
from QUANTAXIS.ashare.quotes import PytdxQuoteClient, make_manual_quote
from QUANTAXIS.ashare.runner import dump_run_result, run_once_from_config
from QUANTAXIS.backtest.data import fetch_ashare_bars, fetch_ashare_portfolio_bars, load_multi_ohlcv_csv, load_ohlcv_csv
from QUANTAXIS.backtest.engine import run_backtest
from QUANTAXIS.backtest.plot import save_backtest_figure
from QUANTAXIS.backtest.strategy import RecursiveQTransformerStrategy, StrategyConfig


STYLE_GROUPS = {
    "600104": "auto",
    "601238": "auto",
    "601633": "auto",
    "601127": "auto",
    "601012": "new_energy",
    "600732": "new_energy",
    "603799": "new_energy",
    "600460": "chip",
    "600584": "chip",
    "601117": "legacy_92",
}


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
    parser.add_argument("--multi-csv")
    parser.add_argument("--symbol")
    parser.add_argument("--symbols", nargs="+")
    parser.add_argument("--start")
    parser.add_argument("--end")
    parser.add_argument("--adjust", default="")
    parser.add_argument("--source", default="auto", choices=["auto", "akshare", "pytdx"])
    parser.add_argument("--frequency", default="day", choices=["1min", "5min", "15min", "30min", "60min", "day"])
    parser.add_argument("--portfolio-size", type=int, default=3)
    parser.add_argument("--sequence-length", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=8)
    parser.add_argument("--fractal-window", type=int, default=5)
    parser.add_argument("--buy-threshold", type=float, default=0.02)
    parser.add_argument("--sell-threshold", type=float, default=-0.03)
    parser.add_argument("--trade-size", type=int, default=100)
    parser.add_argument("--rank-temperature", type=float, default=6.0)
    parser.add_argument("--max-position-weight", type=float, default=0.18)
    parser.add_argument("--rebalance-buffer", type=float, default=0.02)
    parser.add_argument("--gross-exposure", type=float, default=0.95)
    parser.add_argument("--hold-threshold", type=float, default=-0.01)
    parser.add_argument("--min-holding-bars", type=int, default=6)
    parser.add_argument("--fft-event-threshold", type=float, default=0.50)
    parser.add_argument("--fft-regime-threshold", type=float, default=0.35)
    parser.add_argument("--per-group-limit", type=int, default=1)
    parser.add_argument("--group", action="append", default=[], help="override group as SYMBOL:GROUP")
    parser.add_argument("--initial-cash", type=float, default=1_000_000)
    parser.add_argument("--allow-short", action="store_true")
    parser.add_argument("--export-equity")
    parser.add_argument("--plot")
    parser.set_defaults(command="backtest")


def _load_backtest_frame(args: argparse.Namespace) -> pd.DataFrame:
    if args.csv:
        return load_ohlcv_csv(args.csv)
    if args.multi_csv:
        return load_multi_ohlcv_csv(args.multi_csv)
    if args.symbols and args.start and args.end:
        return fetch_ashare_portfolio_bars(args.symbols, args.start, args.end, frequency=args.frequency, adjust=args.adjust, source=args.source)
    if args.symbol and args.start and args.end:
        return fetch_ashare_bars(args.symbol, args.start, args.end, frequency=args.frequency, adjust=args.adjust, source=args.source)
    raise ValueError("backtest requires --csv/--multi-csv or --symbol(s)/--start/--end")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="quantaxis-a")
    subparsers = parser.add_subparsers(dest="command", required=True)
    _add_quote_parser(subparsers)
    _add_paper_order_parser(subparsers, "paper-buy")
    _add_paper_order_parser(subparsers, "paper-sell")
    _add_run_parser(subparsers)
    _add_backtest_parser(subparsers)
    return parser


def _resolve_symbol_groups(args: argparse.Namespace) -> dict[str, str]:
    symbols = list(args.symbols or ([args.symbol] if args.symbol else []))
    groups = {symbol: STYLE_GROUPS.get(symbol, "default") for symbol in symbols}
    for item in args.group:
        symbol, sep, group = item.partition(":")
        if not sep or not symbol or not group:
            raise ValueError(f"invalid --group value: {item!r}, expected SYMBOL:GROUP")
        groups[symbol] = group
    return groups


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
        rank_temperature=args.rank_temperature,
        max_position_weight=args.max_position_weight,
        rebalance_buffer=args.rebalance_buffer,
        gross_exposure=args.gross_exposure,
        hold_threshold=args.hold_threshold,
        min_holding_bars=args.min_holding_bars,
        fft_event_threshold=args.fft_event_threshold,
        fft_regime_threshold=args.fft_regime_threshold,
        per_group_limit=args.per_group_limit,
        symbol_groups=_resolve_symbol_groups(args),
        allow_short=args.allow_short,
    )
    strategy = RecursiveQTransformerStrategy(config)
    bars_per_year = 252 if args.frequency == "day" else {"1min": 252 * 240, "5min": 252 * 48, "15min": 252 * 16, "30min": 252 * 8, "60min": 252 * 4}[args.frequency]
    result = run_backtest(frame, strategy, initial_cash=args.initial_cash, bars_per_year=bars_per_year, portfolio_size=args.portfolio_size)
    if args.export_equity:
        export_path = Path(args.export_equity)
        export_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(result.equity_curve).to_csv(export_path, index=False)
    if args.plot:
        title = None
        if args.csv:
            title = Path(args.csv).stem
        elif args.multi_csv:
            title = Path(args.multi_csv).stem
        elif args.symbols:
            title = "-".join(args.symbols[:3]) + ("..." if len(args.symbols) > 3 else "")
        else:
            title = args.symbol or "backtest"
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
