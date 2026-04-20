from __future__ import annotations

import argparse

from QUANTAXIS.QAStrategy.qactabase import QAStrategyCtaBase


class DemoLegacyCta(QAStrategyCtaBase):
    def on_bar(self, bar):
        price = float(bar["close"])
        position = self.get_positions(self.get_code())
        self.plot("close", price, "line")
        self.plot("position_long", position.volume_long, "line")
        if self.bar_id == 1:
            self.send_order(direction="BUY", offset="OPEN", price=price, volume=100)
        elif self.bar_id == 5 and position.volume_long > 0:
            self.send_order(direction="SELL", offset="CLOSE", price=price, volume=100)

    def on_tick(self, tick):
        pass

    def on_1min_bar(self):
        pass

    def on_5min_bar(self):
        pass

    def on_15min_bar(self):
        pass

    def on_30min_bar(self):
        pass


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the legacy CTA base on local CSV data with a SQLite-backed QIFI account."
    )
    parser.add_argument("--csv", default="data/sample_ohlcv.csv", help="OHLCV csv with datetime/open/high/low/close/volume")
    parser.add_argument("--code", default="000001", help="Instrument code used by the strategy account.")
    parser.add_argument("--sqlite-path", default="outputs/legacy_cta_demo.sqlite3", help="SQLite path for persisted account snapshots.")
    parser.add_argument("--plot", default="outputs/legacy_cta_demo.png", help="PNG output path for the legacy strategy figure.")
    parser.add_argument("--init-cash", type=float, default=100000.0, help="Initial cash for the local account.")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    strategy = DemoLegacyCta(
        code=args.code,
        frequence="day",
        start="2020-01-01",
        end="2024-12-31",
        init_cash=args.init_cash,
        backtest_backend="sqlite",
        sqlite_path=args.sqlite_path,
        backtest_csv=args.csv,
    )
    snapshot = strategy.run_backtest()
    figure = strategy.save_plot(args.plot, title=f"Legacy CTA {args.code}")
    print("account_cookie:", snapshot["account_cookie"])
    print("trading_day:", snapshot["trading_day"])
    print("available:", round(snapshot["accounts"]["available"], 2))
    print("balance:", round(snapshot["accounts"]["balance"], 2))
    print("plot:", figure)


if __name__ == "__main__":
    main()
