from __future__ import annotations

import argparse

import pandas as pd

from QUANTAXIS.backtest.data import load_ohlcv_csv
from QUANTAXIS.QAStrategy.qamultibase import QAStrategyStockBase


class DemoMultiStockStrategy(QAStrategyStockBase):
    def user_init(self):
        self._seen = {}

    def on_bar(self, bar):
        code = bar.name[1]
        price = float(bar["close"])
        position = self.get_positions(code)
        seen = self._seen.get(code, 0) + 1
        self._seen[code] = seen
        self.plot(f"{code}_close", price, "line")
        self.plot(f"{code}_position_long", position.volume_long, "line")
        if seen == 1:
            self.send_order(direction="BUY", offset="OPEN", code=code, price=price, volume=100)
        elif seen == 5 and position.volume_long > 0:
            self.send_order(direction="SELL", offset="CLOSE", code=code, price=price, volume=100)

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


def _build_demo_frame(csv_path: str, codes: list[str]) -> pd.DataFrame:
    base = load_ohlcv_csv(csv_path)
    frames: list[pd.DataFrame] = []
    for idx, code in enumerate(codes):
        frame = base.copy()
        scale = 1.0 + idx * 0.02
        for column in ["open", "high", "low", "close"]:
            frame[column] = (frame[column] * scale).round(2)
        frame["code"] = code
        frames.append(frame)
    return pd.concat(frames, ignore_index=True).sort_values(["datetime", "code"]).reset_index(drop=True)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the legacy multi-symbol stock strategy base on local data with SQLite snapshots."
    )
    parser.add_argument("--csv", default="data/sample_ohlcv.csv", help="Base OHLCV csv used to construct a demo multi-symbol panel.")
    parser.add_argument("--codes", nargs="+", default=["000001", "000002"], help="Codes to include in the local backtest.")
    parser.add_argument("--sqlite-path", default="outputs/legacy_multisymbol_demo.sqlite3", help="SQLite path for persisted account snapshots.")
    parser.add_argument("--plot", default="outputs/legacy_multisymbol_demo.png", help="PNG output path for the legacy strategy figure.")
    parser.add_argument("--init-cash", type=float, default=200000.0, help="Initial cash for the local account.")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    frame = _build_demo_frame(args.csv, args.codes)
    strategy = DemoMultiStockStrategy(
        code=args.codes,
        frequence="day",
        start="2020-01-01",
        end="2024-12-31",
        init_cash=args.init_cash,
        backtest_backend="sqlite",
        sqlite_path=args.sqlite_path,
        backtest_data=frame,
    )
    snapshot = strategy.run_backtest()
    figure = strategy.save_plot(args.plot, title="Legacy Multi-Symbol CTA")
    print("account_cookie:", snapshot["account_cookie"])
    print("trading_day:", snapshot["trading_day"])
    print("available:", round(snapshot["accounts"]["available"], 2))
    print("balance:", round(snapshot["accounts"]["balance"], 2))
    print("position_keys:", ",".join(sorted(snapshot["positions"].keys())))
    print("plot:", figure)


if __name__ == "__main__":
    main()
