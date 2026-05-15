"""OKX 全天候实盘/模拟盘交易示例.

快速启动（模拟盘）:
  export OKX_API_KEY=xxx OKX_SECRET_KEY=xxx OKX_PASSPHRASE=xxx
  uv run python examples/okx_live_trading.py --simulated --bar 1H

实盘去掉 --simulated 参数。
"""

from __future__ import annotations

import argparse
import logging
import os
import signal
import sys
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

from QUANTAXIS import (
    EngineConfig,
    LiveRiskConfig,
    OKXClient,
    OKXConfig,
    OKXTrader,
    RecursiveQTransformerStrategy,
    StrategyConfig,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="OKX 全天候交易机器人")
    p.add_argument("--inst-id", default="BTC-USDT-SWAP", help="交易品种")
    p.add_argument("--bar", default="1H", choices=["1m","5m","15m","30m","1H","4H","1D"])
    p.add_argument("--leverage", type=int, default=3)
    p.add_argument("--simulated", action="store_true", help="使用模拟盘")
    p.add_argument("--dry-run", action="store_true", help="只运行单次评估，不下单")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    cfg = OKXConfig(
        api_key=os.environ.get("OKX_API_KEY", ""),
        secret_key=os.environ.get("OKX_SECRET_KEY", ""),
        passphrase=os.environ.get("OKX_PASSPHRASE", ""),
        simulated=args.simulated,
    )

    strategy = RecursiveQTransformerStrategy(StrategyConfig(
        sequence_length=32,
        buy_threshold=0.03,
        sell_threshold=-0.03,
        allow_short=True,
    ))

    risk = LiveRiskConfig(
        max_position_usdt=500.0,
        max_daily_loss_usdt=100.0,
        max_order_usdt=200.0,
        min_signal_threshold=0.03,
        cooldown_bars=3,
    )

    trader = OKXTrader(
        config=cfg,
        strategy=strategy,
        inst_id=args.inst_id,
        td_mode="cross",
        leverage=args.leverage,
        risk_config=risk,
        on_order=lambda o: print(f"[ORDER] {o.side} {o.sz} @ {o.inst_id} status={o.status}"),
    )

    # ── Dry-run mode: single evaluation, no orders placed ──
    if args.dry_run:
        result = trader.run_once(bar=args.bar)
        print("\n=== Dry-Run Result ===")
        for k, v in result.items():
            print(f"  {k}: {v}")
        return

    # ── Live/Paper mode: start 24/7 loop ──
    mode = "SIMULATED" if args.simulated else "LIVE"
    print(f"\n[{mode}] Starting OKX trader: {args.inst_id} bar={args.bar} leverage={args.leverage}x")
    print("Press Ctrl+C to stop.\n")

    session = trader.start(bar=args.bar)

    def _shutdown(sig, frame):
        print("\nStopping trader...")
        session.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    # Keep main thread alive; session loop is daemonic
    while session.is_running:
        time.sleep(5)
        if session.equity_log:
            last = session.equity_log[-1]
            print(f"  equity={last['equity']:.2f} price={last['price']:.4f} signal={last['signal']:.4f}")


if __name__ == "__main__":
    main()
