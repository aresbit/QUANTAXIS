# QUANTAXIS

这个仓库已经被裁成一个纯本地、面向中国 A 股执行的最小版本。

保留内容：

- `pytdx` 行情
- `paper` 本地撮合账户
- `easytrader` 实盘适配口
- YAML 批量下单入口
- 本地 CSV 回测
- 递归 `quant transformer` 风格研究策略
- 历史 A 股日线抓取
- 回测图直接导出 PNG

移除内容：

- Web
- Docker
- 文档站
- 回测/因子/分析/多市场杂项
- Rust 桥接和其他非交易路径

## 用 `uv`

安装依赖：

```bash
uv sync
```

如果要接实盘：

```bash
uv sync --extra live
```

查看行情：

```bash
uv run python -m QUANTAXIS quote 000001
```

本地纸面下单：

```bash
uv run python -m QUANTAXIS paper-buy 000001 100 --price 12.34
```

配置批量执行：

```bash
uv run python -m QUANTAXIS run --config config.paper.yaml

回测：

```bash
uv run python -m QUANTAXIS backtest --csv data/sample_ohlcv.csv
```

直接抓 A 股历史数据回测并画图：

```bash
uv run python -m QUANTAXIS backtest \
  --symbol 000001 \
  --start 2020-01-01 \
  --end 2024-12-31 \
  --plot outputs/000001_backtest.png
```
```

## 配置

`config.paper.yaml`

```yaml
quote:
  provider: pytdx

broker:
  kind: paper
  initial_cash: 1000000

orders:
  - symbol: "000001"
    side: buy
    amount: 100
    price: 12.34
```

`config.easytrader.yaml`

```yaml
quote:
  provider: pytdx

broker:
  kind: easytrader
  client: ths
  prepare:
    exe_path: "C:/同花顺软件/xiadan.exe"

orders:
  - symbol: "000001"
    side: buy
    amount: 100
    price: market
```

## 说明

- 当前环境里 `paper` 路径可以完全本地验证。
- `quote` 和 `price: market` 依赖 TDX 网络连通性。
- `easytrader` 需要你自己的券商终端和登录态，只能在你的交易机上联调。
- A 股手数默认限制为 `100` 的整数倍。
- 回测 CSV 需要列：`datetime,open,high,low,close,volume`
- 历史日线回测支持 `AKShare` 抓取：`--symbol/--start/--end`
- 可视化支持 `--plot path.png`，输出价格/买卖点/信号/净值/回撤
- 当前研究策略不是任何机构内部专有模型，而是一个可调参的本地研究骨架，融合了递归状态更新和缠论分型/笔强度特征。
