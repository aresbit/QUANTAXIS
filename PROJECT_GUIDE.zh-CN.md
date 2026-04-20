# QUANTAXIS 项目上手指南

这个仓库现在是一个面向中国 A 股研究的极简版本，核心用途只有三类：

- 获取 A 股行情
- 在本地做纸面交易
- 在本地做历史回测并输出图表

它不是原来那个大而全的 QUANTAXIS 平台，而是一个已经裁剪过的研究内核。

## 1. 环境准备

项目使用 `uv` 管理 Python 环境和依赖。

先确认你本机已经安装 `uv`：

```bash
uv --version
```

安装基础依赖：

```bash
uv sync
```

如果你要跑历史数据回测和图表输出，需要安装研究依赖：

```bash
uv sync --extra research
```

如果你以后要接 `easytrader` 实盘适配口，再安装：

```bash
uv sync --extra live
```

## 2. 常用命令

查看可用命令：

```bash
uv run python -m QUANTAXIS --help
```

查看回测命令帮助：

```bash
uv run python -m QUANTAXIS backtest --help
```

## 3. 最短回测流程

### 方式 A：使用项目自带示例 CSV

```bash
uv run python -m QUANTAXIS backtest \
  --csv data/sample_ohlcv.csv \
  --plot outputs/sample_backtest.png \
  --export-equity outputs/sample_equity.csv
```

这条命令会做三件事：

- 读取 `data/sample_ohlcv.csv`
- 运行策略回测
- 输出图表和净值曲线 CSV

### 方式 B：直接抓取 A 股历史数据回测

```bash
uv run python -m QUANTAXIS backtest \
  --symbol 000001 \
  --start 2020-01-01 \
  --end 2024-12-31 \
  --plot outputs/000001_backtest.png \
  --export-equity outputs/000001_equity.csv
```

这条命令是当前项目最常用的入口。

它的含义是：

- `uv run`
  使用 `uv` 管理的虚拟环境执行命令，不污染系统 Python。

- `python -m QUANTAXIS`
  以模块方式启动项目主入口。

- `backtest`
  调用回测子命令。

- `--symbol 000001`
  指定要回测的股票代码。这里的 `000001` 是平安银行。

- `--start 2020-01-01`
  回测起始日期。

- `--end 2024-12-31`
  回测结束日期。

- `--plot outputs/000001_backtest.png`
  把回测图保存到这个 PNG 文件。

- `--export-equity outputs/000001_equity.csv`
  把净值曲线、信号和仓位序列导出到这个 CSV 文件。

## 4. 回测输入数据要求

如果你使用 `--csv`，CSV 至少要有这几列：

```text
datetime,open,high,low,close,volume
```

字段中文含义：

- `datetime`：时间
- `open`：开盘价
- `high`：最高价
- `low`：最低价
- `close`：收盘价
- `volume`：成交量

## 5. 回测输出内容

回测结束后，终端会输出一段 JSON 结果。

常见字段含义如下：

- `bars`
  使用了多少根 K 线

- `trades`
  总成交次数

- `final_equity`
  回测结束时账户总资产

- `total_return`
  总收益率

- `annual_return`
  年化收益率

- `max_drawdown`
  最大回撤

- `sharpe`
  夏普比率

- `equity_curve`
  每根 K 线对应的净值、信号、仓位轨迹

- `trades_log`
  每一笔交易的明细

## 6. 图表中文含义

如果你指定了 `--plot`，程序会输出一张四联图。

### 第一张：价格与买卖点

含义：

- 蓝线：收盘价走势
- 绿色上三角：买点
- 红色下三角：卖点

你主要看两件事：

- 策略是不是在明显弱势时乱买
- 策略的买卖点是否和趋势方向基本一致

### 第二张：信号曲线

含义：

- 橙线：策略信号值
- 黑色水平线：0 轴

中文理解：

- 信号大于 0，说明模型偏多
- 信号小于 0，说明模型偏空
- 信号越接近阈值，说明越接近开仓或平仓条件

这张图用来判断：

- 信号是否平滑
- 信号是否频繁抖动
- 信号是否过度滞后

### 第三张：净值曲线

含义：

- 绿色线：账户总资产变化

中文理解：

- 向上越稳定越好
- 突然向下说明有损失或回撤
- 横着走说明没有仓位或者策略没有赚到钱

这是判断策略是否赚钱的主图。

### 第四张：回撤图

含义：

- 红色阴影：从历史净值高点回落的幅度

中文理解：

- 越接近 0 越好
- 越往下说明回撤越大
- 如果长时间在深回撤区，说明策略恢复能力差

这是判断策略风险是否可接受的主图。

## 7. 当前策略是什么

当前回测默认使用：

- 递归状态更新
- attention 风格序列聚合
- 缠论相关结构特征

更具体一点：

- 用最近一段 K 线序列生成上下文
- 用分型、结构偏置、波动和成交量特征描述市场状态
- 用一个递归 hidden state 累积历史信息
- 最后输出一个 `signal`

这个 `signal` 决定是否开仓、持仓、平仓。

## 8. 常见调参入口

你现在最常调的参数有：

```bash
uv run python -m QUANTAXIS backtest \
  --symbol 000001 \
  --start 2020-01-01 \
  --end 2024-12-31 \
  --sequence-length 64 \
  --hidden-dim 16 \
  --buy-threshold 0.05 \
  --sell-threshold -0.05 \
  --trade-size 100 \
  --plot outputs/000001_backtest.png
```

参数中文解释：

- `--sequence-length`
  看多少根历史 K 线

- `--hidden-dim`
  递归状态向量维度

- `--buy-threshold`
  多头开仓阈值

- `--sell-threshold`
  平仓或空头阈值

- `--trade-size`
  每次交易多少股

- `--allow-short`
  是否允许做空

说明：

- A 股普通股票研究时，通常先不要开 `--allow-short`
- 阈值越低，交易越频繁
- 阈值越高，交易越保守

## 9. 推荐使用顺序

建议按这个顺序上手：

1. 先跑示例 CSV，确认环境没问题。
2. 再跑真实 A 股历史数据。
3. 看图，不要先盯收益率。
4. 先调阈值和序列长度，再调 hidden 维度。
5. 等图看起来合理后，再做更系统的参数搜索。

## 10. 常见问题

### 1. 为什么没有出图？

先确认你装了研究依赖：

```bash
uv sync --extra research
```

### 2. 为什么抓不到历史数据？

原因通常有三类：

- 当前网络不通
- 数据源接口临时变化
- 股票代码或日期范围有问题

### 3. 为什么有信号但没有交易？

常见原因：

- 阈值太高
- `trade_size` 太大
- 当前配置是 `long-only`
- 信号没有真正穿越阈值

### 4. 为什么收益很差？

这是正常现象。现在这版是研究骨架，不是成品策略。

你应该优先检查：

- 信号是否稳定
- 买卖点是否合理
- 回撤是否过大
- 参数是否过拟合

## 11. 你接下来最值得做的事

如果你准备继续深入，优先顺序建议是：

1. 先拿 3 到 5 只 A 股做横向验证。
2. 增加 walk-forward 回测。
3. 增加参数搜索。
4. 把当前固定 readout 改成可训练参数。
5. 把缠论特征从轻量版升级为真正的笔/线段/中枢状态机。

## 12. 一条够用的命令

如果你只记一条命令，就记这个：

```bash
uv run python -m QUANTAXIS backtest \
  --symbol 000001 \
  --start 2020-01-01 \
  --end 2024-12-31 \
  --plot outputs/000001_backtest.png \
  --export-equity outputs/000001_equity.csv
```

它会直接给你：

- 历史数据
- 回测结果
- 中文可解释的图
- 后续可分析的净值 CSV
