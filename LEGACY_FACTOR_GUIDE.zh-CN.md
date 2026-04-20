# QUANTAXIS Legacy 因子库上手指南

这套因子链路是旧 QUANTAXIS 生态里的研究基础设施，不是“下载后直接本地单文件回测”的轻量模块。它适合下面这类场景：

- 你准备把因子值长期存进 ClickHouse
- 你已经有股票日线、复权、行业等基础表
- 你希望复用 `QAFactor -> featureView -> featureAnalysis -> featurebacktest` 这条旧研究链

如果你只是想本地快速试策略，优先使用 [PROJECT_GUIDE.zh-CN.md](/home/ares/yys/QUANTAXIS/PROJECT_GUIDE.zh-CN.md) 里的新 `backtest` 入口。

## 1. 安装

```bash
uv sync --extra legacy
```

如果你还要使用新回测画图，可以一起装：

```bash
uv sync --extra legacy --extra research
```

## 2. 这套链路依赖什么

最关键的不是 Python 包，而是数据底座。

Python 依赖：

- `clickhouse-driver`
- `pymongo`
- `motor`
- `qaenv`
- `alphalens`
- `statsmodels`
- `scipy`

基础服务：

- ClickHouse
- MongoDB

环境变量来源：

- `qaenv`

代码里默认直接从 `qaenv` 取这些配置：

- `clickhouse_ip`
- `clickhouse_port`
- `clickhouse_user`
- `clickhouse_password`
- `mongo_ip`

## 3. 数据库约定

因子值默认保存在 ClickHouse 的 `factor` 数据库。

核心元数据表：

- `factor.factormetadata`

单因子表默认结构：

- `date`
- `code`
- `factor`

行情与研究辅助表默认走 ClickHouse 的 `quantaxis` 数据库，例如：

- `quantaxis.stock_cn_day`
- `quantaxis.stock_adj`
- `quantaxis.citis_industry`
- `quantaxis.index_weight`

这些约定直接写在下面几个文件里：

- [QUANTAXIS/QAFactor/feature.py](/home/ares/yys/QUANTAXIS/QUANTAXIS/QAFactor/feature.py)
- [QUANTAXIS/QAFactor/featureView.py](/home/ares/yys/QUANTAXIS/QUANTAXIS/QAFactor/featureView.py)
- [QUANTAXIS/QAFetch/QAClickhouse.py](/home/ares/yys/QUANTAXIS/QUANTAXIS/QAFetch/QAClickhouse.py)
- [QUANTAXIS/QAFactor/featureAnalysis.py](/home/ares/yys/QUANTAXIS/QUANTAXIS/QAFactor/featureAnalysis.py)
- [QUANTAXIS/QAFactor/featurebacktest.py](/home/ares/yys/QUANTAXIS/QUANTAXIS/QAFactor/featurebacktest.py)

## 4. 因子链路怎么走

### `QASingleFactor_DailyBase`

这是单日频因子基类。你自己继承它，实现 `calc()`，返回一个包含以下列的 `DataFrame`：

- `date`
- `code`
- `factor`

它会负责：

- 注册因子元数据
- 初始化 ClickHouse 因子表
- 将因子值写入 `factor.<factor_name>`

### `QAFeatureView`

它负责读回因子库，适合做：

- 查看已存在因子
- 拉取单因子
- 组合多个因子

### `QAFeatureAnalysis`

它负责分析层，适合做：

- forward return 对齐
- rank 化
- IC / IR 统计
- Alphalens tear sheet
- 行业中性化前的数据准备

### `QAFeatureBacktest`

它是一个旧式因子回测器，逻辑是：

- 每天按因子分位数选股票
- 次日开盘买入
- 持有固定天数后卖出
- 通过 `QIFI_Account` 记账

注意它不是通用策略回测器，而是偏“横截面选股因子回测”。

## 5. 最小使用示例

仓库里已经放了一个最小脚本：

- [examples/legacy_factor_demo.py](/home/ares/yys/QUANTAXIS/examples/legacy_factor_demo.py)

只看帮助：

```bash
uv run python examples/legacy_factor_demo.py --help
```

列出当前因子库：

```bash
uv run python examples/legacy_factor_demo.py list-factors
```

查看单个因子前几行：

```bash
uv run python examples/legacy_factor_demo.py show-factor MA10 --head 20
```

注意：这些命令只有在你的 `qaenv` 和 ClickHouse 表都可用时才会成功。

## 6. 什么时候用新回测，什么时候用旧因子链

用新回测：

- 你只想拿 CSV 或 AKShare 历史数据快速测策略
- 你要直接出 PNG 图
- 你不想维护 Mongo/ClickHouse/qaenv

用旧因子链：

- 你已经接受 ClickHouse + Mongo 的旧栈
- 你在做日频横截面因子研究
- 你要复用 `QAFeatureAnalysis` 和 `QAFeatureBacktest`

## 7. 常见坑

- `QAFactor` 能 import 不等于能直接跑；真正的门槛在数据库表和 `qaenv`
- `featureAnalysis` 依赖 `alphalens`
- `featurebacktest` 默认使用 `QACKClient` 取 ClickHouse 数据，不会自动回退到 CSV
- 这套旧链更偏“研究基础设施”，不是零配置产品

## 8. 推荐实践

- 新策略原型先用新 `backtest`
- 因子稳定后，再决定是否入旧 `QAFactor` 库
- 不要拿旧因子链替代现在的轻量回测入口；两者解决的是不同问题
