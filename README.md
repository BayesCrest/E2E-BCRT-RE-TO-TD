# E2E-BCRT-RE-TO-TD

从研究到交易的端到端动量交易自动化系统示例。

## 功能概览

- 使用 `yfinance` 或本地 CSV 文件加载历史价格数据。
- 根据设定的动量窗口对标的排序，并可限制持仓数量与单一权重上限。
- 简化的投资组合管理与资金曲线跟踪，考虑交易成本。
- 命令行界面支持自定义交易参数并输出回测结果。

## 安装依赖

```bash
pip install -r requirements.txt
```

## 运行示例

```bash
python main.py \
  --tickers AAPL MSFT QQQ \
  --start 2018-01-01 \
  --end 2023-01-01 \
  --lookback 126 \
  --top-n 3 \
  --rebalance-days 5 \
  --capital 100000
```

如需在离线环境运行，可准备包含 `Adj Close`/`Close` 列的 CSV 文件并放入目录，例如 `data/AAPL.csv`。然后使用：

```bash
python main.py --tickers AAPL MSFT --start 2018-01-01 --end 2023-01-01 --csv-dir data/
```

## 目录结构

```
momentum_trader/
  ├── broker.py          # 执行层（模拟券商）
  ├── config.py          # 配置对象定义
  ├── data.py            # 数据加载实现
  ├── engine.py          # 交易引擎协调模块
  ├── portfolio.py       # 投资组合与订单管理
  ├── strategy.py        # 动量策略逻辑
  └── __init__.py
main.py                  # 命令行入口
requirements.txt         # 依赖列表
```

## 注意事项

- 默认使用 Yahoo Finance 数据，需要网络访问；若无网络请使用 CSV 模式。
- 示例程序用于教学与研究目的，未包含真实交易所需的风控、委托处理等复杂逻辑。
