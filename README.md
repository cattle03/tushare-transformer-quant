# tushare-transformer-quant

> 用 **TuShare** 抓取 A 股数据，用 **Transformer** 做量化选股与回测的开源研究框架。
> A reproducible pipeline that turns TuShare market data into Transformer-based
> cross-sectional stock-return predictions and honest, leakage-aware backtests.

<p align="left">
  <img alt="python" src="https://img.shields.io/badge/python-3.9%2B-blue">
  <img alt="license" src="https://img.shields.io/badge/license-MIT-green">
  <img alt="torch" src="https://img.shields.io/badge/pytorch-%E2%89%A52.0-ee4c2c">
</p>

---

## 这个项目做了什么

它以「通信 / 算力」主题板块为例，完整实现了一条可复现的量化研究流水线：

1. **抓数据** — 按同花顺概念板块取成分股并集，拉取日线、每日指标、复权因子，计算前复权价。
2. **造因子** — 生成 32 个量价 / 估值 Alpha 因子（`alpha01 … alpha32`、`alpha_w_005` 等）。
3. **筛因子** — 只用**训练期**数据做单因子诊断（分组收益、IC / ICIR、多空回撤、单调性），按 `|ICIR|` 选出 top-K 再送进模型，避免未来信息选择偏差。
4. **训模型** — 每只股票按固定长度（默认 20 天）滑窗，送入 Transformer 编码器回归未来 5 日收益；采用**带隔离带（purge/embargo）的 walk-forward 交叉验证**，杜绝训练/验证边界泄漏。
5. **做回测** — 十分位分组、固定 Top-N 组合、板块等权基准对冲、以及**非重叠 5 日调仓**（挤掉重叠收益带来的 Sharpe 虚高），并把结果导出到 Excel + 图。

它把一个原本面向单机脚本的项目，整理成了结构清晰、可安装、可配置、可复现的开源包。

```
TuShare ─▶ fetch ─▶ qfq(前复权) ─▶ 32 Alpha 因子
                                     │
                        训练期因子质量筛选 (IC/ICIR/多空/单调)
                                     │
                        逐股滑窗序列 (date × stock)
                                     │
                 purged walk-forward CV · Transformer 回归
                                     │
        ┌────────────┬──────────────┼───────────────┐
      十分位分组     Top-N       等权基准对冲      非重叠5日
        └────────────┴──────────────┴───────────────┘
                                     │
                             results.xlsx + figs/
```

---

## 目录结构

```
tushare-transformer-quant/
├── tstq/
│   ├── __init__.py      # 最早设置确定性数值环境（必须在 numpy/torch 之前）
│   ├── config.py        # 全部超参数 / 板块 / 路径的 Config（支持 YAML）
│   ├── client.py        # Tushare 连接（token 走环境变量）
│   ├── fetch.py         # 股票池 + 日线 + 每日指标（断点续传）
│   ├── adjust.py        # 复权因子 + 前复权价
│   ├── factors.py       # AlphaLib(32 因子) + 因子质量诊断/筛选
│   ├── dataset.py       # 因子表构建 + 标准化 + 滑窗序列
│   ├── model.py         # Transformer 模型 / Dataset / epoch runner
│   ├── backtest.py      # 分组 / Top-N / 对冲 / 非重叠 绩效
│   ├── pipeline.py      # 端到端训练 + 回测编排
│   └── cli.py           # 命令行入口：tstq fetch|qfq|train
├── tests/test_smoke.py  # 纯函数冒烟测试（无需联网 / GPU）
├── config.example.yaml
├── requirements.txt
├── pyproject.toml
├── .env.example
└── LICENSE
```

---

## 安装

```bash
git clone https://github.com/cattle03/tushare-transformer-quant.git
cd tushare-transformer-quant

python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -e .            # 或：pip install -r requirements.txt
```

> 需要 Python ≥ 3.9。有 NVIDIA GPU 并安装 CUDA 版 PyTorch 会快很多，但 CPU 也能跑（缩小 `epochs` / `n_splits` 即可）。

---

## 配置 Token

本项目**不包含任何 token**，请通过环境变量提供：

```bash
# 方式一：复制 .env
cp .env.example .env
# 然后把 TUSHARE_TOKEN=... 改成你自己的 token（https://tushare.pro/user/token）

# 方式二：直接设环境变量
# Windows (PowerShell)
setx TUSHARE_TOKEN "your_token"
# macOS / Linux
export TUSHARE_TOKEN="your_token"
```

任何代码里都不应出现明文 token。抓取 / 复权阶段才会用到它，训练阶段完全离线。

---

## 快速开始

```bash
# 1) 抓数据：板块成分并集 -> 日线 + 每日指标
python -m tstq fetch --data-dir data

# 2) 抓复权因子并计算前复权价（产出 comm_price_qfq.csv）
python -m tstq qfq --data-dir data

# 3) 训练 + 回测（结果写进 outputs/）
python -m tstq train --data-dir data --split-date 20250101 --epochs 5
```

安装后也可以直接用 `tstq` 命令（等价于 `python -m tstq`）：

```bash
tstq train --config config.yaml --epochs 10 --top-n 20
```

> **已有数据？** 如果你之前已经跑过 `fetch` + `qfq`，只要把 `--data-dir` 指向包含
> `comm_price_qfq.csv` / `comm_daily_basic.csv` 的目录，就能直接 `train`。

---

## 输出

运行 `train` 后：

- `outputs/results.xlsx` — 多 sheet：指标汇总、分组绩效（全部/可成交）、Top-N、对冲、非重叠绩效、因子列表、因子质量筛选。
- `outputs/figs/*.png` — 分组净值、多空净值、Top-N 净值、对冲净值、非重叠净值等。

关键指标解读：

| 指标 | 含义 |
| --- | --- |
| `OOF RankIC / ICIR` | 交叉验证集上逐日截面 Spearman IC 的均值 / 信息比率，衡量**排序能力** |
| `Test RankIC / ICIR` | 测试集上的同口径指标 |
| `decile_perf` | 按预测分 10 档的收益 / 夏普 / 回撤，看是否单调 |
| `topn_perf` | 每日买预测最高的 N 只（默认 20）的绩效 |
| `hedge_perf` | 相对板块等权基准的超额绩效 |
| `nonoverlap_perf` | 非重叠 5 日调仓绩效，最接近“真实可交易”的评估 |

> `nonoverlap` 的有效独立样本 ≈ 交易日 / 5，Sharpe 的置信区间较宽，**不要过度解读**。

---

## 示例数据与结果（examples/）

`examples/` 目录里放了：

- `results.xlsx`：一次回测的输出示例（多 sheet：指标、分组、Top-N、对冲、非重叠等）
- `comm_price_qfq.sample.csv` / `comm_daily_basic.sample.csv`：前复权日线与每日指标的**极小格式样本**（各 5 只股票）

> ⚠️ 这些样本只用于**展示 CSV 格式 / 结果长什么样**，股票太少，**不能用于真实训练或回测**。
> 要真正跑，请用 `tstq fetch` + `tstq qfq` 拉取完整数据（见「快速开始」）。
> 请勿把完整行情数据提交到公开仓库。

## 换成你自己的主题

在 `config.py`（或 `config.yaml`）里改 `boards` 即可换主题，例如：

```python
boards = {
    "半导体": "881121.TI",
    "芯片概念": "885756.TI",
    # ...
}
```

流水线其余部分完全复用。`boards` 的代码是 Tushare 的同花顺板块指数代码（`*.TI`）。

---

## 方法论要点（为什么不是“作弊回测”）

- **只用训练期统计量**做特征标准化与标签缩尾，测试期信息不泄漏。
- **带隔离带的 walk-forward CV**：输入窗口 + 5 日标签跨度 => 至少 25 个交易日 purge gap，切断训练/验证交界的前视。
- **按验证集 RankIC 选模型**，而不是 MSE —— 我们关心横截面排序。
- **因子筛选只在训练期做**，避免“用未来挑因子”。
- **剔除买入日开盘封板 / 不可成交样本**，Top-N 与可成交分组同口径对比。
- **非重叠 5 日评估**，消除重叠收益对 Sharpe 的放大。
- **全流程固定随机种子 + 确定性 cuBLAS / 单线程 BLAS**，尽量可复现。

---

## 常见问题

- **`FileNotFoundError: comm_price_qfq.csv`** — 先跑 `fetch` 和 `qfq`，或用 `--data-dir` 指向已有数据目录。
- **`No Tushare token found`** — 没设 `TUSHARE_TOKEN`（见上文）。
- **显存不足 / 太慢** — 调小 `batch_size` / `epochs` / `n_splits`，或减小 `seq_len` / `d_model`。
- **Windows 多进程报错** — 已默认 `num_workers=0`；如需多进程可显式 `--num-workers 4`（Linux 更稳）。
- **数据抓取被限频** — 调大 `fetch_sleep`，抓取本身支持断点续传。

---

## 免责声明

本项目仅用于**量化研究与学习**，不构成任何投资建议。回测存在过拟合、幸存者偏差、交易成本与滑点未完全建模等固有局限。历史表现不代表未来收益。请自行承担使用风险。

---

## 贡献 & 许可

欢迎 issue / PR：新增因子、改进模型、加交易成本模型、支持更多板块等。

本项目基于 [MIT License](LICENSE) 开源。

数据来源：[TuShare](https://tushare.pro/)。请遵守其服务条款与数据使用规范。
