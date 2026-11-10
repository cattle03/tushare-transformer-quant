# examples — 示例数据与结果

> ⚠️ **重要提示（务必先读）**
>
> - 这里的 `*.sample.csv` 只是从真实数据里截取的**极小样本**（每份仅 **5 只股票、约 9300 行**），
>   唯一用途是**展示 CSV 的列结构 / 长什么样**。
> - 5 只股票**太少**，无法形成十分位分组、也无法训练出有意义的模型。**不要拿它跑 `tstq train` 或据此做任何投资判断。**
> - 完整、真实的行情数据体积很大（数百 MB），**不放进仓库**。请按下面“想要真实数据？”自行拉取。
> - 请遵守 TuShare 数据使用条款，**不要把完整行情数据提交到公开仓库**。

## 文件说明

| 文件 | 内容 |
| --- | --- |
| `results.xlsx` | 一次 `tstq train` 的回测输出示例（多个 sheet） |
| `comm_price_qfq.sample.csv` | 前复权日线小样本（5 只股票，约 9300 行） |
| `comm_daily_basic.sample.csv` | 每日指标小样本（同样 5 只股票，日期与上面一一对应） |

两份 CSV 用的是**同一批股票、同一段日期**，方便你对照 `comm_price_qfq.csv` 与 `comm_daily_basic.csv` 的字段。

`results.xlsx` 里的 sheet：

- `metrics`：关键指标汇总（OOF/Test RankIC、ICIR、样本数等）
- `decile_perf` / `decile_perf_tradable`：十分位分组绩效
- `topn_perf` / `hedge_perf`：Top-N 与对冲绩效
- `nonoverlap_perf` / `nonoverlap_returns`：非重叠 5 日绩效
- `features` / `factor_screen`：入模因子列表与因子质量筛选明细

## 想要真实数据？

用 TuShare 自己拉取（需要先配置环境变量 `TUSHARE_TOKEN`）：

```bash
python -m tstq fetch --data-dir data
python -m tstq qfq   --data-dir data
python -m tstq train --data-dir data
```

或者把你**已有的**数据目录直接喂给训练：

```bash
python -m tstq train --data-dir 你的数据目录
# 该目录下需要有：comm_price_qfq.csv、comm_daily_basic.csv
```
