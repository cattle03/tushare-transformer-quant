# examples

这里放**小的示例产物**，方便 clone 下来直接感受输出长什么样。

- `results.xlsx` — 一次 `tstq train` 的回测输出示例，含多个 sheet：
  - `metrics`：关键指标汇总（OOF/Test RankIC、ICIR、样本数等）
  - `decile_perf` / `decile_perf_tradable`：十分位分组绩效
  - `topn_perf` / `hedge_perf`：Top-N 与对冲绩效
  - `nonoverlap_perf` / `nonoverlap_returns`：非重叠 5 日绩效
  - `features` / `factor_screen`：入模因子列表与因子质量筛选明细

> 说明：完整的行情数据（`comm_price_qfq.csv` 等）体积很大，**不提交进仓库**。
> 需要真实数据时请用 `tstq fetch` + `tstq qfq` 自行拉取，或把 `--data-dir`
> 指向你本地的数据目录。请勿把大数据文件 / TuShare 原始数据提交到公开仓库。
