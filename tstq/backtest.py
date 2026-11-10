"""Backtest performance metrics.

Covers: decile-group performance, fixed Top-N portfolio, benchmark-hedged
excess, and non-overlapping 5-day rebalancing that removes the Sharpe inflation
caused by overlapping forward returns.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

PERF_COLS = ["group_name", "return", "sharpe", "std", "winratio", "mdd"]


# ---------------------------------------------------------------------------
# Grouped (decile) performance
# ---------------------------------------------------------------------------
def max_dd_group(returns: pd.Series):
    """Max drawdown on a simple-cumsum NAV (per-period simple returns)."""
    r = returns.cumsum() + 1
    dd = r.div(r.cummax()).sub(1)
    mdd = dd.min()
    end = returns.index[dd.argmin()]
    start = returns.index[r.loc[:end].argmax()]
    return mdd, start, end


def calc_pfmc_groups(ret_pd: pd.DataFrame) -> pd.DataFrame:
    """Annualized return / Sharpe / vol / win-rate / drawdown for deciles 1..10 + diff."""
    rows = []
    for i in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, "diff"]:
        r = ret_pd[i]
        ret_year = r.mean() * 252
        ret_sharpe = r.mean() / r.std() * (252 ** 0.5)
        max_draw, _, _ = max_dd_group(r)
        winratio = (r > 0).sum() / len(r)
        rows.append([i, ret_year, ret_sharpe, r.std(), winratio, max_draw])
    return pd.DataFrame(rows, columns=PERF_COLS)


# ---------------------------------------------------------------------------
# Single-series performance (Top-N / hedged) -- returns must be a 1-col DataFrame
# ---------------------------------------------------------------------------
def max_dd(returns: pd.DataFrame):
    """Max drawdown for a single-column DataFrame (column named 'ret'/'diff')."""
    r = returns.cumsum() + 1
    dd = r.div(r.cummax()).sub(1)
    mdd = dd.min().values[0]
    end = returns.index[dd.values.argmin()]
    start = returns.index[r.loc[:end].values.argmax()]
    return mdd, start, end


def calc_topn_perf(ret_pd: pd.DataFrame) -> pd.DataFrame:
    """Performance summary for a single daily-return series (the Top-N portfolio)."""
    r = ret_pd.copy()
    r.columns = ["ret"]
    series = r["ret"]
    ret_year = series.mean() * 252
    ret_sharpe = series.mean() / series.std() * (252 ** 0.5)
    winratio = (series > 0).sum() / (series != 0).sum()
    max_draw, _, _ = max_dd(r)
    return pd.DataFrame(
        [["Transformer", ret_year, ret_sharpe, series.std(), winratio, max_draw]],
        columns=PERF_COLS,
    )


# ---------------------------------------------------------------------------
# Non-overlapping rebalancing
# ---------------------------------------------------------------------------
def calc_non_overlapping_pfmc(daily_5d_ret, period: int = 5) -> dict:
    """Evaluate overlapping 5-day absolute returns via non-overlapping sampling.

    Builds ``period`` interleaved phase paths (``offset::period``) and averages.
    Annualization factor is ``252/period``; effective independent samples are
    roughly ``len/period`` so Sharpe confidence intervals are wide -- don't
    over-interpret.
    """
    r0 = pd.Series(daily_5d_ret).dropna().sort_index()
    acc = {"return": [], "sharpe": [], "std": [], "winratio": [], "mdd": [], "n_periods": []}
    ann = 252.0 / period
    for offset in range(period):
        sampled = r0.iloc[offset::period]
        if len(sampled) < 2:
            continue
        nav = (1.0 + sampled).cumprod()
        dd = nav / nav.cummax() - 1.0
        sd = sampled.std()
        acc["return"].append(float(sampled.mean() * ann))
        acc["sharpe"].append(float(sampled.mean() / sd * (ann ** 0.5)) if sd > 0 else np.nan)
        acc["std"].append(float(sd))
        acc["winratio"].append(float((sampled > 0).mean()))
        acc["mdd"].append(float(dd.min()))
        acc["n_periods"].append(int(len(sampled)))
    return {
        k: (float(np.nanmean(v)) if np.any(np.isfinite(np.asarray(v, dtype=float))) else np.nan)
        for k, v in acc.items()
    }


# ---------------------------------------------------------------------------
# Helpers to turn a prediction frame into per-date return series
# ---------------------------------------------------------------------------
def daily_topn_ret(df: pd.DataFrame, n: int) -> pd.Series:
    """Each day's mean 5-day absolute return of the top-``n`` predictions."""
    g = df.dropna(subset=["pred", "ret"])
    top = (
        g.sort_values(["trade_date", "pred"], ascending=[True, False])
        .groupby("trade_date")
        .head(n)
    )
    return top.groupby("trade_date")["ret"].mean() - 1.0


def daily_topdecile_ret(df: pd.DataFrame) -> pd.Series:
    """Each day's mean 5-day absolute return of the top decile by prediction."""
    g = df.dropna(subset=["pred", "ret"]).copy()
    g["rk"] = g.groupby("trade_date")["pred"].rank(pct=True, method="dense")
    return g[g["rk"] > 0.9].groupby("trade_date")["ret"].mean() - 1.0
