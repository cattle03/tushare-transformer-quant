"""Alpha factor generation + single-factor quality diagnostics/screening.

Ports the factor logic from the original ``transformer.py``:
  * rolling time-series helpers (ts_sum / sma / ts_max ...)
  * ``AlphaLib``: builds the 32 price/volume/valuation factors from pivoted data
  * ``evaluate_alpha`` / ``pass_quality_gate`` / ``screen_factor_quality``:
    decile / IC / ICIR / drawdown / monotonicity diagnostics used to pick the
    factor subset that actually gets fed into the Transformer.

All pivoted frames use ``index=trade_date`` and ``columns=ts_code`` so that
cross-sectional operations (``rank``, ``amount.sum(1)``) work as intended.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Rolling time-series helpers (operate column-wise on date x stock frames)
# ---------------------------------------------------------------------------
def ts_sum(df, window=10):
    return df.rolling(window).sum()


def sma(df, window=10):
    return df.rolling(window).mean()


def ts_min(df, window=10):
    return df.rolling(window).min()


def ts_median(df, window=10):
    return df.rolling(window).median()


def ts_max(df, window=10):
    return df.rolling(window).max()


def delta(df, period=1):
    return df.diff(period)


def delay(df, period=1):
    return df.shift(period)


def rank(df):
    return df.rank(axis=1, pct=True)


def ts_argmax(df, window=10):
    return df.rolling(window).apply(np.argmax) + 1


# ---------------------------------------------------------------------------
# Alpha factor library
# ---------------------------------------------------------------------------
class AlphaLib:
    def __init__(self, daily_info: Dict[str, pd.DataFrame]):
        self.open = daily_info["open"]
        self.high = daily_info["high"]
        self.low = daily_info["low"]
        self.close = daily_info["close"]
        self.vwap = daily_info["vwap"]
        self.volume = daily_info["volume"]
        self.amount = daily_info["amount"]
        self.pe_ttm = daily_info["pe_ttm"]
        self.total_mv = daily_info["total_mv"]
        self.pb = daily_info["pb"]
        self.turnover_rate = daily_info["turnover_rate"]
        self.volume_ratio = daily_info["volume_ratio"]
        self.dv_ttm = daily_info["dv_ttm"]

    def calcu_alpha(self) -> Dict[str, pd.DataFrame]:
        d = {}
        d["alpha01"] = ((self.close / sma(self.close, 10) - 1) * self.amount).div(self.amount.sum(1), axis=0)
        d["alpha02"] = sma(self.amount, 5).div(self.amount.sum(1), axis=0)
        d["alpha06"] = sma((self.close / sma(self.close, 5) - 1) * self.amount, 5).div(self.amount.sum(1), axis=0)
        d["alpha07"] = ts_max((self.close / sma(self.close, 15) - 1) * self.amount, 15).div(self.amount.sum(1), axis=0)
        d["alpha08"] = ts_min((self.close / ts_max(self.close, 60) - 1) * self.amount, 10).div(self.amount.sum(1), axis=0)
        d["alpha09"] = ts_max((self.close / ts_min(self.close, 60) - 1) * self.amount, 50).div(self.amount.sum(1), axis=0)
        d["alpha10"] = ts_max((self.close / ts_min(self.close, 20) - 1) * self.amount, 15).div(self.amount.sum(1), axis=0)
        d["alpha12"] = ts_max((ts_max(self.high, 30) / ts_min(self.low, 30) - 1) * self.amount, 20).div(self.amount.sum(1), axis=0)
        d["alpha13"] = ts_max(self.high - self.low, 30) / ts_min(self.close + self.open, 30)
        d["alpha14"] = ts_sum(self.amount * (self.close - self.open), 5)
        d["alpha15"] = ts_max(self.amount * (self.high - self.low), 15)
        d["alpha16"] = ts_sum(self.amount * (self.close - self.vwap), 5)
        d["alpha17"] = ts_min(self.amount * (self.low - self.vwap), 15)
        d["alpha18"] = ts_min(self.amount * (self.open - self.vwap), 15)
        d["alpha19"] = ts_min(self.amount * (self.open - self.low), 10)
        d["alpha20"] = ts_max(self.amount * (self.close - self.low), 10)
        d["alpha21"] = ts_median(self.amount, 15) / ts_sum(self.amount, 15)
        d["alpha23"] = ts_max(self.amount, 15) / ts_min(self.amount, 10)
        d["alpha24"] = ts_sum(self.amount.div(self.amount.sum(1), axis=0), 5)
        d["alpha28"] = (ts_max(self.close, 5) / delay(self.close, 5)) * ts_min(self.close, 5) / self.close

        tmp2 = abs((self.close - self.open) / (self.high - self.low + 0.01))
        d["alpha29"] = ts_max(tmp2, 6) / ts_min(tmp2, 6)
        d["alpha30"] = tmp2 / delay(tmp2, 4)

        tmp3 = abs((self.low - self.open) / (self.close - self.low + 0.01))
        d["alpha31"] = ts_max(tmp3, 4) / ts_min(tmp3, 4)

        tmp4 = abs((self.high - self.open) / (self.close - self.low + 0.01))
        d["alpha32"] = ts_max(tmp4, 2) / ts_min(tmp4, 2)

        d["alpha_w_005"] = (rank(self.open - (ts_sum(self.vwap, 10) / 10))) * (-1 * abs(rank(self.close - self.vwap)))
        d["alpha_pe"] = -1 * self.pe_ttm
        d["alpha_total_mv"] = self.total_mv
        d["alpha_pb"] = -1 * self.pb
        d["alpha_turnover_rate"] = self.turnover_rate
        d["alpha_volume_ratio"] = self.volume_ratio
        d["alpha_vwap"] = self.vwap
        d["alpha_dv_ttm"] = -1 * self.dv_ttm
        return d


# ---------------------------------------------------------------------------
# Quality evaluation of a single factor
# ---------------------------------------------------------------------------
DEFAULT_QUALITY_THRESHOLDS = {
    "min_diff_sharpe": 1.5,
    "min_diff_return": 0.0,
    "min_abs_ic": 0.02,
    "min_abs_icir": 0.3,
    "max_diff_mdd": -0.30,
    "min_diff_winratio": 0.50,
    "require_monotonic": True,
}


def bin_factor_rank(x):
    """Map a 0~1 percentile rank to a 1~10 decile bucket."""
    if np.isnan(x):
        return np.nan
    return min(10, int(x * 10) + 1)


def build_ret_pd(alpha: pd.DataFrame) -> pd.DataFrame:
    """Per-period returns by factor decile + a long-short ``diff`` column."""
    ret_pd = (alpha.groupby(["factor_rank_bin", "trade_date"]).ret1.mean() - 1.00).unstack().T
    if ret_pd[1].mean() > ret_pd[10].mean():
        ret_pd["diff"] = ret_pd[1] - ret_pd[10]
    else:
        ret_pd["diff"] = ret_pd[10] - ret_pd[1]
    return ret_pd


def aq_max_dd(returns):
    """Max drawdown + dates on the compounded NAV (robust to overlapping returns)."""
    nav = (1.0 + returns.fillna(0.0)).cumprod()
    dd = nav.div(nav.cummax()).sub(1.0)
    mdd = dd.min()
    end = returns.index[dd.argmin()]
    start = returns.index[nav.loc[:end].argmax()]
    return mdd, start, end


def aq_calc_pfmc(ret_pd: pd.DataFrame, groups=(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, "diff"), horizon=1) -> pd.DataFrame:
    """Annualized return / Sharpe / vol / win-rate / drawdown per decile."""
    periods_per_year = 252.0 / float(horizon)
    rows = []
    for i in groups:
        r = ret_pd[i]
        mean_p = r.mean()
        std_p = r.std()
        ret_year = mean_p * periods_per_year
        ret_sharpe = (mean_p / std_p * (periods_per_year ** 0.5)) if std_p > 0 else np.nan
        max_draw, _, _ = aq_max_dd(r)
        winratio = (r > 0).sum() / len(r)
        rows.append([i, ret_year, ret_sharpe, std_p, winratio, max_draw])
    return pd.DataFrame(rows, columns=["group_name", "return", "sharpe", "std", "winratio", "mdd"])


def calc_ic_series(alpha: pd.DataFrame) -> pd.DataFrame:
    """Daily cross-sectional IC (factor vs. ret1) series indexed by trade_date."""
    ic = alpha.groupby(["trade_date"])[["factor", "ret1"]].corr()["factor"].reset_index()
    ic = ic[ic.level_1 == "ret1"]
    ic = ic[["trade_date", "factor"]].set_index("trade_date")
    return ic.rename(columns={"factor": "IC"})


def calc_icir(ic_series: pd.DataFrame) -> Dict[str, float]:
    ic_mean = ic_series["IC"].mean()
    ic_std = ic_series["IC"].std()
    icir = ic_mean / ic_std if ic_std != 0 else np.nan
    return {"ic_mean": ic_mean, "ic_std": ic_std, "icir": icir}


def evaluate_alpha(alpha: pd.DataFrame, horizon: int = 1) -> dict:
    """Full single-factor diagnostic bundle (deciles, IC series, summary)."""
    alpha = alpha.copy()
    if "factor_rank" not in alpha.columns:
        alpha["factor_rank"] = alpha.groupby(["trade_date", "industry"]).factor.rank(pct=True)
    if "factor_rank_bin" not in alpha.columns:
        alpha["factor_rank_bin"] = alpha.factor_rank.apply(bin_factor_rank)

    ret_pd = build_ret_pd(alpha)
    performance = aq_calc_pfmc(ret_pd, horizon=horizon)
    ic_series = calc_ic_series(alpha)
    ic_stats = calc_icir(ic_series)

    diff_row = performance[performance.group_name == "diff"].iloc[0]
    group_returns = (
        performance[performance.group_name != "diff"].sort_values("group_name")["return"].values
    )
    inc = np.all(np.diff(group_returns) >= 0)
    dec = np.all(np.diff(group_returns) <= 0)
    monotonic = bool(inc or dec)

    summary = {
        "diff_return": float(diff_row["return"]),
        "diff_sharpe": float(diff_row["sharpe"]),
        "diff_std": float(diff_row["std"]),
        "diff_winratio": float(diff_row["winratio"]),
        "diff_mdd": float(diff_row["mdd"]),
        "ic_mean": float(ic_stats["ic_mean"]),
        "ic_std": float(ic_stats["ic_std"]),
        "icir": float(ic_stats["icir"]),
        "monotonic": monotonic,
    }
    return {
        "factor_rank": alpha,
        "ret_PD": ret_pd,
        "performance": performance,
        "ic_series": ic_series,
        "ic_mean": ic_stats["ic_mean"],
        "ic_std": ic_stats["ic_std"],
        "icir": ic_stats["icir"],
        "summary": summary,
    }


def pass_quality_gate(summary: dict, thresholds: dict | None = None) -> Tuple[bool, dict]:
    """Apply the hard quality gate; returns (passed, detail)."""
    t = dict(DEFAULT_QUALITY_THRESHOLDS)
    if thresholds:
        t.update(thresholds)

    checks = {
        "diff_sharpe>=threshold": summary["diff_sharpe"] >= t["min_diff_sharpe"],
        "diff_return>=threshold": summary["diff_return"] >= t["min_diff_return"],
        "|IC|>=threshold": abs(summary["ic_mean"]) >= t["min_abs_ic"],
        "|ICIR|>=threshold": abs(summary["icir"]) >= t["min_abs_icir"],
        "diff_mdd acceptable": summary["diff_mdd"] >= t["max_diff_mdd"],
        "diff_winratio>=threshold": summary["diff_winratio"] >= t["min_diff_winratio"],
    }
    if t.get("require_monotonic", False):
        checks["deciles monotonic"] = bool(summary["monotonic"])

    passed = all(checks.values())
    return passed, {"thresholds": t, "checks": checks, "summary": summary}


_EVAL_HORIZON = {"ret1": 1, "ret5": 5, "ret": 5}


def screen_factor_quality(
    data: pd.DataFrame,
    features: list[str],
    split_date: str,
    ret_col: str = "ret5",
    thresholds: dict | None = None,
    top_k: int = 8,
    min_winratio: float = 0.50,
    max_mdd: float = -0.95,
    min_abs_icir: float = 0.05,
    industry: str = "communication",
) -> Tuple[pd.DataFrame, list[str]]:
    """Diagnose every candidate factor on *training-period* data and pick top-K by |ICIR|.

    Selection logic (aligned with the backtest口径):
      1) hard-exclude factors whose long-short win-rate < min_winratio or whose
         long-short max-drawdown < max_mdd (fragile / near-zero factors);
      2) rank the survivors by |ICIR| and keep the top ``top_k``.
    ``passed`` (vs DEFAULT thresholds) is informational only, never used to pick.

    Screening on the training window avoids using future data to select factors.
    """
    horizon = _EVAL_HORIZON.get(ret_col, 1)
    screen = data[data["trade_date"] < split_date]
    rows = []
    for feat in tqdm(features, desc="factor quality screen"):
        alpha = pd.DataFrame({
            "ts_code": screen["ts_code"].values,
            "trade_date": screen["trade_date"].values,
            "industry": industry,
            "factor": pd.to_numeric(screen[feat], errors="coerce").replace([np.inf, -np.inf], np.nan).values,
            "ret1": pd.to_numeric(screen[ret_col], errors="coerce").replace([np.inf, -np.inf], np.nan).values,
        })
        try:
            res = evaluate_alpha(alpha, horizon=horizon)
            s = res["summary"]
            passed, _ = pass_quality_gate(s, thresholds=thresholds)
            rows.append({
                "feature": feat,
                "diff_sharpe": s["diff_sharpe"],
                "diff_return": s["diff_return"],
                "diff_winratio": s["diff_winratio"],
                "diff_mdd": s["diff_mdd"],
                "ic_mean": s["ic_mean"],
                "ic_std": s["ic_std"],
                "icir": s["icir"],
                "monotonic": s["monotonic"],
                "passed": bool(passed),
                "error": "",
            })
        except Exception as exc:  # noqa: BLE001 - all-NaN / constant factors cannot be diagnosed
            rows.append({
                "feature": feat, "diff_sharpe": np.nan, "diff_return": np.nan,
                "diff_winratio": np.nan, "diff_mdd": np.nan, "ic_mean": np.nan,
                "ic_std": np.nan, "icir": np.nan, "monotonic": False, "passed": False,
                "error": str(exc),
            })

    report = pd.DataFrame(rows)
    report["abs_icir"] = report["icir"].abs()

    hard_ok = (
        (report["diff_winratio"] >= min_winratio)
        & (report["diff_mdd"] >= max_mdd)
        & (report["abs_icir"] >= min_abs_icir)
    )
    report["eligible"] = hard_ok

    eligible = report.loc[hard_ok].sort_values("abs_icir", ascending=False, na_position="last")
    selected_feats = eligible["feature"].head(top_k).tolist()

    if len(selected_feats) < 2:
        print("[warn] fewer than 2 factors survived hard-exclusion; fall back to top-K by |ICIR|")
        selected_feats = (
            report.sort_values("abs_icir", ascending=False, na_position="last")["feature"]
            .head(top_k)
            .tolist()
        )

    selected_set = set(selected_feats)
    selected = [f for f in features if f in selected_set]   # preserve original order
    return report, selected
