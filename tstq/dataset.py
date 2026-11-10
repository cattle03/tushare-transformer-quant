"""Build the factor/target table and the per-stock sequence tensors.

Two stages:
  * ``build_factor_table(cfg)`` -- load qfq prices + daily-basic, add forward
    return targets, pivot to date x stock frames, compute all alpha factors,
    merge targets back. Returns the tidy long table + the raw price frame.
  * ``prepare_and_build_sequences(...)`` -- standardize features using *train*
    statistics only, winsorize the training label, and slice fixed-length
    windows per stock. Drops windows whose price-factor coverage is too low.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from .config import ALL_FEATURES, VALUATION_FEATURES, Config
from .factors import AlphaLib


# ---------------------------------------------------------------------------
# Load + targets
# ---------------------------------------------------------------------------
def _limit_ratio(code: str) -> float:
    """Board-specific daily limit (non-ST), used to flag untradable buy days."""
    if code.startswith(("688", "689")):     # STAR market
        return 0.20
    if code.startswith(("300", "301")):     # ChiNext (registration-based)
        return 0.20
    if code.startswith(("8", "4", "920")):  # Beijing Stock Exchange
        return 0.30
    return 0.10                             # main board


def load_price_basic(cfg: Config) -> pd.DataFrame:
    """Load qfq bars, attach a constant industry tag, merge daily-basic indicators."""
    price = pd.read_csv(
        f"{cfg.data_dir}/{cfg.price_file}", dtype={"trade_date": "str"}
    )
    price = (
        price.drop_duplicates(["ts_code", "trade_date"])
        .sort_values(["ts_code", "trade_date"])
        .reset_index(drop=True)
    )

    # Single-industry theme: use code as name, constant industry/market so that
    # cross-sectional ranking by (trade_date, industry) degenerates to by-date.
    price["name"] = price["ts_code"]
    price["industry"] = "communication"
    price["market"] = "communication"

    price = price[
        ["ts_code", "name", "market", "industry", "trade_date", "open", "high", "low",
         "close", "pre_close", "change", "pct_chg", "vol", "amount"]
    ]

    basic = pd.read_csv(
        f"{cfg.data_dir}/{cfg.basic_file}", dtype={"trade_date": "str"}
    )
    return pd.merge(price, basic, how="left", on=["ts_code", "trade_date"])


def add_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Add ret1 / ret5 / ret targets and tradability flag (t+1 open buy)."""
    df = df.copy()
    g = df.groupby("ts_code")

    # t+1 open relative to t close -> can we actually fill at the open?
    df["open_up_t1"] = g.open.shift(-1) / g.close.shift(0)
    df["limit_pct"] = df["ts_code"].map(_limit_ratio)
    df["buy_tradable"] = (
        (df["open_up_t1"] < 1 + df["limit_pct"] - 0.005)
        & (df["open_up_t1"] > 1 - df["limit_pct"] + 0.005)
    ).astype(bool)

    # t close -> t+1 close
    df["ret1"] = g.close.shift(-1) / g.close.shift(0)
    # t+1 open buy -> t+5 close sell (the modeling target)
    df["ret5"] = g.close.shift(-5) / g.open.shift(-1)
    df["ret"] = df["ret5"]
    return df


def _make_daily_info(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Pivot each raw field to a date x stock frame for factor computation."""
    def pivot(col: str) -> pd.DataFrame:
        return pd.pivot(df, index="trade_date", columns="ts_code", values=col)

    info = {
        "open": pivot("open"),
        "close": pivot("close"),
        "high": pivot("high"),
        "low": pivot("low"),
        "volume": pivot("vol"),
        "amount": pivot("amount"),
    }
    # vwap: amount(thousand yuan)*1000 / (vol(lots)*100 + 1) -> yuan per share
    info["vwap"] = (info["amount"] * 1000) / (info["volume"] * 100 + 1)
    info["pe_ttm"] = pivot("pe_ttm")
    info["total_mv"] = pivot("total_mv")
    info["pb"] = pivot("pb")
    info["turnover_rate"] = pivot("turnover_rate")
    info["volume_ratio"] = pivot("volume_ratio")
    info["dv_ttm"] = pivot("dv_ttm")
    return info


def _build_alpha_table(daily_info: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    alpha_dict = AlphaLib(daily_info).calcu_alpha()
    data = pd.DataFrame()
    for k, v in tqdm(alpha_dict.items(), desc="building factor table"):
        tmp = v.unstack().reset_index().rename(columns={0: k})
        if len(data) == 0:
            data = tmp
        else:
            data = pd.merge(data, tmp, how="left", on=["ts_code", "trade_date"])
    return data


def build_factor_table(cfg: Config) -> Tuple[pd.DataFrame, pd.DataFrame, List[str], List[str], List[str]]:
    """Return (data, price_df, all_features, valuation_features, price_features)."""
    price_df = add_targets(load_price_basic(cfg))
    daily_info = _make_daily_info(price_df)
    data = _build_alpha_table(daily_info)

    data = pd.merge(
        data,
        price_df[["trade_date", "ts_code", "name", "ret1", "ret5",
                  "ret", "open_up_t1", "buy_tradable"]],
        how="left", on=["trade_date", "ts_code"],
    )

    # Sanity guard against degenerate targets (all ~0 / constant -> fake convergence).
    print("[sanity] price ret5 describe:\n", price_df["ret5"].describe())
    print(
        f"[sanity] data.ret mean={data['ret'].mean():.4f} std={data['ret'].std():.4f} "
        f"nan={int(data['ret'].isna().sum())} zeros={int((data['ret'] == 0).sum())}"
    )
    if not (data["ret"].std() > 0.01):
        raise ValueError(
            "ret target degenerate (std too small / all zero). Check ret5/ret and the merge."
        )

    price_features = [f for f in ALL_FEATURES if f not in set(VALUATION_FEATURES)]
    return data, price_df, list(ALL_FEATURES), list(VALUATION_FEATURES), price_features


# ---------------------------------------------------------------------------
# Standardize + slice windows
# ---------------------------------------------------------------------------
@dataclass
class SequenceBundle:
    data: pd.DataFrame
    arrays_by_code: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]
    train_samples: List[Tuple[str, int]]
    test_samples: List[Tuple[str, int]]
    test_meta: List[Tuple[str, str]]
    ret_lo: float
    ret_hi: float
    n_valid_dropped: int


def prepare_and_build_sequences(
    data: pd.DataFrame,
    features: List[str],
    price_features: List[str],
    cfg: Config,
) -> SequenceBundle:
    data = data.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)

    # alpha "divide by 0" yields inf (not nan); normalize to nan first.
    data[features] = data[features].replace([np.inf, -np.inf], np.nan)
    data["ret"] = data["ret"].replace([np.inf, -np.inf], np.nan)
    data["ret1"] = data["ret1"].replace([np.inf, -np.inf], np.nan)

    # Standardize with *training-only* statistics to avoid leakage.
    train_rows = data["trade_date"] < cfg.split_date
    feat_mean = data.loc[train_rows, features].mean().fillna(0.0)
    feat_std = (
        data.loc[train_rows, features].std().fillna(1.0)
        .replace(0.0, 1.0).clip(lower=1e-6)
    )

    # Capture raw NaN mask of price factors *before* fillna (needed for valid_ratio).
    raw_nan_mask = data[price_features].isna().values   # (N, n_price), row-aligned to data

    data[features] = (data[features].fillna(feat_mean) - feat_mean) / feat_std
    data[features] = data[features].clip(-10.0, 10.0)   # guard against fp16 overflow

    # Winsorize the training label with train-only 1%/99% quantiles.
    train_ret = data.loc[train_rows, "ret"].dropna()
    ret_lo, ret_hi = train_ret.quantile(0.01), train_ret.quantile(0.99)
    data["ret_label"] = data["ret"].clip(lower=ret_lo, upper=ret_hi)
    print(f"[winsor] ret_label clipped to [{ret_lo:.4f}, {ret_hi:.4f}] (train 1%/99%)")

    arrays_by_code: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    train_samples: List[Tuple[str, int]] = []
    test_samples: List[Tuple[str, int]] = []
    test_meta: List[Tuple[str, str]] = []
    n_valid_dropped = 0

    for code, g in tqdm(data.groupby("ts_code", sort=False), desc="building sequences"):
        feat = g[features].values.astype(np.float32)     # (T, F)
        ret = g["ret_label"].values.astype(np.float32)    # (T,)
        dates = g["trade_date"].values                    # (T,)
        arrays_by_code[code] = (feat, ret, dates)
        mask = raw_nan_mask[g.index.values]               # (T, n_price), True = missing

        T = len(dates)
        if T < cfg.seq_len:
            continue
        for end in range(cfg.seq_len - 1, T):
            # Drop windows where price factors are mostly missing (no-information windows).
            if (~mask[end - cfg.seq_len + 1: end + 1]).mean() < cfg.min_valid_ratio:
                n_valid_dropped += 1
                continue
            d = dates[end]
            if d < cfg.split_date:
                if np.isfinite(ret[end]):
                    train_samples.append((code, end))
            elif d > cfg.split_date:
                test_samples.append((code, end))
                test_meta.append((code, d))

    print(
        f"train samples: {len(train_samples)}, test samples: {len(test_samples)}, "
        f"valid_ratio dropped: {n_valid_dropped}"
    )

    return SequenceBundle(
        data=data,
        arrays_by_code=arrays_by_code,
        train_samples=train_samples,
        test_samples=test_samples,
        test_meta=test_meta,
        ret_lo=ret_lo,
        ret_hi=ret_hi,
        n_valid_dropped=n_valid_dropped,
    )
