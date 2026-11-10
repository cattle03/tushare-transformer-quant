"""Fetch adjustment factors and compute forward-adjusted (qfq) prices.

Forward-adjust formula:
    qfq_price(t) = raw_price(t) * adj_factor(t) / adj_factor(latest)

Output (under ``cfg.data_dir``):
  - comm_adj_factor.csv   adj_factor per (ts_code, trade_date)
  - comm_price_qfq.csv    forward-adjusted daily bars (input to the model)
  - raw/adj/*.parquet     per-stock shards (resume support)
"""

from __future__ import annotations

import os
import time

import numpy as np
import pandas as pd

from .client import connect
from .config import Config


def _fetch_with_retry(func, retries: int, **kwargs) -> pd.DataFrame:
    last: Exception | None = None
    for i in range(retries):
        try:
            df = func(**kwargs)
            return df if df is not None else pd.DataFrame()
        except Exception as exc:  # noqa: BLE001
            last = exc
            time.sleep(1.5 * (i + 1))
    assert last is not None
    raise last


def fetch_adj(pro, codes: list[str], cfg: Config, raw_adj: str) -> None:
    total = len(codes)
    done, fail = 0, []
    for idx, code in enumerate(codes, 1):
        f = os.path.join(raw_adj, f"{code}.parquet")
        if os.path.exists(f):
            done += 1
            continue
        try:
            df = _fetch_with_retry(
                pro.adj_factor,
                cfg.max_retry,
                ts_code=code,
                start_date=cfg.start_date,
                end_date=cfg.end_date,
            )
            (df if df is not None else pd.DataFrame()).to_parquet(f, index=False)
            done += 1
        except Exception as exc:  # noqa: BLE001
            fail.append(code)
            print(f"[fail] {code}: {repr(exc)[:100]}")
        time.sleep(cfg.fetch_sleep)
        if idx % 50 == 0 or idx == total:
            print(f"  adj progress {idx}/{total}  ok {done}  fail {len(fail)}")
    print(f"adj fetch done: ok {done}/{total}, fail {len(fail)}")
    if fail:
        print("failed codes:", fail)


def merge_adj(codes: list[str], raw_adj: str, out_dir: str) -> pd.DataFrame:
    parts = []
    for code in codes:
        f = os.path.join(raw_adj, f"{code}.parquet")
        if os.path.exists(f):
            df = pd.read_parquet(f)
            if not df.empty:
                parts.append(df)
    if not parts:
        return pd.DataFrame(columns=["ts_code", "trade_date", "adj_factor"])
    adj = pd.concat(parts, ignore_index=True)
    adj = adj.drop_duplicates(["ts_code", "trade_date"])
    adj["trade_date"] = adj["trade_date"].astype(str)
    adj = adj.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
    adj.to_csv(os.path.join(out_dir, "comm_adj_factor.csv"), index=False, encoding="utf-8-sig")
    print(f"  comm_adj_factor.csv: {len(adj)} rows, {adj['ts_code'].nunique()} stocks")
    return adj


def compute_qfq(adj: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    price_path = os.path.join(cfg.data_dir, cfg.raw_price_file)
    price = pd.read_csv(price_path, dtype={"ts_code": str, "trade_date": str})
    df = price.merge(adj, on=["ts_code", "trade_date"], how="left")
    df = df.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)

    # Fill missing factors from neighbouring rows of the same stock.
    df["adj_factor"] = df.groupby("ts_code")["adj_factor"].ffill().bfill()
    df["adj_factor"] = df["adj_factor"].fillna(1.0)

    latest = df.groupby("ts_code")["adj_factor"].transform("last")
    ratio = df["adj_factor"] / latest.replace(0, np.nan)
    ratio = ratio.fillna(1.0)

    for col in ["open", "high", "low", "close", "pre_close"]:
        df[col + "_q"] = df[col] * ratio

    # Use the previous adjusted close as pre_close so ex-dividend days stay continuous.
    df["pre_close_q"] = df.groupby("ts_code")["close_q"].shift(1)
    first_mask = df.groupby("ts_code").cumcount() == 0
    df.loc[first_mask, "pre_close_q"] = df.loc[first_mask, "pre_close"] * ratio[first_mask]

    df["change"] = df["close_q"] - df["pre_close_q"]
    df["pct_chg"] = np.where(
        df["pre_close_q"] != 0,
        df["change"] / df["pre_close_q"] * 100,
        np.nan,
    )

    out = df[
        ["ts_code", "trade_date", "open_q", "high_q", "low_q", "close_q",
         "pre_close_q", "change", "pct_chg", "vol", "amount"]
    ].copy()
    out = out.rename(columns={
        "open_q": "open", "high_q": "high", "low_q": "low",
        "close_q": "close", "pre_close_q": "pre_close",
    })
    out_path = os.path.join(cfg.data_dir, cfg.adj_qfq_file)
    out.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"  {cfg.adj_qfq_file}: {len(out)} rows, {out['ts_code'].nunique()} stocks")
    return out


def build_qfq(cfg: Config, pro=None) -> None:
    """Run the full adjustment pipeline: factors -> merged -> forward-adjusted prices."""
    raw_adj = os.path.join(cfg.data_dir, "raw", "adj")
    os.makedirs(raw_adj, exist_ok=True)

    uni_path = os.path.join(cfg.data_dir, cfg.universe_file)
    if not os.path.exists(uni_path):
        raise FileNotFoundError(
            f"universe file not found: {uni_path}. Run the fetch stage first."
        )
    codes = pd.read_csv(uni_path, dtype={"ts_code": str})["ts_code"].tolist()
    print(f"universe: {len(codes)} stocks\n")

    pro = pro or connect()
    print("fetching adjustment factors ...")
    fetch_adj(pro, codes, cfg, raw_adj)

    print("\nmerging factors ...")
    adj = merge_adj(codes, raw_adj, cfg.data_dir)

    print("\ncomputing forward-adjusted prices ...")
    compute_qfq(adj, cfg)
    print("\nqfq stage complete. output dir:", cfg.data_dir)
