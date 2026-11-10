"""Fetch a themed stock universe + daily bars + daily-basic indicators.

Resumable: each stock's intermediate result is written to a parquet shard and
skipped on subsequent runs. The final step concatenates shards into the two
CSV files consumed by the model pipeline.

Output (under ``cfg.data_dir``):
  - comm_universe.csv       constituent list (with board membership)
  - comm_price.csv          un-adjusted daily bars
  - comm_daily_basic.csv    daily valuation / liquidity indicators
  - raw/price/*.parquet     per-stock shards (resume support)
  - raw/basic/*.parquet
"""

from __future__ import annotations

import os
import time

import pandas as pd

from .client import connect
from .config import BASIC_FIELDS, Config


def _ensure_dirs(cfg: Config) -> dict[str, str]:
    raw_price = os.path.join(cfg.data_dir, "raw", "price")
    raw_basic = os.path.join(cfg.data_dir, "raw", "basic")
    os.makedirs(cfg.data_dir, exist_ok=True)
    os.makedirs(raw_price, exist_ok=True)
    os.makedirs(raw_basic, exist_ok=True)
    return {"raw_price": raw_price, "raw_basic": raw_basic}


def _fetch_with_retry(func, retries: int, **kwargs) -> pd.DataFrame:
    last: Exception | None = None
    for i in range(retries):
        try:
            df = func(**kwargs)
            return df if df is not None else pd.DataFrame()
        except Exception as exc:  # noqa: BLE001 - network calls fail in many ways
            last = exc
            time.sleep(1.5 * (i + 1))
    assert last is not None
    raise last


def build_universe(pro, cfg: Config) -> pd.DataFrame:
    """Union of every configured concept board's constituents."""
    rows = []
    for board_name, board_code in cfg.boards.items():
        try:
            df = pro.ths_member(ts_code=board_code)
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] ths_member {board_name}({board_code}) failed: {exc}")
            continue
        if df is None or df.empty:
            continue
        for code in df["con_code"].astype(str):
            rows.append((code, board_name))
        print(f"  + {board_name:<16} members {df['con_code'].nunique()}")

    mem = pd.DataFrame(rows, columns=["ts_code", "board"])
    boards_map = mem.groupby("ts_code")["board"].apply(lambda s: ";".join(sorted(set(s))))
    universe = boards_map.reset_index().rename(columns={"board": "boards"})
    universe = universe[universe["ts_code"].str.contains(r"\.(SH|SZ|BJ)$", regex=True)]

    basic = pro.stock_basic(
        exchange="",
        list_status="L",
        fields="ts_code,name,industry,market,list_date",
    )
    universe = universe.merge(basic, on="ts_code", how="left")
    return universe.sort_values("ts_code").reset_index(drop=True)


def fetch_one(pro, ts_code: str, cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    price = _fetch_with_retry(
        pro.daily,
        cfg.max_retry,
        ts_code=ts_code,
        start_date=cfg.start_date,
        end_date=cfg.end_date,
    )
    time.sleep(cfg.fetch_sleep)
    basic = _fetch_with_retry(
        pro.daily_basic,
        cfg.max_retry,
        ts_code=ts_code,
        start_date=cfg.start_date,
        end_date=cfg.end_date,
        fields=BASIC_FIELDS,
    )
    time.sleep(cfg.fetch_sleep)
    return price, basic


def _load_or_build_universe(pro, cfg: Config) -> pd.DataFrame:
    uni_path = os.path.join(cfg.data_dir, cfg.universe_file)
    if os.path.exists(uni_path):
        universe = pd.read_csv(uni_path, dtype={"ts_code": str})
        print(f"reuse existing universe: {len(universe)} stocks")
        return universe
    print("building themed stock universe ...")
    universe = build_universe(pro, cfg)
    universe.to_csv(uni_path, index=False, encoding="utf-8-sig")
    print(f"universe saved: {uni_path} ({len(universe)} stocks)")
    return universe


def _merge_shards(codes: list[str], raw_dir: str, out_path: str, label: str) -> None:
    parts = []
    for code in codes:
        f = os.path.join(raw_dir, f"{code}.parquet")
        if os.path.exists(f):
            df = pd.read_parquet(f)
            if not df.empty:
                parts.append(df)
    if not parts:
        print(f"  [skip] no data for {label}")
        return
    merged = pd.concat(parts, ignore_index=True)
    merged = (
        merged.drop_duplicates()
        .sort_values(["ts_code", "trade_date"])
        .reset_index(drop=True)
    )
    merged.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"  {os.path.basename(out_path)}: {len(merged)} rows, {merged['ts_code'].nunique()} stocks")


def fetch_all(cfg: Config, pro=None) -> None:
    """Run the full fetch: universe -> per-stock bars/basic -> merged CSVs."""
    dirs = _ensure_dirs(cfg)
    pro = pro or connect()

    universe = _load_or_build_universe(pro, cfg)
    codes = universe["ts_code"].tolist()
    total = len(codes)
    print(f"fetching {total} stocks, {cfg.start_date}~{cfg.end_date}\n")

    done, fail = 0, []
    for idx, code in enumerate(codes, 1):
        p_file = os.path.join(dirs["raw_price"], f"{code}.parquet")
        b_file = os.path.join(dirs["raw_basic"], f"{code}.parquet")
        if os.path.exists(p_file) and os.path.exists(b_file):
            done += 1
            continue
        try:
            price, basic = fetch_one(pro, code, cfg)
            (price if not price.empty else pd.DataFrame()).to_parquet(p_file, index=False)
            (basic if not basic.empty else pd.DataFrame()).to_parquet(b_file, index=False)
            done += 1
        except Exception as exc:  # noqa: BLE001
            fail.append(code)
            print(f"[fail] {code}: {repr(exc)[:100]}")

        if idx % 50 == 0 or idx == total:
            print(f"  progress {idx}/{total}  ok {done}  fail {len(fail)}")

    print(f"\nfetch done: ok {done}/{total}, fail {len(fail)}")
    if fail:
        print("failed codes:", fail)

    print("\nmerging shards ...")
    _merge_shards(codes, dirs["raw_price"], os.path.join(cfg.data_dir, cfg.raw_price_file), "price")
    _merge_shards(codes, dirs["raw_basic"], os.path.join(cfg.data_dir, cfg.basic_file), "basic")
    print("\nfetch stage complete. output dir:", cfg.data_dir)
