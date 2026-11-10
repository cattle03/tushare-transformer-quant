"""End-to-end training + backtest orchestration.

Flow:
  factor table -> factor quality screening (train window) -> per-stock windows
  -> purged walk-forward CV (Transformer) -> ensemble test prediction
  -> decile / Top-N / hedged / non-overlapping backtests -> plots + Excel export.

Benchmark is the equal-weight return of all stocks in the theme (no Tushare
index dependency). Everything is driven by a single :class:`~tstq.config.Config`.
"""

from __future__ import annotations

import os
import time

import matplotlib

matplotlib.use("Agg")  # non-interactive backend: plt.show() is a no-op, figures saved to fig_dir
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from torch import nn  # noqa: E402
from tqdm import tqdm  # noqa: E402

from .backtest import (  # noqa: E402
    calc_non_overlapping_pfmc,
    calc_pfmc_groups,
    calc_topn_perf,
    daily_topdecile_ret,
    daily_topn_ret,
)
from .config import Config  # noqa: E402
from .dataset import build_factor_table, prepare_and_build_sequences  # noqa: E402
from .factors import screen_factor_quality  # noqa: E402
from .model import StockSeqDataset, build_model, make_loader, run_epoch  # noqa: E402


def _seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
            torch.backends.cuda.enable_flash_sdp(False)
            torch.backends.cuda.enable_mem_efficient_sdp(False)
            torch.backends.cuda.enable_math_sdp(True)
        except Exception:  # noqa: BLE001 - older torch without these knobs
            pass


def save_show(fig_dir: str, name: str) -> None:
    path = os.path.join(fig_dir, f"{name}.png")
    plt.savefig(path, bbox_inches="tight")
    print("saved figure:", path)
    plt.show()
    plt.close()


def _mean_rank_ic(pred, dates, act) -> float:
    """Mean per-day cross-sectional Spearman IC (used to select the best epoch)."""
    df = pd.DataFrame({"d": dates, "p": pred, "a": act}).dropna()
    if len(df) == 0:
        return float("nan")
    ic = (
        df.groupby("d")[["p", "a"]]
        .apply(lambda g: g["p"].corr(g["a"], method="spearman") if len(g) > 1 else np.nan)
        .dropna()
    )
    return float(ic.mean()) if len(ic) else float("nan")


def _bin(x):
    if np.isnan(x):
        return np.nan
    return min(10, int(x * 10) + 1)


def run(cfg: Config) -> dict:
    _seed_everything(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = cfg.use_amp and device.type == "cuda"
    print("Using device:", device, "| AMP:", use_amp)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    t_start = time.time()

    # ------------------------------------------------------------------ data
    data, _price_df, all_features, valuation_features, price_features = build_factor_table(cfg)
    features = all_features

    # -------------------------------------------------------- factor screen
    factor_screen_report = None
    if cfg.factor_screen_enable:
        factor_screen_report, selected = screen_factor_quality(
            data, features,
            split_date=cfg.split_date,
            ret_col=cfg.factor_eval_ret,
            top_k=cfg.factor_top_k,
            min_winratio=cfg.factor_min_winrate,
            max_mdd=cfg.factor_max_mdd,
            min_abs_icir=cfg.factor_min_abs_icir,
        )
        rep_show = factor_screen_report.sort_values(
            by=["eligible", "abs_icir"], ascending=[False, False], na_position="last"
        )
        print(
            "\n[factor screen] target=%s | passed gate %d/%d | eligible %d | top-%d by |ICIR|"
            % (cfg.factor_eval_ret, int(factor_screen_report["passed"].sum()), len(features),
               int(factor_screen_report["eligible"].sum()), len(selected))
        )
        print(rep_show[["feature", "diff_sharpe", "diff_return", "diff_winratio",
                        "diff_mdd", "ic_mean", "icir", "monotonic", "eligible", "passed"]]
              .to_string(index=False))

        if selected:
            features = selected
            valuation_features = [f for f in valuation_features if f in set(features)]
            price_features = [f for f in features if f not in valuation_features]
            print("features fed to Transformer:", features)
            if len(features) < 2:
                raise RuntimeError(
                    "fewer than 2 factors after screening; relax factor_min_winrate / "
                    "factor_max_mdd or increase factor_top_k"
                )
        else:
            print("[warn] no factors after screening, keeping all")

    # ------------------------------------------------------------- sequences
    bundle = prepare_and_build_sequences(data, features, price_features, cfg)
    data = bundle.data
    arrays_by_code = bundle.arrays_by_code
    train_samples = bundle.train_samples
    test_samples = bundle.test_samples
    test_meta = bundle.test_meta
    ret_lo, ret_hi = bundle.ret_lo, bundle.ret_hi

    # --------------------------------------- purged walk-forward CV training
    train_samples.sort(key=lambda s: arrays_by_code[s[0]][2][s[1]])
    sample_dates = np.array([arrays_by_code[c][2][e] for c, e in train_samples])
    train_y_all = np.array(
        [arrays_by_code[c][1][e] - 1.0 for c, e in train_samples], dtype=np.float32
    )
    oof_predictions = np.full(len(train_samples), np.nan, dtype=np.float32)
    modellist = []

    # input window SEQ_LEN + label span 5 -> >= 25 trading-day purge/embargo gap
    purge_gap_days = cfg.seq_len + 5
    uniq_dates = np.sort(np.unique(sample_dates))
    n_dates = len(uniq_dates)

    folds = []
    edges = np.linspace(0, n_dates, cfg.n_splits + 2).astype(int)
    for k in range(cfg.n_splits):
        val_lo, val_hi = edges[k + 1], edges[k + 2]
        if val_hi <= val_lo:
            continue
        purge_end = val_lo - purge_gap_days
        if purge_end <= 0:
            continue
        train_last_date = uniq_dates[purge_end - 1]
        val_first_date = uniq_dates[val_lo]
        val_last_date = uniq_dates[val_hi - 1]
        trn_idx = np.where(sample_dates <= train_last_date)[0]
        val_idx = np.where((sample_dates >= val_first_date) & (sample_dates <= val_last_date))[0]
        if len(trn_idx) < cfg.batch_size or len(val_idx) == 0:
            continue
        folds.append((trn_idx, val_idx))

    if not folds:
        raise RuntimeError(
            "not enough training data to build purged walk-forward folds; "
            "check split_date / data length"
        )
    print(f"purged walk-forward folds: {len(folds)} (gap={purge_gap_days} trading days)")

    loss_fn = nn.MSELoss()
    for fold, (trn_idx, val_idx) in enumerate(folds):
        fold_t0 = time.time()
        trn_samples = [train_samples[i] for i in trn_idx]
        val_samples = [train_samples[i] for i in val_idx]

        train_ds = StockSeqDataset(arrays_by_code, trn_samples, cfg.seq_len)
        val_ds = StockSeqDataset(arrays_by_code, val_samples, cfg.seq_len)
        train_loader = make_loader(train_ds, cfg, shuffle=True)
        val_loader = make_loader(val_ds, cfg, shuffle=False)

        model = build_model(len(features), cfg).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

        # Select the epoch by validation RankIC (we care about cross-sectional
        # ordering, not raw MSE).
        best_ic, best_state, best_pred = -np.inf, None, None
        for epoch in range(cfg.epochs):
            tr_loss, _ = run_epoch(
                model, train_loader, opt, loss_fn, device=device, use_amp=use_amp,
                scaler=scaler, target_scale=cfg.target_scale, train=True,
                desc=f"fold {fold + 1}/{len(folds)} epoch {epoch + 1}/{cfg.epochs} train",
            )
            val_loss, val_pred = run_epoch(
                model, val_loader, opt, loss_fn, device=device, use_amp=use_amp,
                scaler=scaler, target_scale=cfg.target_scale, train=False,
                desc=f"fold {fold + 1}/{len(folds)} epoch {epoch + 1}/{cfg.epochs} val",
            )
            val_ic = _mean_rank_ic(val_pred, sample_dates[val_idx], train_y_all[val_idx])
            print(f"[fold {fold + 1}] epoch {epoch + 1}/{cfg.epochs} "
                  f"train_loss={tr_loss:.6f} val_loss={val_loss:.6f} val_RankIC={val_ic:.4f} "
                  f"elapsed={time.time() - fold_t0:.1f}s")
            improved = (best_pred is None) or (np.isfinite(val_ic) and val_ic > best_ic)
            if improved:
                if np.isfinite(val_ic):
                    best_ic = val_ic
                best_pred = val_pred.copy()
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if best_state is not None:
            model.load_state_dict(best_state)
        oof_predictions[val_idx] = best_pred if best_pred is not None else val_pred
        modellist.append(model)
        print(f"[fold {fold + 1}] done (best val RankIC={best_ic:.4f}), "
              f"total elapsed {time.time() - t_start:.1f}s")

    # -------------------------------------------------------------- OOF eval
    oof_df = pd.DataFrame({"date": sample_dates, "act": train_y_all,
                           "pred": oof_predictions}).dropna(subset=["act", "pred"])
    rmse = np.sqrt(np.nanmean((oof_df["act"] - oof_df["pred"]) ** 2))
    print(f"OOF RMSE = {rmse:.6f}")
    oof_ic = (
        oof_df.groupby("date")[["pred", "act"]]
        .apply(lambda g: g["pred"].corr(g["act"], method="spearman") if len(g) > 1 else np.nan)
        .dropna()
    )
    if len(oof_ic):
        print(f"OOF RankIC mean={oof_ic.mean():.4f} std={oof_ic.std():.4f} "
              f"ICIR={oof_ic.mean() / oof_ic.std():.4f} days={len(oof_ic)}")

    oof_metrics = {
        "OOF_RMSE": float(rmse),
        "OOF_RankIC_mean": float(oof_ic.mean()) if len(oof_ic) else np.nan,
        "OOF_RankIC_std": float(oof_ic.std()) if len(oof_ic) else np.nan,
        "OOF_ICIR": float(oof_ic.mean() / oof_ic.std()) if len(oof_ic) and oof_ic.std() > 0 else np.nan,
        "OOF_days": int(len(oof_ic)),
    }

    # -------------------------------------------------- ensemble test predict
    test_ds = StockSeqDataset(arrays_by_code, test_samples, cfg.seq_len)
    test_loader = make_loader(test_ds, cfg, shuffle=False)

    test_pred_sum = np.zeros(len(test_samples), dtype=np.float64)
    for model in tqdm(modellist, desc="ensemble predict"):
        model.eval()
        fold_preds = []
        with torch.no_grad(), torch.amp.autocast("cuda", enabled=use_amp):
            for xb, _yb in tqdm(test_loader, desc="  test batch", unit="batch", leave=False):
                xb = xb.to(device, non_blocking=True)
                fold_preds.append(model(xb).float().detach().cpu().numpy())
        test_pred_sum += np.concatenate(fold_preds) / cfg.target_scale
    test_pred = test_pred_sum / len(modellist)

    test = data[data.trade_date > cfg.split_date].copy()
    pred_df = pd.DataFrame(test_meta, columns=["ts_code", "trade_date"])
    pred_df["pred"] = test_pred
    test = test.merge(pred_df, on=["ts_code", "trade_date"], how="left")
    # Crop the last 5 trading days (unobservable future returns) for the whole
    # cross-section, not per stock, to avoid look-ahead bias.
    tail_dates = set(sorted(test["trade_date"].unique())[-5:])
    test = test[~test["trade_date"].isin(tail_dates)].copy()
    date_calc = sorted(test.trade_date.unique())
    print(f"total elapsed so far: {time.time() - t_start:.1f}s")

    # ------------------------------------------------------ decile groupings
    test["industry"] = "communication"
    test["factor_rank"] = test.groupby(["trade_date"]).pred.rank(pct=True, method="dense")
    test["factor_rank_bin"] = test.factor_rank.apply(_bin)
    print(test.factor_rank_bin.value_counts())

    test_ic = (
        test.dropna(subset=["pred", "ret"])
        .groupby("trade_date")[["pred", "ret"]]
        .apply(lambda g: g["pred"].corr(g["ret"], method="spearman") if len(g) > 1 else np.nan)
        .dropna()
    )
    if len(test_ic):
        print(f"[test] RankIC mean={test_ic.mean():.4f} std={test_ic.std():.4f} "
              f"ICIR={test_ic.mean() / test_ic.std():.4f} days={len(test_ic)}")

    test_ic_metrics = {
        "Test_RankIC_mean": float(test_ic.mean()) if len(test_ic) else np.nan,
        "Test_RankIC_std": float(test_ic.std()) if len(test_ic) else np.nan,
        "Test_ICIR": float(test_ic.mean() / test_ic.std()) if len(test_ic) and test_ic.std() > 0 else np.nan,
        "Test_days": int(len(test_ic)),
    }

    ((test.groupby(["factor_rank_bin", "trade_date"]).ret.mean() - 1.0)
     .unstack().T.div(5).cumsum().plot(rot=45, grid=True))
    save_show(cfg.fig_dir, "group_cumret")

    # long-short (top decile minus bottom decile), 5-day returns / 5
    long_ret = ((test[test.factor_rank_bin == 10].groupby("trade_date").ret.mean() - 1.0) / 5).cumsum()
    short_ret = ((test[test.factor_rank_bin == 1].groupby("trade_date").ret.mean() - 1.0) / 5).cumsum()
    (long_ret - short_ret).plot(rot=45, grid=True)
    save_show(cfg.fig_dir, "long_short")

    ret_pd = (test.groupby(["factor_rank_bin", "trade_date"]).ret.mean() - 1.00).unstack().T.div(5)
    if ret_pd[1].mean() > ret_pd[10].mean():
        ret_pd["diff"] = ret_pd[1] - ret_pd[10]
    else:
        ret_pd["diff"] = ret_pd[10] - ret_pd[1]

    perf_groups = calc_pfmc_groups(ret_pd)
    print(perf_groups)

    # tradable-only deciles (exclude open-limit-up / unfillable samples)
    test_tr = test[test["buy_tradable"].eq(True)]
    ret_pd_tr = (test_tr.groupby(["factor_rank_bin", "trade_date"]).ret.mean() - 1.0).unstack().T.div(5)
    ret_pd_tr["diff"] = (
        (ret_pd_tr[10] - ret_pd_tr[1]) if ret_pd_tr[10].mean() >= ret_pd_tr[1].mean()
        else (ret_pd_tr[1] - ret_pd_tr[10])
    )
    perf_groups_tr = calc_pfmc_groups(ret_pd_tr)
    print("[tradable decile performance]")
    print(perf_groups_tr)

    # ------------------------------------------------------------ fixed Top-N
    tmp = test.copy()
    tmp = tmp[tmp["buy_tradable"].eq(True)]
    num = cfg.top_n
    tmp = (
        tmp.sort_values(["trade_date", "pred"], ascending=[True, False])
        .groupby("trade_date").head(num).reset_index(drop=True)
    )
    ret_pd1 = pd.DataFrame(tmp.groupby("trade_date").ret.mean() - 1.000) / 5
    ret_pd1.loc["20250101":, ].cumsum().plot(rot=45, title="2025 year")
    save_show(cfg.fig_dir, "topN_cumret")

    perf_topn = calc_topn_perf(ret_pd1)
    print(perf_topn)
    print(ret_pd1.tail(10))

    # --------------------------------------------------------------- hedging
    # Benchmark: equal-weight mean return of all theme stocks
    # (t+1 open buy, t+5 close sell, scaled to daily).
    dfindex = pd.DataFrame(test.groupby("trade_date").ret.mean() - 1.000) / 5
    dfindex.columns = ["pct_chg"]
    dfindex = dfindex.sort_index()

    ret_pd_concat = pd.merge(ret_pd1, dfindex, left_index=True, right_index=True)
    ret_pd_concat.columns = ["ret", "dfindex"]
    ret_pd_concat["diff"] = ret_pd_concat["ret"] - ret_pd_concat["dfindex"]
    corr_mat = ret_pd_concat.corr()
    ret_pd_concat[["ret", "dfindex"]].cumsum().plot(rot=45, title="2025-2026 year")
    save_show(cfg.fig_dir, "hedge_cumret")

    perf_hedge = calc_topn_perf(ret_pd_concat[["diff"]])
    print(perf_hedge)

    # --------------------------------------------- non-overlapping 5-day perf
    _test_tr = test[test["buy_tradable"].eq(True)]
    topn_daily = daily_topn_ret(_test_tr, cfg.top_n)
    dec_daily = daily_topdecile_ret(_test_tr)
    bench_all_daily = test.dropna(subset=["ret"]).groupby("trade_date")["ret"].mean() - 1.0
    bench_tr_daily = _test_tr.dropna(subset=["ret"]).groupby("trade_date")["ret"].mean() - 1.0

    _idx = topn_daily.index.intersection(bench_all_daily.index)
    nonoverlap_returns = pd.DataFrame({
        "TopN": topn_daily,
        "decile10": dec_daily,
        "bench_all": bench_all_daily,
        "bench_tradable": bench_tr_daily,
    }).reindex(_idx)
    nonoverlap_returns["excess_all"] = nonoverlap_returns["TopN"] - nonoverlap_returns["bench_all"]
    nonoverlap_returns["excess_tradable"] = nonoverlap_returns["TopN"] - nonoverlap_returns["bench_tradable"]

    series_map = {
        "TopN(tradable)": nonoverlap_returns["TopN"],
        "decile10(tradable)": nonoverlap_returns["decile10"],
        "bench(all eqw)": nonoverlap_returns["bench_all"],
        "bench(tradable eqw)": nonoverlap_returns["bench_tradable"],
        "excess(TopN-all)": nonoverlap_returns["excess_all"],
        "excess(TopN-tradable)": nonoverlap_returns["excess_tradable"],
    }
    perf_nonoverlap = pd.DataFrame(
        {name: calc_non_overlapping_pfmc(s, cfg.non_overlap_step) for name, s in series_map.items()}
    ).T
    print(f"[non-overlap {cfg.non_overlap_step}-day rebalance | 5-phase avg | "
          f"effective ~{len(_idx) // cfg.non_overlap_step} periods]")
    print(perf_nonoverlap)

    _s0_topn = nonoverlap_returns["TopN"].dropna().sort_index().iloc[0::cfg.non_overlap_step]
    _s0_bench = nonoverlap_returns["bench_all"].dropna().sort_index().iloc[0::cfg.non_overlap_step]
    _nav0 = pd.DataFrame({"TopN": (1 + _s0_topn).cumprod(),
                          "bench_all": (1 + _s0_bench).cumprod()})
    _nav0.plot(rot=45, title="non-overlap cumulative NAV (phase 0)", grid=True)
    save_show(cfg.fig_dir, "nonoverlap_nav")

    # ----------------------------------------------------------- Excel export
    metrics_all = {
        "SplitDate": cfg.split_date, "SeqLen": cfg.seq_len, "BatchSize": cfg.batch_size,
        "Epochs": cfg.epochs, "Folds": len(folds), "PurgeGapDays": purge_gap_days,
        "WinsorLow": float(ret_lo), "WinsorHigh": float(ret_hi),
        "MinValidRatio": cfg.min_valid_ratio,
        "TrainSamples": len(train_samples), "TestSamples": len(test_samples),
        **oof_metrics, **test_ic_metrics,
    }
    metrics_df = pd.DataFrame(list(metrics_all.items()), columns=["metric", "value"])

    out_path = os.path.join(cfg.out_dir, "results.xlsx")
    with pd.ExcelWriter(out_path, engine="openpyxl") as xw:
        metrics_df.to_excel(xw, sheet_name="metrics", index=False)
        perf_groups.to_excel(xw, sheet_name="decile_perf", index=False)
        perf_groups_tr.to_excel(xw, sheet_name="decile_perf_tradable", index=False)
        perf_topn.to_excel(xw, sheet_name="topn_perf", index=False)
        perf_hedge.to_excel(xw, sheet_name="hedge_perf", index=False)
        corr_mat.to_excel(xw, sheet_name="hedge_corr")
        ret_pd1.tail(10).to_excel(xw, sheet_name="topn_daily_tail")
        ret_pd_concat.to_excel(xw, sheet_name="hedge_daily")
        perf_nonoverlap.to_excel(xw, sheet_name="nonoverlap_perf")
        nonoverlap_returns.to_excel(xw, sheet_name="nonoverlap_returns")
        pd.DataFrame({"feature": features}).to_excel(xw, sheet_name="features", index=False)
        if factor_screen_report is not None:
            factor_screen_report.to_excel(xw, sheet_name="factor_screen", index=False)
    print("results exported:", out_path)

    print(f"total elapsed: {time.time() - t_start:.1f}s")

    return {
        "oof_metrics": oof_metrics,
        "test_ic_metrics": test_ic_metrics,
        "perf_topn": perf_topn,
        "perf_hedge": perf_hedge,
        "perf_nonoverlap": perf_nonoverlap,
        "features": features,
        "results_path": out_path,
    }
