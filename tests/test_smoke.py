"""Fast smoke tests for the pure (no-network, no-torch) building blocks.

Run with:  pytest -q
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from tstq.backtest import calc_non_overlapping_pfmc, max_dd_group
from tstq.factors import bin_factor_rank


def test_bin_factor_rank_buckets():
    assert bin_factor_rank(float("nan")) is None or np.isnan(bin_factor_rank(float("nan")))
    assert bin_factor_rank(0.0) == 1
    assert bin_factor_rank(0.95) == 10
    assert bin_factor_rank(1.0) == 10


def test_max_dd_group_monotonic_decline():
    idx = pd.RangeIndex(5)
    # constant -10% per period -> drawdown close to -1 + (1-0.1)^5
    r = pd.Series([-0.1] * 5, index=idx)
    mdd, start, end = max_dd_group(r)
    assert -1.0 < mdd < 0.0
    assert start == idx[0]


def test_non_overlapping_pfmc_shapes():
    rng = np.random.default_rng(0)
    rets = pd.Series(rng.normal(0.002, 0.02, size=250))
    out = calc_non_overlapping_pfmc(rets, period=5)
    assert set(out) == {"return", "sharpe", "std", "winratio", "mdd", "n_periods"}
    assert np.isfinite(out["n_periods"])
    assert out["n_periods"] >= 40
