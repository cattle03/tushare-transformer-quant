"""Typed configuration with sane defaults + optional YAML override.

Every tunable that used to be a module-level global in the original scripts
lives here, so the whole pipeline can be driven from a single object.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional


# Tushare / 同花顺 THS concept-board codes for the "通信 / 算力" theme.
# Swap these for any other theme to reuse the whole pipeline.
DEFAULT_BOARDS: Dict[str, str] = {
    "5G": "885556.TI",
    "共封装光学(CPO)": "886033.TI",
    "光纤概念": "886084.TI",
    "东数西算(算力)": "885957.TI",
    "数据中心(AIDC)": "885887.TI",
    "液冷服务器": "886044.TI",
    "算力租赁": "886050.TI",
    "卫星通信": "865176.TI",
    "通信设备": "881129.TI",
    "通信服务": "881162.TI",
}

# Full candidate factor set produced by factors.AlphaLib.calcu_alpha().
ALL_FEATURES = [
    "alpha01", "alpha02", "alpha06", "alpha07",
    "alpha08", "alpha09", "alpha10", "alpha12", "alpha13", "alpha14",
    "alpha15", "alpha16", "alpha17", "alpha18", "alpha19", "alpha20",
    "alpha21", "alpha23", "alpha24", "alpha28", "alpha29", "alpha30",
    "alpha31", "alpha32", "alpha_w_005", "alpha_pe", "alpha_total_mv",
    "alpha_pb", "alpha_turnover_rate", "alpha_volume_ratio", "alpha_vwap",
    "alpha_dv_ttm",
]

# Valuation / liquidity factors sourced from daily_basic -- structurally
# sparse, so the valid_ratio gate only looks at price/volume factors.
VALUATION_FEATURES = [
    "alpha_pe", "alpha_total_mv", "alpha_pb", "alpha_turnover_rate",
    "alpha_volume_ratio", "alpha_vwap", "alpha_dv_ttm",
]

BASIC_FIELDS = (
    "ts_code,trade_date,turnover_rate,turnover_rate_f,volume_ratio,pe,pe_ttm,pb,ps,"
    "ps_ttm,dv_ratio,dv_ttm,total_share,float_share,free_share,total_mv,circ_mv"
)


@dataclass
class Config:
    # ----- data locations -------------------------------------------------
    data_dir: str = "data"                 # where raw/merged CSVs live
    out_dir: str = "outputs"               # results.xlsx lives here
    fig_dir: str = "outputs/figs"          # PNGs live here

    # ----- merged CSV file names -----------------------------------------
    price_file: str = "comm_price_qfq.csv"         # qfq daily bars (model input)
    basic_file: str = "comm_daily_basic.csv"
    universe_file: str = "comm_universe.csv"
    raw_price_file: str = "comm_price.csv"         # un-adjusted bars (qfq step input)
    adj_file: str = "comm_adj_factor.csv"
    adj_qfq_file: str = "comm_price_qfq.csv"

    # ----- fetch settings -------------------------------------------------
    boards: Dict[str, str] = field(default_factory=lambda: dict(DEFAULT_BOARDS))
    start_date: str = "20190101"
    end_date: str = "20261107"
    fetch_sleep: float = 0.08
    max_retry: int = 5

    # ----- model / training ----------------------------------------------
    split_date: str = "20250101"   # train < split_date, test > split_date
    seed: int = 42
    seq_len: int = 20
    batch_size: int = 512
    epochs: int = 5
    lr: float = 5e-4
    weight_decay: float = 1e-4
    d_model: int = 64
    nhead: int = 4
    num_layers: int = 2
    dim_ff: int = 128
    dropout: float = 0.1
    n_splits: int = 5
    target_scale: float = 100.0
    use_amp: bool = True
    num_workers: Optional[int] = None   # None -> auto (0 on Windows, else 4)

    # ----- factor screening ----------------------------------------------
    factor_screen_enable: bool = True
    factor_eval_ret: str = "ret5"       # 'ret1' | 'ret5' | 'ret'
    factor_top_k: int = 8
    factor_min_winrate: float = 0.50
    factor_max_mdd: float = -0.95
    factor_min_abs_icir: float = 0.05
    min_valid_ratio: float = 0.7

    # ----- backtest -------------------------------------------------------
    top_n: int = 20
    non_overlap_step: int = 5

    # ----- runtime helpers ------------------------------------------------
    def resolved_num_workers(self) -> int:
        if self.num_workers is not None:
            return self.num_workers
        return 0 if os.name == "nt" else 4

    def prepare_dirs(self) -> None:
        for d in (self.data_dir, self.out_dir, self.fig_dir):
            os.makedirs(d, exist_ok=True)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def load_config(path: Optional[str] = None, **overrides: Any) -> Config:
    """Build a Config from defaults, an optional YAML file, then kwargs.

    Precedence: defaults < YAML file < keyword overrides < environment vars
    for the handful of runtime-only knobs (e.g. COMM_DIR).
    """
    values: Dict[str, Any] = {}
    if path:
        import yaml  # lazy: only required when a config file is actually used

        with open(path, "r", encoding="utf-8") as fh:
            values = yaml.safe_load(fh) or {}

    values.update({k: v for k, v in overrides.items() if v is not None})

    # Environment override keeps parity with the original scripts.
    env_data_dir = os.environ.get("COMM_DIR")
    if env_data_dir and "data_dir" not in values:
        values["data_dir"] = env_data_dir

    cfg = Config(**values)
    cfg.prepare_dirs()
    return cfg
