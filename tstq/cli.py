"""Command-line interface: ``tstq {fetch,qfq,train}``.

Examples
--------
    tstq fetch --data-dir data
    tstq qfq   --data-dir data
    tstq train --data-dir data --split-date 20250101 --epochs 5

Any option not supplied falls back to the YAML config ``--config`` and then to
the built-in defaults. The Tushare token is read from ``$TUSHARE_TOKEN``.
"""

from __future__ import annotations

import argparse
import os
import sys

from .config import Config, load_config


def _load_dotenv(path: str = ".env") -> None:
    """Minimal .env loader (no external dependency): KEY=VALUE per line."""
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            os.environ.setdefault(key, value)


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--config", default=None, help="YAML config file")
    p.add_argument("--data-dir", dest="data_dir", default=None)
    p.add_argument("--out-dir", dest="out_dir", default=None)
    p.add_argument("--fig-dir", dest="fig_dir", default=None)
    p.add_argument("--start-date", dest="start_date", default=None)
    p.add_argument("--end-date", dest="end_date", default=None)


def _add_train(p: argparse.ArgumentParser) -> None:
    p.add_argument("--split-date", dest="split_date", default=None)
    p.add_argument("--seq-len", dest="seq_len", type=int, default=None)
    p.add_argument("--batch-size", dest="batch_size", type=int, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--n-splits", dest="n_splits", type=int, default=None)
    p.add_argument("--top-n", dest="top_n", type=int, default=None)
    p.add_argument("--factor-top-k", dest="factor_top_k", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--num-workers", dest="num_workers", type=int, default=None)
    p.add_argument("--use-amp", dest="use_amp", action=argparse.BooleanOptionalAction, default=None)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tstq", description="Tushare + Transformer A-share quant")
    sub = parser.add_subparsers(dest="command", required=True)

    f = sub.add_parser("fetch", help="fetch universe / daily bars / daily-basic from Tushare")
    _add_common(f)

    q = sub.add_parser("qfq", help="fetch adjustment factors and compute forward-adjusted prices")
    _add_common(q)

    t = sub.add_parser("train", help="train the Transformer and run the backtests")
    _add_common(t)
    _add_train(t)

    return parser


def _overrides(args: argparse.Namespace) -> dict:
    skip = {"command", "config"}
    out = {}
    for k, v in vars(args).items():
        if k in skip or v is None:
            continue
        if hasattr(Config, k):
            out[k] = v
    return out


def main(argv: list[str] | None = None) -> None:
    _load_dotenv()
    parser = build_parser()
    args = parser.parse_args(argv)

    cfg = load_config(args.config, **_overrides(args))

    if args.command == "fetch":
        from .fetch import fetch_all

        fetch_all(cfg)
    elif args.command == "qfq":
        from .adjust import build_qfq

        build_qfq(cfg)
    elif args.command == "train":
        from .pipeline import run

        run(cfg)
    else:  # pragma: no cover - argparse guarantees a valid command
        parser.error(f"unknown command: {args.command}")


if __name__ == "__main__":
    sys.exit(main())
