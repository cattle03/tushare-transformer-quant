"""Tushare + Transformer A-share quant research toolkit.

Importing this package pins the numeric libraries to single-threaded execution
*before* numpy / pandas / torch are imported elsewhere. This keeps floating
point reductions (and therefore the NaN masks that drive sample filtering)
deterministic run-to-run, which matters a lot for reproducible backtests.
"""

import os as _os

# Deterministic BLAS / OpenMP behaviour -- must run before numpy import.
for _v in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    _os.environ.setdefault(_v, "1")
# Required for deterministic cuBLAS ops on GPU (set before any CUDA call).
_os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

__version__ = "0.1.0"

__all__ = ["__version__"]
