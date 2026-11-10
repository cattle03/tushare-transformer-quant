"""Thin Tushare connection helper.

The token is read from the ``TUSHARE_TOKEN`` environment variable so it never
lives in source control. See ``.env.example`` for the expected format.
"""

from __future__ import annotations

import os


def connect(direct_connect: bool = True, token: str | None = None):
    """Return a Tushare ``pro_api`` handle.

    Parameters
    ----------
    direct_connect:
        Kept for backwards compatibility with the original helper. When a
        custom ``TUSHARE_API_URL`` is configured this routes requests to it.
    token:
        Explicit token; falls back to ``$TUSHARE_TOKEN``.
    """
    import tushare as ts

    token = token or os.environ.get("TUSHARE_TOKEN")
    if not token:
        raise RuntimeError(
            "No Tushare token found. Set the TUSHARE_TOKEN environment variable "
            "(e.g. `setx TUSHARE_TOKEN your_token` on Windows, or put it in a .env "
            "file and load it). Get a token at https://tushare.pro/user/token"
        )

    pro = ts.pro_api(token)

    api_url = os.environ.get("TUSHARE_API_URL")
    if direct_connect and api_url:
        # Mirror the original "private server" pattern when an URL is supplied.
        pro._DataApi__http_url = api_url
        pro._DataApi__token = token
    return pro
