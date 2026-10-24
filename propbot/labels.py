"""Triple-barrier labels: what would a long and a short trade have done?

For each closed bar t the trade enters at the next bar's open with
SL = sl_atr * ATR and TP = tp_atr * ATR. Walking forward up to
horizon_bars, the first barrier touched decides the result. If SL and TP
are both touched inside one bar we assume the SL came first (pessimistic).
Unresolved trades are closed at market at the horizon. Spread and
commission are charged in R units.

Class: 2 = long wins, 0 = short wins, 1 = neither (no trade).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _side(o, h, l, c, entry, sl_d, tp_d, H, direction):
    n = len(o)
    r = np.full(n, np.nan)
    tp_hit = np.zeros(n, bool)
    open_ = np.ones(n, bool)
    rr = tp_d / sl_d
    for j in range(1, H + 1):
        idx = np.arange(n) + j
        ok = idx < n
        ii = np.where(ok, idx, n - 1)
        if direction == 1:
            sl_now = l[ii] <= entry - sl_d
            tp_now = h[ii] >= entry + tp_d
        else:
            sl_now = h[ii] >= entry + sl_d
            tp_now = l[ii] <= entry - tp_d
        sl_now &= ok & open_
        tp_now &= ok & open_ & ~sl_now
        r[sl_now] = -1.0
        r[tp_now] = rr[tp_now]
        tp_hit |= tp_now
        open_ &= ~(sl_now | tp_now)
    end = np.arange(n) + H
    has_end = end < n
    last = c[np.minimum(end, n - 1)]
    timeout = open_ & has_end
    r[timeout] = (direction * (last - entry) / sl_d)[timeout]
    return r, tp_hit


def make_labels(df: pd.DataFrame, a: pd.Series, settings: dict, symbol: str) -> pd.DataFrame:
    p = settings["labels"]
    spec = settings["symbols"][symbol]
    H = int(p["horizon_bars"])
    o, h, l, c = (df[x].to_numpy(float) for x in ("open", "high", "low", "close"))
    av = a.to_numpy(float)
    entry = np.r_[o[1:], np.nan]                 # next bar open
    sl_d = p["sl_atr"] * av
    tp_d = p["tp_atr"] * av
    # _side walks bars t+1 .. t+H (the entry bar is included)
    r_long, w_long = _side(o, h, l, c, entry, sl_d, tp_d, H, 1)
    r_short, w_short = _side(o, h, l, c, entry, sl_d, tp_d, H, -1)

    # costs in R: spread (points) + round-turn commission per lot expressed per unit
    spread_px = df["spread"].to_numpy(float) * spec["point"]
    comm_px = spec.get("commission_per_lot", 0.0) / spec["contract_size"]
    if spec.get("pnl_conversion") == "base_usd":
        comm_px = comm_px * c  # commission in USD -> price units of quote ccy
    cost_r = (spread_px + comm_px) / sl_d
    r_long = r_long - cost_r
    r_short = r_short - cost_r

    y = np.ones(len(df), dtype=float)
    y[w_long & (r_long > 0)] = 2
    y[w_short & (r_short > 0)] = 0
    invalid = np.isnan(r_long) | np.isnan(r_short) | ~np.isfinite(sl_d) | (sl_d <= 0)
    y[invalid] = np.nan
    return pd.DataFrame({"y": y, "r_long": r_long, "r_short": r_short})
