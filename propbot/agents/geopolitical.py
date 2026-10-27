"""Geopolitical / macro-risk agent (market-implied).

There is no free, point-in-time archive of geopolitical headlines going back
to 2016, and a hand-made list of past crises would leak hindsight into the
backtest. So this agent reads geopolitical stress the way it shows up in gold
itself, from information available at the bar close:

  - weekend / session gaps (shocks that happen while the market is closed)
  - volatility shock: last day's range vs the 20-day norm
  - safe-haven momentum: 5- and 20-day returns in daily-ATR units
  - distance from the 1-year high (crisis bids tend to run at all-time highs)
  - today's range expansion so far

They feed the "geo" expert (prefix gp_) and a composite stress score that the
coordinator uses to cut size or block counter-trend trades in a crisis.
A live headline feed can be added later through `GeopoliticalAgent.external_score`.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

EPS = 1e-12


def trading_day(times: pd.Series) -> pd.Series:
    """Trading day that starts at 17:00 New York time (gold's daily break)."""
    t = pd.DatetimeIndex(times).tz_localize("UTC").tz_convert("America/New_York") + pd.Timedelta(hours=7)
    return pd.Series(np.asarray(t.date), index=times.index)


def geo_features(df: pd.DataFrame, a: pd.Series, f: dict, utc_offset_h: float = 0.0) -> None:
    t = df["time"] - pd.Timedelta(hours=utc_offset_h)
    day = trading_day(t)
    o, h, l, c = (df[x] for x in ("open", "high", "low", "close"))

    # completed trading days only (shifted by one day -> no look-ahead)
    d = pd.DataFrame({"day": day, "o": o, "h": h, "l": l, "c": c}).groupby("day").agg(
        o=("o", "first"), h=("h", "max"), l=("l", "min"), c=("c", "last"))
    pc = d["c"].shift(1)
    tr = pd.concat([d["h"] - d["l"], (d["h"] - pc).abs(), (d["l"] - pc).abs()], axis=1).max(axis=1)
    datr = tr.ewm(alpha=1 / 14, adjust=False, min_periods=10).mean()
    daily = pd.DataFrame({
        "gp_vol_shock": tr / (tr.rolling(20, min_periods=10).mean() + EPS),
        "gp_ret5": (d["c"] - d["c"].shift(5)) / (datr + EPS),
        "gp_ret20": (d["c"] - d["c"].shift(20)) / (datr + EPS),
        "gp_dist_1y_high": (d["c"] - d["h"].rolling(250, min_periods=20).max()) / (datr + EPS),
        "gp_open_gap": (d["o"] - pc) / (datr + EPS),
        "gp_datr_pct": datr / d["c"],
    })
    daily["gp_datr_regime"] = daily["gp_datr_pct"] / (daily["gp_datr_pct"].rolling(250, min_periods=20).mean() + EPS)
    # the open gap of *today* is known at today's first bar; everything else from yesterday
    gap_today = daily["gp_open_gap"]
    daily = daily.shift(1)
    daily["gp_open_gap"] = gap_today
    m = daily.reindex(day.values)
    for col in daily.columns:
        f[col] = m[col].to_numpy()

    # intraday: today's range so far vs. the daily ATR known at the open
    hi = h.groupby(day).cummax(); lo = l.groupby(day).cummin()
    datr_prev = (datr.shift(1)).reindex(day.values).to_numpy()
    f["gp_range_so_far"] = ((hi - lo).to_numpy()) / (datr_prev + EPS)

    # weekend gap: bar opening after >= 36 h of closed market
    gap_h = df["time"].diff().dt.total_seconds() / 3600
    wk = (gap_h >= 36).to_numpy()
    gap_sz = np.where(wk, (o - c.shift(1)).to_numpy() / (a.to_numpy() + EPS), np.nan)
    last_gap = pd.Series(gap_sz).ffill().fillna(0.0).to_numpy()
    f["gp_weekend_gap_atr"] = np.clip(last_gap, -20, 20)

    stress = (np.nan_to_num(f["gp_vol_shock"], nan=1.0) - 1.0) * 1.5 \
        + np.abs(np.nan_to_num(f["gp_open_gap"])) * 2.0 \
        + np.abs(np.nan_to_num(f["gp_ret5"])) * 0.4 \
        + np.clip(np.nan_to_num(f["gp_datr_regime"], nan=1.0) - 1.0, 0, None) * 1.5
    f["gp_stress"] = stress


@dataclass
class GeopoliticalAgent:
    """Turns the stress score into a regime, a size multiplier and a directional bias."""
    cfg: dict

    def regime(self, stress: float) -> str:
        cs = self.cfg.get("crisis_score", 2.5)
        if not np.isfinite(stress):
            return "calm"
        return "crisis" if stress >= cs else "elevated" if stress >= cs / 2 else "calm"

    def assess(self, stress: float, ret5: float) -> dict:
        reg = self.regime(stress)
        mult = self.cfg.get("crisis_risk_mult", 0.5) if reg == "crisis" else 1.0
        bias = 0
        if reg == "crisis" and np.isfinite(ret5) and abs(ret5) > 1.0:
            bias = 1 if ret5 > 0 else -1          # safe-haven flow direction
        return {"regime": reg, "risk_mult": mult, "bias": bias}

    def allows(self, direction: int, view: dict) -> bool:
        if view["regime"] != "crisis" or not self.cfg.get("block_counter_trend_in_crisis", True):
            return True
        return view["bias"] == 0 or direction == view["bias"]

    @staticmethod
    def external_score(headline_risk: float | None) -> float:
        """Hook for a live geopolitical headline score (0 = calm). Not used in backtests."""
        return float(headline_risk or 0.0)
