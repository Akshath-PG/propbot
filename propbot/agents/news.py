"""Forex-news agent.

Two jobs:
  1. Features for the "news" expert (prefix nw_, built in features.news_features):
     time to / since high-impact events and the signed surprise of the last one.
  2. A blackout gate: no new position shortly before or after a high-impact
     event in the configured currencies (gold reacts to USD data). HERO allows
     news trading, but a 1% floating-loss limit makes a spike through the stop
     an account-ending event, so the agent stays out of those minutes.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


class NewsAgent:
    def __init__(self, calendar: pd.DataFrame | None, cfg: dict):
        self.cfg = cfg
        ev = calendar if calendar is not None else pd.DataFrame(columns=["time", "currency", "impact"])
        ev = ev[ev["currency"].isin(cfg.get("currencies", ["USD"])) & (ev["impact"] >= 3)]
        self.t = np.sort(pd.to_datetime(ev["time"]).to_numpy())
        self.before = np.timedelta64(int(cfg.get("blackout_before_min", 30)), "m")
        self.after = np.timedelta64(int(cfg.get("blackout_after_min", 15)), "m")

    def blackout_mask(self, entry_times, hold_until=None) -> np.ndarray:
        """True where a trade entered at entry_time would open inside a blackout window."""
        t = np.asarray(pd.to_datetime(entry_times), dtype="datetime64[ns]")
        if len(self.t) == 0:
            return np.zeros(len(t), bool)
        j = np.searchsorted(self.t, t - self.after, side="left")       # first event not long past
        nxt = self.t[np.minimum(j, len(self.t) - 1)]
        return (j < len(self.t)) & (nxt - self.before <= t)

    def blocked(self, entry_time) -> bool:
        return bool(self.blackout_mask([entry_time])[0])
