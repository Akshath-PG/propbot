"""Generate SYNTHETIC bars + calendar to smoke-test the pipeline without MT5.

The fake market has volatility clustering, sessions, and one deliberately
planted edge: after a liquidity sweep of the 20-bar high/low, price tends to
revert. A working pipeline should find that edge (mostly via the ICT expert).
Results on this data say nothing about real markets.

    python scripts/make_synthetic.py --out <folder> --start 2022-01-01
"""
import argparse
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd


def gen(symbol, start, end, p0, vol, seed, edge=0.35):
    rng = np.random.default_rng(seed)
    t = pd.date_range(start, end, freq="15min")
    t = t[t.dayofweek < 5]
    n = len(t)
    hour = t.hour.to_numpy()
    sess = np.where((hour >= 9) & (hour < 19), 1.6, 0.7)
    out = np.zeros((n, 4)); p = p0; s2 = vol ** 2
    hi_q, lo_q = deque(maxlen=20), deque(maxlen=20)
    bias_left, bias_dir = 0, 0
    for i in range(n):
        s2 = 0.97 * s2 + 0.03 * vol ** 2 * rng.lognormal(0, 0.6)
        sig = np.sqrt(s2) * sess[i]
        mu = bias_dir * edge * sig if bias_left > 0 else 0.0
        bias_left -= 1
        r = mu + sig * rng.standard_t(5) / np.sqrt(5 / 3)
        o = p; c = p * np.exp(r)
        h = max(o, c) * np.exp(abs(rng.normal(0, sig * 0.6)))
        l = min(o, c) * np.exp(-abs(rng.normal(0, sig * 0.6)))
        if len(hi_q) == 20:
            ph, pl = max(hi_q), min(lo_q)
            if h > ph and c < ph:
                bias_left, bias_dir = 10, -1
            elif l < pl and c > pl:
                bias_left, bias_dir = 10, 1
        hi_q.append(h); lo_q.append(l)
        out[i] = (o, h, l, c); p = c
    df = pd.DataFrame(out, columns=["open", "high", "low", "close"])
    df.insert(0, "time", t)
    df["tick_volume"] = (rng.poisson(300, n) * sess).astype(int)
    df["spread"] = np.where(symbol == "XAUUSD", 25, 12)
    return df


def calendar(start, end, seed=3):
    rng = np.random.default_rng(seed)
    rows = []
    for d in pd.date_range(start, end, freq="D"):
        if d.dayofweek in (2, 4):
            f = rng.normal(2, 1)
            rows.append((d + pd.Timedelta(hours=15, minutes=30), "USD", 3, "US Data", f + rng.normal(0, 0.5), f, f))
        if d.dayofweek == 3:
            f = rng.normal(1, 1)
            rows.append((d + pd.Timedelta(hours=12), "EUR", 3, "EU Data", f + rng.normal(0, 0.5), f, f))
        if d.dayofweek in (0, 1):
            rows.append((d + pd.Timedelta(hours=17), "USD", 2, "US Minor", np.nan, np.nan, np.nan))
    return pd.DataFrame(rows, columns=["time", "currency", "impact", "event", "actual", "forecast", "previous"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--start", default="2022-01-01")
    ap.add_argument("--end", default="2026-10-23")
    ap.add_argument("--edge", type=float, default=0.35, help="0 = pure random walk (no edge)")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    for sym, p0, vol, seed in (("XAUUSD", 1800.0, 0.0012, 1), ("EURUSD", 1.10, 0.0006, 2)):
        gen(sym, a.start, a.end, p0, vol, seed, a.edge).to_csv(out / f"{sym}_M15.csv", index=False)
        print("wrote", sym)
    calendar(a.start, a.end).to_csv(out / "calendar.csv", index=False)
    print("wrote calendar")
