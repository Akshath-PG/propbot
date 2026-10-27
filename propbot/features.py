"""Feature engineering, grouped into the strategy "experts".

Every feature at bar t uses only bars <= t (no look-ahead). Swing points are
only used after they are confirmed (k bars later), higher-timeframe bars only
after they have closed, and news "actual" values only after release time.

Groups
  trend     : EMAs, slopes, efficiency, higher-timeframe trend
  ict       : FVG / IFVG, liquidity sweeps (fakeouts), BOS / CHoCH,
              displacement, premium/discount, session (Asia range) sweeps
  orderflow : tick-volume based proxies (see README: true order flow needs futures data)
  reversal  : RSI, divergence, stretch, wicks, exhaustion
  news      : time to/from high-impact events, signed surprise
  geo       : market-implied geopolitical / macro stress (agents/geopolitical.py)
  context   : volatility regime, session, symbol (used by the combiner)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import TF_MINUTES

EPS = 1e-12


# ------------------------------------------------------------------ helpers
def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    pc = df["close"].shift(1)
    tr = pd.concat([df["high"] - df["low"], (df["high"] - pc).abs(), (df["low"] - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    return 100 - 100 / (1 + up / (dn + EPS))


def bars_since(flag: np.ndarray, cap: int = 200) -> np.ndarray:
    out = np.full(len(flag), cap, dtype=float)
    last = -10**9
    for i, f in enumerate(flag):
        if f:
            last = i
        out[i] = min(i - last, cap)
    return out


# ------------------------------------------------------------------ groups
def trend_features(df, a, f):
    c = df["close"]
    for n in (20, 50, 200):
        e = c.ewm(span=n, adjust=False).mean()
        f[f"tr_ema{n}_dist"] = (c - e) / a
        f[f"tr_ema{n}_slope"] = (e - e.shift(5)) / a
    for n in (4, 16, 64):
        f[f"tr_ret_{n}"] = (c - c.shift(n)) / a
    move = (c - c.shift(20)).abs()
    path = c.diff().abs().rolling(20).sum()
    f["tr_efficiency20"] = move / (path + EPS)


def htf_features(df, f, htfs, tf):
    base_min = TF_MINUTES[tf]
    s = df.set_index("time")
    close_time = (df["time"] + pd.Timedelta(minutes=base_min)).to_frame("t")
    for htf in htfs:
        m = TF_MINUTES[htf]
        r = s.resample(f"{m}min", label="right", closed="left").agg(
            {"open": "first", "high": "max", "low": "min", "close": "last"}).dropna()
        ra = atr(r, 14)
        e20 = r["close"].ewm(span=20, adjust=False).mean()
        e50 = r["close"].ewm(span=50, adjust=False).mean()
        h = pd.DataFrame({
            f"tr_{htf}_ema20_dist": (r["close"] - e20) / ra,
            f"tr_{htf}_ema50_dist": (r["close"] - e50) / ra,
            f"tr_{htf}_slope": (e20 - e20.shift(3)) / ra,
            f"tr_{htf}_ret3": (r["close"] - r["close"].shift(3)) / ra,
        })
        h.index.name = "t"
        merged = pd.merge_asof(close_time, h.reset_index(), on="t", direction="backward")
        for col in h.columns:
            f[col] = merged[col].values


def ict_features(df, a, f, utc_hour, utc_date, min_gap_atr=0.25, k=3):
    o, h, l, c = (df[x].to_numpy() for x in ("open", "high", "low", "close"))
    av = a.to_numpy()
    n = len(df)

    # ---- FVG / IFVG tracking
    dist_bull = np.full(n, 10.0); dist_bear = np.full(n, 10.0)
    in_bull = np.zeros(n); in_bear = np.zeros(n)
    n_bull = np.zeros(n); n_bear = np.zeros(n)
    ifvg_bull_evt = np.zeros(n, bool); ifvg_bear_evt = np.zeros(n, bool)
    fvg_bull_evt = np.zeros(n, bool); fvg_bear_evt = np.zeros(n, bool)
    zones = []  # [dir(+1 bull/-1 bear), bottom, top, created, inverted]
    for t in range(2, n):
        if not np.isfinite(av[t]) or av[t] <= 0:
            continue
        # new zones formed by candles t-2..t
        if l[t] - h[t - 2] > min_gap_atr * av[t]:
            zones.append([1, h[t - 2], l[t], t, False]); fvg_bull_evt[t] = True
        if l[t - 2] - h[t] > min_gap_atr * av[t]:
            zones.append([-1, h[t], l[t - 2], t, False]); fvg_bear_evt[t] = True
        keep = []
        for z in zones:
            d, bot, top, t0, inv = z
            if t - t0 > 500:
                continue
            if z[3] != t:
                # a close through the far side inverts the gap (IFVG)
                if d == 1 and c[t] < bot:
                    if inv:  # already inverted once -> discard
                        continue
                    z[0], z[4] = -1, True; ifvg_bear_evt[t] = True
                elif d == -1 and c[t] > top:
                    if inv:
                        continue
                    z[0], z[4] = 1, True; ifvg_bull_evt[t] = True
            keep.append(z)
        zones = keep[-40:]
        bull = [z for z in zones if z[0] == 1 and z[2] <= c[t] + av[t] * 0.5]
        bear = [z for z in zones if z[0] == -1 and z[1] >= c[t] - av[t] * 0.5]
        n_bull[t] = len(bull); n_bear[t] = len(bear)
        if bull:
            top = max(z[2] for z in bull)
            dist_bull[t] = min((c[t] - top) / av[t], 10)
            in_bull[t] = float(any(l[t] <= z[2] and c[t] >= z[1] for z in bull))
        if bear:
            bot = min(z[1] for z in bear)
            dist_bear[t] = min((bot - c[t]) / av[t], 10)
            in_bear[t] = float(any(h[t] >= z[1] and c[t] <= z[2] for z in bear))
    f["ict_dist_bull_fvg"] = dist_bull; f["ict_dist_bear_fvg"] = dist_bear
    f["ict_in_bull_fvg"] = in_bull; f["ict_in_bear_fvg"] = in_bear
    f["ict_n_bull_fvg"] = n_bull; f["ict_n_bear_fvg"] = n_bear
    f["ict_since_bull_fvg"] = bars_since(fvg_bull_evt, 100)
    f["ict_since_bear_fvg"] = bars_since(fvg_bear_evt, 100)
    f["ict_since_bull_ifvg"] = bars_since(ifvg_bull_evt, 100)
    f["ict_since_bear_ifvg"] = bars_since(ifvg_bear_evt, 100)

    # ---- liquidity sweeps / fakeouts of recent highs & lows
    hs, ls, cs = df["high"], df["low"], df["close"]
    for N in (20, 96):
        ph = hs.rolling(N).max().shift(1); pl = ls.rolling(N).min().shift(1)
        sweep_hi = ((hs > ph) & (cs < ph)).to_numpy()
        sweep_lo = ((ls < pl) & (cs > pl)).to_numpy()
        f[f"ict_sweep_hi{N}_mag"] = np.where(sweep_hi, (hs - ph) / a, 0.0)
        f[f"ict_sweep_lo{N}_mag"] = np.where(sweep_lo, (pl - ls) / a, 0.0)
        f[f"ict_since_sweep_hi{N}"] = bars_since(sweep_hi, 100)
        f[f"ict_since_sweep_lo{N}"] = bars_since(sweep_lo, 100)
        f[f"ict_breakout_hi{N}"] = ((cs > ph).astype(float)).to_numpy()
        f[f"ict_breakout_lo{N}"] = ((cs < pl).astype(float)).to_numpy()

    # ---- Asian range sweep during London / New York
    is_asia = (utc_hour < 7)
    tmp = pd.DataFrame({"d": utc_date, "h": np.where(is_asia, h, np.nan),
                        "l": np.where(is_asia, l, np.nan)})
    asia_hi = tmp.groupby("d")["h"].cummax().groupby(tmp["d"]).ffill().to_numpy()
    asia_lo = tmp.groupby("d")["l"].cummin().groupby(tmp["d"]).ffill().to_numpy()
    after = ~is_asia & np.isfinite(asia_hi)
    f["ict_asia_sweep_hi"] = (after & (h > asia_hi) & (c < asia_hi)).astype(float)
    f["ict_asia_sweep_lo"] = (after & (l < asia_lo) & (c > asia_lo)).astype(float)
    f["ict_dist_asia_hi"] = np.where(after, (c - asia_hi) / av, 0.0)
    f["ict_dist_asia_lo"] = np.where(after, (c - asia_lo) / av, 0.0)

    # ---- market structure: confirmed swings, BOS, CHoCH
    win = 2 * k + 1
    is_ph = (hs == hs.rolling(win, center=True).max())
    is_pl = (ls == ls.rolling(win, center=True).min())
    # a pivot at i is only known at i+k
    sh = hs.where(is_ph).shift(k).ffill().to_numpy()
    sl = ls.where(is_pl).shift(k).ffill().to_numpy()
    state = np.zeros(n); bos_up = np.zeros(n, bool); bos_dn = np.zeros(n, bool)
    choch_up = np.zeros(n, bool); choch_dn = np.zeros(n, bool)
    st = 0
    for t in range(1, n):
        if np.isfinite(sh[t]) and c[t] > sh[t] and c[t - 1] <= sh[t]:
            bos_up[t] = True; choch_up[t] = st == -1; st = 1
        elif np.isfinite(sl[t]) and c[t] < sl[t] and c[t - 1] >= sl[t]:
            bos_dn[t] = True; choch_dn[t] = st == 1; st = -1
        state[t] = st
    f["ict_struct_state"] = state
    f["ict_since_bos_up"] = bars_since(bos_up, 100)
    f["ict_since_bos_dn"] = bars_since(bos_dn, 100)
    f["ict_since_choch_up"] = bars_since(choch_up, 100)
    f["ict_since_choch_dn"] = bars_since(choch_dn, 100)
    f["ict_dist_swing_hi"] = (c - sh) / av
    f["ict_dist_swing_lo"] = (c - sl) / av

    # ---- displacement and premium / discount
    body = (cs - df["open"]) / a
    f["ict_body_atr"] = body
    f["ict_since_disp_up"] = bars_since((body > 1.5).to_numpy(), 100)
    f["ict_since_disp_dn"] = bars_since((body < -1.5).to_numpy(), 100)
    rh = hs.rolling(96).max(); rl = ls.rolling(96).min()
    f["ict_premium_discount"] = (cs - rl) / (rh - rl + EPS)


def orderflow_features(df, a, f, point):
    o, h, l, c, v = (df[x] for x in ("open", "high", "low", "close", "tick_volume"))
    rng = (h - l).replace(0, np.nan)
    vm = v.rolling(96).mean(); vs = v.rolling(96).std()
    f["of_vol_z"] = (v - vm) / (vs + EPS)
    delta = v * (c - o) / rng
    f["of_delta_20"] = delta.rolling(20).sum() / (v.rolling(20).sum() + EPS)
    f["of_delta_5"] = delta.rolling(5).sum() / (v.rolling(5).sum() + EPS)
    f["of_clv"] = ((c - l) - (h - c)) / rng
    f["of_absorption"] = ((f["of_vol_z"] > 1.5) & ((c - o).abs() / rng < 0.3)).astype(float)
    f["of_delta_div"] = np.sign((c - c.shift(20)).fillna(0)) * -np.sign(f["of_delta_20"].fillna(0))
    f["of_spread_atr"] = df["spread"] * point / a


def reversal_features(df, a, f):
    c, h, l, o = df["close"], df["high"], df["low"], df["open"]
    r = rsi(c, 14)
    f["rv_rsi14"] = r
    f["rv_rsi_slope"] = r - r.shift(3)
    sma = c.rolling(20).mean(); sd = c.rolling(20).std()
    f["rv_bb_z"] = (c - sma) / (sd + EPS)
    rng = (h - l).replace(0, np.nan)
    f["rv_upper_wick"] = (h - np.maximum(o, c)) / rng
    f["rv_lower_wick"] = (np.minimum(o, c) - l) / rng
    up = (c > o).astype(int); dn = (c < o).astype(int)
    f["rv_consec_up"] = up.groupby((up != up.shift()).cumsum()).cumsum() * up
    f["rv_consec_dn"] = dn.groupby((dn != dn.shift()).cumsum()).cumsum() * dn
    at_hi = h >= h.rolling(20).max(); at_lo = l <= l.rolling(20).min()
    f["rv_bear_div"] = (at_hi & (r < r.rolling(20).max() - 5)).astype(float)
    f["rv_bull_div"] = (at_lo & (r > r.rolling(20).min() + 5)).astype(float)
    f["rv_range_atr"] = (h - l) / a


def news_features(df, f, cal, currencies, symbol, tf):
    n = len(df)
    names = ["nw_mins_to_high", "nw_mins_since_high", "nw_surprise", "nw_high_today", "nw_mins_to_med"]
    if cal is None or cal.empty:
        f["nw_mins_to_high"] = 1440.0; f["nw_mins_since_high"] = 1440.0
        f["nw_surprise"] = 0.0; f["nw_high_today"] = 0.0; f["nw_mins_to_med"] = 1440.0
        return
    ev = cal[cal["currency"].isin(currencies)]
    t_close = (df["time"] + pd.Timedelta(minutes=TF_MINUTES[tf])).to_numpy()
    base = symbol[:3]

    def mins(x):
        return x / np.timedelta64(1, "m")

    for imp, nm_to in ((3, "nw_mins_to_high"), (2, "nw_mins_to_med")):
        et = ev.loc[ev["impact"] >= imp, "time"].to_numpy()
        if len(et) == 0:
            f[nm_to] = 1440.0; continue
        j = np.searchsorted(et, t_close, side="right")
        nxt = np.where(j < len(et), mins(et[np.minimum(j, len(et) - 1)] - t_close), 1440)
        f[nm_to] = np.clip(nxt, 0, 1440)

    hi = ev[ev["impact"] >= 3].copy()
    if hi.empty:
        f["nw_mins_since_high"] = 1440.0; f["nw_surprise"] = 0.0; f["nw_high_today"] = 0.0
        return
    hi["sign"] = np.where(hi["currency"] == base, 1.0, -1.0)
    if symbol.startswith("US") and not symbol.startswith("USD"):  # US indices
        hi["sign"] = 1.0
    surpr = (hi["actual"] - hi["forecast"]) / (hi["forecast"].abs() + hi["previous"].abs() * 0.1 + 1e-9)
    hi["s"] = (surpr.clip(-5, 5) * hi["sign"]).fillna(0.0)
    et = hi["time"].to_numpy()
    j = np.searchsorted(et, t_close, side="right") - 1  # last event released at or before bar close
    valid = j >= 0
    since = np.where(valid, mins(t_close - et[np.maximum(j, 0)]), 1440)
    f["nw_mins_since_high"] = np.clip(since, 0, 1440)
    f["nw_surprise"] = np.where(valid & (since <= 240), hi["s"].to_numpy()[np.maximum(j, 0)], 0.0)
    day = pd.Series(pd.to_datetime(t_close)).dt.normalize().to_numpy()
    k0 = np.searchsorted(et, day, side="left")
    f["nw_high_today"] = np.maximum(j + 1 - k0, 0).astype(float)


def context_features(df, a, f, utc_hour, symbol_id):
    f["ctx_atr_pct"] = a / df["close"]
    f["ctx_vol_regime"] = a / (a.rolling(500, min_periods=50).mean() + EPS)
    f["ctx_hour_sin"] = np.sin(2 * np.pi * utc_hour / 24)
    f["ctx_hour_cos"] = np.cos(2 * np.pi * utc_hour / 24)
    f["ctx_london"] = ((utc_hour >= 7) & (utc_hour < 10)).astype(float)
    f["ctx_newyork"] = ((utc_hour >= 12) & (utc_hour < 15)).astype(float)
    f["ctx_asia"] = (utc_hour < 7).astype(float)
    f["ctx_dow"] = df["time"].dt.dayofweek.to_numpy().astype(float)
    f["ctx_symbol"] = float(symbol_id)


GROUP_PREFIX = {"trend": "tr_", "ict": "ict_", "orderflow": "of_",
                "reversal": "rv_", "news": "nw_", "geo": "gp_", "context": "ctx_"}
EXPERTS = ["trend", "ict", "orderflow", "reversal", "news"]


def experts_for(settings: dict) -> list[str]:
    return list(settings.get("training", {}).get("experts") or EXPERTS)


def feature_groups(columns) -> dict:
    return {g: [c for c in columns if c.startswith(p)] for g, p in GROUP_PREFIX.items()}


def build_features(df: pd.DataFrame, settings: dict, symbol: str,
                   calendar: pd.DataFrame | None = None) -> pd.DataFrame:
    """Return a frame aligned with df: time, symbol, atr + all feature columns."""
    spec = settings["symbols"][symbol]
    tf = settings["timeframe"]
    a = atr(df, settings["labels"]["atr_period"])
    utc_time = df["time"] - pd.Timedelta(hours=settings.get("server_utc_offset_hours", 0))
    utc_hour = utc_time.dt.hour.to_numpy() + utc_time.dt.minute.to_numpy() / 60
    utc_date = utc_time.dt.date.to_numpy()

    f: dict = {}
    trend_features(df, a, f)
    htf_features(df, f, settings.get("higher_timeframes", []), tf)
    ict_features(df, a, f, utc_hour, utc_date)
    orderflow_features(df, a, f, spec["point"])
    reversal_features(df, a, f)
    news_features(df, f, calendar, spec.get("currencies", []), symbol, tf)
    if "geo" in experts_for(settings):
        from .agents.geopolitical import geo_features
        geo_features(df, a, f, settings.get("server_utc_offset_hours", 0))
    context_features(df, a, f, utc_hour, list(settings["symbols"]).index(symbol))

    out = pd.DataFrame({k: np.asarray(v, dtype=float) for k, v in f.items()})
    out.insert(0, "time", df["time"].values)
    out.insert(1, "symbol", symbol)
    out.insert(2, "atr", a.values)
    return out.replace([np.inf, -np.inf], np.nan)
