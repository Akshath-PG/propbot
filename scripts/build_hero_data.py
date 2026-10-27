"""Build the XAUUSD M15 history + economic calendar used by the HERO model,
from public GitHub mirrors (works without MT5 and without market-data APIs).

    python scripts/build_hero_data.py               # -> data/XAUUSD_M15.csv, data/calendar.csv

Sources (cloned into data/external/, which is git-ignored):

  Dypoi/XAUUSD_Dataset                  Dukascopy XAUUSD M1 bid+ask, 2016-09 -> 2026-09-01 (UTC)
  getdata-finance/xauusd-15m-...        XAUUSD 15m, used only for 2026-09-02 -> 2026-09-09
  navtemmt/xauusd-autodata              XAUUSD 1m (FX feed), 2026-09-09 -> today
  spoluan/forex-factory-scraper         ForexFactory calendar 2016-2019 (times in GMT+8)
  ehsanrs2/forexfactory-scraper         ForexFactory high-impact calendar 2020 -> 2026-01 (tz-aware)

After the ForexFactory snapshot ends, the calendar is extended with the
scheduled US events whose dates follow fixed rules (FOMC decisions, NFP).

The two late gold feeds agree with Dukascopy spot to within ~0.2 from
2026-08-30 on (checked on the overlap), so they are appended without an
offset. They have no bid/ask, so their spread is set to the median
Dukascopy spread of the last 30 days.

All bar times are UTC. Set server_utc_offset_hours: 0 in the settings file.
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

SOURCES = {
    "dukascopy": "https://github.com/Dypoi/XAUUSD_Dataset",
    "getdata15m": "https://github.com/getdata-finance/xauusd-15m-ohlcv-metals-historical-data",
    "navtemmt": "https://github.com/navtemmt/xauusd-autodata",
    "ff_2010s": "https://github.com/spoluan/forex-factory-scraper",
    "ff_2020s": "https://github.com/ehsanrs2/forexfactory-scraper",
}

POINT = 0.01


# ------------------------------------------------------------------ fetch
def fetch(name: str, url: str, ext: Path, paths: list[str] | None = None) -> tuple[Path, str]:
    dest = ext / name
    if not (dest / ".git").exists():
        subprocess.run(["git", "clone", "-q", "--depth", "1", "--filter=blob:none", "--no-checkout", url, str(dest)],
                       check=True)
    else:
        subprocess.run(["git", "-C", str(dest), "fetch", "-q", "--depth", "1", "origin", "HEAD"], check=True)
        subprocess.run(["git", "-C", str(dest), "reset", "-q", "--soft", "FETCH_HEAD"], check=True)
    subprocess.run(["git", "-C", str(dest), "checkout", "-q", "HEAD", "--", *(paths or ["."])], check=True)
    head = subprocess.run(["git", "-C", str(dest), "rev-parse", "--short", "HEAD"],
                          capture_output=True, text=True, check=True).stdout.strip()
    return dest, head


# ------------------------------------------------------------------ bars
def m15(df: pd.DataFrame, cols: dict) -> pd.DataFrame:
    return df.resample("15min", label="left", closed="left").agg(cols).dropna(subset=["open"])


def dukascopy_bars(folder: Path) -> pd.DataFrame:
    frames = []
    for f in sorted(folder.glob("XAUUSD_M1_*.csv")):
        d = pd.read_csv(f, parse_dates=["timestamp"]).set_index("timestamp")
        d["spread_px"] = d["close_ask"] - d["close_bid"]
        d["vol"] = d["volume_bid"] + d["volume_ask"]
        b = m15(d.rename(columns={"open_bid": "open", "high_bid": "high", "low_bid": "low", "close_bid": "close"}),
                {"open": "first", "high": "max", "low": "min", "close": "last", "vol": "sum", "spread_px": "median"})
        frames.append(b)
        print(f"  {f.name}: {len(b):,} M15 bars")
    b = pd.concat(frames)
    b = b[~b.index.duplicated(keep="last")].sort_index()
    b["tick_volume"] = np.maximum((b.pop("vol") * 1000).round(), 1).astype(int)
    b["spread"] = (b.pop("spread_px") / POINT).round().clip(lower=1).astype(int)
    return b


def ohlc_csv(path: Path, time_col: str, utc: bool) -> pd.DataFrame:
    d = pd.read_csv(path)
    t = pd.to_datetime(d[time_col], utc=utc)
    d.index = t.dt.tz_localize(None) if utc else t
    d = d.rename(columns=str.lower)
    return d[["open", "high", "low", "close", "volume"]].astype(float).sort_index()


# ------------------------------------------------------------------ calendar
def _num(x):
    if pd.isna(x):
        return np.nan
    s = str(x).strip().replace(",", "")
    m = re.match(r"^([<>]?)(-?\d+(?:\.\d+)?)([KMBT%]?)$", s)
    if not m:
        return np.nan
    v = float(m.group(2))
    return v * {"K": 1e3, "M": 1e6, "B": 1e9, "T": 1e12}.get(m.group(3), 1.0)


def calendar_2010s(folder: Path) -> pd.DataFrame:
    rows = []
    for y in range(2016, 2020):
        d = pd.read_csv(folder / f"forex_factory_calendar_{y}.csv")
        d = d[d["Impact"].eq("High") & ~d["Time"].astype(str).str.contains("Day|Tentative", na=True)]
        # this scraper wrote times in GMT+8
        t = pd.to_datetime(d["Combined DateTime"]) - pd.Timedelta(hours=8)
        rows.append(pd.DataFrame({"time": t.values, "currency": d["Currency"].values, "impact": 3,
                                  "event": d["Event"].values, "actual": d["Actual"].map(_num).values,
                                  "forecast": d["Forecast"].map(_num).values,
                                  "previous": d["Previous"].map(_num).values}))
    return pd.concat(rows, ignore_index=True)


def calendar_2020s(path: Path) -> pd.DataFrame:
    d = pd.read_csv(path)
    d = d[d["Impact"].astype(str).str.startswith("High")]
    t = pd.to_datetime(d["DateTime"], utc=True, format="ISO8601").dt.tz_localize(None)
    return pd.DataFrame({"time": t.values, "currency": d["Currency"].values, "impact": 3,
                         "event": d["Event"].values, "actual": d["Actual"].map(_num).values,
                         "forecast": d["Forecast"].map(_num).values, "previous": d["Previous"].map(_num).values})


# FOMC statement days (second day of each meeting), 14:00 New York time.
FOMC_2026 = ["2026-01-28", "2026-03-18", "2026-04-29", "2026-06-17", "2026-07-29",
             "2026-09-16", "2026-10-28", "2026-12-09"]


def scheduled_us_events(start, end) -> pd.DataFrame:
    """Rule-based US high-impact events for months the scraped calendar does not cover."""
    ny = "America/New_York"
    rows = []
    for d in FOMC_2026:
        t = pd.Timestamp(f"{d} 14:00").tz_localize(ny).tz_convert("UTC").tz_localize(None)
        rows.append((t, "USD", 3, "FOMC Statement (scheduled)"))
    for ms in pd.date_range(pd.Timestamp(start).replace(day=1), end, freq="MS"):
        first_fri = ms + pd.Timedelta(days=(4 - ms.dayofweek) % 7)
        t = pd.Timestamp(f"{first_fri.date()} 08:30").tz_localize(ny).tz_convert("UTC").tz_localize(None)
        rows.append((t, "USD", 3, "Non-Farm Employment Change (scheduled)"))
    ev = pd.DataFrame(rows, columns=["time", "currency", "impact", "event"])
    ev = ev[(ev["time"] > pd.Timestamp(start)) & (ev["time"] <= pd.Timestamp(end))]
    for c in ("actual", "forecast", "previous"):
        ev[c] = np.nan
    return ev


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(ROOT / "data"))
    a = ap.parse_args()
    data = Path(a.data_dir); ext = data / "external"; ext.mkdir(parents=True, exist_ok=True)

    print("Fetching sources ...")
    duk, h1 = fetch("dukascopy", SOURCES["dukascopy"], ext)
    g15, h2 = fetch("getdata15m", SOURCES["getdata15m"], ext, ["XAUUSD_15m.csv"])
    nav, h3 = fetch("navtemmt", SOURCES["navtemmt"], ext, ["data/XAUUSD_data.csv"])
    ff1, h4 = fetch("ff_2010s", SOURCES["ff_2010s"], ext, ["datasets"])
    ff2, h5 = fetch("ff_2020s", SOURCES["ff_2020s"], ext, ["high_impact_events_calendar.csv"])

    print("Building XAUUSD M15 ...")
    bars = dukascopy_bars(duk)
    duk_end = bars.index.max()
    med_spread = int(bars.loc[bars.index > duk_end - pd.Timedelta(days=30), "spread"].median())

    g = m15(ohlc_csv(g15 / "XAUUSD_15m.csv", "datetime", utc=True),
            {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
    n = m15(ohlc_csv(nav / "data" / "XAUUSD_data.csv", "datetime", utc=False),
            {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
    nav_start = n.index.min()
    ext_bars = pd.concat([g[(g.index > duk_end) & (g.index < nav_start)], n[n.index > duk_end]])
    ext_bars = ext_bars[~ext_bars.index.duplicated(keep="last")]
    ext_bars["tick_volume"] = np.maximum(ext_bars.pop("volume").round(), 1).astype(int)
    ext_bars["spread"] = med_spread
    bars = pd.concat([bars, ext_bars]).sort_index()
    bars = bars[(bars["high"] >= bars["low"]) & (bars["close"] > 0)]
    bars.index.name = "time"
    out = data / "XAUUSD_M15.csv"
    bars.reset_index()[["time", "open", "high", "low", "close", "tick_volume", "spread"]].to_csv(out, index=False)
    print(f"  wrote {out}: {len(bars):,} bars {bars.index.min()} -> {bars.index.max()} "
          f"(Dukascopy to {duk_end}, then {len(ext_bars):,} appended bars, spread {med_spread} pts)")

    print("Building calendar ...")
    c1 = calendar_2010s(ff1 / "datasets")
    c2 = calendar_2020s(ff2 / "high_impact_events_calendar.csv")
    c2 = c2[c2["time"] >= pd.Timestamp("2020-01-01")]
    ff_end = c2["time"].max()
    c3 = scheduled_us_events(ff_end, bars.index.max() + pd.Timedelta(days=60))
    cal = pd.concat([c1[c1["time"] < pd.Timestamp("2020-01-01")], c2, c3], ignore_index=True)
    cal = cal.sort_values("time").drop_duplicates(["time", "currency", "event"]).reset_index(drop=True)
    cal.to_csv(data / "calendar.csv", index=False)
    print(f"  wrote {data / 'calendar.csv'}: {len(cal):,} high-impact events {cal['time'].min()} -> "
          f"{cal['time'].max()} (ForexFactory to {ff_end}, rule-based after)")

    (data / "SOURCES.txt").write_text("\n".join(
        f"{k}: {SOURCES[k]} @ {h}" for k, h in
        zip(["dukascopy", "getdata15m", "navtemmt", "ff_2010s", "ff_2020s"], [h1, h2, h3, h4, h5])) + "\n")


if __name__ == "__main__":
    sys.exit(main())
