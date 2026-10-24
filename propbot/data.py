"""Price history and economic calendar loading.

Bars come from CSV files in data/ (written by scripts/download_history.py,
which uses the MetaTrader5 Python package on Windows) or directly from a
running MT5 terminal. Calendar events come from data/calendar.csv, written
by mql5/ExportCalendar.mq5 (the MT5 Python package has no calendar API).
"""
from __future__ import annotations

from pathlib import Path
import pandas as pd

from .config import path_in_root

TF_MINUTES = {"M1": 1, "M5": 5, "M15": 15, "M30": 30, "H1": 60, "H4": 240, "D1": 1440}

BAR_COLS = ["time", "open", "high", "low", "close", "tick_volume", "spread"]


def bars_path(settings: dict, symbol: str, tf: str) -> Path:
    return path_in_root(settings, settings["data_dir"]) / f"{symbol}_{tf}.csv"


def load_bars(settings: dict, symbol: str, tf: str | None = None,
              start: str | None = None, end: str | None = None) -> pd.DataFrame:
    tf = tf or settings["timeframe"]
    path = bars_path(settings, symbol, tf)
    if not path.exists():
        raise FileNotFoundError(
            f"No history file {path}. Run scripts/download_history.py on the "
            f"computer with MT5, or put a CSV with columns {BAR_COLS} there.")
    df = pd.read_csv(path)
    df["time"] = pd.to_datetime(df["time"])
    if "spread" not in df:
        df["spread"] = 0
    if "tick_volume" not in df:
        df["tick_volume"] = 1
    df = df[BAR_COLS].sort_values("time").drop_duplicates("time")
    if start:
        df = df[df["time"] >= pd.Timestamp(start)]
    if end:
        df = df[df["time"] <= pd.Timestamp(end) + pd.Timedelta(days=1)]
    return df.reset_index(drop=True)


def load_calendar(settings: dict, path: str | None = None) -> pd.DataFrame:
    """Columns: time (server time), currency, impact (1-3), event, actual, forecast, previous."""
    p = path_in_root(settings, path or settings["live"]["calendar_file"])
    cols = ["time", "currency", "impact", "event", "actual", "forecast", "previous"]
    if not p.exists():
        print(f"[warn] no calendar file at {p}; news features and news rules disabled")
        return pd.DataFrame(columns=cols)
    cal = pd.read_csv(p)
    cal["time"] = pd.to_datetime(cal["time"])
    for c in ("actual", "forecast", "previous", "impact"):
        cal[c] = pd.to_numeric(cal.get(c), errors="coerce")
    return cal[cols].sort_values("time").reset_index(drop=True)


# ------------------------------------------------------------------ MT5 access
def mt5_connect():
    try:
        import MetaTrader5 as mt5  # Windows only
    except ImportError as e:
        raise RuntimeError("MetaTrader5 package not installed (pip install MetaTrader5; Windows only)") from e
    if not mt5.initialize():
        raise RuntimeError(f"MT5 initialize failed: {mt5.last_error()}. Is the terminal open and logged in?")
    return mt5


def mt5_timeframe(mt5, tf: str):
    return getattr(mt5, f"TIMEFRAME_{tf}")


def mt5_bars(mt5, symbol: str, tf: str, count: int) -> pd.DataFrame:
    rates = mt5.copy_rates_from_pos(symbol, mt5_timeframe(mt5, tf), 0, count)
    if rates is None or len(rates) == 0:
        raise RuntimeError(f"No bars for {symbol} {tf}: {mt5.last_error()}")
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    return df[BAR_COLS]


def mt5_history(mt5, symbol: str, tf: str, start: str, end: str | None) -> pd.DataFrame:
    import datetime as dt
    t0 = pd.Timestamp(start).to_pydatetime()
    t1 = (pd.Timestamp(end) if end else pd.Timestamp.now() + pd.Timedelta(days=1)).to_pydatetime()
    frames = []
    # pull year by year; brokers limit single requests
    cur = t0
    while cur < t1:
        nxt = min(cur + dt.timedelta(days=365), t1)
        r = mt5.copy_rates_range(symbol, mt5_timeframe(mt5, tf), cur, nxt)
        if r is not None and len(r):
            frames.append(pd.DataFrame(r))
        cur = nxt
    if not frames:
        raise RuntimeError(f"No history for {symbol} {tf}: {mt5.last_error()}")
    df = pd.concat(frames)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    return df[BAR_COLS].drop_duplicates("time").sort_values("time").reset_index(drop=True)
