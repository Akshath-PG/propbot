"""Build the pooled multi-symbol dataset (features + labels)."""
from __future__ import annotations

import pandas as pd

from .data import load_bars, load_calendar
from .features import build_features
from .labels import make_labels


def build_dataset(settings: dict, start: str | None = None, end: str | None = None,
                  symbols: list[str] | None = None, verbose: bool = True):
    """Returns (dataset, bars_by_symbol). dataset rows are sorted by time."""
    cal = load_calendar(settings)
    frames, bars = [], {}
    for sym in symbols or list(settings["symbols"]):
        try:
            df = load_bars(settings, sym, start=start, end=end)
        except FileNotFoundError as e:
            print(f"[skip] {e}")
            continue
        if len(df) < 1000:
            print(f"[skip] {sym}: only {len(df)} bars")
            continue
        feats = build_features(df, settings, sym, cal)
        labs = make_labels(df, feats["atr"], settings, sym)
        px = df[["open", "high", "low", "close", "spread"]].reset_index(drop=True)
        frames.append(pd.concat([feats, labs, px.add_prefix("px_")], axis=1))
        bars[sym] = df
        if verbose:
            vc = labs["y"].value_counts(normalize=True).round(3).to_dict()
            print(f"  {sym}: {len(df):,} bars  {df['time'].iloc[0]} -> {df['time'].iloc[-1]}  label mix {vc}")
    if not frames:
        raise RuntimeError("No symbol data loaded.")
    ds = pd.concat(frames, ignore_index=True).sort_values(["time", "symbol"]).reset_index(drop=True)
    return ds, bars
