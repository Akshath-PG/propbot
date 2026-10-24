"""Download price history from your MT5 terminal into data/*.csv (Windows only).

Before running: open MT5, log in, and in Tools > Options > Charts set
"Max bars in chart" to Unlimited, then open each symbol's chart once so the
terminal downloads its history.

    python scripts/download_history.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from propbot.config import load_settings  # noqa: E402
from propbot.data import bars_path, mt5_connect, mt5_history  # noqa: E402


def main():
    s = load_settings()
    mt5 = mt5_connect()
    try:
        for sym in s["symbols"]:
            if not mt5.symbol_select(sym, True):
                print(f"[skip] {sym}: not available at this broker (check the exact symbol name)")
                continue
            df = mt5_history(mt5, sym, s["timeframe"], s["history_start"], None)
            p = bars_path(s, sym, s["timeframe"]); p.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(p, index=False)
            print(f"{sym}: {len(df):,} bars  {df['time'].iloc[0]} -> {df['time'].iloc[-1]}  -> {p}")
    finally:
        mt5.shutdown()


if __name__ == "__main__":
    main()
