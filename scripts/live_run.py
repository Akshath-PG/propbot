"""Start the live / demo bot (Windows, MT5 terminal open and logged in).

    python scripts/live_run.py            # uses settings.yaml (dry_run: true by default)
    python scripts/live_run.py --real     # required for a non-demo account with dry_run: false
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from propbot.config import load_settings  # noqa: E402
from propbot.live import LiveTrader  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--settings", default="config/settings.yaml")
    ap.add_argument("--real", action="store_true", help="confirm trading a non-demo account")
    args = ap.parse_args()
    LiveTrader(load_settings(args.settings), allow_real=args.real).run()


if __name__ == "__main__":
    main()
