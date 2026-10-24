"""Self-learning cycle: champion / challenger.

Run monthly (Windows Task Scheduler or by hand). It:
  1. retrains a CHALLENGER on all data up to `holdout_weeks` ago,
  2. tests champion and challenger on those most recent weeks
     (unseen by both),
  3. promotes the challenger only if it is tradeable, did not breach,
     and its expectancy is at least as good as the champion's.
The live bot always loads models/champion.joblib, so weights never change
in the middle of a trading day and never change without passing a test.

    python scripts/retrain.py --holdout-weeks 8
"""
import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from propbot.config import load_settings, path_in_root  # noqa: E402
from propbot.data import load_calendar  # noqa: E402
from propbot.dataset import build_dataset  # noqa: E402
from propbot.evaluate import evaluate_model  # noqa: E402
from propbot.model import EnsembleModel, train_ensemble  # noqa: E402


def score(res):
    sq = res["signal_quality"]; st = res["funded_run"]["status"]
    return {"expectancy_r": sq["expectancy_r"], "signals": sq["signals"],
            "net_pct": res["funded_run"]["summary"].get("net_pct", 0.0),
            "breached": bool(st.get("breached"))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--settings", default="config/settings.yaml")
    ap.add_argument("--holdout-weeks", type=int, default=8)
    args = ap.parse_args()

    s = load_settings(args.settings)
    ds, _ = build_dataset(s, start=s["history_start"])
    cut = ds["time"].max() - pd.Timedelta(weeks=args.holdout_weeks)
    s_ch = dict(s, train_end=str(cut.date()))
    print(f"Challenger trains up to {cut.date()}, both models tested after that.")
    challenger = train_ensemble(ds, s_ch)

    mdir = path_in_root(s, s["model_dir"]); rdir = path_in_root(s, s["report_dir"])
    cal = load_calendar(s)
    start = cut + pd.Timedelta(days=1)
    res_ch = evaluate_model(challenger, ds, s, cal, start, None, rdir, tag="retrain_challenger")
    sc_ch = score(res_ch)

    champ_path = mdir / "champion.joblib"
    if champ_path.exists():
        champion = EnsembleModel.load(champ_path)
        sc_cp = score(evaluate_model(champion, ds, s, cal, start, None, rdir, tag="retrain_champion"))
    else:
        sc_cp = None

    promote = (challenger.tradeable and not sc_ch["breached"] and
               (sc_cp is None or sc_cp["breached"] or sc_ch["expectancy_r"] >= sc_cp["expectancy_r"]))
    print(json.dumps({"challenger": sc_ch, "champion": sc_cp, "promote": promote}, indent=2))
    stamp = time.strftime("%Y%m%d_%H%M")
    challenger.save(mdir / f"challenger_{stamp}.joblib")
    if promote:
        if champ_path.exists():
            shutil.copy(champ_path, mdir / f"retired_{stamp}.joblib")
        challenger.save(champ_path)
        print("Challenger promoted to champion.")
    else:
        print("Champion kept.")


if __name__ == "__main__":
    main()
