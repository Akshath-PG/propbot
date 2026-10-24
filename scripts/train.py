"""Train the ensemble on history up to train_end, then test on the unseen period.

    python scripts/train.py                 # train + evaluate on 2026
    python scripts/train.py --no-eval       # train only
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from propbot.config import load_settings, path_in_root  # noqa: E402
from propbot.data import load_calendar  # noqa: E402
from propbot.dataset import build_dataset  # noqa: E402
from propbot.evaluate import evaluate_model  # noqa: E402
from propbot.model import train_ensemble  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--settings", default="config/settings.yaml")
    ap.add_argument("--no-eval", action="store_true")
    args = ap.parse_args()

    s = load_settings(args.settings)
    t0 = time.time()
    print("Building dataset ...")
    ds, _ = build_dataset(s, start=s["history_start"], end=s.get("test_end"))

    print("Training (walk-forward) ...")
    model = train_ensemble(ds, s)
    mdir = path_in_root(s, s["model_dir"]); mdir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M")
    model.save(mdir / f"model_{stamp}.joblib")
    model.save(mdir / "champion.joblib")
    rdir = path_in_root(s, s["report_dir"]); rdir.mkdir(parents=True, exist_ok=True)
    rep = {k: v for k, v in model.report.items() if k != "threshold_table"}
    (rdir / "training_report.json").write_text(json.dumps(model.report, indent=2, default=str))
    print(json.dumps(rep, indent=2, default=str))
    if not model.tradeable:
        print("\n[!] No confidence threshold met the expectancy / trade-count bar on validation.\n"
              "    The model is marked NOT TRADEABLE; the live runner will refuse to place orders.")

    if not args.no_eval:
        print(f"\nTesting on unseen data from {s['test_start']} ...")
        cal = load_calendar(s)
        res = evaluate_model(model, ds, s, cal, s["test_start"], s.get("test_end"), rdir, tag="holdout_2026")
        print(json.dumps({k: res[k] for k in ("period", "signal_quality", "pass_rate", "breach_rate")}, indent=2))
        print(json.dumps(res["funded_run"]["summary"], indent=2, default=str))
        print(f"\nReport: {rdir / 'holdout_2026_report.md'}")
    print(f"Done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
