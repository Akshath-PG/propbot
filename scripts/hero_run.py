"""Multi-agent model for the GFT Instant HERO $5K account: walk-forward training,
policy search on 2018-2024, one untouched test on the last 2 years.

    python scripts/build_hero_data.py          # once: price history + calendar
    python scripts/hero_run.py                 # full run (~25 min on 4 cores)
    python scripts/hero_run.py --quick         # small grid, fewer seeds

Stages
  0. Research agent  : researched HERO rules -> hard limits
  1. Strategy agents : ML experts (trend, ICT, order flow, reversal, news, geo) +
                       combiner, trained walk-forward: each 2-year block is predicted by
                       a model that only saw data before it
  2. Risk policy     : Monte Carlo search on 2018-11 -> 2024-10 with RANDOM trade
                       directions (several seeds). This picks the plan that gives the most
                       payouts per $17 account without relying on any market prediction.
  3. Direction       : the direction source with the best payout rate on 2018-11 -> 2024-10
                       (ML combiner, H4 / daily trend agents, a majority vote of agents,
                       always long), compared against the random-direction baseline
  4. Test            : 2024-11 -> today, touched once. A fresh account every week
                       (first-payout rate), one continuous trader, the random baseline
                       distribution, and ablations of the news / geo agents.
"""
from __future__ import annotations

import argparse
import copy
import itertools
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from propbot.agents.coordinator import Coordinator, Market  # noqa: E402
from propbot.agents.geopolitical import GeopoliticalAgent  # noqa: E402
from propbot.agents.news import NewsAgent  # noqa: E402
from propbot.agents.research import ResearchAgent  # noqa: E402
from propbot.agents.strategy import StrategyAgents  # noqa: E402
from propbot.config import load_settings, path_in_root  # noqa: E402
from propbot.data import load_calendar  # noqa: E402
from propbot.dataset import build_dataset  # noqa: E402
from propbot.hero_report import write_report  # noqa: E402

# walk-forward blocks: (model trained up to, predicts from, predicts to)
WF_BLOCKS = [("2018-10-31", "2018-11-01", "2020-10-31"),
             ("2020-10-31", "2020-11-01", "2022-10-31"),
             ("2022-10-31", "2022-11-01", "2024-10-31"),
             ("2024-10-31", "2024-11-01", None)]
SEARCH = ("2018-11-01", "2024-10-31")
ERAS = {"2018-11..2020-10": ("2018-11-01", "2020-10-31"), "2020-11..2022-10": ("2020-11-01", "2022-10-31"),
        "2022-11..2024-10": ("2022-11-01", "2024-10-31")}

GRID = {
    "risk_pct": [0.5, 0.65, 0.8],            # loss at the stop, % of balance (floating limit is 1%)
    "tp_pct": [0.52, 0.6, 0.7],              # win, % of balance (>= 0.5% = a profitable day)
    "min_sl_atr": [2.0, 3.5],                # stop never tighter than this x ATR(M15)
    "max_trades_per_day": [1, 2],
    "day_stop_pct": [0.5, 1.0],
    "breakeven_at": [0.0, 0.5, 0.8],         # move stop to entry after this fraction of the target
    "cycle_target_pct": [3.5, 4.0],          # stop adding risk once this much profit qualifies
}
QUICK = {"risk_pct": [0.65, 0.8], "tp_pct": [0.52, 0.6], "min_sl_atr": [2.0], "max_trades_per_day": [1, 2],
         "day_stop_pct": [0.5], "breakeven_at": [0.0, 0.5], "cycle_target_pct": [3.5]}
FIXED_RISK = {"exit_mode": "money", "day_target_pct": 0.5}   # done for the day after the first profitable close
ML_QUANTILE = 0.8                                             # ML timing: top 20% of confidences

_G: dict = {}  # shared with pool workers (fork)


# ------------------------------------------------------------------ direction sources
def hourly_pred(ds: pd.DataFrame, direction: np.ndarray) -> pd.DataFrame:
    """One candidate per hour; direction +1 / -1, 0 = no trade."""
    on = (ds["time"].dt.minute == 0).to_numpy() & (direction != 0)
    p = pd.DataFrame({"p_long": np.where(direction > 0, 1.0, 0.0), "p_short": np.where(direction < 0, 1.0, 0.0)})
    p.loc[~on, ["p_long", "p_short"]] = np.nan
    return p


def ml_mask(ml: pd.DataFrame, thresholds: pd.Series) -> np.ndarray:
    conf = ml[["p_long", "p_short"]].max(axis=1).to_numpy()
    return np.isfinite(conf) & (conf >= thresholds.to_numpy())


def direction_sources(ds: pd.DataFrame, ml: pd.DataFrame, thresholds: pd.Series) -> dict:
    ml_dir = np.where(ml["p_long"] >= ml["p_short"], 1, -1)
    ml_on = ml_mask(ml, thresholds)
    h4 = np.sign(ds["tr_H4_ema50_dist"].fillna(0)).to_numpy()
    d20 = np.sign(ds["gp_ret20"].fillna(0)).to_numpy()
    vote = np.sign(ml_dir + h4 + d20)                       # majority of three agents
    src = {
        "ml_combiner": _masked(ml_dir, ml_on),
        "trend_h4_agent": hourly_pred(ds, h4),
        "trend_daily_agent": hourly_pred(ds, d20),
        "agents_majority_vote": _masked(vote, ml_on),
        "always_long": hourly_pred(ds, np.ones(len(ds))),
    }
    return src


def _masked(direction: np.ndarray, on: np.ndarray) -> pd.DataFrame:
    p = pd.DataFrame({"p_long": np.where(direction > 0, 1.0, 0.0), "p_short": np.where(direction < 0, 1.0, 0.0)})
    p.loc[~on | (direction == 0), ["p_long", "p_short"]] = np.nan
    return p


# ------------------------------------------------------------------ helpers
def summarize_roll(roll: pd.DataFrame, price: float) -> dict:
    n = len(roll)
    if n == 0:
        return {"accounts": 0}
    out = roll["outcome"].value_counts()
    paid = roll["payout"]
    res = {"accounts": n, "payout_rate": round(out.get("payout", 0) / n, 3),
           "breach_rate": round(out.get("breached", 0) / n, 3), "still_open": round(out.get("still_open", 0) / n, 3),
           "avg_payout_usd": round(float(paid[paid > 0].mean()), 2) if (paid > 0).any() else 0.0,
           "ev_per_account_usd": round(float(paid.mean() - price), 2)}
    d = roll.loc[roll["outcome"] == "payout", "days"]
    res["median_days_to_payout"] = float(d.median()) if len(d) else None
    return res


def make_coord(s, research, cal, risk_cfg):
    a = s["agents"]
    return Coordinator(s, research.limits, NewsAgent(cal, a["news"]), GeopoliticalAgent(a["geopolitical"]), risk_cfg)


def risk_cfg_for(s, policy: dict) -> dict:
    cfg = {**s["agents"]["risk"], **FIXED_RISK, **policy}
    cfg["breakeven_at"] = cfg.get("breakeven_at") or None
    return cfg


def roll_period(coord, mk, pred, ds, t0, t1, price):
    cand, votes = coord.candidates(mk, pred, ds, 0.5)
    return coord.rolling_first_payout(mk, cand, t0, t1), votes


def eval_policy(args):
    """Mean payout statistics of one risk policy over random-direction seeds."""
    policy, seeds = args
    g = _G["ctx"]
    coord = make_coord(g["s"], g["research"], g["cal"], risk_cfg_for(g["s"], policy))
    rows = []
    for seed in seeds:
        d = np.random.default_rng(1000 + seed).choice([-1, 1], len(g["ds"]))
        roll, _ = roll_period(coord, g["mk"], hourly_pred(g["ds"], d), g["ds"], *SEARCH, g["price"])
        rows.append(summarize_roll(roll, g["price"]))
    r = pd.DataFrame(rows)
    return {**policy, "payout_rate": round(r["payout_rate"].mean(), 4), "payout_rate_sd": round(r["payout_rate"].std(), 4),
            "breach_rate": round(r["breach_rate"].mean(), 4), "ev_per_account_usd": round(r["ev_per_account_usd"].mean(), 2),
            "median_days_to_payout": r["median_days_to_payout"].median()}


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--settings", default="config/settings_hero.yaml")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seeds", type=int, default=6)
    args = ap.parse_args()
    t_start = time.time()
    s = load_settings(args.settings)
    rdir = path_in_root(s, s["report_dir"]); rdir.mkdir(parents=True, exist_ok=True)
    mdir = path_in_root(s, s["model_dir"]); mdir.mkdir(parents=True, exist_ok=True)
    grid = QUICK if args.quick else GRID
    seeds = list(range(2 if args.quick else args.seeds))

    # ------------------------------------------------ 0. research agent + data
    research = ResearchAgent(s)
    price = research.limits.price_usd
    print("Research agent - account limits:")
    for line in research.limits.describe():
        print("  " + line)
    cal = load_calendar(s)
    ds, _ = build_dataset(s, start=s["history_start"], verbose=False)
    ds = ds.reset_index(drop=True)
    mk = Market.build(ds, s["symbols"]["XAUUSD"], s["agents"]["risk"]["flatten_before_reset_min"])

    # ------------------------------------------------ 1. walk-forward strategy agents
    ml = pd.DataFrame(np.nan, index=ds.index, columns=["p_long", "p_short"])
    thr = pd.Series(np.inf, index=ds.index)
    reports = {}
    for train_end, p0, p1 in WF_BLOCKS:
        t = time.time()
        sa = StrategyAgents.train(ds, s, train_end, verbose=False)
        pred = sa.signals(ds)
        blk = (ds["time"] >= pd.Timestamp(p0)) & ((ds["time"] <= pd.Timestamp(p1) + pd.Timedelta(days=1))
                                                  if p1 else True)
        ml.loc[blk, ["p_long", "p_short"]] = pred.loc[blk, ["p_long", "p_short"]].to_numpy()
        prior = (ds["time"] < pd.Timestamp(p0)) & (ds["time"] >= pd.Timestamp(p0) - pd.Timedelta(days=365))
        conf_prior = pred.loc[prior, ["p_long", "p_short"]].max(axis=1)
        thr.loc[blk] = float(np.quantile(conf_prior, ML_QUANTILE))
        reports[train_end] = {k: v for k, v in sa.model.report.items() if k != "threshold_table"}
        print(f"  strategy agents trained to {train_end} -> predict {p0}..{p1 or 'today'} "
              f"({time.time() - t:.0f}s, threshold {thr.loc[blk].iloc[0]:.3f})")
        if p1 is None:
            sa.model.save(mdir / "hero_champion.joblib")

    # ------------------------------------------------ 2. risk policy by Monte Carlo (random directions)
    combos = [dict(zip(grid, v)) for v in itertools.product(*grid.values())]
    _G["ctx"] = {"s": s, "research": research, "cal": cal, "mk": mk, "ds": ds, "price": price}
    t = time.time()
    with Pool(args.workers) as pool:
        search = pool.map(eval_policy, [(c, seeds) for c in combos], chunksize=2)
    S = pd.DataFrame(search).sort_values(["ev_per_account_usd", "payout_rate"], ascending=False)
    S.to_csv(rdir / "hero_policy_search.csv", index=False)
    policy = {k: S.iloc[0][k] for k in grid}
    policy["max_trades_per_day"] = int(policy["max_trades_per_day"])
    print(f"Risk policy search: {len(combos)} policies x {len(seeds)} seeds in {time.time() - t:.0f}s")
    print(S.head(8).to_string(index=False))
    risk_cfg = risk_cfg_for(s, policy)
    coord = make_coord(s, research, cal, risk_cfg)

    # ------------------------------------------------ 3. direction source on 2018-11 -> 2024-10
    sources = direction_sources(ds, ml, thr)
    era_rows = []
    for name, p in sources.items():
        cand, _ = coord.candidates(mk, p, ds, 0.5)
        for era, (a, b) in {"all pre-test": SEARCH, **ERAS}.items():
            r = summarize_roll(coord.rolling_first_payout(mk, cand, a, b), price)
            era_rows.append({"source": name, "era": era, **r})
    rnd_rows = []
    for seed in range(20):
        d = np.random.default_rng(5000 + seed).choice([-1, 1], len(ds))
        cand, _ = coord.candidates(mk, hourly_pred(ds, d), ds, 0.5)
        for era, (a, b) in {"all pre-test": SEARCH, **ERAS}.items():
            rnd_rows.append({"era": era, **summarize_roll(coord.rolling_first_payout(mk, cand, a, b), price)})
    rnd = pd.DataFrame(rnd_rows).groupby("era")[["payout_rate", "ev_per_account_usd"]].agg(["mean", "std"])
    E = pd.DataFrame(era_rows)
    pre = E[E["era"] == "all pre-test"].set_index("source")
    choice = pre["ev_per_account_usd"].idxmax()
    base_mu, base_sd = rnd.loc["all pre-test", ("payout_rate", "mean")], rnd.loc["all pre-test", ("payout_rate", "std")]
    edge_z = (pre.loc[choice, "payout_rate"] - base_mu) / base_sd if base_sd > 0 else 0.0
    print("\nDirection sources on the pre-test years:")
    print(E.pivot(index="source", columns="era", values="payout_rate").to_string())
    print(f"random baseline payout rate {base_mu:.3f} +/- {base_sd:.3f}; chosen: {choice} (z = {edge_z:+.2f})")

    # ------------------------------------------------ 4. test (last 2 years)
    t0 = s["test_start"]
    cand, votes = coord.candidates(mk, sources[choice], ds, 0.5)
    roll = coord.rolling_first_payout(mk, cand, t0, None)
    cont = coord.continuous(mk, cand, t0, None)
    test = summarize_roll(roll, price)
    other = {}
    for name, p in sources.items():
        if name != choice:
            c2, _ = coord.candidates(mk, p, ds, 0.5)
            other[name] = summarize_roll(coord.rolling_first_payout(mk, c2, t0, None), price)
    for name, kw in {"without news agent": {"use_news": False}, "without geo agent": {"use_geo": False}}.items():
        c2, _ = coord.candidates(mk, sources[choice], ds, 0.5, **kw)
        other[name] = summarize_roll(coord.rolling_first_payout(mk, c2, t0, None), price)
    rt = []
    for seed in range(30):
        d = np.random.default_rng(9000 + seed).choice([-1, 1], len(ds))
        c2, _ = coord.candidates(mk, hourly_pred(ds, d), ds, 0.5)
        rt.append(summarize_roll(coord.rolling_first_payout(mk, c2, t0, None), price))
    rt = pd.DataFrame(rt)
    baseline_test = {"payout_rate_mean": round(rt["payout_rate"].mean(), 3), "payout_rate_sd": round(rt["payout_rate"].std(), 3),
                     "payout_rate_p10": round(rt["payout_rate"].quantile(0.1), 3),
                     "payout_rate_p90": round(rt["payout_rate"].quantile(0.9), 3),
                     "ev_mean": round(rt["ev_per_account_usd"].mean(), 2)}

    t_end = ds["time"].max()
    meta = {"policy": {**FIXED_RISK, **policy}, "direction_source": choice, "direction_edge_z": round(float(edge_z), 2),
            "labels": s["labels"], "ml_quantile": ML_QUANTILE, "test_period": [t0, str(t_end)],
            "search_period": list(SEARCH), "walk_forward": WF_BLOCKS, "limits": research.limits.__dict__,
            "policy_warnings": research.check_policy(risk_cfg), "final_model_report": reports[WF_BLOCKS[-1][0]],
            "runtime_s": round(time.time() - t_start)}
    results = {"test": {"summary": test, "votes": dict(votes)}, "test_other_sources": other,
               "test_random_baseline": baseline_test,
               "pretest_by_era": E.to_dict("records"),
               "pretest_random_baseline": {f"{e} {a}_{b}": round(float(v), 3) for (a, b), col in rnd.items()
                                           for e, v in col.items()}}
    roll.to_csv(rdir / "hero_test_rolling_accounts.csv", index=False)
    cont["trades"].to_csv(rdir / "hero_test_trades.csv", index=False)
    cont["payouts"].to_csv(rdir / "hero_test_payouts.csv", index=False)
    cont["accounts"].to_csv(rdir / "hero_test_accounts.csv", index=False)
    E.to_csv(rdir / "hero_pretest_direction_sources.csv", index=False)
    (rdir / "hero_results.json").write_text(json.dumps({**meta, **results}, indent=2, default=str))
    write_report(rdir, meta, results, roll, cont, S, E, rnd, research.limits)
    print(json.dumps({k: results[k] for k in ("test", "test_other_sources", "test_random_baseline")}, indent=2, default=str))
    print(f"\nReport: {rdir / 'hero_report.md'}   ({time.time() - t_start:.0f}s)")


if __name__ == "__main__":
    main()
