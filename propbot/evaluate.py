"""Out-of-sample evaluation and report writing."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .backtest import run_backtest, summarize
from .config import load_firm
from .model import EnsembleModel, evaluate_threshold


def evaluate_model(model: EnsembleModel, ds: pd.DataFrame, settings: dict, calendar,
                   start, end=None, out_dir: Path | None = None, tag: str = "holdout") -> dict:
    test = ds[ds["time"] >= pd.Timestamp(start)]
    if end is not None:
        test = test[test["time"] < pd.Timestamp(end)]
    test = test.reset_index(drop=True)
    if test.empty:
        raise RuntimeError(f"No data after {start} to test on.")
    pred = model.predict(test)
    initial = settings["initial_balance"]

    # 1) raw signal quality on unseen data vs. what validation promised
    valid = test["y"].notna()
    n, exp, tot, wr = evaluate_threshold(pred[valid], test.loc[valid, "r_long"], test.loc[valid, "r_short"],
                                         model.threshold)
    signal = {"threshold": model.threshold, "signals": n, "expectancy_r": round(exp, 3),
              "win_rate": round(wr, 3), "validation_expectancy_r": model.report["selection_fold_at_threshold"]["expectancy_r"]}

    # 2) continuous trading under FUNDED rules
    funded_rules = load_firm(settings, "funded")
    run = run_backtest(test, pred, settings, funded_rules, calendar, model.threshold)
    funded = {"status": run["status"], "summary": summarize(run["trades"], run["equity"], initial)}

    # 3) a fresh EVALUATION challenge started at the beginning of every month
    eval_rules = load_firm(settings, "evaluation")
    max_days = eval_rules.get("max_calendar_days")
    challenges = []
    t_last = test["time"].max()
    for ms in pd.date_range(test["time"].min().normalize(), t_last, freq="MS"):
        if ms < test["time"].min() - pd.Timedelta(days=1):
            continue
        me = ms + pd.Timedelta(days=max_days) if max_days else None
        r = run_backtest(test, pred, settings, eval_rules, calendar, model.threshold,
                         start=ms, end=me, stop_on_pass=True)
        s = r["status"]
        days = (pd.Timestamp(s["end"]) - ms).days if s.get("end") else None
        challenges.append({"start": str(ms.date()), "outcome": s.get("outcome"), "calendar_days": days,
                           "profit_pct": s.get("profit_pct"), "trading_days": s.get("days_traded"),
                           "breach": s.get("breached")})
    ch = pd.DataFrame(challenges)
    result = {"tag": tag, "period": [str(test["time"].min()), str(t_last)], "signal_quality": signal,
              "funded_run": funded, "challenges": challenges,
              "pass_rate": round(float((ch["outcome"] == "passed").mean()), 3) if len(ch) else None,
              "breach_rate": round(float((ch["outcome"] == "breach").mean()), 3) if len(ch) else None}

    if out_dir:
        out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
        run["trades"].to_csv(out_dir / f"{tag}_trades.csv", index=False)
        run["equity"].to_csv(out_dir / f"{tag}_equity.csv", index=False)
        (out_dir / f"{tag}_result.json").write_text(json.dumps(result, indent=2, default=str))
        _plot(run["equity"], out_dir / f"{tag}_equity.png", initial, funded_rules)
        (out_dir / f"{tag}_report.md").write_text(render_report(model, result))
    return result


def _plot(eq: pd.DataFrame, path, initial, rules):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    if eq.empty:
        return
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(eq["time"], eq["equity"], lw=1, label="equity")
    ax.plot(eq["time"], eq["balance"], lw=1, alpha=0.7, label="balance")
    ax.axhline(initial, color="grey", lw=0.8, ls=":")
    ax.axhline(initial * (1 - rules["max_drawdown_pct"] / 100), color="red", lw=0.8, ls="--", label="max drawdown floor")
    ax.set_title("Out-of-sample equity (funded rules)"); ax.legend(loc="upper left")
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def render_report(model: EnsembleModel, res: dict) -> str:
    rp = model.report
    L = [f"# PropBot report - {res['tag']}", "",
         f"Training period: {rp['train_period'][0]} -> {rp['train_period'][1]}  ({rp['rows']:,} rows)",
         f"Test period (never seen in training): {res['period'][0]} -> {res['period'][1]}", "",
         "## 1. Did the model learn anything? (walk-forward validation inside training period)", "",
         "| Expert | train log-loss | validation log-loss | baseline | skill % | verdict |",
         "|---|---|---|---|---|---|"]
    for g, v in rp["experts"].items():
        L.append(f"| {g} | {v['train_ll']} | {v['val_ll']} | {v['baseline_ll']} | {v['skill_pct']} | {v['verdict']} |")
    L += ["", "Skill % = how much better than guessing the class mix. Near 0 means no edge; a large",
          "train/validation gap means memorising (overfit).", "",
          "## 2. Learned weights of each expert (combiner)", ""]
    for g, w in sorted(rp["expert_weights_pct"].items(), key=lambda x: -x[1]):
        L.append(f"- {g}: {w}%")
    sel = rp["selection_fold_at_threshold"]
    L += ["", "## 3. Confidence threshold", "",
          f"Chosen threshold: **{rp['threshold']}** (probability of reaching TP before SL). "
          f"Tradeable: **{rp['tradeable']}**",
          f"On the last validation fold: {sel['trades']} signals, win rate {sel['win_rate']}, "
          f"expectancy {sel['expectancy_r']} R per trade.", ""]
    sq = res["signal_quality"]
    L += ["## 4. Unseen-data check (2026)", "",
          f"Signals at threshold: {sq['signals']}, win rate {sq['win_rate']}, expectancy {sq['expectancy_r']} R "
          f"(validation promised {sq['validation_expectancy_r']} R).", ""]
    s = res["funded_run"]["summary"]; st = res["funded_run"]["status"]
    L += ["### Continuous trading with FUNDED rules", ""]
    if s.get("trades"):
        L += [f"- Trades: {s['trades']}, win rate {s['win_rate']}, avg {s['avg_r']} R, profit factor {s['profit_factor']}",
              f"- Net: {s['net_pct']}%, max drawdown {s['max_dd_pct']}%",
              f"- Outcome: {st['outcome']}" + (f" ({st['breached']})" if st.get("breached") else ""),
              f"- News-window violations: {st['news_violations']}", "", "| Month | Return % |", "|---|---|"]
        L += [f"| {m} | {v} |" for m, v in s["monthly_pct"].items()]
        L += ["", f"Exit reasons: {s['exit_reasons']}", f"P&L by symbol: {s['by_symbol']}"]
    else:
        L.append("- No trades taken.")
    L += ["", "### Evaluation challenge started on the 1st of every month", "",
          f"Pass rate: **{res['pass_rate']}**, breach rate: **{res['breach_rate']}**", "",
          "| Start | Outcome | Days | Profit % | Trading days |", "|---|---|---|---|---|"]
    for c in res["challenges"]:
        L.append(f"| {c['start']} | {c['outcome']} | {c['calendar_days']} | {c['profit_pct']} | {c['trading_days']} |")
    L += ["", "_Backtests use pessimistic fills but still cannot model every live condition "
          "(slippage spikes, requotes, broker-specific rules). Forward-test on demo before any paid challenge._"]
    return "\n".join(L)
