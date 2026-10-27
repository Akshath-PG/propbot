"""Markdown report + chart for scripts/hero_run.py."""
from __future__ import annotations

from pathlib import Path

import pandas as pd


def _pct(x):
    return "-" if x is None or pd.isna(x) else f"{100 * x:.1f}%"


def _row(name: str, r: dict) -> str:
    md = r.get("median_days_to_payout")
    md = "-" if md is None or pd.isna(md) else f"{md:.0f}"
    return (f"| {name} | {r.get('accounts', 0)} | {_pct(r.get('payout_rate'))} | {_pct(r.get('breach_rate'))} | "
            f"{_pct(r.get('still_open'))} | {md} | ${r.get('ev_per_account_usd', 0):+.2f} |")


HDR = ["| Run | Accounts | First payout | Failed | Still open | Median days to payout | EV per $17 account |",
       "|---|---|---|---|---|---|---|"]


def write_report(rdir: Path, meta: dict, results: dict, roll: pd.DataFrame, cont: dict, search: pd.DataFrame,
                 eras: pd.DataFrame, rnd: pd.DataFrame, limits) -> None:
    rdir = Path(rdir)
    T = results["test"]["summary"]
    B = results["test_random_baseline"]
    acc, pay, tr = cont["accounts"], cont["payouts"], cont["trades"]
    received = float(pay["trader_gets"].sum()) if len(pay) else 0.0
    fees = len(acc) * limits.price_usd
    fr = meta["final_model_report"]
    pol = meta["policy"]
    L = [
        "# Multi-agent model - GFT Instant HERO $5K (XAUUSD)",
        "",
        f"Test period (last 2 years, touched once): **{meta['test_period'][0]} -> {meta['test_period'][1][:10]}**. "
        f"Every choice (risk policy, direction source) was made on {meta['search_period'][0]} -> "
        f"{meta['search_period'][1]} with walk-forward models.",
        "",
        "## Headline",
        "",
        f"- **A fresh $17 account started every week of the last 2 years ({T['accounts']} accounts): "
        f"{_pct(T['payout_rate'])} reached the first payout**, {_pct(T['breach_rate'])} failed, "
        f"{_pct(T['still_open'])} were still running when the data ends.",
        f"- Median time to the first payout: {T.get('median_days_to_payout')} days. "
        f"Expected value per account (payout to you minus the fee): **${T['ev_per_account_usd']:+.2f}**.",
        f"- One trader running the bot continuously, re-buying after every failed account: {len(acc)} accounts "
        f"(${fees:,.0f} in fees), **{len(pay)} payouts, ${received:,.0f} received, net ${received - fees:+,.0f}**.",
        f"- Same risk plan with RANDOM trade directions on the same 2 years (30 runs): payout rate "
        f"{_pct(B['payout_rate_mean'])} on average (10-90% range {_pct(B['payout_rate_p10'])} - "
        f"{_pct(B['payout_rate_p90'])}), EV ${B['ev_mean']:+.2f}.",
        "",
        f"**How to read this:** the risk agent's plan is what lets a HERO account pay out often enough to beat the "
        f"$17 fee - random trade directions under the same plan already average {_pct(B['payout_rate_mean'])}. "
        f"The direction source chosen on 2018-2024 ({meta['direction_source']}) was well ahead of random there "
        f"(z = {meta['direction_edge_z']:+.1f}), but on the test years it scored {_pct(T['payout_rate'])} vs "
        f"{_pct(B['payout_rate_mean'])} for random: no reliable directional edge out of sample. "
        "Results swing a lot with luck: treat every rate here as +/- 10 points.",
        "",
        "## 1. Research agent - the account rules modelled",
        "",
    ]
    L += [f"- {x}" for x in limits.describe()]
    L += ["- Passing = first payout: at least 14 days, 6 profitable days (>= +0.5%), no day above 15% of the profit,",
          "  no rule broken. HERO has no profit target, so the payout is the only 'pass'.",
          "- 'Instant Haiku' is not a GFT product name; the $17 / $5K price matches **Instant HERO**. "
          "Sources and open questions: `config/firms/gft_instant_hero.yaml`.",
          "",
          "## 2. How the agents work together",
          "",
          "```",
          "research agent ---- hard limits (3% daily, 5% trailing on equity, 1% floating, payout rules)",
          "strategy agents --- trend / ICT / order-flow / reversal / news / geo experts -> combiner (walk-forward)",
          "trend agents ------ H4 trend, daily trend; majority vote with the combiner",
          "coordinator ------- picks the direction source, session hours, room before the 5 PM NY reset",
          "news agent -------- no entries 30 min before / 15 min after high-impact USD events",
          "geopolitical agent  market-implied stress regime: half size in a crisis, no trades against the safe-haven flow",
          "risk agent -------- stop = risk_pct of balance (inside the 1% floating limit), win = tp_pct (a profitable",
          "                    day), done for the day after one win, day stop, breakeven stop, then wait for payout",
          "execution --------- next-bar entry + spread; worst price of every bar checked against every limit",
          "```",
          "",
          "Signals through each stage in the test period: " + ", ".join(f"{k} {v:,}" for k, v in results["test"]["votes"].items()),
          "",
          "## 3. Chosen setup (from 2018-2024 only)",
          "",
          f"- Risk policy: risk {pol['risk_pct']}% at the stop, win {pol['tp_pct']}%, stop >= {pol['min_sl_atr']} x ATR(M15), "
          f"max {pol['max_trades_per_day']} trades/day, day stop {pol['day_stop_pct']}%, "
          f"breakeven at {pol['breakeven_at'] or 'off'}, cycle target {pol['cycle_target_pct']}%",
          f"- Direction source: **{meta['direction_source']}**",
          ""]
    if meta.get("policy_warnings"):
        L += [f"- WARNING: {w}" for w in meta["policy_warnings"]] + [""]
    cols = [c for c in search.columns if c not in ("median_days_to_payout",)]
    L += [f"Top risk policies (Monte Carlo, random directions, {meta['search_period'][0]} -> {meta['search_period'][1]}):",
          "", "| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in search.head(8).iterrows():
        L.append("| " + " | ".join(f"{r[c]:g}" if isinstance(r[c], (int, float)) else str(r[c]) for c in cols) + " |")
    piv = eras.pivot(index="source", columns="era", values="payout_rate")
    L += ["", "Direction sources, first-payout rate by era (walk-forward, before the test):", "",
          "| Source | " + " | ".join(piv.columns) + " |", "|---|" + "---|" * len(piv.columns)]
    for i, r in piv.iterrows():
        L.append(f"| {i} | " + " | ".join(_pct(x) for x in r) + " |")
    L.append("| random direction (mean of 20) | " + " | ".join(
        f"{_pct(rnd.loc[e, ('payout_rate', 'mean')])} +/- {100 * rnd.loc[e, ('payout_rate', 'std')]:.0f}"
        for e in piv.columns) + " |")

    L += ["", "## 4. Test - last 2 years", ""] + HDR + [_row(f"**Full model ({meta['direction_source']})**", T)]
    for k, v in results["test_other_sources"].items():
        L.append(_row(k.replace("_", " "), v))
    L.append(f"| random direction, same risk agent (mean of 30) | {T['accounts']} | {_pct(B['payout_rate_mean'])} | - | - | - | ${B['ev_mean']:+.2f} |")
    L += ["", "Rows other than the first are shown for information only; none of them was used to choose anything."]
    if len(roll):
        fail = roll.loc[roll["outcome"] == "breached", "breach"].value_counts()
        if len(fail):
            L += ["", "Why accounts failed:", ""] + [f"- {k}: {v}" for k, v in fail.items()]
            L += ["", "('abandoned' = equity sat so close to the trailing floor that even a 0.01 lot stop would "
                  "have broken it, for 10 trading days: the account is finished; the buffer stopped it before a "
                  "rule breach.)"]
        r2 = roll.copy()
        r2["quarter"] = pd.PeriodIndex(pd.to_datetime(r2["start"]), freq="Q").astype(str)
        q = r2.groupby("quarter")["outcome"].value_counts().unstack(fill_value=0)
        L += ["", "Weekly accounts by start quarter:", "", "| Quarter | " + " | ".join(q.columns) + " |",
              "|---|" + "---|" * len(q.columns)]
        L += [f"| {i} | " + " | ".join(str(x) for x in r) + " |" for i, r in q.iterrows()]
    L += ["", "### Continuous trader (one account at a time, re-buy after a failure)", ""]
    if len(acc):
        L += ["| # | Start | End | Result | Payouts | Paid to trader | Trades |", "|---|---|---|---|---|---|---|"]
        for j, a in acc.iterrows():
            L.append(f"| {j + 1} | {str(a['start'])[:10]} | {str(a['end'])[:10]} | {a['breach'] if isinstance(a['breach'], str) else 'running'} | "
                     f"{a['payouts']} | ${a['paid']:,.0f} | {a['trades']} |")
    if len(pay):
        L += ["", "| Payout date | Withdrawn | You get | Cycle days | Profitable days | Biggest day |",
              "|---|---|---|---|---|---|"]
        L += [f"| {str(p['time'])[:10]} | ${p['profit_withdrawn']:,.0f} | ${p['trader_gets']:,.0f} | {p['cycle_days']} | "
              f"{p['profitable_days']} | ${p['max_day']:,.0f} |" for _, p in pay.iterrows()]
    if len(tr):
        L += ["", f"Trades: {len(tr)}, win rate {(tr['pnl'] > 0).mean():.1%}, exits {tr['reason'].value_counts().to_dict()}"]
    L += ["", "## 5. Strategy agents (ML experts), final model trained to 2024-10-31", "",
          "| Expert | validation log-loss | baseline | skill % | verdict |", "|---|---|---|---|---|"]
    for g, v in fr["experts"].items():
        L.append(f"| {g} | {v['val_ll']} | {v['baseline_ll']} | {v['skill_pct']} | {v['verdict']} |")
    L += ["", f"Combiner weights: {fr['expert_weights_pct']}", "",
          "Skill is a few percent better than guessing the class mix, and after spread the raw signal expectancy "
          "is about zero. Research for this model also tested gold's 'Asian session drift' and daily momentum: "
          "both disappear with true mid prices and real spreads (the drift seen in bid prices is the spread "
          "narrowing after the 6 PM reopen).",
          "",
          "## 6. Limits of this test", "",
          "- Data: Dukascopy XAUUSD M1 bid/ask to 2026-09-01, then two public feeds to 2026-10-27; GFT's own "
          "prices and spreads will differ, mostly around news and the daily reopen.",
          "- Fills: next-bar open + spread; stop assumed first when stop and target fall in the same bar; the "
          "equity high-water mark takes each bar's best price before its worst (strict for the trailing limit). "
          "No extra slippage.",
          "- Calendar: ForexFactory high-impact events to 2026-01-30, then rule-based FOMC / NFP dates.",
          "- Rules: the 5% trailing limit is modelled without a lock at the start balance; payout cap 4% ($200) per "
          "14-day cycle; trader gets 90%. GFT can change any of this.",
          "- Weekly accounts overlap in time, so their outcomes are correlated.",
          "- Forward-test on a demo before buying accounts, and confirm GFT allows your own EA on HERO."]
    (rdir / "hero_report.md").write_text("\n".join(L) + "\n")
    _plot(rdir / "hero_test_equity.png", tr, pay, limits)


def _plot(path: Path, tr: pd.DataFrame, pay: pd.DataFrame, limits):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    if tr.empty:
        return
    fig, ax = plt.subplots(figsize=(11, 4.5))
    for a, g in tr.groupby("account"):
        ax.plot(g["close_time"], g["balance"], lw=1, label=f"account {a}" if a <= 8 else None)
    for _, p in pay.iterrows():
        ax.axvline(p["time"], color="green", lw=0.6, alpha=0.6)
    ax.axhline(limits.initial, color="grey", lw=0.8, ls=":")
    ax.axhline(limits.initial - limits.trailing_dd, color="red", lw=0.8, ls="--", label="5% below start")
    ax.set_title("HERO $5K - continuous trader, last 2 years (green lines = payouts)")
    ax.legend(loc="upper left", fontsize=7)
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)
