"""Bar-by-bar, multi-symbol backtester that runs the real rule engine.

Pessimistic by design: entries at next bar open plus spread, SL assumed
first when SL and TP fall in the same bar, drawdown checked against the
worst price inside each bar (that is how "hidden" floating breaches happen).
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .data import TF_MINUTES
from .rules import NewsWindows, RuleEngine, SessionGuard, firm_days


def _conv(spec: dict, price: float) -> float:
    """Money per 1.0 price move per 1 unit -> account currency (USD)."""
    return 1.0 / price if spec.get("pnl_conversion") == "base_usd" else 1.0


def run_backtest(ds: pd.DataFrame, pred: pd.DataFrame, settings: dict, rules: dict,
                 calendar: pd.DataFrame | None, threshold: float,
                 start=None, end=None, stop_on_pass: bool = False,
                 initial: float | None = None) -> dict:
    tf_min = TF_MINUTES[settings["timeframe"]]
    bar = pd.Timedelta(minutes=tf_min)
    specs = settings["symbols"]
    tcfg, lab = settings["trading"], settings["labels"]
    initial = initial or settings["initial_balance"]

    d = ds[["time", "symbol", "atr", "px_open", "px_high", "px_low", "px_close", "px_spread"]].copy()
    d = pd.concat([d, pred[["direction", "confidence"]]], axis=1)
    d = d.sort_values(["symbol", "time"])
    d["next_open"] = d.groupby("symbol")["px_open"].shift(-1)
    d["next_time"] = d.groupby("symbol")["time"].shift(-1)
    if start is not None:
        d = d[d["time"] >= pd.Timestamp(start)]
    if end is not None:
        d = d[d["time"] < pd.Timestamp(end)]
    d = d.sort_values(["time", "symbol"]).reset_index(drop=True)
    if d.empty:
        return {"status": {"error": "no data in range"}, "trades": pd.DataFrame(), "equity": pd.DataFrame()}

    times = d["time"].drop_duplicates().reset_index(drop=True)
    days = dict(zip(times, firm_days(times, rules, settings.get("server_utc_offset_hours", 0))))

    rule = RuleEngine(rules, initial)
    guard = SessionGuard(settings["session_guard"], threshold)
    news = NewsWindows(calendar, rules, specs)
    ncfg = rules.get("news") or {}
    weekend_ok = rules.get("weekend_holding", False)
    monthly_cap = (rules.get("monthly_target_pct") or [None, None])[1]

    positions, trades, curve = [], [], []
    last_px: dict[str, float] = {}
    cur_day, month_key, month_start_bal = None, None, initial
    outcome = None

    def close(pos, price, t, reason, in_news=False):
        spec = specs[pos["sym"]]
        gross = pos["dir"] * (price - pos["entry"]) * pos["lots"] * spec["contract_size"] * _conv(spec, price)
        pnl = gross - spec.get("commission_per_lot", 0.0) * pos["lots"]
        pnl = rule.on_close(pnl, cur_day, in_news)
        guard.on_close(pnl, rule.day_pnl.get(cur_day, 0.0), initial)
        trades.append({**{k: pos[k] for k in ("sym", "dir", "entry", "sl", "tp", "lots", "open_time", "conf", "risk")},
                       "exit": price, "close_time": t, "reason": reason, "pnl": pnl,
                       "r": pnl / pos["risk"] if pos["risk"] else 0.0, "news_violation": in_news})

    for T, rows in d.groupby("time", sort=True):
        day = days[T]
        if day != cur_day:
            cur_day = day
            rule.new_day(day, rule.balance, rule.equity)
            guard.new_day(day)
        mk = (T.year, T.month)
        if mk != month_key:
            month_key, month_start_bal = mk, rule.balance

        by_sym = {r.symbol: r for r in rows.itertuples(index=False)}
        for s, r in by_sym.items():
            last_px[s] = r.px_close

        # ---------- manage open positions on this bar
        still = []
        for pos in positions:
            r = by_sym.get(pos["sym"])
            if r is None or T < pos["active_from"]:
                still.append(pos); continue
            o, h, l, c = r.px_open, r.px_high, r.px_low, r.px_close
            exit_px, reason = None, None
            if pos["dir"] == 1:
                if l <= pos["sl"]:
                    exit_px, reason = min(o, pos["sl"]), "sl"
                elif h >= pos["tp"]:
                    exit_px, reason = max(o, pos["tp"]), "tp"
            else:
                if h >= pos["sl"]:
                    exit_px, reason = max(o, pos["sl"]), "sl"
                elif l <= pos["tp"]:
                    exit_px, reason = min(o, pos["tp"]), "tp"
            if exit_px is None and (T - pos["open_time"]) >= bar * lab["horizon_bars"]:
                exit_px, reason = c, "time"
            gap = (r.next_time - T) if pd.notna(r.next_time) else pd.Timedelta(0)
            if exit_px is None and not weekend_ok and gap > pd.Timedelta(hours=12):
                exit_px, reason = c, "weekend"
            if exit_px is None and ncfg.get("flatten_before_window") and news.enabled and pd.notna(r.next_time):
                if news.in_window(pos["sym"], r.next_time, r.next_time + bar):
                    exit_px, reason = c, "pre-news"
            if exit_px is not None:
                in_news = bool(ncfg.get("no_close_in_window")) and news.in_window(pos["sym"], T, T + bar)
                close(pos, exit_px, T, reason, in_news)
            else:
                still.append(pos)
        positions = still

        # ---------- equity, worst-case equity inside the bar, rule check
        floating = worst = 0.0
        for pos in positions:
            spec = specs[pos["sym"]]
            r = by_sym.get(pos["sym"])
            mark = last_px.get(pos["sym"], pos["entry"])
            adverse = (r.px_low if pos["dir"] == 1 else r.px_high) if r is not None and T >= pos["active_from"] else mark
            k = pos["lots"] * spec["contract_size"]
            floating += pos["dir"] * (mark - pos["entry"]) * k * _conv(spec, mark)
            worst += pos["dir"] * (adverse - pos["entry"]) * k * _conv(spec, adverse)
        equity = rule.balance + floating
        st = rule.check(rule.balance + worst, equity, worst, T)
        if st == "breach":
            outcome = "breach"
            for pos in positions:
                close(pos, last_px[pos["sym"]], T, "breach")
            positions = []
            curve.append((T, rule.balance, rule.balance))
            break
        if st == "flatten":
            for pos in positions:
                close(pos, last_px[pos["sym"]], T, "rule-flatten")
            positions = []
            guard.halt()
        curve.append((T, rule.balance, rule.balance + floating if positions else rule.balance))

        if stop_on_pass and rule.passed():
            outcome = "passed"
            for pos in positions:
                close(pos, last_px[pos["sym"]], T, "passed")
            positions = []
            break

        # ---------- new entries
        if guard.halted_today:
            continue
        target_hit = rule.target_reached()
        cap = rule.daily_profit_cap()
        if cap is not None and rule.day_pnl.get(cur_day, 0.0) >= cap:
            continue
        if monthly_cap and rule.balance - month_start_bal >= initial * monthly_cap / 100:
            continue
        cands = sorted((r for r in rows.itertuples(index=False)
                        if np.isfinite(r.confidence) and r.confidence >= guard.threshold()),
                       key=lambda r: -r.confidence)
        for r in cands:
            if guard.trades_today >= tcfg["max_trades_per_day"] or len(positions) >= tcfg["max_open_trades"]:
                break
            if target_hit and guard.trades_today >= 1:
                break  # target done: only a single minimum trade per day to collect trading days
            if tcfg.get("one_position_per_symbol", True) and any(p["sym"] == r.symbol for p in positions):
                continue
            if pd.isna(r.next_open) or not np.isfinite(r.atr) or r.atr <= 0:
                continue
            if (r.next_time - T) > pd.Timedelta(hours=12) and not weekend_ok:
                continue  # don't open into a weekend gap
            if ncfg.get("no_open_in_window") and news.in_window(r.symbol, r.next_time, r.next_time + bar):
                continue
            spec = specs[r.symbol]
            spread_px = r.px_spread * spec["point"]
            direction = int(r.direction)
            entry = r.next_open + spread_px if direction == 1 else r.next_open - spread_px
            sl_d, tp_d = lab["sl_atr"] * r.atr, lab["tp_atr"] * r.atr
            per_lot = sl_d * spec["contract_size"] * _conv(spec, entry) + spec.get("commission_per_lot", 0.0)
            open_risk = sum(p["risk"] for p in positions)
            want = equity * tcfg["risk_per_trade_pct"] / 100
            budget = min(want, rule.risk_budget(open_risk))
            lots = math.floor(budget / per_lot / spec["lot_step"]) * spec["lot_step"]
            if target_hit:
                lots = spec["min_lot"]
            lots = min(lots, spec["max_lot"])
            if lots < spec["min_lot"] or lots * per_lot > rule.risk_budget(open_risk) + 1e-9:
                continue
            positions.append({"sym": r.symbol, "dir": direction, "entry": entry,
                              "sl": entry - direction * sl_d, "tp": entry + direction * tp_d,
                              "lots": round(lots, 2), "open_time": r.next_time, "active_from": r.next_time,
                              "conf": float(r.confidence), "risk": lots * per_lot})
            guard.trades_today += 1

    # close leftovers at last price
    T_end = d["time"].iloc[-1]
    for pos in positions:
        close(pos, last_px[pos["sym"]], T_end, "end")
    if outcome is None:
        outcome = "passed" if rule.passed() else "running"

    tr = pd.DataFrame(trades)
    eq = pd.DataFrame(curve, columns=["time", "balance", "equity"])
    status = rule.status()
    status["outcome"] = outcome
    status["start"], status["end"] = str(d["time"].iloc[0]), str(eq["time"].iloc[-1]) if len(eq) else None
    return {"status": status, "trades": tr, "equity": eq, "rule": rule}


def summarize(tr: pd.DataFrame, eq: pd.DataFrame, initial: float) -> dict:
    if tr.empty:
        return {"trades": 0}
    peak = eq["equity"].cummax()
    dd = ((eq["equity"] - peak) / peak).min()
    gp, gl = tr.loc[tr.pnl > 0, "pnl"].sum(), -tr.loc[tr.pnl < 0, "pnl"].sum()
    eqm = eq.set_index("time")["balance"].resample("ME").last().dropna()
    monthly = (eqm.pct_change().fillna(eqm.iloc[0] / initial - 1) * 100).round(2)
    return {"trades": int(len(tr)), "win_rate": round(float((tr.pnl > 0).mean()), 3),
            "avg_r": round(float(tr["r"].mean()), 3), "profit_factor": round(float(gp / gl), 2) if gl else None,
            "net_pct": round(100 * tr["pnl"].sum() / initial, 2), "max_dd_pct": round(100 * float(dd), 2),
            "monthly_pct": {str(k.date())[:7]: float(v) for k, v in monthly.items()},
            "by_symbol": tr.groupby("sym")["pnl"].sum().round(2).to_dict(),
            "exit_reasons": tr["reason"].value_counts().to_dict()}
