"""Coordinator: runs the agents together on every bar and simulates the
instant account (HERO) with payouts.

Per closed M15 bar:
  1. Strategy agents  -> direction + confidence (combiner of the experts)
  2. Coordinator      -> confidence >= threshold, session hours, enough time
                         left before the 5 PM NY reset
  3. News agent       -> veto inside a high-impact USD blackout window
  4. Geopolitical     -> regime; crisis = smaller size, no trades against the
                         safe-haven flow
  5. Risk agent       -> day / cycle plan and lot size inside every limit
  6. Execution        -> enter next bar open (+spread), SL / TP / time exit,
                         flat before the daily reset; every bar is checked
                         against the floating-loss, trailing and daily limits
                         using the worst price inside the bar
Payouts are requested on the first day the account qualifies.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .geopolitical import GeopoliticalAgent, trading_day
from .news import NewsAgent
from .research import AccountLimits
from .risk import AccountState, RiskAgent

BAR = np.timedelta64(15, "m")
BREACHES = ("floating-loss 1%", "trailing max loss", "daily loss")


@dataclass
class Market:
    """Bar arrays for one symbol plus the per-bar trading window."""
    time: np.ndarray
    o: np.ndarray
    h: np.ndarray
    l: np.ndarray
    c: np.ndarray
    sp: np.ndarray            # spread in price units
    atr: np.ndarray
    day: np.ndarray           # trading-day ordinal (17:00 NY roll)
    last_ok: np.ndarray       # last bar index a trade opened at i+1 may stay open to
    hour: np.ndarray          # UTC hour of the entry bar
    spec: dict

    @classmethod
    def build(cls, ds: pd.DataFrame, spec: dict, flatten_min: int, utc_offset_h: float = 0.0) -> "Market":
        d = ds.reset_index(drop=True)
        t = pd.to_datetime(d["time"]) - pd.Timedelta(hours=utc_offset_h)
        ny = pd.DatetimeIndex(t).tz_localize("UTC").tz_convert("America/New_York")
        close_ny = ny + pd.Timedelta(minutes=15)
        day = np.asarray([x.toordinal() for x in trading_day(t)])
        # minutes from the bar close to the next 17:00 NY
        mins = ((17 * 60 - (close_ny.hour * 60 + close_ny.minute)) % 1440).to_numpy()
        ok = mins >= flatten_min
        idx = np.arange(len(d))
        last = pd.Series(np.where(ok, idx, -1)).groupby(day).transform("max").to_numpy()
        last = np.where(last < 0, idx, last)
        return cls(time=t.to_numpy(), o=d["px_open"].to_numpy(float), h=d["px_high"].to_numpy(float),
                   l=d["px_low"].to_numpy(float), c=d["px_close"].to_numpy(float),
                   sp=d["px_spread"].to_numpy(float) * spec["point"], atr=d["atr"].to_numpy(float),
                   day=day, last_ok=last, hour=((t + pd.Timedelta(minutes=15)).dt.hour).to_numpy(), spec=spec)


class Coordinator:
    ABANDON_AFTER_DAYS = 10   # account too close to its floor to place even the minimum lot

    def __init__(self, settings: dict, limits: AccountLimits, news: NewsAgent, geo: GeopoliticalAgent,
                 risk_cfg: dict):
        self.s, self.L, self.news, self.geo = settings, limits, news, geo
        self.risk_cfg = risk_cfg
        self.risk = RiskAgent(risk_cfg, limits)
        lab = settings["labels"]
        self.sl_atr, self.tp_atr, self.H = lab["sl_atr"], lab["tp_atr"], int(lab["horizon_bars"])

    # ------------------------------------------------------------ candidates
    def candidates(self, mk: Market, pred: pd.DataFrame, ds: pd.DataFrame, threshold: float,
                   use_news: bool = True, use_geo: bool = True) -> tuple[pd.DataFrame, Counter]:
        n = len(mk.time)
        p_long, p_short = pred["p_long"].to_numpy(), pred["p_short"].to_numpy()
        direction = np.where(p_long >= p_short, 1, -1)
        conf = np.maximum(p_long, p_short)
        votes = Counter()
        m = np.isfinite(conf) & (conf >= threshold)
        votes["strategy_signals"] = int(m.sum())
        nxt = np.r_[mk.time[1:], mk.time[-1]]
        cont = np.r_[(mk.time[1:] - mk.time[:-1]) == BAR, False]
        lo_h, hi_h = self.risk_cfg.get("session_hours_utc", [0, 24])
        sess = (mk.hour >= lo_h) & (mk.hour < hi_h)
        room = (mk.last_ok - (np.arange(n) + 1)) >= 4
        ok_atr = np.isfinite(mk.atr) & (mk.atr > 0)
        timing = cont & sess & room & ok_atr
        votes["vetoed_timing"] = int((m & ~timing).sum())
        m &= timing
        if use_news:
            blk = self.news.blackout_mask(nxt)
            votes["vetoed_news"] = int((m & blk).sum())
            m &= ~blk
        stress = ds["gp_stress"].to_numpy() if "gp_stress" in ds else np.zeros(n)
        ret5 = ds["gp_ret5"].to_numpy() if "gp_ret5" in ds else np.zeros(n)
        mult = np.ones(n)
        if use_geo:
            cs = self.geo.cfg.get("crisis_score", 2.5)
            crisis = np.nan_to_num(stress) >= cs
            bias = np.where(crisis & (np.abs(np.nan_to_num(ret5)) > 1.0), np.sign(np.nan_to_num(ret5)), 0)
            against = crisis & (bias != 0) & (direction != bias)
            if self.geo.cfg.get("block_counter_trend_in_crisis", True):
                votes["vetoed_geo"] = int((m & against).sum())
                m &= ~against
            mult = np.where(crisis, self.geo.cfg.get("crisis_risk_mult", 0.5), 1.0)
            votes["geo_crisis_size_cut"] = int((m & crisis).sum())
        idx = np.where(m)[0]
        return pd.DataFrame({"i": idx, "dir": direction[idx], "conf": conf[idx], "mult": mult[idx]}), votes

    # ------------------------------------------------------------ execution
    def _trade(self, mk: Market, st: AccountState, i: int, d: int, lots: float, sl_d: float, tp_d: float,
               horizon: int):
        """Open at bar i+1 and walk bar by bar until SL / TP / time / day end, checking
        every firm limit on the worst price inside each bar.
        Returns (pnl, exit_idx, reason, entry, exit_px, equity_hwm)."""
        s, e = i + 1, min(i + horizon, mk.last_ok[i])
        spec, L = mk.spec, self.L
        sp0 = mk.sp[i]
        units = lots * spec["contract_size"]
        comm = spec.get("commission_per_lot", 0.0) * lots
        O, Hh, Lo, C, SP = mk.o[s:e + 1], mk.h[s:e + 1], mk.l[s:e + 1], mk.c[s:e + 1], mk.sp[s:e + 1]
        if d == 1:
            entry = O[0] + sp0
            adv, fav, close_px = Lo, Hh, C
        else:
            entry = O[0]
            adv, fav, close_px = Hh + SP, Lo + SP, C + SP
        sl, tp = entry - d * sl_d, entry + d * tp_d
        sl_arr = np.full(len(O), sl)
        be = self.risk_cfg.get("breakeven_at")
        if be:
            # stop moves to entry from the bar AFTER price has gone be x the target distance our way
            trig = d * (fav - entry) >= be * tp_d
            if trig.any():
                sl_arr[int(np.argmax(trig)) + 1:] = entry
        hit_sl = d * (adv - sl_arr) <= 0
        hit_tp = d * (fav - tp) >= 0
        hit = hit_sl | hit_tp
        if hit.any():
            x = int(np.argmax(hit))
            if hit_sl[x]:
                exit_px = min(O[x], sl_arr[x]) if d == 1 else max(O[x] + SP[x], sl_arr[x])
                reason = "sl" if sl_arr[x] == sl else "breakeven"
            else:
                exit_px, reason = tp, "tp"
        else:
            x, exit_px, reason = len(O) - 1, close_px[-1], "time" if e == i + horizon else "day_end"
        adv = adv[:x + 1].copy(); fav = fav[:x + 1].copy()
        if reason in ("sl", "breakeven"):
            adv[x] = exit_px
        elif reason == "tp":
            fav[x] = exit_px
        worst = st.balance + d * (adv - entry) * units - comm
        best = st.balance + d * (fav - entry) * units - comm
        hwm_path = np.maximum(st.hwm, np.maximum.accumulate(best))
        trail_floor = hwm_path - L.trailing_dd
        if L.trailing_locks:
            trail_floor = np.minimum(trail_floor, L.initial)
        floating = d * (adv - entry) * units
        b_float = -floating >= st.balance * L.floating_limit_pct / 100
        b_trail = worst < trail_floor
        b_day = worst < st.day_ref - L.daily_dd
        breach = b_float | b_trail | b_day
        if breach.any():
            k = int(np.argmax(breach))
            why = "floating-loss 1%" if b_float[k] else "trailing max loss" if b_trail[k] else "daily loss"
            pnl = d * (adv[k] - entry) * units - comm
            return pnl, s + k, why, entry, adv[k], float(hwm_path[k])
        pnl = d * (exit_px - entry) * units - comm
        return pnl, s + x, reason, entry, exit_px, float(hwm_path[-1])

    def run_account(self, mk: Market, cand: pd.DataFrame, start: int, end: int,
                    stop_at_first_payout: bool, log_trades: bool = False) -> AccountState:
        st = AccountState(self.L, pd.Timestamp(mk.time[start]))
        st.new_day(mk.day[start])
        ci, cd, cm = cand["i"].to_numpy(), cand["dir"].to_numpy(), cand["mult"].to_numpy()
        k = int(np.searchsorted(ci, start))
        free = start
        no_room_days, last_room_day = 0, None
        spec = mk.spec
        while k < len(ci) and ci[k] < end:
            i = ci[k]; k += 1
            if i < free:
                continue
            if mk.day[i] != st.day:
                if st.day_pnl != 0.0:
                    st.cycle_day_pnl[st.day] = st.day_pnl
                now = pd.Timestamp(mk.time[i])
                if st.payout_ready(now):
                    st.take_payout(now)
                    if stop_at_first_payout:
                        return st
                st.new_day(mk.day[i])
            why = self.risk.day_open(st)
            if why:
                st.refusals[why] += 1
                continue
            lots, sl_d, tp_d, horizon, why = self.risk.plan(st, mk.atr[i], cm[k - 1], spec, self.sl_atr,
                                                            self.tp_atr, self.H)
            if why == "too_volatile":
                st.refusals[why] += 1
                continue
            if why == "no_room":
                st.refusals[why] += 1
                if st.day != last_room_day:
                    last_room_day, no_room_days = st.day, no_room_days + 1
                if no_room_days >= self.ABANDON_AFTER_DAYS:
                    st.breached, st.breach_time = "abandoned: no room for the minimum lot", pd.Timestamp(mk.time[i])
                    return st
                continue
            no_room_days = 0
            pnl, x, reason, entry, exit_px, hwm = self._trade(mk, st, i, int(cd[k - 1]), lots, sl_d, tp_d, horizon)
            st.balance += pnl
            st.day_pnl += pnl
            st.trades_today += 1
            st.hwm = max(hwm, st.balance)
            if log_trades:
                st.trades.append({"open_time": pd.Timestamp(mk.time[i + 1]), "close_time": pd.Timestamp(mk.time[x]),
                                  "dir": int(cd[k - 1]), "lots": lots, "entry": round(entry, 3),
                                  "exit": round(float(exit_px), 3), "reason": reason, "pnl": round(pnl, 2),
                                  "balance": round(st.balance, 2)})
            if reason in BREACHES:
                st.breached, st.breach_time = reason, pd.Timestamp(mk.time[x])
                return st
            free = x + 1
        if st.day_pnl != 0.0:
            st.cycle_day_pnl[st.day] = st.day_pnl
        now = pd.Timestamp(mk.time[min(end, len(mk.time) - 1)])
        if st.payout_ready(now):
            st.take_payout(now)
        return st

    # ------------------------------------------------------------ evaluation
    def rolling_first_payout(self, mk: Market, cand: pd.DataFrame, t0, t1, every_days: int = 7) -> pd.DataFrame:
        """Buy a fresh account every `every_days` (Monday) between t0 and t1; follow each one until
        its first payout, a breach, or the end of the data."""
        times = pd.DatetimeIndex(mk.time)
        end = int(np.searchsorted(mk.time, np.datetime64(pd.Timestamp(t1)))) if t1 is not None else len(mk.time) - 1
        end = min(end, len(mk.time) - 1)
        rows = []
        for d0 in pd.date_range(pd.Timestamp(t0), times[end], freq=f"{every_days}D"):
            start = int(np.searchsorted(mk.time, np.datetime64(d0)))
            if start >= end:
                break
            st = self.run_account(mk, cand, start, end, stop_at_first_payout=True)
            if st.payouts:
                out, t_end = "payout", st.payouts[0]["time"]
            elif st.breached:
                out, t_end = "breached", st.breach_time
            else:
                out, t_end = "still_open", times[end]
            rows.append({"start": d0.date(), "outcome": out, "days": (t_end - d0).days,
                         "breach": st.breached, "payout": st.payouts[0]["trader_gets"] if st.payouts else 0.0,
                         "balance_end": round(st.balance, 2)})
        return pd.DataFrame(rows)

    def continuous(self, mk: Market, cand: pd.DataFrame, t0, t1) -> dict:
        """One trader over the whole period: trade, take every payout, buy a new
        account the next day after a breach."""
        times = pd.DatetimeIndex(mk.time)
        end = len(mk.time) - 1 if t1 is None else min(int(np.searchsorted(mk.time, np.datetime64(pd.Timestamp(t1)))),
                                                      len(mk.time) - 1)
        start = int(np.searchsorted(mk.time, np.datetime64(pd.Timestamp(t0))))
        accounts, payouts, trades = [], [], []
        while start < end:
            st = self.run_account(mk, cand, start, end, stop_at_first_payout=False, log_trades=True)
            accounts.append({"start": times[start], "end": st.breach_time or times[end], "breach": st.breached,
                             "payouts": len(st.payouts), "paid": round(sum(p["trader_gets"] for p in st.payouts), 2),
                             "trades": len(st.trades)})
            payouts += st.payouts
            trades += [{**t, "account": len(accounts)} for t in st.trades]
            if not st.breached:
                break
            nxt = np.datetime64(pd.Timestamp(st.breach_time).normalize() + pd.Timedelta(days=1))
            start = int(np.searchsorted(mk.time, nxt))
        return {"accounts": pd.DataFrame(accounts), "payouts": pd.DataFrame(payouts), "trades": pd.DataFrame(trades)}
