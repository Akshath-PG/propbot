"""Prop-firm rule engine, news windows and the daily session guard.

The same classes are used by the backtester and the live runner, so what
you test is exactly what trades. The rule engine has the final word:
it can block a trade, shrink it, or order everything closed.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


# =================================================================== days
def firm_days(times: pd.Series | pd.DatetimeIndex, rules: dict, server_utc_offset: float) -> np.ndarray:
    """Map server-time timestamps to the firm's trading-day date (vectorised)."""
    t = pd.DatetimeIndex(times) - pd.Timedelta(hours=server_utc_offset)
    local = t.tz_localize("UTC").tz_convert(rules.get("daily_reset_timezone") or "UTC")
    hh, mm = (int(x) for x in (rules.get("daily_reset_time") or "00:00").split(":"))
    local = local - pd.Timedelta(hours=hh, minutes=mm)
    return np.asarray(local.date)


# =================================================================== rules
@dataclass
class RuleEngine:
    rules: dict
    initial: float
    balance: float = None
    equity: float = None
    hwm: float = None
    day: object = None
    day_ref: float = None
    day_pnl: dict = field(default_factory=dict)        # closed P&L per trading day
    days_traded: set = field(default_factory=set)
    breached: str | None = None
    violations: list = field(default_factory=list)

    def __post_init__(self):
        self.balance = self.equity = self.hwm = self.initial
        self.buffer = float(self.rules.get("safety_buffer") or 1.0)

    # ---------------------------------------------------------- day handling
    def new_day(self, day, balance: float, equity: float):
        if self.rules["max_drawdown_type"] == "eod_trailing":
            self.hwm = max(self.hwm, balance)
        self.day = day
        basis = self.rules.get("daily_drawdown_basis", "balance")
        self.day_ref = {"balance": balance, "equity": equity}.get(basis, max(balance, equity))
        self.day_pnl.setdefault(day, 0.0)

    # ---------------------------------------------------------- floors
    def daily_floor(self, buffered: bool) -> float:
        base = self.initial if self.rules.get("daily_drawdown_of", "initial") == "initial" else self.day_ref
        allowed = base * self.rules["daily_drawdown_pct"] / 100
        return self.day_ref - allowed * (self.buffer if buffered else 1.0)

    def max_floor(self, buffered: bool) -> float:
        allowed = self.initial * self.rules["max_drawdown_pct"] / 100 * (self.buffer if buffered else 1.0)
        if self.rules["max_drawdown_type"] == "static":
            return self.initial - allowed
        floor = self.hwm - allowed
        if self.rules.get("trailing_locks_at_initial", True):
            floor = min(floor, self.initial)
        return floor

    def floor(self, buffered: bool) -> float:
        return max(self.daily_floor(buffered), self.max_floor(buffered))

    # ---------------------------------------------------------- monitoring
    def check(self, equity_worst: float, equity_now: float, floating: float, t=None) -> str:
        """Returns 'ok', 'flatten' (close everything, stop for today) or 'breach'."""
        if self.breached:
            return "breach"
        self.equity = equity_now
        if self.rules["max_drawdown_type"] == "trailing":
            basis = equity_now if self.rules.get("trailing_basis", "equity") == "equity" else self.balance
            self.hwm = max(self.hwm, basis)
        if equity_worst < self.floor(buffered=False):
            self.breached = f"drawdown breach at {t}: equity {equity_worst:,.2f} < floor {self.floor(False):,.2f}"
            return "breach"
        fl = self.rules.get("max_floating_loss_pct")
        if fl:
            lim = self.initial * fl / 100
            if floating < -lim:
                self.breached = f"floating-loss breach at {t}: {floating:,.2f}"
                return "breach"
            if floating < -lim * self.buffer:
                return "flatten"
        if equity_worst < self.floor(buffered=True):
            return "flatten"
        return "ok"

    def risk_budget(self, open_risk: float) -> float:
        """Max extra money that may be lost at SL so that, even if every open
        trade AND the new one hit their stops, equity stays above the buffered floor."""
        room = self.equity - self.floor(buffered=True) - open_risk
        cap = self.rules.get("max_risk_per_trade_pct")
        if cap:
            room = min(room, self.initial * cap / 100)
        return max(room, 0.0)

    def on_close(self, pnl: float, day, in_news_window: bool = False):
        if in_news_window and self.rules.get("news", {}).get("enabled"):
            effect = self.rules["news"].get("violation_effect", "void_profit")
            self.violations.append({"day": str(day), "pnl": pnl, "effect": effect})
            if effect == "breach":
                self.breached = f"news-window violation on {day}"
            elif effect == "void_profit" and pnl > 0:
                pnl = 0.0  # firm removes the profit
        self.balance += pnl
        self.day_pnl[day] = self.day_pnl.get(day, 0.0) + pnl
        self.days_traded.add(day)
        return pnl

    # ---------------------------------------------------------- objectives
    def profit(self) -> float:
        return self.balance - self.initial

    def target_amount(self) -> float | None:
        pt = self.rules.get("profit_target_pct")
        return self.initial * pt / 100 if pt else None

    def consistency_ok(self) -> bool:
        cp = self.rules.get("consistency_pct")
        if not cp or self.profit() <= 0:
            return True
        return max(self.day_pnl.values()) <= cp / 100 * self.profit()

    def daily_profit_cap(self) -> float | None:
        """Stop for the day once today's profit could break the consistency rule."""
        cp = self.rules.get("consistency_pct")
        if not cp:
            return None
        ref = max(self.target_amount() or 0.0, self.profit())
        return 0.9 * cp / 100 * ref

    def target_reached(self) -> bool:
        tgt = self.target_amount()
        return tgt is not None and self.profit() >= tgt

    def passed(self) -> bool:
        return (self.target_reached() and not self.breached and self.consistency_ok()
                and len(self.days_traded) >= (self.rules.get("min_trading_days") or 0))

    def status(self) -> dict:
        return {"balance": round(self.balance, 2), "profit_pct": round(100 * self.profit() / self.initial, 2),
                "days_traded": len(self.days_traded), "breached": self.breached,
                "target_reached": self.target_reached(), "consistency_ok": self.consistency_ok(),
                "passed": self.passed(), "news_violations": len(self.violations)}


# =================================================================== news
class NewsWindows:
    """Restricted windows around high-impact events, per symbol."""

    def __init__(self, calendar: pd.DataFrame, rules: dict, symbols: dict):
        self.cfg = rules.get("news") or {}
        self.enabled = bool(self.cfg.get("enabled")) and calendar is not None and not calendar.empty
        self.win: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        if not self.enabled:
            return
        levels = self.cfg.get("impact_levels", [3])
        before = pd.Timedelta(minutes=self.cfg.get("minutes_before", 2))
        after = pd.Timedelta(minutes=self.cfg.get("minutes_after", 2))
        ev = calendar[calendar["impact"].isin(levels)]
        for sym, spec in symbols.items():
            t = ev.loc[ev["currency"].isin(spec.get("currencies", [])), "time"].drop_duplicates().sort_values()
            self.win[sym] = ((t - before).to_numpy(), (t + after).to_numpy())

    def in_window(self, sym: str, t0, t1=None) -> bool:
        """True if [t0, t1] overlaps any restricted window for this symbol."""
        if not self.enabled or sym not in self.win:
            return False
        s, e = self.win[sym]
        t0 = np.datetime64(pd.Timestamp(t0)); t1 = np.datetime64(pd.Timestamp(t1 if t1 is not None else t0))
        i = np.searchsorted(e, t0, side="left")
        return bool(i < len(s) and s[i] <= t1)

    def next_window_start(self, sym: str, t):
        if not self.enabled or sym not in self.win:
            return None
        s, e = self.win[sym]
        i = np.searchsorted(s, np.datetime64(pd.Timestamp(t)), side="left")
        return pd.Timestamp(s[i]) if i < len(s) else None


# =================================================================== session guard
@dataclass
class SessionGuard:
    """Your own extra safety on top of the firm: stop after a losing streak or
    a soft daily loss, then demand higher confidence the next day."""
    cfg: dict
    base_threshold: float
    day: object = None
    consec_losses: int = 0
    halted_today: bool = False
    raised: bool = False
    trades_today: int = 0

    def new_day(self, day):
        if self.halted_today:
            self.raised = True
        self.day, self.halted_today, self.trades_today = day, False, 0
        self.consec_losses = 0

    def threshold(self) -> float:
        return self.base_threshold + (self.cfg["threshold_raise"] if self.raised else 0.0)

    def on_close(self, pnl: float, day_pnl: float, initial: float):
        if pnl < 0:
            self.consec_losses += 1
        else:
            self.consec_losses = 0
            if self.cfg.get("reset_after_win", True):
                self.raised = False
        if self.consec_losses >= self.cfg["max_consecutive_losses"]:
            self.halted_today = True
        if day_pnl <= -initial * self.cfg["daily_soft_loss_pct"] / 100:
            self.halted_today = True

    def halt(self):
        self.halted_today = True
