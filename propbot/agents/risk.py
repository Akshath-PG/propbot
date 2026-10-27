"""Risk agent: sizes every trade and runs the day / payout-cycle plan for an
instant account with a floating-loss limit, a trailing max loss, profitable-day
and consistency requirements.

Policy (all values in % of the account, set in settings.agents.risk):
  - risk_pct of balance at the stop, never more than the buffered floating-loss
    limit and never more than the room left to the daily / trailing floors;
    lots are rounded DOWN to the lot step (on a $5K gold account one 0.01 lot
    step is often 0.2-0.3% of the account, so this matters)
  - size never goes up after a loss (no martingale)
  - stop for the day once the day is profitable (day_target_pct), at
    day_stop_pct loss, or after max_trades_per_day
  - stop adding risk once this cycle's profit reaches cycle_target_pct and
    the consistency rule is met: then just wait for the payout date
"""
from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field

from .research import AccountLimits


@dataclass
class AccountState:
    limits: AccountLimits
    start_time: object
    balance: float = 0.0
    hwm: float = 0.0
    cycle_start_time: object = None
    cycle_start_balance: float = 0.0
    day: object = None
    day_ref: float = 0.0
    day_pnl: float = 0.0
    trades_today: int = 0
    cycle_day_pnl: dict = field(default_factory=dict)
    breached: str | None = None
    breach_time: object = None
    payouts: list = field(default_factory=list)
    trades: list = field(default_factory=list)
    refusals: Counter = field(default_factory=Counter)

    def __post_init__(self):
        self.balance = self.hwm = self.cycle_start_balance = self.limits.initial
        self.cycle_start_time = self.start_time

    # ---------------------------------------------------------- cycle stats
    def cycle_profit(self) -> float:
        return self.balance - self.cycle_start_balance

    def profitable_days(self) -> int:
        return sum(v >= self.limits.profitable_day for v in self.cycle_day_pnl.values())

    def max_day(self) -> float:
        return max([0.0, *self.cycle_day_pnl.values()])

    def consistency_ok(self) -> bool:
        p = self.cycle_profit()
        return p > 0 and self.max_day() <= self.limits.consistency * p + 1e-9

    def payout_ready(self, now) -> bool:
        L = self.limits
        return (self.breached is None and (now - self.cycle_start_time).days >= L.cycle_days
                and self.profitable_days() >= L.min_profitable_days and self.consistency_ok()
                and min(self.cycle_profit(), L.payout_cap) >= max(L.min_payout, 1.0))

    def take_payout(self, now) -> float:
        amt = min(self.cycle_profit(), self.limits.payout_cap)
        self.balance -= amt
        self.payouts.append({"time": now, "profit_withdrawn": round(amt, 2),
                             "trader_gets": round(amt * self.limits.split, 2),
                             "cycle_days": (now - self.cycle_start_time).days,
                             "profitable_days": self.profitable_days(), "max_day": round(self.max_day(), 2)})
        self.cycle_start_time, self.cycle_start_balance = now, self.balance
        self.cycle_day_pnl = {}
        self.hwm = self.balance
        return amt

    # ---------------------------------------------------------- days
    def new_day(self, day):
        self.day, self.day_ref, self.day_pnl, self.trades_today = day, self.balance, 0.0, 0

    def floors(self) -> tuple[float, float]:
        L = self.limits
        trail = self.hwm - L.trailing_dd
        if L.trailing_locks:
            trail = min(trail, L.initial)
        return self.day_ref - L.daily_dd, trail


class RiskAgent:
    def __init__(self, cfg: dict, limits: AccountLimits):
        self.cfg, self.L = cfg, limits

    def day_open(self, st: AccountState) -> str | None:
        """Reason the agent refuses any new trade right now, or None."""
        c, i = self.cfg, self.L.initial
        if st.trades_today >= c["max_trades_per_day"]:
            return "max_trades"
        if st.day_pnl >= i * c["day_target_pct"] / 100:
            return "day_target"
        if st.day_pnl <= -i * c["day_stop_pct"] / 100:
            return "day_stop"
        if st.cycle_profit() >= i * c["cycle_target_pct"] / 100 and st.consistency_ok() \
                and st.profitable_days() >= self.L.min_profitable_days:
            return "wait_payout"
        return None

    def room(self, st: AccountState) -> float:
        """Money that may be lost before a buffered firm limit (daily, trailing, floating)."""
        L = self.L
        day_floor, trail_floor = st.floors()
        room = min(st.balance - day_floor, st.balance - trail_floor) * L.buffer
        return max(min(room, st.balance * L.floating_limit_pct / 100 * L.buffer), 0.0)

    def plan(self, st: AccountState, atr: float, risk_mult: float, spec: dict, sl_atr: float, tp_atr: float,
             horizon: int) -> tuple[float, float, float, int, str]:
        """(lots, sl_distance, tp_distance, horizon_bars, reason); lots = 0 means no trade.

        exit_mode "atr"  : SL / TP at sl_atr / tp_atr x ATR, lots from the risk budget.
        exit_mode "money": the stop costs exactly the risk budget and a win pays tp_pct of
                           the balance (just above a "profitable day", inside the consistency
                           cap); the stop is never tighter than min_sl_atr x ATR. Held up to
                           money_horizon_bars, and always flat before the daily reset.
        reason: "ok" | "no_room" (the account is too close to a floor for the minimum lot)
                | "too_volatile" (the minimum lot at the minimum stop costs more than risk_pct,
                  only when min_lot_up_to_room is false)
        """
        c = self.cfg
        comm = spec.get("commission_per_lot", 0.0)
        unit = spec["contract_size"]
        money = c.get("exit_mode", "atr") == "money"
        min_d = (c.get("min_sl_atr", sl_atr) if money else sl_atr) * atr
        min_lot_risk = spec["min_lot"] * (min_d * unit + comm)
        room = self.room(st)
        if room < min_lot_risk:
            return 0.0, 0.0, 0.0, horizon, "no_room"
        budget = min(st.balance * c["risk_pct"] / 100 * risk_mult, room)
        lots = math.floor(budget / (min_d * unit + comm) / spec["lot_step"] + 1e-9) * spec["lot_step"]
        lots = round(min(lots, spec["max_lot"]), 2)
        if lots < spec["min_lot"] - 1e-12:
            if not c.get("min_lot_up_to_room", True):
                return 0.0, 0.0, 0.0, horizon, "too_volatile"
            # a $5K account cannot trade less than the minimum lot; take it if the
            # stop still fits inside the buffered floating / drawdown room
            lots, budget = spec["min_lot"], min_lot_risk
        if money:
            sl_d = (budget - comm * lots) / (lots * unit)
            tp_d = (st.balance * c["tp_pct"] / 100 + comm * lots) / (lots * unit)
            return lots, sl_d, tp_d, int(c.get("money_horizon_bars", 96)), "ok"
        return lots, min_d, tp_atr * atr, horizon, "ok"
