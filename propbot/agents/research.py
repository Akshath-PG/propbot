"""Research agent: turns the researched firm profile into hard limits the
other agents must respect, and explains what "passing" means for this account.
"""
from __future__ import annotations

from dataclasses import dataclass

from ..config import load_firm


@dataclass(frozen=True)
class AccountLimits:
    name: str
    initial: float
    price_usd: float
    daily_dd: float              # money
    trailing_dd: float           # money below the equity high-water mark
    trailing_locks: bool
    floating_limit_pct: float    # % of balance
    consistency: float           # fraction (0.15)
    cycle_days: int
    min_profitable_days: int
    profitable_day: float        # money
    payout_cap: float            # money per cycle
    split: float
    min_payout: float
    buffer: float
    reset_time: str
    reset_tz: str

    def describe(self) -> list[str]:
        i = self.initial
        return [
            f"Account: {self.name}, ${i:,.0f} balance, price ${self.price_usd:g}",
            f"Daily loss limit: ${self.daily_dd:,.0f} ({100 * self.daily_dd / i:g}%), resets {self.reset_time} {self.reset_tz}",
            f"Max loss: ${self.trailing_dd:,.0f} ({100 * self.trailing_dd / i:g}%) trailing the equity high"
            + (" (locks at start balance)" if self.trailing_locks else " (no lock assumed)"),
            f"Floating loss: open trades may never be down {self.floating_limit_pct:g}% of balance (account closed)",
            f"Payout: every {self.cycle_days} days, needs {self.min_profitable_days} days of >= ${self.profitable_day:,.0f} "
            f"profit and no day above {100 * self.consistency:g}% of total profit; cap ${self.payout_cap:,.0f}, "
            f"split {100 * self.split:g}%",
        ]


class ResearchAgent:
    def __init__(self, settings: dict):
        self.settings = settings
        self.rules = load_firm(settings)
        r, p = self.rules, self.rules.get("payout") or {}
        i = float(settings["initial_balance"])
        self.limits = AccountLimits(
            name=r["name"], initial=i, price_usd=float(_firm_key(settings, "price_usd") or 0.0),
            daily_dd=i * r["daily_drawdown_pct"] / 100, trailing_dd=i * r["max_drawdown_pct"] / 100,
            trailing_locks=bool(r.get("trailing_locks_at_initial")),
            floating_limit_pct=float(r.get("max_floating_loss_pct") or 100.0),
            consistency=float(r.get("consistency_pct") or 100.0) / 100,
            cycle_days=int(p.get("reward_cycle_days", 14)), min_profitable_days=int(p.get("min_profitable_days", 0)),
            profitable_day=i * float(p.get("profitable_day_pct", 0.0)) / 100,
            payout_cap=i * float(p.get("cap_pct_of_initial", 100.0)) / 100, split=float(p.get("profit_split", 0.8)),
            min_payout=float(p.get("min_payout_usd", 0.0)), buffer=float(r.get("safety_buffer") or 1.0),
            reset_time=r.get("daily_reset_time") or "00:00", reset_tz=r.get("daily_reset_timezone") or "UTC")

    def check_policy(self, risk_cfg: dict) -> list[str]:
        """Warnings when a risk policy could break a firm rule by construction."""
        L, w = self.limits, []
        if risk_cfg["risk_pct"] >= L.floating_limit_pct * L.buffer:
            w.append(f"risk_pct {risk_cfg['risk_pct']} >= {L.buffer:.0%} of the floating-loss limit")
        if risk_cfg["day_stop_pct"] + risk_cfg["risk_pct"] >= 100 * L.daily_dd / L.initial * L.buffer:
            w.append("day stop + one more loss could reach the buffered daily limit")
        return w


def _firm_key(settings: dict, key: str):
    import yaml
    from ..config import path_in_root
    with open(path_in_root(settings, settings["firm_file"]), encoding="utf-8") as f:
        return yaml.safe_load(f).get(key)
