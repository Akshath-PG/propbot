"""Loading settings and firm rule profiles."""
from __future__ import annotations

from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parent.parent


def load_settings(path: str | Path = "config/settings.yaml") -> dict:
    path = Path(path)
    if not path.is_absolute():
        path = ROOT / path
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["_root"] = str(ROOT)
    return cfg


def load_firm(settings: dict, phase: str | None = None) -> dict:
    """Return the rule set for one phase, with firm-wide keys merged in."""
    path = Path(settings["firm_file"])
    if not path.is_absolute():
        path = ROOT / path
    with open(path, "r", encoding="utf-8") as f:
        firm = yaml.safe_load(f)
    phase = phase or settings.get("phase", "evaluation")
    if phase not in firm["phases"]:
        raise ValueError(f"Phase '{phase}' not in firm file {path}")
    rules = dict(firm["phases"][phase])
    for key in ("name", "account_currency", "daily_reset_time",
                "daily_reset_timezone", "safety_buffer"):
        rules[key] = firm.get(key)
    rules["phase"] = phase
    validate_rules(rules)
    return rules


REQUIRED = ["daily_drawdown_pct", "max_drawdown_pct", "max_drawdown_type"]


def validate_rules(rules: dict) -> None:
    missing = [k for k in REQUIRED if rules.get(k) is None]
    if missing:
        raise ValueError(f"Firm rules missing required keys: {missing}")
    if rules["max_drawdown_type"] not in ("static", "trailing", "eod_trailing"):
        raise ValueError("max_drawdown_type must be static | trailing | eod_trailing")
    if rules.get("ea_allowed") is False:
        raise ValueError("Firm profile says EAs are not allowed in this phase. "
                         "The bot will not run on it.")


def path_in_root(settings: dict, rel: str) -> Path:
    p = Path(rel)
    return p if p.is_absolute() else Path(settings["_root"]) / p
