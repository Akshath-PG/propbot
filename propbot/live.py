"""Live / demo trading loop on MetaTrader 5 (Windows).

Every poll (default 5 s):
  - read balance, equity and this bot's open positions
  - feed the rule engine; on 'flatten' or 'breach' close everything and halt
  - record trades that closed since the last poll (rule engine + session guard)
  - close positions before a news window if the firm requires it
On every new closed bar:
  - rebuild features for each symbol, ask the model, apply all filters,
    size the trade from the remaining risk budget, send the order
    (or only log it when dry_run is true).
State (days traded, daily P&L, high-water mark, guard) survives restarts.
"""
from __future__ import annotations

import csv
import json
import math
import time
from pathlib import Path

import pandas as pd

from .config import load_firm, path_in_root
from .data import TF_MINUTES, load_calendar, mt5_bars, mt5_connect
from .features import build_features
from .model import EnsembleModel
from .rules import NewsWindows, RuleEngine, SessionGuard, firm_days


class LiveTrader:
    def __init__(self, settings: dict, allow_real: bool = False, force_untradeable: bool = False):
        self.s = settings
        self.rules = load_firm(settings)
        self.model = EnsembleModel.load(path_in_root(settings, settings["model_dir"]) / "champion.joblib")
        self.dry = settings["live"]["dry_run"]
        self.magic = settings["live"]["magic_number"]
        self.tf = settings["timeframe"]
        self.bar = pd.Timedelta(minutes=TF_MINUTES[self.tf])
        self.mt5 = mt5_connect()
        acc = self.mt5.account_info()
        is_demo = acc.trade_mode == self.mt5.ACCOUNT_TRADE_MODE_DEMO
        print(f"Account {acc.login} @ {acc.server}  balance {acc.balance:,.2f}  demo={is_demo}  dry_run={self.dry}")
        if not is_demo and not self.dry and not allow_real:
            raise SystemExit("Real/prop account with dry_run=false: start with --real to confirm.")
        if not self.model.tradeable and not self.dry and not force_untradeable:
            raise SystemExit("Champion model is marked NOT tradeable. Retrain, or run with dry_run=true.")
        for sym in settings["symbols"]:
            self.mt5.symbol_select(sym, True)

        self.state_path = path_in_root(settings, settings["report_dir"]) / "live_state.json"
        self.log_path = path_in_root(settings, settings["live"]["log_file"])
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.rule = RuleEngine(self.rules, settings["initial_balance"])
        self.guard = SessionGuard(settings["session_guard"], self.model.threshold)
        self._load_state()
        self.known: dict[int, dict] = {}
        self.last_bar = None
        self.cal_mtime = None
        self._refresh_calendar()

    # ------------------------------------------------------------ persistence
    def _load_state(self):
        if not self.state_path.exists():
            return
        st = json.loads(self.state_path.read_text())
        self.rule.hwm = st.get("hwm", self.rule.hwm)
        self.rule.day_pnl = {pd.Timestamp(k).date(): v for k, v in st.get("day_pnl", {}).items()}
        self.rule.days_traded = {pd.Timestamp(d).date() for d in st.get("days_traded", [])}
        self.rule.breached = st.get("breached")
        self.guard.raised = st.get("guard_raised", False)
        print(f"Restored state: {len(self.rule.days_traded)} trading days, breached={self.rule.breached}")

    def _save_state(self):
        st = {"hwm": self.rule.hwm, "day_pnl": {str(k): v for k, v in self.rule.day_pnl.items()},
              "days_traded": sorted(str(d) for d in self.rule.days_traded), "breached": self.rule.breached,
              "guard_raised": self.guard.raised, "day": str(self.rule.day), "halted": self.guard.halted_today}
        self.state_path.write_text(json.dumps(st, indent=2))

    def log(self, **row):
        row = {"ts": pd.Timestamp.now().isoformat(timespec="seconds"), **row}
        new = not self.log_path.exists()
        with open(self.log_path, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(row))
            if new:
                w.writeheader()
            w.writerow(row)
        print(" | ".join(f"{k}={v}" for k, v in row.items()))

    def _refresh_calendar(self):
        p = path_in_root(self.s, self.s["live"]["calendar_file"])
        m = p.stat().st_mtime if p.exists() else None
        if m != self.cal_mtime:
            self.cal = load_calendar(self.s)
            self.news = NewsWindows(self.cal, self.rules, self.s["symbols"])
            self.cal_mtime = m

    # ------------------------------------------------------------ helpers
    def server_now(self) -> pd.Timestamp:
        sym = next(iter(self.s["symbols"]))
        return pd.Timestamp(self.mt5.symbol_info_tick(sym).time, unit="s")

    def my_positions(self):
        ps = self.mt5.positions_get() or []
        return [p for p in ps if p.magic == self.magic]

    def money_per_price_unit(self, sym) -> float:
        info = self.mt5.symbol_info(sym)
        return info.trade_tick_value / info.trade_tick_size  # per 1.0 lot

    def close_position(self, p, reason):
        tick = self.mt5.symbol_info_tick(p.symbol)
        is_buy = p.type == self.mt5.POSITION_TYPE_BUY
        req = {"action": self.mt5.TRADE_ACTION_DEAL, "symbol": p.symbol, "volume": p.volume,
               "type": self.mt5.ORDER_TYPE_SELL if is_buy else self.mt5.ORDER_TYPE_BUY,
               "position": p.ticket, "price": tick.bid if is_buy else tick.ask, "deviation": 30,
               "magic": self.magic, "comment": f"close:{reason}"[:31],
               "type_filling": self._filling(p.symbol)}
        res = self.mt5.order_send(req) if not self.dry else None
        self.log(event="close", symbol=p.symbol, ticket=p.ticket, reason=reason,
                 result=getattr(res, "retcode", "dry"))

    def close_all(self, reason):
        for p in self.my_positions():
            self.close_position(p, reason)

    def _filling(self, sym):
        fm = self.mt5.symbol_info(sym).filling_mode
        if fm & 1:
            return self.mt5.ORDER_FILLING_FOK
        if fm & 2:
            return self.mt5.ORDER_FILLING_IOC
        return self.mt5.ORDER_FILLING_RETURN

    # ------------------------------------------------------------ main loop
    def step(self):
        now = self.server_now()
        day = firm_days(pd.DatetimeIndex([now]), self.rules, self.s.get("server_utc_offset_hours", 0))[0]
        acc = self.mt5.account_info()
        if day != self.rule.day:
            self.rule.balance = acc.balance
            self.rule.new_day(day, acc.balance, acc.equity)
            self.guard.new_day(day)
            self.log(event="new_day", day=day, balance=acc.balance, threshold=round(self.guard.threshold(), 3))
        self.rule.balance = acc.balance

        # trades closed since last poll
        pos = self.my_positions()
        open_tickets = {p.ticket for p in pos}
        for tk in [t for t in self.known if t not in open_tickets]:
            deals = self.mt5.history_deals_get(position=tk) or []
            pnl = sum(d.profit + d.commission + d.swap for d in deals)
            in_news = bool(self.rules.get("news", {}).get("no_close_in_window")) and \
                self.news.in_window(self.known[tk]["symbol"], now - pd.Timedelta(seconds=30), now)
            self.rule.balance -= pnl          # on_close adds it back and books the day
            self.rule.on_close(pnl, day, in_news)
            self.guard.on_close(pnl, self.rule.day_pnl.get(day, 0.0), self.rule.initial)
            self.log(event="closed", ticket=tk, symbol=self.known[tk]["symbol"], pnl=round(pnl, 2),
                     day_pnl=round(self.rule.day_pnl.get(day, 0.0), 2), consec_losses=self.guard.consec_losses,
                     halted=self.guard.halted_today)
            del self.known[tk]
        for p in pos:
            self.known.setdefault(p.ticket, {"symbol": p.symbol})

        # firm rules on live equity
        floating = acc.equity - acc.balance
        st = self.rule.check(acc.equity, acc.equity, floating, now)
        if st in ("flatten", "breach"):
            self.close_all(st)
            self.guard.halt()
            self.log(event=st, equity=acc.equity, floor=round(self.rule.floor(False), 2), reason=self.rule.breached)
            self._save_state()
            return

        # news: flatten before a window if required
        ncfg = self.rules.get("news") or {}
        if ncfg.get("flatten_before_window") and self.news.enabled:
            for p in pos:
                if self.news.in_window(p.symbol, now, now + pd.Timedelta(minutes=2)):
                    self.close_position(p, "pre-news")

        # new closed bar?
        first = next(iter(self.s["symbols"]))
        b = mt5_bars(self.mt5, first, self.tf, 3)
        last_closed = b["time"].iloc[-2]
        if self.last_bar is not None and last_closed <= self.last_bar:
            return
        self.last_bar = last_closed
        self._refresh_calendar()
        self.on_bar(now, day, acc)
        self._save_state()

    def on_bar(self, now, day, acc):
        tcfg = self.s["trading"]
        if self.rule.breached or self.guard.halted_today:
            return
        if self.rule.target_reached() and self.rules["phase"] == "evaluation" and \
                len(self.rule.days_traded) >= (self.rules.get("min_trading_days") or 0):
            self.log(event="target_reached", status=self.rule.status())
            return
        cap = self.rule.daily_profit_cap()
        if cap is not None and self.rule.day_pnl.get(day, 0.0) >= cap:
            return

        rows = []
        for sym in self.s["symbols"]:
            try:
                df = mt5_bars(self.mt5, sym, self.tf, 1500).iloc[:-1]  # drop forming bar
            except RuntimeError as e:
                self.log(event="data_error", symbol=sym, error=str(e)); continue
            feats = build_features(df, self.s, sym, self.cal)
            x = feats.iloc[[-1]]
            pred = self.model.predict(x).iloc[0]
            experts = {k: round(float(v), 3) for k, v in pred.items() if k.endswith(("_long", "_short"))
                       and not k.startswith("p_")}
            rows.append((sym, x.iloc[0], pred, experts))

        pos = self.my_positions()
        th = self.guard.threshold()
        for sym, x, pred, experts in sorted(rows, key=lambda r: -r[2]["confidence"]):
            conf, direction = float(pred["confidence"]), int(pred["direction"])
            decision = "skip"
            why = ""
            if conf < th:
                why = f"conf {conf:.3f} < {th:.3f}"
            elif self.guard.trades_today >= tcfg["max_trades_per_day"] or len(pos) >= tcfg["max_open_trades"]:
                why = "trade limits"
            elif any(p.symbol == sym for p in pos):
                why = "already in position"
            elif (self.rules.get("news") or {}).get("no_open_in_window") and \
                    self.news.in_window(sym, now, now + self.bar):
                why = "news window"
            elif not self.rules.get("weekend_holding", False) and now.weekday() == 4 and now.hour >= 21:
                why = "weekend close approaching"
            else:
                decision, why = self.place(sym, direction, float(x["atr"]), conf, pos, acc)
            self.log(event="signal", symbol=sym, direction=direction, confidence=round(conf, 3),
                     threshold=round(th, 3), decision=decision, why=why, experts=json.dumps(experts))

    def place(self, sym, direction, atr_val, conf, pos, acc):
        lab, tcfg = self.s["labels"], self.s["trading"]
        info, tick = self.mt5.symbol_info(sym), self.mt5.symbol_info_tick(sym)
        price = tick.ask if direction == 1 else tick.bid
        sl_d, tp_d = lab["sl_atr"] * atr_val, lab["tp_atr"] * atr_val
        per_lot = sl_d * self.money_per_price_unit(sym) + self.s["symbols"][sym].get("commission_per_lot", 0.0)
        open_risk = 0.0
        for p in pos:
            if p.sl:
                open_risk += max(abs(p.price_open - p.sl) * self.money_per_price_unit(p.symbol) * p.volume, 0)
        self.rule.equity = acc.equity
        budget = min(acc.equity * tcfg["risk_per_trade_pct"] / 100, self.rule.risk_budget(open_risk))
        lots = math.floor(budget / per_lot / info.volume_step) * info.volume_step
        if self.rule.target_reached():
            lots = info.volume_min
        lots = round(min(lots, info.volume_max), 2)
        if lots < info.volume_min:
            return "skip", f"risk budget too small ({budget:.2f})"
        sl = price - direction * sl_d; tp = price + direction * tp_d
        digits = info.digits
        req = {"action": self.mt5.TRADE_ACTION_DEAL, "symbol": sym, "volume": lots,
               "type": self.mt5.ORDER_TYPE_BUY if direction == 1 else self.mt5.ORDER_TYPE_SELL,
               "price": price, "sl": round(sl, digits), "tp": round(tp, digits), "deviation": 20,
               "magic": self.magic, "comment": f"pb c={conf:.2f}", "type_time": self.mt5.ORDER_TIME_GTC,
               "type_filling": self._filling(sym)}
        if self.dry:
            self.guard.trades_today += 1
            return "dry_run", f"lots {lots} sl {req['sl']} tp {req['tp']} risk {lots * per_lot:.2f}"
        res = self.mt5.order_send(req)
        if res is None or res.retcode != self.mt5.TRADE_RETCODE_DONE:
            return "rejected", f"{getattr(res, 'retcode', None)} {getattr(res, 'comment', self.mt5.last_error())}"
        self.guard.trades_today += 1
        self.known[res.order] = {"symbol": sym}
        return "opened", f"ticket {res.order} lots {lots} sl {req['sl']} tp {req['tp']}"

    def run(self):
        print(f"Running. Threshold {self.model.threshold}, firm {self.rules['name']} ({self.rules['phase']}).")
        try:
            while True:
                try:
                    self.step()
                except Exception as e:  # keep running; log the problem
                    self.log(event="error", error=repr(e))
                time.sleep(self.s["live"]["poll_seconds"])
        finally:
            self._save_state()
            self.mt5.shutdown()
