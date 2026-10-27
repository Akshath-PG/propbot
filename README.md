# PropBot

An AI trading bot for MT5 prop-firm accounts. Several strategy "experts"
(trend, ICT, order flow, reversal, news) each give an opinion, a learned
combiner weighs them into one confidence score, and a rule engine built from
your firm's rules has the final veto on every trade.

```
MT5 history + calendar ─► features ─► 5 experts ─► combiner (learned weights)
                                                       │ confidence
                                     session guard ◄───┘
                                           │
                                     firm rule engine  (can block / shrink / close all)
                                           │
                                     MT5 orders  (or log only, when dry_run)
```

## Folder map

| Path | What it is |
|---|---|
| `config/settings.yaml` | symbols, dates, labels, training, risk, session guard, live options |
| `config/firms/example_firm.yaml` | template firm profile (evaluation + funded rules, news windows) |
| `propbot/features.py` | all features, with exact ICT definitions (FVG, IFVG, sweeps, BOS/CHoCH ...) |
| `propbot/labels.py` | triple-barrier labels: did a long / short reach TP before SL? |
| `propbot/model.py` | experts, combiner, purged walk-forward training, threshold choice |
| `propbot/rules.py` | firm rule engine, news windows, session guard |
| `propbot/backtest.py` | bar-by-bar backtest running the real rule engine |
| `propbot/evaluate.py` | 2026 out-of-sample test + monthly challenge simulations + report |
| `propbot/live.py` | live / demo loop on MT5 |
| `mql5/CalendarExporter.mq5` | exports MT5's economic calendar to CSV |
| `scripts/` | `download_history.py`, `train.py`, `retrain.py`, `live_run.py`, `make_synthetic.py` |

## Setup (Windows PC with MT5)

1. Install Python 3.11+ and run `pip install -r requirements.txt`.
2. In MT5: Tools > Options > Charts > **Max bars in chart = Unlimited**. Open an
   M15 chart of every symbol once so the history downloads.
3. Edit `config/settings.yaml`: symbol names exactly as your broker shows them,
   `server_utc_offset_hours`, account size.
4. Copy `config/firms/example_firm.yaml` to e.g. `config/firms/myfirm.yaml`, fill
   it from the firm's official rules page, set `firm_file` to it.
5. Calendar: copy `mql5/CalendarExporter.mq5` into `MQL5/Experts/` (File > Open
   Data Folder), compile it in MetaEditor, attach it to any chart. It writes
   `%APPDATA%\MetaQuotes\Terminal\Common\Files\calendar.csv`. Point
   `live.calendar_file` in settings to that full path (or copy the file to `data/`).
6. `python scripts/download_history.py`

## Train and test

```
python scripts/train.py
```

This trains on everything up to **31 Dec 2025** and then tests on **1 Jan 2026
onwards**, which the model never saw. You get `reports/holdout_2026_report.md`,
an equity chart, every trade in CSV, and a fresh evaluation challenge simulated
from the 1st of each month in 2026.

### Why not a random 30 % / 70 % split

Neighbouring bars are almost the same, so a random split lets the model see the
"future" during training. It then looks excellent in testing and fails live.
Instead:

- **Purged walk-forward folds** inside 2018-2025: train on the past, validate on
  the next block, with a gap so no trade outcome leaks across.
- **Overfit check:** for every expert the report compares training vs validation
  log-loss and flags `OVERFIT` when training is much better.
- **Underfit check:** the report flags `UNDERFIT / no edge` when validation is no
  better than guessing.
- **Final exam:** the 2026 period is touched only once, after everything is fixed.

### What "confidence" means here

Confidence is the model's probability that a trade reaches TP (3 x ATR) before
SL (1.5 x ATR). Because the reward is twice the risk, break-even is about 34 %
plus costs, so 0.40-0.50 is already a strong edge. A 0.80-0.90 requirement would
almost never trigger.

The threshold is therefore not fixed by hand. Training picks the one with the
best total expectancy on the last validation fold, subject to a minimum number of
trades per month. If no threshold earns at least `min_expectancy_r`, the model is
marked **not tradeable** and the live bot refuses to place orders.

## Safety layers (in order)

1. **Session guard** (yours)
   - Stops for the day after `max_consecutive_losses` losses in a row, or at
     `daily_soft_loss_pct`.
   - The next day it requires `threshold + threshold_raise` confidence until a
     trade wins.
2. **Firm rule engine**
   - Daily drawdown (balance, equity or the higher of both at reset).
   - Max drawdown: static, trailing, or end-of-day trailing, with lock at the
     starting balance.
   - Floating-loss (hidden breach) limit.
   - Consistency cap, minimum trading days, max risk per trade.
   - Weekend holding, news windows.
   - The bot acts at `safety_buffer` (default 80 %) of every limit.
   - Before each trade it checks that, even if all open trades and the new one hit
     their stops, equity stays above the floor.
3. **Monthly band (funded):** stops adding risk once the month reaches the top of
   `monthly_target_pct`.

## Self-learning (champion / challenger)

```
python scripts/retrain.py --holdout-weeks 8
```

Run it monthly, for example with Windows Task Scheduler on a weekend. It:

1. Retrains a challenger on all data except the last 8 weeks.
2. Tests both the challenger and the current champion on those 8 weeks.
3. Promotes the challenger only if it is tradeable, did not breach, and is at
   least as good as the champion.

Weights never change mid-day and never change without passing a test. Old models
are kept in `models/`.

## Going live

```
python scripts/live_run.py
```

`dry_run: true` is the default: it logs every signal, with each expert's opinion,
to `reports/live_log.csv` without sending orders.

Recommended order:

1. Dry run for 1-2 weeks.
2. Demo account with `dry_run: false` for at least a month.
3. Compare the demo results with what the backtest predicted.
4. Only then use a cheap evaluation.

A non-demo account with `dry_run: false` needs `--real`.

## Smoke test without MT5

```
python scripts/make_synthetic.py --out data_synth            # fake market with one planted edge
python scripts/make_synthetic.py --out data_rw --edge 0      # pure random walk (no edge)
```

Point a copy of `settings.yaml` at either folder (`data_dir`, `calendar_file`)
and run `train.py --settings <copy>`.

- On the planted-edge data the bot should find the edge.
- On the random walk it should report no edge and refuse to trade.

The second test is how you know the pipeline is not leaking future data.

## GFT Instant HERO $5K ($17) - multi-agent model

A separate pipeline for Goat Funded Trader's Instant HERO account (the $17 /
$5K instant account). HERO has no evaluation: "passing" means reaching the first
payout (after 14+ days, 6 profitable days of >= 0.5%, no day above 15% of the
profit) without breaking the 3% daily, 5% trailing or 1% floating-loss rule.
Rules, sources and open questions: `config/firms/gft_instant_hero.yaml`.

```
python scripts/build_hero_data.py    # XAUUSD M15 2016 -> today + economic calendar, from public GitHub mirrors
python scripts/hero_run.py           # walk-forward training, policy search, 2-year test -> reports/hero_report.md
```

| Agent | File | Job |
|---|---|---|
| Research | `propbot/agents/research.py` | turns the researched firm rules into hard limits |
| Strategy (ML experts) | `propbot/agents/strategy.py`, `model.py` | trend, ICT, order-flow, reversal, news and geo experts + combiner |
| Trend | `scripts/hero_run.py` | H4 and daily trend direction, majority vote with the combiner |
| News | `propbot/agents/news.py` | no entries around high-impact USD events |
| Geopolitical | `propbot/agents/geopolitical.py` | market-implied stress regime (gaps, volatility shocks, safe-haven runs) |
| Risk | `propbot/agents/risk.py` | sizing inside every limit, profitable-day / consistency / payout plan |
| Coordinator | `propbot/agents/coordinator.py` | runs them together bar by bar and simulates the account and its payouts |

Settings: `config/settings_hero.yaml` (all bar times UTC). The report states
plainly how much of the result comes from the risk plan and how much from
direction calls, by comparing against random-direction trading under the same plan.

## Honest limitations

- **Order flow:** spot-forex MT5 gives tick volume only. The order-flow expert uses
  proxies; real delta needs a futures data feed (GC for gold, 6E for EUR ...).
- **News surprise:** the sign is learned from data. Some indicators are "higher is
  worse" (e.g. unemployment), and the model has to learn that per currency.
- **Backtest limits:** backtests are pessimistic on fills but cannot model every
  live condition (slippage spikes, requotes, server outages).
- **Firm rules:** check that your firm allows EAs/bots in the phase you are in.
  Re-read its rules before every new challenge; they change.
- **No guarantee:** past results, including 2026 out-of-sample, do not guarantee
  future profit. The rule engine protects the account; it cannot create an edge.
