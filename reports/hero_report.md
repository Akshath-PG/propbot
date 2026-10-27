# Multi-agent model - GFT Instant HERO $5K (XAUUSD)

Test period (last 2 years, touched once): **2024-11-01 -> 2026-10-27**. Every choice (risk policy, direction source) was made on 2018-11-01 -> 2024-10-31 with walk-forward models.

## Headline

- **A fresh $17 account started every week of the last 2 years (104 accounts): 29.8% reached the first payout**, 63.5% failed, 6.7% were still running when the data ends.
- Median time to the first payout: 40.0 days. Expected value per account (payout to you minus the fee): **$+33.97**.
- One trader running the bot continuously, re-buying after every failed account: 8 accounts ($136 in fees), **5 payouts, $868 received, net $+732**.
- Same risk plan with RANDOM trade directions on the same 2 years (30 runs): payout rate 25.9% on average (10-90% range 11.2% - 39.4%), EV $+27.21.

**How to read this:** the risk agent's plan is what lets a HERO account pay out often enough to beat the $17 fee - random trade directions under the same plan already average 25.9%. The direction source chosen on 2018-2024 (always_long) was well ahead of random there (z = +2.9), but on the test years it scored 29.8% vs 25.9% for random: no reliable directional edge out of sample. Results swing a lot with luck: treat every rate here as +/- 10 points.

## 1. Research agent - the account rules modelled

- Account: GFT Instant HERO 5K, $5,000 balance, price $17
- Daily loss limit: $150 (3%), resets 17:00 America/New_York
- Max loss: $250 (5%) trailing the equity high (no lock assumed)
- Floating loss: open trades may never be down 1% of balance (account closed)
- Payout: every 14 days, needs 6 days of >= $25 profit and no day above 15% of total profit; cap $200, split 90%
- Passing = first payout: at least 14 days, 6 profitable days (>= +0.5%), no day above 15% of the profit,
  no rule broken. HERO has no profit target, so the payout is the only 'pass'.
- 'Instant Haiku' is not a GFT product name; the $17 / $5K price matches **Instant HERO**. Sources and open questions: `config/firms/gft_instant_hero.yaml`.

## 2. How the agents work together

```
research agent ---- hard limits (3% daily, 5% trailing on equity, 1% floating, payout rules)
strategy agents --- trend / ICT / order-flow / reversal / news / geo experts -> combiner (walk-forward)
trend agents ------ H4 trend, daily trend; majority vote with the combiner
coordinator ------- picks the direction source, session hours, room before the 5 PM NY reset
news agent -------- no entries 30 min before / 15 min after high-impact USD events
geopolitical agent  market-implied stress regime: half size in a crisis, no trades against the safe-haven flow
risk agent -------- stop = risk_pct of balance (inside the 1% floating limit), win = tp_pct (a profitable
                    day), done for the day after one win, day stop, breakeven stop, then wait for payout
execution --------- next-bar entry + spread; worst price of every bar checked against every limit
```

Signals through each stage in the test period: strategy_signals 60,039, vetoed_timing 23,578, vetoed_news 1,730, vetoed_geo 616, geo_crisis_size_cut 1,098

## 3. Chosen setup (from 2018-2024 only)

- Risk policy: risk 0.8% at the stop, win 0.52%, stop >= 3.5 x ATR(M15), max 2 trades/day, day stop 1.0%, breakeven at 0.8, cycle target 3.5%
- Direction source: **always_long**

- WARNING: risk_pct 0.8 >= 80% of the floating-loss limit

Top risk policies (Monte Carlo, random directions, 2018-11-01 -> 2024-10-31):

| risk_pct | tp_pct | min_sl_atr | max_trades_per_day | day_stop_pct | breakeven_at | cycle_target_pct | payout_rate | payout_rate_sd | breach_rate | ev_per_account_usd |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.8 | 0.52 | 3.5 | 2 | 1 | 0.8 | 3.5 | 0.305 | 0.064 | 0.6752 | 35.21 |
| 0.8 | 0.52 | 3.5 | 2 | 1 | 0.8 | 4 | 0.302 | 0.0617 | 0.6785 | 34.81 |
| 0.8 | 0.52 | 3.5 | 2 | 1 | 0 | 3.5 | 0.2877 | 0.0673 | 0.6953 | 32.05 |
| 0.8 | 0.52 | 3.5 | 2 | 1 | 0 | 4 | 0.2862 | 0.0652 | 0.6963 | 31.93 |
| 0.8 | 0.52 | 3.5 | 2 | 0.5 | 0 | 3.5 | 0.2838 | 0.0717 | 0.6923 | 31.42 |
| 0.8 | 0.52 | 3.5 | 1 | 0.5 | 0 | 3.5 | 0.2833 | 0.0722 | 0.6922 | 31.36 |
| 0.8 | 0.52 | 3.5 | 1 | 1 | 0 | 3.5 | 0.2833 | 0.0722 | 0.6922 | 31.36 |
| 0.8 | 0.52 | 3.5 | 2 | 0.5 | 0 | 4 | 0.2822 | 0.0708 | 0.6933 | 31.32 |

Direction sources, first-payout rate by era (walk-forward, before the test):

| Source | 2018-11..2020-10 | 2020-11..2022-10 | 2022-11..2024-10 | all pre-test |
|---|---|---|---|---|
| agents_majority_vote | 16.2% | 29.5% | 25.7% | 24.6% |
| always_long | 44.8% | 9.5% | 44.8% | 34.2% |
| ml_combiner | 22.9% | 44.8% | 29.5% | 33.9% |
| trend_daily_agent | 14.3% | 7.6% | 25.7% | 16.0% |
| trend_h4_agent | 10.5% | 18.1% | 28.6% | 18.5% |
| random direction (mean of 20) | 19.5% +/- 8 | 19.7% +/- 9 | 22.0% +/- 7 | 21.2% +/- 4 |

## 4. Test - last 2 years

| Run | Accounts | First payout | Failed | Still open | Median days to payout | EV per $17 account |
|---|---|---|---|---|---|---|
| **Full model (always_long)** | 104 | 29.8% | 63.5% | 6.7% | 40 | $+33.97 |
| ml combiner | 104 | 32.7% | 59.6% | 7.7% | 43 | $+39.31 |
| trend h4 agent | 104 | 48.1% | 40.4% | 11.5% | 47 | $+65.80 |
| trend daily agent | 104 | 38.5% | 55.8% | 5.8% | 40 | $+48.90 |
| agents majority vote | 104 | 33.7% | 57.7% | 8.7% | 38 | $+40.65 |
| without news agent | 104 | 29.8% | 63.5% | 6.7% | 40 | $+33.72 |
| without geo agent | 104 | 26.0% | 67.3% | 6.7% | 39 | $+27.11 |
| random direction, same risk agent (mean of 30) | 104 | 25.9% | - | - | - | $+27.21 |

Rows other than the first are shown for information only; none of them was used to choose anything.

Why accounts failed:

- abandoned: no room for the minimum lot: 66

('abandoned' = equity sat so close to the trailing floor that even a 0.01 lot stop would have broken it, for 10 trading days: the account is finished; the buffer stopped it before a rule breach.)

Weekly accounts by start quarter:

| Quarter | breached | payout | still_open |
|---|---|---|---|
| 2024Q4 | 6 | 3 | 0 |
| 2025Q1 | 8 | 5 | 0 |
| 2025Q2 | 13 | 0 | 0 |
| 2025Q3 | 4 | 9 | 0 |
| 2025Q4 | 2 | 11 | 0 |
| 2026Q1 | 13 | 0 | 0 |
| 2026Q2 | 10 | 3 | 0 |
| 2026Q3 | 10 | 0 | 3 |
| 2026Q4 | 0 | 0 | 4 |

### Continuous trader (one account at a time, re-buy after a failure)

| # | Start | End | Result | Payouts | Paid to trader | Trades |
|---|---|---|---|---|---|---|
| 1 | 2024-11-01 | 2025-01-13 | abandoned: no room for the minimum lot | 0 | $0 | 54 |
| 2 | 2025-01-14 | 2025-03-25 | abandoned: no room for the minimum lot | 2 | $346 | 56 |
| 3 | 2025-03-26 | 2025-07-28 | abandoned: no room for the minimum lot | 0 | $0 | 111 |
| 4 | 2025-07-29 | 2026-04-07 | abandoned: no room for the minimum lot | 3 | $521 | 190 |
| 5 | 2026-04-08 | 2026-05-14 | abandoned: no room for the minimum lot | 0 | $0 | 21 |
| 6 | 2026-05-15 | 2026-07-10 | abandoned: no room for the minimum lot | 0 | $0 | 30 |
| 7 | 2026-07-12 | 2026-09-24 | abandoned: no room for the minimum lot | 0 | $0 | 62 |
| 8 | 2026-09-25 | 2026-10-27 | running | 0 | $0 | 33 |

| Payout date | Withdrawn | You get | Cycle days | Profitable days | Biggest day |
|---|---|---|---|---|---|
| 2025-02-03 | $200 | $180 | 20 | 10 | $27 |
| 2025-02-25 | $185 | $166 | 22 | 11 | $27 |
| 2025-09-09 | $196 | $177 | 42 | 17 | $27 |
| 2025-10-17 | $182 | $164 | 38 | 17 | $27 |
| 2026-01-14 | $200 | $180 | 89 | 33 | $27 |

Trades: 557, win rate 52.1%, exits {'tp': 236, 'sl': 162, 'day_end': 106, 'breakeven': 53}

## 5. Strategy agents (ML experts), final model trained to 2024-10-31

| Expert | validation log-loss | baseline | skill % | verdict |
|---|---|---|---|---|
| trend | 1.041 | 1.0497 | 0.83 | OVERFIT (train much better than validation) |
| ict | 1.0338 | 1.0497 | 1.51 | OVERFIT (train much better than validation) |
| orderflow | 1.0379 | 1.0497 | 1.12 | ok |
| reversal | 1.0459 | 1.0497 | 0.36 | ok |
| news | 1.0225 | 1.0497 | 2.59 | ok |
| geo | 1.0462 | 1.0497 | 0.33 | OVERFIT (train much better than validation) |

Combiner weights: {'trend': 3.2, 'ict': 9.1, 'orderflow': 1.9, 'reversal': 1.2, 'news': 6.9, 'geo': 3.9, 'context': 73.7}

Skill is a few percent better than guessing the class mix, and after spread the raw signal expectancy is about zero. Research for this model also tested gold's 'Asian session drift' and daily momentum: both disappear with true mid prices and real spreads (the drift seen in bid prices is the spread narrowing after the 6 PM reopen).

## 6. Limits of this test

- Data: Dukascopy XAUUSD M1 bid/ask to 2026-09-01, then two public feeds to 2026-10-27; GFT's own prices and spreads will differ, mostly around news and the daily reopen.
- Fills: next-bar open + spread; stop assumed first when stop and target fall in the same bar; the equity high-water mark takes each bar's best price before its worst (strict for the trailing limit). No extra slippage.
- Calendar: ForexFactory high-impact events to 2026-01-30, then rule-based FOMC / NFP dates.
- Rules: the 5% trailing limit is modelled without a lock at the start balance; payout cap 4% ($200) per 14-day cycle; trader gets 90%. GFT can change any of this.
- Weekly accounts overlap in time, so their outcomes are correlated.
- Forward-test on a demo before buying accounts, and confirm GFT allows your own EA on HERO.
