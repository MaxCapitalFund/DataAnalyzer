
# Strategy One-Sheet (Trade Data)

**Strategy:** SuperSignal_v7_PreMarket45_43  
**Instrument:** Unknown  
**Timeframe:** 180d:15m  
**Run Date:** 2025-09-09  
**Session Basis:** New York time (ET). **Metrics Scope:** RTH (09:30–16:00)  
**Initial Capital (float):** $2500  
**Commission (RT / contract):** $4.04  
**Point Value:** $5.00 per point per contract

> **Note:** Stop-loss normalization is enabled: losses below **−$100** are capped at −$100 and the overage is added back as a positive correction (TOS discrepancy fix). All metrics use **AdjustedNetPL**.

---

## Key Performance Indicators (KPI's)
- **Net Profit (after commissions, adjusted):** $3771.80
- **Gross Profit (before commissions):** $1607.50
- **Total Return (Net, adjusted):** 150.87%
- **Total Return (Gross):** 64.30%
- **Win Rate:** 49.11%
- **Profit Factor (Adjusted):** 1.76
- **Max Drawdown:** $1048.08 (35.73%)
- **CAGR:** 276.95%
- **Sharpe (annualized):** 2.04
- **Sortino (annualized):** 7.75

---

## Performance (returns & consistency)
| Metric | Result | Definition / Formula |
|---|---:|---|
| Net Profit ($, Adjusted) | $3771.80 | Σ(AdjustedNetPLᵢ) |
| Gross Profit ($) | $1607.50 | Σ(GrossPLᵢ) |
| Total Return (%) | 150.87% | (Net Profit ÷ Initial Capital) × 100 |
| Total Return (Gross %) | 64.30% | (Gross Profit ÷ Initial Capital) × 100 |
| Average Monthly Return | 18.70% | mean(Monthly equity % change) |
| CAGR | 276.95% | (Ending Equity ÷ Initial Capital)^(365 ÷ Days) − 1 |
| Profit Factor (Adjusted) | 1.76 | (Σ profits) ÷ |Σ losses| |
| Win Rate (%) | 49.11% | (# wins ÷ total trades) × 100 |
| Avg Win ($, Adjusted) | $158.57 | mean(AdjustedNetPL | >0) |
| Avg Loss ($, Adjusted) | $-86.84 | mean(AdjustedNetPL | <0) |
| **Avg Win ÷ Avg Loss** | 1.83 | |Avg Win| ÷ |Avg Loss| |
| **Largest Win ÷ Largest Loss** | 12.45 | |Largest Win| ÷ |Largest Loss| |
| Avg Win (pts / contract) | 31.71 | AdjustedNetPL ÷ (point_value × |Qty|) |
| Avg Loss (pts / contract) | -17.37 | AdjustedNetPL ÷ (point_value × |Qty|) |
| Largest Win (pts/contract) | 248.94 | max over trades |
| Largest Loss (pts/contract) | -20.00 | min over trades |
| Expectancy per Trade ($, Adjusted) | $33.68 | mean(AdjustedNetPLᵢ) |

---

## Risk (drawdowns & risk-adjusted)
| Metric | Result | Definition / Formula |
|---|---:|---|
| Max Drawdown ($) | $1048.08 | max(peak − Equity) |
| Max Drawdown (%) | 35.73% | min(Equity ÷ peak − 1) × 100 |
| Average Drawdown (%) | -8.18% | mean(Drawdownₜ) × 100 |
| Recovery Factor | 3.60 | Net Profit ÷ Max DD ($) |
| Sharpe (per-trade proxy) | 0.19 | mean(rᵢ) ÷ stdev(rᵢ) |
| Sortino (per-trade proxy) | 0.73 | mean(rᵢ) ÷ stdev(min(rᵢ,0)) |
| Largest Winning Trade ($) | $1244.71 | max(AdjustedNetPLᵢ) |
| Largest Losing Trade ($) | $-100.00 | min(AdjustedNetPLᵢ) |
| Volatility of Trade Returns | 0.07 | stdev(per-trade returns) |

## Exit Method Breakdown (RTH)
| Exit | Trades | Avg NetPL ($, Adj.) | Profit Factor (Adj.) |
|---|---:|---:|---:|
| Close | 112 | 33.68 | 1.76 |

### Exit Method × Strategy Bucket
| Bucket | Exit | Trades | Avg Adj. NetPL | Sum Adj. NetPL |
|---|---|---:|---:|---:|
| Premarket45 | Close | 112 | 33.68 | 3771.80 |

---

## Long vs Short Breakdown (RTH)
| Direction | Trades | Win Rate | Avg NetPL ($) | Profit Factor |
|---|---:|---:|---:|---:|
| Long | 0 | n/a | n/a | n/a |
| Short | 0 | n/a | n/a | n/a |

### Extra (by direction)
**Longs:** wins=0, losses=0, total win=$0.00, total loss=$0.00, avg win=$n/a, avg loss=$n/a, largest win (pts/contract)=n/a, largest loss (pts/contract)=n/a, return=n/a

**Shorts:** wins=0, losses=0, total win=$0.00, total loss=$0.00, avg win=$n/a, avg loss=$n/a, largest win (pts/contract)=n/a, largest loss (pts/contract)=n/a, return=n/a

---

## Streaks (Adjusted)
- **Max Winning Streak:** 5 trades (index range [48, 52])
- **Max Losing Streak:** 11 trades (index range [62, 72])
- See files: `max_win_streak_trades.csv`, `max_loss_streak_trades.csv`.

---

## Visuals & Tables (investor-friendly)
- **Equity Curve (last 180 days):** `equity_curve_180d.png`
- **Drawdown Curve:** `drawdown_curve.png`
- **Trade P/L Histogram:** `pl_histogram.png`
- **Monthly Performance Table:** `monthly_performance.csv`
- **DOW KPIs:** `dow_kpis.csv` (flags rows with low sample size < 30)
- **Hold-Time KPIs:** `hold_kpis.csv` (flags rows with low sample size < 30)
- **Session KPIs:** `session_kpis.csv` (flags rows with low sample size < 30)
- **Trade Distribution Heatmap (Count):** `heatmap_dow_hour_count.png`
- **Top Worst Trades:** `top_worst_trades.csv` (N=10)

### Monthly Performance Preview (last 6)
| Month | NetPL ($) | Return (%) |
|---|---:|---:|
| 2025-03-31 | 1791.52 | 71.66 |
| 2025-04-30 | 2756.14 | 110.25 |
| 2025-05-31 | -848.08 | -33.92 |
| 2025-06-30 | 467.39 | 18.70 |
| 2025-07-31 | -40.69 | -1.63 |
| 2025-08-31 | 398.15 | 15.93 |
