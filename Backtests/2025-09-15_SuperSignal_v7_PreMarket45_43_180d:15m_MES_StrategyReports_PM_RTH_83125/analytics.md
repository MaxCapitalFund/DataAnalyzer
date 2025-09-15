
# Strategy One-Sheet (Trade Data)

**Strategy:** SuperSignal_v7_PreMarket45_43  
**Instrument:** Unknown  
**Timeframe:** 180d:15m  
**Run Date:** 2025-09-15  
**Session Basis:** New York time (ET). **Metrics Scope:** RTH (09:30–16:00)  
**Initial Capital (float):** $2500  
**Commission (RT / contract):** $4.04  
**Point Value:** $5.00 per point per contract

> **Note:** Stop-loss normalization is enabled: losses below **−$100** are capped at −$100 and the overage is added back as a positive correction (TOS discrepancy fix). All metrics use **AdjustedNetPL**.

---

## Key Performance Indicators (KPI's)
- **Net Profit (after commissions, adjusted):** $5478.75
- **Gross Profit (before commissions):** $2296.25
- **Total Return (Net, adjusted):** 219.15%
- **Total Return (Gross):** 91.85%
- **Win Rate:** 50.57%
- **Profit Factor (Adjusted):** 1.73
- **Max Drawdown:** $825.11 (22.28%)
- **CAGR:** 429.96%
- **Sharpe (annualized):** 2.63
- **Sortino (annualized):** 9.00

---

## Performance (returns & consistency)
| Metric | Result | Definition / Formula |
|---|---:|---|
| Net Profit ($, Adjusted) | $5478.75 | Σ(AdjustedNetPLᵢ) |
| Gross Profit ($) | $2296.25 | Σ(GrossPLᵢ) |
| Total Return (%) | 219.15% | (Net Profit ÷ Initial Capital) × 100 |
| Total Return (Gross %) | 91.85% | (Gross Profit ÷ Initial Capital) × 100 |
| Average Monthly Return | 20.33% | mean(Monthly equity % change) |
| CAGR | 429.96% | (Ending Equity ÷ Initial Capital)^(365 ÷ Days) − 1 |
| Profit Factor (Adjusted) | 1.73 | (Σ profits) ÷ |Σ losses| |
| Win Rate (%) | 50.57% | (# wins ÷ total trades) × 100 |
| Avg Win ($, Adjusted) | $147.32 | mean(AdjustedNetPL | >0) |
| Avg Loss ($, Adjusted) | $-87.04 | mean(AdjustedNetPL | <0) |
| **Avg Win ÷ Avg Loss** | 1.69 | |Avg Win| ÷ |Avg Loss| |
| **Largest Win ÷ Largest Loss** | 12.45 | |Largest Win| ÷ |Largest Loss| |
| Avg Win (pts / contract) | 29.46 | AdjustedNetPL ÷ (point_value × |Qty|) |
| Avg Loss (pts / contract) | -17.41 | AdjustedNetPL ÷ (point_value × |Qty|) |
| Largest Win (pts/contract) | 248.94 | max over trades |
| Largest Loss (pts/contract) | -20.00 | min over trades |
| Expectancy per Trade ($, Adjusted) | $31.49 | mean(AdjustedNetPLᵢ) |

---

## Risk (drawdowns & risk-adjusted)
| Metric | Result | Definition / Formula |
|---|---:|---|
| Max Drawdown ($) | $825.11 | max(peak − Equity) |
| Max Drawdown (%) | 22.28% | min(Equity ÷ peak − 1) × 100 |
| Average Drawdown (%) | -5.22% | mean(Drawdownₜ) × 100 |
| Recovery Factor | 6.64 | Net Profit ÷ Max DD ($) |
| Sharpe (per-trade proxy) | 0.20 | mean(rᵢ) ÷ stdev(rᵢ) |
| Sortino (per-trade proxy) | 0.69 | mean(rᵢ) ÷ stdev(min(rᵢ,0)) |
| Largest Winning Trade ($) | $1244.71 | max(AdjustedNetPLᵢ) |
| Largest Losing Trade ($) | $-100.00 | min(AdjustedNetPLᵢ) |
| Volatility of Trade Returns | 0.06 | stdev(per-trade returns) |

## Exit Method Breakdown (RTH)
| Exit | Trades | Avg NetPL ($, Adj.) | Profit Factor (Adj.) |
|---|---:|---:|---:|
| Close | 174 | 31.49 | 1.73 |

### Exit Method × Strategy Bucket
| Bucket | Exit | Trades | Avg Adj. NetPL | Sum Adj. NetPL |
|---|---|---:|---:|---:|
| Premarket45 | Close | 174 | 31.49 | 5478.75 |

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
- **Max Winning Streak:** 7 trades (index range [71, 77])
- **Max Losing Streak:** 7 trades (index range [78, 84])
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
- **Top Best Trades:** `top_best_trades.csv` (N=10)

### Monthly Performance Preview (last 6)
| Month | NetPL ($) | Return (%) |
|---|---:|---:|
| 2025-03-31 | 2508.82 | 100.35 |
| 2025-04-30 | 3395.27 | 135.81 |
| 2025-05-31 | -725.11 | -29.00 |
| 2025-06-30 | 392.86 | 15.71 |
| 2025-07-31 | -189.93 | -7.60 |
| 2025-08-31 | 256.03 | 10.24 |
