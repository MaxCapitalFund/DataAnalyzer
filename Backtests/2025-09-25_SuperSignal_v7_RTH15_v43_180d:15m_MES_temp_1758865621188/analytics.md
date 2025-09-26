
# Strategy One-Sheet (Trade Data)

**Strategy:** SuperSignal_v7_RTH15_v43  
**Instrument:** Unknown  
**Timeframe:** 180d:15m  
**Run Date:** 2025-09-25  
**Session Basis:** New York time (ET). **Metrics Scope:** RTH (09:30–16:00)  
**Initial Capital (float):** $2500  
**Commission (RT / contract):** $4.04  
**Point Value:** $5.00 per point per contract

> **Note:** Stop-loss normalization is enabled: losses below **−$100** are capped at −$100 and the overage is added back as a positive correction (TOS discrepancy fix). All metrics use **AdjustedNetPL**.

---

## Key Performance Indicators (KPI's)
- **Net Profit (after commissions, adjusted):** $3501.82
- **Gross Profit (before commissions):** $1871.25
- **Total Return (Net, adjusted):** 140.07%
- **Total Return (Gross):** 74.85%
- **Win Rate:** 46.53%
- **Profit Factor (Adjusted):** 1.71
- **Max Drawdown:** $1402.36 (19.11%)
- **CAGR:** 241.93%
- **Sharpe (annualized):** 1.72
- **Sortino (annualized):** 7.32

---

## Performance (returns & consistency)
| Metric | Result | Definition / Formula |
|---|---:|---|
| Net Profit ($, Adjusted) | $3501.82 | Σ(AdjustedNetPLᵢ) |
| Gross Profit ($) | $1871.25 | Σ(GrossPLᵢ) |
| Total Return (%) | 140.07% | (Net Profit ÷ Initial Capital) × 100 |
| Total Return (Gross %) | 74.85% | (Gross Profit ÷ Initial Capital) × 100 |
| Average Monthly Return | 14.52% | mean(Monthly equity % change) |
| CAGR | 241.93% | (Ending Equity ÷ Initial Capital)^(365 ÷ Days) − 1 |
| Profit Factor (Adjusted) | 1.71 | (Σ profits) ÷ |Σ losses| |
| Win Rate (%) | 46.53% | (# wins ÷ total trades) × 100 |
| Avg Win ($, Adjusted) | $178.78 | mean(AdjustedNetPL | >0) |
| Avg Loss ($, Adjusted) | $-90.76 | mean(AdjustedNetPL | <0) |
| **Avg Win ÷ Avg Loss** | 1.97 | |Avg Win| ÷ |Avg Loss| |
| **Largest Win ÷ Largest Loss** | 12.93 | |Largest Win| ÷ |Largest Loss| |
| Avg Win (pts / contract) | 35.76 | AdjustedNetPL ÷ (point_value × |Qty|) |
| Avg Loss (pts / contract) | -18.15 | AdjustedNetPL ÷ (point_value × |Qty|) |
| Largest Win (pts/contract) | 258.69 | max over trades |
| Largest Loss (pts/contract) | -20.00 | min over trades |
| Expectancy per Trade ($, Adjusted) | $34.67 | mean(AdjustedNetPLᵢ) |

---

## Risk (drawdowns & risk-adjusted)
| Metric | Result | Definition / Formula |
|---|---:|---|
| Max Drawdown ($) | $1402.36 | max(peak − Equity) |
| Max Drawdown (%) | 19.11% | min(Equity ÷ peak − 1) × 100 |
| Average Drawdown (%) | -7.38% | mean(Drawdownₜ) × 100 |
| Recovery Factor | 2.50 | Net Profit ÷ Max DD ($) |
| Sharpe (per-trade proxy) | 0.17 | mean(rᵢ) ÷ stdev(rᵢ) |
| Sortino (per-trade proxy) | 0.74 | mean(rᵢ) ÷ stdev(min(rᵢ,0)) |
| Largest Winning Trade ($) | $1293.46 | max(AdjustedNetPLᵢ) |
| Largest Losing Trade ($) | $-100.00 | min(AdjustedNetPLᵢ) |
| Volatility of Trade Returns | 0.08 | stdev(per-trade returns) |

## Exit Method Breakdown (RTH)
| Exit | Trades | Avg NetPL ($, Adj.) | Profit Factor (Adj.) |
|---|---:|---:|---:|
| Close | 101 | 34.67 | 1.71 |

### Exit Method × Strategy Bucket
| Bucket | Exit | Trades | Avg Adj. NetPL | Sum Adj. NetPL |
|---|---|---:|---:|---:|
| RTH15 | Close | 101 | 34.67 | 3501.82 |

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
- **Max Winning Streak:** 7 trades (index range [28, 34])
- **Max Losing Streak:** 7 trades (index range [59, 65])
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
| 2025-04-30 | 3589.89 | 143.60 |
| 2025-05-31 | -676.74 | -27.07 |
| 2025-06-30 | -361.74 | -14.47 |
| 2025-07-31 | 191.43 | 7.66 |
| 2025-08-31 | 6.43 | 0.26 |
| 2025-09-30 | -490.40 | -19.62 |
