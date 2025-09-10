
# Strategy One-Sheet (Trade Data)

**Strategy:** SuperSignal_v7_RTH15_v43  
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
- **Net Profit (after commissions, adjusted):** $4256.73
- **Gross Profit (before commissions):** $2626.25
- **Total Return (Net, adjusted):** 170.27%
- **Total Return (Gross):** 105.05%
- **Win Rate:** 49.48%
- **Profit Factor (Adjusted):** 1.94
- **Max Drawdown:** $1239.73 (15.88%)
- **CAGR:** 305.99%
- **Sharpe (annualized):** 2.09
- **Sortino (annualized):** 8.97

---

## Performance (returns & consistency)
| Metric | Result | Definition / Formula |
|---|---:|---|
| Net Profit ($, Adjusted) | $4256.73 | Σ(AdjustedNetPLᵢ) |
| Gross Profit ($) | $2626.25 | Σ(GrossPLᵢ) |
| Total Return (%) | 170.27% | (Net Profit ÷ Initial Capital) × 100 |
| Total Return (Gross %) | 105.05% | (Gross Profit ÷ Initial Capital) × 100 |
| Average Monthly Return | 12.51% | mean(Monthly equity % change) |
| CAGR | 305.99% | (Ending Equity ÷ Initial Capital)^(365 ÷ Days) − 1 |
| Profit Factor (Adjusted) | 1.94 | (Σ profits) ÷ |Σ losses| |
| Win Rate (%) | 49.48% | (# wins ÷ total trades) × 100 |
| Avg Win ($, Adjusted) | $183.12 | mean(AdjustedNetPL | >0) |
| Avg Loss ($, Adjusted) | $-92.51 | mean(AdjustedNetPL | <0) |
| **Avg Win ÷ Avg Loss** | 1.98 | |Avg Win| ÷ |Avg Loss| |
| **Largest Win ÷ Largest Loss** | 12.93 | |Largest Win| ÷ |Largest Loss| |
| Avg Win (pts / contract) | 36.62 | AdjustedNetPL ÷ (point_value × |Qty|) |
| Avg Loss (pts / contract) | -18.50 | AdjustedNetPL ÷ (point_value × |Qty|) |
| Largest Win (pts/contract) | 258.69 | max over trades |
| Largest Loss (pts/contract) | -20.00 | min over trades |
| Expectancy per Trade ($, Adjusted) | $43.88 | mean(AdjustedNetPLᵢ) |

---

## Risk (drawdowns & risk-adjusted)
| Metric | Result | Definition / Formula |
|---|---:|---|
| Max Drawdown ($) | $1239.73 | max(peak − Equity) |
| Max Drawdown (%) | 15.88% | min(Equity ÷ peak − 1) × 100 |
| Average Drawdown (%) | -5.48% | mean(Drawdownₜ) × 100 |
| Recovery Factor | 3.43 | Net Profit ÷ Max DD ($) |
| Sharpe (per-trade proxy) | 0.22 | mean(rᵢ) ÷ stdev(rᵢ) |
| Sortino (per-trade proxy) | 0.92 | mean(rᵢ) ÷ stdev(min(rᵢ,0)) |
| Largest Winning Trade ($) | $1293.46 | max(AdjustedNetPLᵢ) |
| Largest Losing Trade ($) | $-100.00 | min(AdjustedNetPLᵢ) |
| Volatility of Trade Returns | 0.08 | stdev(per-trade returns) |

## Exit Method Breakdown (RTH)
| Exit | Trades | Avg NetPL ($, Adj.) | Profit Factor (Adj.) |
|---|---:|---:|---:|
| Close | 97 | 43.88 | 1.94 |

### Exit Method × Strategy Bucket
| Bucket | Exit | Trades | Avg Adj. NetPL | Sum Adj. NetPL |
|---|---|---:|---:|---:|
| RTH15 | Close | 97 | 43.88 | 4256.73 |

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
- **Max Winning Streak:** 7 trades (index range [35, 41])
- **Max Losing Streak:** 7 trades (index range [66, 72])
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
| 2025-04-30 | 3589.89 | 143.60 |
| 2025-05-31 | -676.74 | -27.07 |
| 2025-06-30 | -361.74 | -14.47 |
| 2025-07-31 | 191.43 | 7.66 |
| 2025-08-31 | 6.43 | 0.26 |
| 2025-09-30 | -200.00 | -8.00 |
