
# Strategy One-Sheet (Trade Data)

**Strategy:** EMA_Inversion_33  
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
- **Net Profit (after commissions, adjusted):** $-3530.97
- **Gross Profit (before commissions):** $-4497.50
- **Total Return (Net, adjusted):** -141.24%
- **Total Return (Gross):** -179.90%
- **Win Rate:** 18.82%
- **Profit Factor (Adjusted):** 0.15
- **Max Drawdown:** $3430.97 (142.96%)
- **CAGR:** n/a
- **Sharpe (annualized):** -7.13
- **Sortino (annualized):** -9.71

---

## Performance (returns & consistency)
| Metric | Result | Definition / Formula |
|---|---:|---|
| Net Profit ($, Adjusted) | $-3530.97 | Σ(AdjustedNetPLᵢ) |
| Gross Profit ($) | $-4497.50 | Σ(GrossPLᵢ) |
| Total Return (%) | -141.24% | (Net Profit ÷ Initial Capital) × 100 |
| Total Return (Gross %) | -179.90% | (Gross Profit ÷ Initial Capital) × 100 |
| Average Monthly Return | 55.99% | mean(Monthly equity % change) |
| CAGR | n/a | (Ending Equity ÷ Initial Capital)^(365 ÷ Days) − 1 |
| Profit Factor (Adjusted) | 0.15 | (Σ profits) ÷ |Σ losses| |
| Win Rate (%) | 18.82% | (# wins ÷ total trades) × 100 |
| Avg Win ($, Adjusted) | $38.77 | mean(AdjustedNetPL | >0) |
| Avg Loss ($, Adjusted) | $-60.16 | mean(AdjustedNetPL | <0) |
| **Avg Win ÷ Avg Loss** | 0.64 | |Avg Win| ÷ |Avg Loss| |
| **Largest Win ÷ Largest Loss** | 1.92 | |Largest Win| ÷ |Largest Loss| |
| Avg Win (pts / contract) | 7.75 | AdjustedNetPL ÷ (point_value × |Qty|) |
| Avg Loss (pts / contract) | -12.03 | AdjustedNetPL ÷ (point_value × |Qty|) |
| Largest Win (pts/contract) | 38.44 | max over trades |
| Largest Loss (pts/contract) | -20.00 | min over trades |
| Expectancy per Trade ($, Adjusted) | $-41.54 | mean(AdjustedNetPLᵢ) |

---

## Risk (drawdowns & risk-adjusted)
| Metric | Result | Definition / Formula |
|---|---:|---|
| Max Drawdown ($) | $3430.97 | max(peak − Equity) |
| Max Drawdown (%) | 142.96% | min(Equity ÷ peak − 1) × 100 |
| Average Drawdown (%) | -82.69% | mean(Drawdownₜ) × 100 |
| Recovery Factor | -1.03 | Net Profit ÷ Max DD ($) |
| Sharpe (per-trade proxy) | -0.78 | mean(rᵢ) ÷ stdev(rᵢ) |
| Sortino (per-trade proxy) | -1.07 | mean(rᵢ) ÷ stdev(min(rᵢ,0)) |
| Largest Winning Trade ($) | $192.21 | max(AdjustedNetPLᵢ) |
| Largest Losing Trade ($) | $-100.00 | min(AdjustedNetPLᵢ) |
| Volatility of Trade Returns | 0.02 | stdev(per-trade returns) |

## Exit Method Breakdown (RTH)
| Exit | Trades | Avg NetPL ($, Adj.) | Profit Factor (Adj.) |
|---|---:|---:|---:|
| Close | 85 | -41.54 | 0.15 |

### Exit Method × Strategy Bucket
| Bucket | Exit | Trades | Avg Adj. NetPL | Sum Adj. NetPL |
|---|---|---:|---:|---:|
| Other | Close | 85 | -41.54 | -3530.97 |

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
- **Max Winning Streak:** 2 trades (index range [38, 39])
- **Max Losing Streak:** 16 trades (index range [0, 15])
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
| 2025-03-31 | -649.15 | -25.97 |
| 2025-04-30 | -564.53 | -22.58 |
| 2025-05-31 | -289.24 | -11.57 |
| 2025-06-30 | -321.07 | -12.84 |
| 2025-07-31 | -112.90 | -4.52 |
| 2025-08-31 | -251.74 | -10.07 |
