
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
- **Net Profit (after commissions, adjusted):** $2528.38
- **Gross Profit (before commissions):** $1335.00
- **Total Return (Net, adjusted):** 101.14%
- **Total Return (Gross):** 53.40%
- **Win Rate:** 56.00%
- **Profit Factor (Adjusted):** 1.82
- **Max Drawdown:** $763.48 (13.87%)
- **CAGR:** 175.15%
- **Sharpe (annualized):** 2.26
- **Sortino (annualized):** 6.19

---

## Performance (returns & consistency)
| Metric | Result | Definition / Formula |
|---|---:|---|
| Net Profit ($, Adjusted) | $2528.38 | Σ(AdjustedNetPLᵢ) |
| Gross Profit ($) | $1335.00 | Σ(GrossPLᵢ) |
| Total Return (%) | 101.14% | (Net Profit ÷ Initial Capital) × 100 |
| Total Return (Gross %) | 53.40% | (Gross Profit ÷ Initial Capital) × 100 |
| Average Monthly Return | 9.00% | mean(Monthly equity % change) |
| CAGR | 175.15% | (Ending Equity ÷ Initial Capital)^(365 ÷ Days) − 1 |
| Profit Factor (Adjusted) | 1.82 | (Σ profits) ÷ |Σ losses| |
| Win Rate (%) | 56.00% | (# wins ÷ total trades) × 100 |
| Avg Win ($, Adjusted) | $133.22 | mean(AdjustedNetPL | >0) |
| Avg Loss ($, Adjusted) | $-92.94 | mean(AdjustedNetPL | <0) |
| **Avg Win ÷ Avg Loss** | 1.43 | |Avg Win| ÷ |Avg Loss| |
| **Largest Win ÷ Largest Loss** | 4.50 | |Largest Win| ÷ |Largest Loss| |
| Avg Win (pts / contract) | 26.64 | AdjustedNetPL ÷ (point_value × |Qty|) |
| Avg Loss (pts / contract) | -18.59 | AdjustedNetPL ÷ (point_value × |Qty|) |
| Largest Win (pts/contract) | 89.94 | max over trades |
| Largest Loss (pts/contract) | -20.00 | min over trades |
| Expectancy per Trade ($, Adjusted) | $33.71 | mean(AdjustedNetPLᵢ) |

---

## Risk (drawdowns & risk-adjusted)
| Metric | Result | Definition / Formula |
|---|---:|---|
| Max Drawdown ($) | $763.48 | max(peak − Equity) |
| Max Drawdown (%) | 13.87% | min(Equity ÷ peak − 1) × 100 |
| Average Drawdown (%) | -3.93% | mean(Drawdownₜ) × 100 |
| Recovery Factor | 3.31 | Net Profit ÷ Max DD ($) |
| Sharpe (per-trade proxy) | 0.26 | mean(rᵢ) ÷ stdev(rᵢ) |
| Sortino (per-trade proxy) | 0.72 | mean(rᵢ) ÷ stdev(min(rᵢ,0)) |
| Largest Winning Trade ($) | $449.71 | max(AdjustedNetPLᵢ) |
| Largest Losing Trade ($) | $-100.00 | min(AdjustedNetPLᵢ) |
| Volatility of Trade Returns | 0.05 | stdev(per-trade returns) |

## Exit Method Breakdown (RTH)
| Exit | Trades | Avg NetPL ($, Adj.) | Profit Factor (Adj.) |
|---|---:|---:|---:|
| Close | 75 | 33.71 | 1.82 |

### Exit Method × Strategy Bucket
| Bucket | Exit | Trades | Avg Adj. NetPL | Sum Adj. NetPL |
|---|---|---:|---:|---:|
| RTH15 | Close | 75 | 33.71 | 2528.38 |

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
- **Max Winning Streak:** 6 trades (index range [13, 18])
- **Max Losing Streak:** 5 trades (index range [56, 60])
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
| 2025-03-31 | 717.30 | 28.69 |
| 2025-04-30 | 1423.26 | 56.93 |
| 2025-05-31 | -188.95 | -7.56 |
| 2025-06-30 | -274.53 | -10.98 |
| 2025-07-31 | -132.03 | -5.28 |
| 2025-08-31 | 119.51 | 4.78 |
