
# Strategy One-Sheet (Trade Data)

**Strategy:** SuperSignal_v7_RTH15_v43  
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
- **Net Profit (after commissions, adjusted):** $7369.89
- **Gross Profit (before commissions):** $5302.50
- **Total Return (Net, adjusted):** 294.80%
- **Total Return (Gross):** 212.10%
- **Win Rate:** 50.00%
- **Profit Factor (Adjusted):** 1.85
- **Max Drawdown:** $1656.85 (15.31%)
- **CAGR:** 592.54%
- **Sharpe (annualized):** 3.01
- **Sortino (annualized):** 11.12

---

## Performance (returns & consistency)
| Metric | Result | Definition / Formula |
|---|---:|---|
| Net Profit ($, Adjusted) | $7369.89 | Σ(AdjustedNetPLᵢ) |
| Gross Profit ($) | $5302.50 | Σ(GrossPLᵢ) |
| Total Return (%) | 294.80% | (Net Profit ÷ Initial Capital) × 100 |
| Total Return (Gross %) | 212.10% | (Gross Profit ÷ Initial Capital) × 100 |
| Average Monthly Return | 18.09% | mean(Monthly equity % change) |
| CAGR | 592.54% | (Ending Equity ÷ Initial Capital)^(365 ÷ Days) − 1 |
| Profit Factor (Adjusted) | 1.85 | (Σ profits) ÷ |Σ losses| |
| Win Rate (%) | 50.00% | (# wins ÷ total trades) × 100 |
| Avg Win ($, Adjusted) | $163.97 | mean(AdjustedNetPL | >0) |
| Avg Loss ($, Adjusted) | $-88.77 | mean(AdjustedNetPL | <0) |
| **Avg Win ÷ Avg Loss** | 1.85 | |Avg Win| ÷ |Avg Loss| |
| **Largest Win ÷ Largest Loss** | 12.93 | |Largest Win| ÷ |Largest Loss| |
| Avg Win (pts / contract) | 32.79 | AdjustedNetPL ÷ (point_value × |Qty|) |
| Avg Loss (pts / contract) | -17.75 | AdjustedNetPL ÷ (point_value × |Qty|) |
| Largest Win (pts/contract) | 258.69 | max over trades |
| Largest Loss (pts/contract) | -20.00 | min over trades |
| Expectancy per Trade ($, Adjusted) | $37.60 | mean(AdjustedNetPLᵢ) |

---

## Risk (drawdowns & risk-adjusted)
| Metric | Result | Definition / Formula |
|---|---:|---|
| Max Drawdown ($) | $1656.85 | max(peak − Equity) |
| Max Drawdown (%) | 15.31% | min(Equity ÷ peak − 1) × 100 |
| Average Drawdown (%) | -5.32% | mean(Drawdownₜ) × 100 |
| Recovery Factor | 4.45 | Net Profit ÷ Max DD ($) |
| Sharpe (per-trade proxy) | 0.22 | mean(rᵢ) ÷ stdev(rᵢ) |
| Sortino (per-trade proxy) | 0.81 | mean(rᵢ) ÷ stdev(min(rᵢ,0)) |
| Largest Winning Trade ($) | $1293.46 | max(AdjustedNetPLᵢ) |
| Largest Losing Trade ($) | $-100.00 | min(AdjustedNetPLᵢ) |
| Volatility of Trade Returns | 0.07 | stdev(per-trade returns) |

## Exit Method Breakdown (RTH)
| Exit | Trades | Avg NetPL ($, Adj.) | Profit Factor (Adj.) |
|---|---:|---:|---:|
| Close | 196 | 37.60 | 1.85 |

### Exit Method × Strategy Bucket
| Bucket | Exit | Trades | Avg Adj. NetPL | Sum Adj. NetPL |
|---|---|---:|---:|---:|
| RTH15 | Close | 196 | 37.60 | 7369.89 |

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
- **Max Winning Streak:** 8 trades (index range [18, 25])
- **Max Losing Streak:** 8 trades (index range [104, 111])
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
| 2025-04-30 | 4746.90 | 189.88 |
| 2025-05-31 | -1372.23 | -54.89 |
| 2025-06-30 | 93.06 | 3.72 |
| 2025-07-31 | 83.24 | 3.33 |
| 2025-08-31 | 513.53 | 20.54 |
| 2025-09-30 | -272.12 | -10.88 |
