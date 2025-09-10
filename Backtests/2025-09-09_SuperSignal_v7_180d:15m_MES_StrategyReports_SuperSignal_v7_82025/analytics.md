
# Strategy One-Sheet (Trade Data)

**Strategy:** SuperSignal_v7  
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
- **Net Profit (after commissions, adjusted):** $6751.72
- **Gross Profit (before commissions):** $5748.75
- **Total Return (Net, adjusted):** 270.07%
- **Total Return (Gross):** 229.95%
- **Win Rate:** 55.23%
- **Profit Factor (Adjusted):** 1.65
- **Max Drawdown:** $1016.02 (17.36%)
- **CAGR:** 532.21%
- **Sharpe (annualized):** 3.63
- **Sortino (annualized):** 9.17

---

## Performance (returns & consistency)
| Metric | Result | Definition / Formula |
|---|---:|---|
| Net Profit ($, Adjusted) | $6751.72 | Σ(AdjustedNetPLᵢ) |
| Gross Profit ($) | $5748.75 | Σ(GrossPLᵢ) |
| Total Return (%) | 270.07% | (Net Profit ÷ Initial Capital) × 100 |
| Total Return (Gross %) | 229.95% | (Gross Profit ÷ Initial Capital) × 100 |
| Average Monthly Return | 19.40% | mean(Monthly equity % change) |
| CAGR | 532.21% | (Ending Equity ÷ Initial Capital)^(365 ÷ Days) − 1 |
| Profit Factor (Adjusted) | 1.65 | (Σ profits) ÷ |Σ losses| |
| Win Rate (%) | 55.23% | (# wins ÷ total trades) × 100 |
| Avg Win ($, Adjusted) | $101.13 | mean(AdjustedNetPL | >0) |
| Avg Loss ($, Adjusted) | $-75.47 | mean(AdjustedNetPL | <0) |
| **Avg Win ÷ Avg Loss** | 1.34 | |Avg Win| ÷ |Avg Loss| |
| **Largest Win ÷ Largest Loss** | 5.26 | |Largest Win| ÷ |Largest Loss| |
| Avg Win (pts / contract) | 20.23 | AdjustedNetPL ÷ (point_value × |Qty|) |
| Avg Loss (pts / contract) | -15.09 | AdjustedNetPL ÷ (point_value × |Qty|) |
| Largest Win (pts/contract) | 105.19 | max over trades |
| Largest Loss (pts/contract) | -20.00 | min over trades |
| Expectancy per Trade ($, Adjusted) | $22.06 | mean(AdjustedNetPLᵢ) |

---

## Risk (drawdowns & risk-adjusted)
| Metric | Result | Definition / Formula |
|---|---:|---|
| Max Drawdown ($) | $1016.02 | max(peak − Equity) |
| Max Drawdown (%) | 17.36% | min(Equity ÷ peak − 1) × 100 |
| Average Drawdown (%) | -4.13% | mean(Drawdownₜ) × 100 |
| Recovery Factor | 6.65 | Net Profit ÷ Max DD ($) |
| Sharpe (per-trade proxy) | 0.21 | mean(rᵢ) ÷ stdev(rᵢ) |
| Sortino (per-trade proxy) | 0.53 | mean(rᵢ) ÷ stdev(min(rᵢ,0)) |
| Largest Winning Trade ($) | $525.96 | max(AdjustedNetPLᵢ) |
| Largest Losing Trade ($) | $-100.00 | min(AdjustedNetPLᵢ) |
| Volatility of Trade Returns | 0.04 | stdev(per-trade returns) |

## Exit Method Breakdown (RTH)
| Exit | Trades | Avg NetPL ($, Adj.) | Profit Factor (Adj.) |
|---|---:|---:|---:|
| Close | 306 | 22.06 | 1.65 |

### Exit Method × Strategy Bucket
| Bucket | Exit | Trades | Avg Adj. NetPL | Sum Adj. NetPL |
|---|---|---:|---:|---:|
| Other | Close | 306 | 22.06 | 6751.72 |

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
- **Max Winning Streak:** 6 trades (index range [27, 32])
- **Max Losing Streak:** 6 trades (index range [266, 271])
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
| 2025-03-31 | 2363.51 | 94.54 |
| 2025-04-30 | 3493.62 | 139.74 |
| 2025-05-31 | 160.43 | 6.42 |
| 2025-06-30 | -511.58 | -20.46 |
| 2025-07-31 | -138.70 | -5.55 |
| 2025-08-31 | 381.21 | 15.25 |
