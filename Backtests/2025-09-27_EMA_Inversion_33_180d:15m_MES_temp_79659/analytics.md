
# Strategy One-Sheet (Trade Data)

**Strategy:** EMA_Inversion_33  
**Instrument:** Unknown  
**Timeframe:** 180d:15m  
**Run Date:** 2025-09-27  
**Session Basis:** New York time (ET). **Metrics Scope:** RTH (09:30–16:00)  
**Initial Capital (float):** $2500  
**Commission (RT / contract):** $4.04  
**Point Value:** $5.00 per point per contract

> **Note:** Stop-loss normalization is enabled: losses below **−$100** are capped at −$100 and the overage is added back as a positive correction (TOS discrepancy fix). All metrics use **AdjustedNetPL**.

---

## Key Performance Indicators (KPI's)
- **Net Profit (after commissions, adjusted):** $8614.42
- **Gross Profit (before commissions):** $7338.75
- **Total Return (Net, adjusted):** 344.58%
- **Total Return (Gross):** 293.55%
- **Win Rate:** 46.53%
- **Profit Factor (Adjusted):** 1.96
- **Max Drawdown:** $649.50 (22.74%)
- **CAGR:** 718.71%
- **Sharpe (annualized):** 4.02
- **Sortino (annualized):** 13.17

---

## Performance (returns & consistency)
| Metric | Result | Definition / Formula |
|---|---:|---|
| Net Profit ($, Adjusted) | $8614.42 | Σ(AdjustedNetPLᵢ) |
| Gross Profit ($) | $7338.75 | Σ(GrossPLᵢ) |
| Total Return (%) | 344.58% | (Net Profit ÷ Initial Capital) × 100 |
| Total Return (Gross %) | 293.55% | (Gross Profit ÷ Initial Capital) × 100 |
| Average Monthly Return | 19.16% | mean(Monthly equity % change) |
| CAGR | 718.71% | (Ending Equity ÷ Initial Capital)^(365 ÷ Days) − 1 |
| Profit Factor (Adjusted) | 1.96 | (Σ profits) ÷ |Σ losses| |
| Win Rate (%) | 46.53% | (# wins ÷ total trades) × 100 |
| Avg Win ($, Adjusted) | $131.40 | mean(AdjustedNetPL | >0) |
| Avg Loss ($, Adjusted) | $-58.40 | mean(AdjustedNetPL | <0) |
| **Avg Win ÷ Avg Loss** | 2.25 | |Avg Win| ÷ |Avg Loss| |
| **Largest Win ÷ Largest Loss** | 6.72 | |Largest Win| ÷ |Largest Loss| |
| Avg Win (pts / contract) | 26.28 | AdjustedNetPL ÷ (point_value × |Qty|) |
| Avg Loss (pts / contract) | -11.68 | AdjustedNetPL ÷ (point_value × |Qty|) |
| Largest Win (pts/contract) | 134.44 | max over trades |
| Largest Loss (pts/contract) | -20.00 | min over trades |
| Expectancy per Trade ($, Adjusted) | $29.91 | mean(AdjustedNetPLᵢ) |

---

## Risk (drawdowns & risk-adjusted)
| Metric | Result | Definition / Formula |
|---|---:|---|
| Max Drawdown ($) | $649.50 | max(peak − Equity) |
| Max Drawdown (%) | 22.74% | min(Equity ÷ peak − 1) × 100 |
| Average Drawdown (%) | -2.94% | mean(Drawdownₜ) × 100 |
| Recovery Factor | 13.26 | Net Profit ÷ Max DD ($) |
| Sharpe (per-trade proxy) | 0.24 | mean(rᵢ) ÷ stdev(rᵢ) |
| Sortino (per-trade proxy) | 0.79 | mean(rᵢ) ÷ stdev(min(rᵢ,0)) |
| Largest Winning Trade ($) | $672.21 | max(AdjustedNetPLᵢ) |
| Largest Losing Trade ($) | $-100.00 | min(AdjustedNetPLᵢ) |
| Volatility of Trade Returns | 0.05 | stdev(per-trade returns) |

## Exit Method Breakdown (RTH)
| Exit | Trades | Avg NetPL ($, Adj.) | Profit Factor (Adj.) |
|---|---:|---:|---:|
| Close | 288 | 29.91 | 1.96 |

### Exit Method × Strategy Bucket
| Bucket | Exit | Trades | Avg Adj. NetPL | Sum Adj. NetPL |
|---|---|---:|---:|---:|
| Other | Close | 288 | 29.91 | 8614.42 |

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
- **Max Winning Streak:** 7 trades (index range [201, 207])
- **Max Losing Streak:** 8 trades (index range [218, 225])
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
| 2025-03-31 | 2232.84 | 89.31 |
| 2025-04-30 | 3608.13 | 144.33 |
| 2025-05-31 | 855.34 | 34.21 |
| 2025-06-30 | -151.09 | -6.04 |
| 2025-07-31 | 2.35 | 0.09 |
| 2025-08-31 | 188.15 | 7.53 |
