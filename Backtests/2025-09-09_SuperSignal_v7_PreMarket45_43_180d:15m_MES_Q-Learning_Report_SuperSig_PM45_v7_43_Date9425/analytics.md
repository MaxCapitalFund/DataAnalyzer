
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
- **Net Profit (after commissions, adjusted):** $4501.89
- **Gross Profit (before commissions):** $3775.00
- **Total Return (Net, adjusted):** 180.08%
- **Total Return (Gross):** 151.00%
- **Win Rate:** 52.89%
- **Profit Factor (Adjusted):** 1.88
- **Max Drawdown:** $986.45 (14.40%)
- **CAGR:** 326.90%
- **Sharpe (annualized):** 2.78
- **Sortino (annualized):** 8.61

---

## Performance (returns & consistency)
| Metric | Result | Definition / Formula |
|---|---:|---|
| Net Profit ($, Adjusted) | $4501.89 | Σ(AdjustedNetPLᵢ) |
| Gross Profit ($) | $3775.00 | Σ(GrossPLᵢ) |
| Total Return (%) | 180.08% | (Net Profit ÷ Initial Capital) × 100 |
| Total Return (Gross %) | 151.00% | (Gross Profit ÷ Initial Capital) × 100 |
| Average Monthly Return | 14.77% | mean(Monthly equity % change) |
| CAGR | 326.90% | (Ending Equity ÷ Initial Capital)^(365 ÷ Days) − 1 |
| Profit Factor (Adjusted) | 1.88 | (Σ profits) ÷ |Σ losses| |
| Win Rate (%) | 52.89% | (# wins ÷ total trades) × 100 |
| Avg Win ($, Adjusted) | $150.49 | mean(AdjustedNetPL | >0) |
| Avg Loss ($, Adjusted) | $-89.99 | mean(AdjustedNetPL | <0) |
| **Avg Win ÷ Avg Loss** | 1.67 | |Avg Win| ÷ |Avg Loss| |
| **Largest Win ÷ Largest Loss** | 7.35 | |Largest Win| ÷ |Largest Loss| |
| Avg Win (pts / contract) | 30.10 | AdjustedNetPL ÷ (point_value × |Qty|) |
| Avg Loss (pts / contract) | -18.00 | AdjustedNetPL ÷ (point_value × |Qty|) |
| Largest Win (pts/contract) | 146.94 | max over trades |
| Largest Loss (pts/contract) | -20.00 | min over trades |
| Expectancy per Trade ($, Adjusted) | $37.21 | mean(AdjustedNetPLᵢ) |

---

## Risk (drawdowns & risk-adjusted)
| Metric | Result | Definition / Formula |
|---|---:|---|
| Max Drawdown ($) | $986.45 | max(peak − Equity) |
| Max Drawdown (%) | 14.40% | min(Equity ÷ peak − 1) × 100 |
| Average Drawdown (%) | -3.79% | mean(Drawdownₜ) × 100 |
| Recovery Factor | 4.56 | Net Profit ÷ Max DD ($) |
| Sharpe (per-trade proxy) | 0.26 | mean(rᵢ) ÷ stdev(rᵢ) |
| Sortino (per-trade proxy) | 0.79 | mean(rᵢ) ÷ stdev(min(rᵢ,0)) |
| Largest Winning Trade ($) | $734.71 | max(AdjustedNetPLᵢ) |
| Largest Losing Trade ($) | $-100.00 | min(AdjustedNetPLᵢ) |
| Volatility of Trade Returns | 0.06 | stdev(per-trade returns) |

## Exit Method Breakdown (RTH)
| Exit | Trades | Avg NetPL ($, Adj.) | Profit Factor (Adj.) |
|---|---:|---:|---:|
| Close | 121 | 37.21 | 1.88 |

### Exit Method × Strategy Bucket
| Bucket | Exit | Trades | Avg Adj. NetPL | Sum Adj. NetPL |
|---|---|---:|---:|---:|
| Premarket45 | Close | 121 | 37.21 | 4501.89 |

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
- **Max Winning Streak:** 6 trades (index range [10, 15])
- **Max Losing Streak:** 5 trades (index range [75, 79])
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
| 2025-04-30 | 2064.89 | 82.60 |
| 2025-05-31 | -786.45 | -31.46 |
| 2025-06-30 | 427.01 | 17.08 |
| 2025-07-31 | 37.48 | 1.50 |
| 2025-08-31 | 733.44 | 29.34 |
| 2025-09-30 | -159.33 | -6.37 |
