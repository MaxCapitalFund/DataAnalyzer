
# Strategy One-Sheet (Trade Data)

**Strategy:** SuperSignal_v7  
**Instrument:** Unknown  
**Timeframe:** 180d:15m  
**Run Date:** 2025-09-05  
**Session Basis:** New York time (ET). **Metrics Scope:** RTH (09:30–16:00)  
**Initial Capital (float):** $2500  
**Commission (RT / contract):** $4.04  
**Point Value:** $5.00 per point per contract

---

## Key Performance Indicators (KPI's)
- **Net Profit (after commissions):** $4512.51
- **Gross Profit (before commissions):** $5748.75
- **Total Return (Net):** 180.50%
- **Total Return (Gross):** 229.95%
- **Win Rate:** 55.23%
- **Profit Factor (Net):** 1.36
- **Max Drawdown:** $1223.76 (18.56%)
- **CAGR:** 327.82%
- **Sharpe (annualized):** 2.19
- **Sortino (annualized):** 4.34

---

## Performance (returns & consistency)
| Metric | Result | Definition / Formula |
|---|---:|---|
| Net Profit ($, Net) | $4512.51 | Σ(NetPLᵢ) |
| Gross Profit ($) | $5748.75 | Σ(GrossPLᵢ) |
| Total Return (%) | 180.50% | (Net Profit ÷ Initial Capital) × 100 |
| Total Return (Gross %) | 229.95% | (Gross Profit ÷ Initial Capital) × 100 |
| Average Monthly Return | 15.48% | mean(Monthly equity % change) |
| CAGR | 327.82% | (Ending Equity ÷ Initial Capital)^(365 ÷ Days) − 1 |
| Profit Factor (Net) | 1.36 | (Σ profits) ÷ |Σ losses| |
| Profit Factor (Gross) | 1.48 | (Σ profits) ÷ |Σ losses| |
| Win Rate (%) | 55.23% | (# wins ÷ total trades) × 100 |
| Avg Win ($, Net) | $101.13 | mean(NetPL | NetPL>0) |
| Avg Loss ($, Net) | $-91.81 | mean(NetPL | NetPL<0) |
| Avg Win ($, Gross) | $103.98 | mean(GrossPL | GrossPL>0) |
| Avg Loss ($, Gross) | $-89.12 | mean(GrossPL | GrossPL<0) |
| **Avg Win ÷ Avg Loss** | 1.10 | |Avg Win| ÷ |Avg Loss| (Net) |
| **Largest Win ÷ Largest Loss** | 1.23 | |Largest Win| ÷ |Largest Loss| (Net) |
| Avg Win (pts / contract) | 20.23 | NetPL ÷ (point_value × |Qty|) |
| Avg Loss (pts / contract) | -18.36 | NetPL ÷ (point_value × |Qty|) |
| Expectancy per Trade ($, Net) | $14.75 | mean(NetPLᵢ) |
| Expectancy per Trade ($, Gross) | $18.79 | mean(GrossPLᵢ) |

---

## Risk (drawdowns & risk-adjusted)
| Metric | Result | Definition / Formula |
|---|---:|---|
| Max Drawdown ($) | $1223.76 | max(peak − Equity) |
| Max Drawdown (%) | 18.56% | min(Equity ÷ peak − 1) × 100 |
| Average Drawdown (%) | -6.67% | mean(Drawdownₜ) × 100 |
| Recovery Factor | 3.69 | Net Profit ÷ Max DD ($) |
| Sharpe (per-trade proxy) | 0.13 | mean(rᵢ) ÷ stdev(rᵢ) |
| Sortino (per-trade proxy) | 0.25 | mean(rᵢ) ÷ stdev(min(rᵢ,0)) |
| Largest Winning Trade ($) | $525.96 | max(NetPLᵢ) |
| Largest Losing Trade ($) | $-429.04 | min(NetPLᵢ) |
| Volatility of Trade Returns | 0.05 | stdev(per-trade returns) |

## Exit Method Breakdown (RTH)
| Exit | Trades | Avg NetPL ($) | Profit Factor |
|---|---:|---:|---:|
| Close | 306 | 14.75 | 1.36 |

---

## Long vs Short Breakdown (RTH)
| Direction | Trades | Win Rate | Avg NetPL ($) | Profit Factor |
|---|---:|---:|---:|---:|
| Long | 0 | n/a | n/a | n/a |
| Short | 0 | n/a | n/a | n/a |

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

### Monthly Performance Preview (last 6)
| Month | NetPL ($) | Return (%) |
|---|---:|---:|
| 2025-03-31 | 2064.27 | 82.57 |
| 2025-04-30 | 2467.64 | 98.71 |
| 2025-05-31 | -124.39 | -4.98 |
| 2025-06-30 | -628.61 | -25.14 |
| 2025-07-31 | -183.99 | -7.36 |
| 2025-08-31 | 372.17 | 14.89 |
