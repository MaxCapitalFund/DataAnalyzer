
# Strategy One-Sheet (Trade Data)

**Strategy:** SuperSignal_v7_RTH15_v43  
**Instrument:** Unknown  
**Timeframe:** 180d:15m  
**Run Date:** 2025-09-05  
**Session Basis:** New York time (ET). **Metrics Scope:** RTH (09:30–16:00)  
**Initial Capital (float):** $2500  
**Commission (RT / contract):** $4.04  
**Point Value:** $5.00 per point per contract

---

## Key Performance Indicators (KPI's)
- **Net Profit (after commissions):** $2234.37
- **Gross Profit (before commissions):** $2626.25
- **Total Return (Net):** 89.37%
- **Total Return (Gross):** 105.05%
- **Win Rate:** 49.48%
- **Profit Factor (Net):** 1.34
- **Max Drawdown:** $2097.45 (31.29%)
- **CAGR:** 145.94%
- **Sharpe (annualized):** 1.01
- **Sortino (annualized):** 2.83

---

## Performance (returns & consistency)
| Metric | Result | Definition / Formula |
|---|---:|---|
| Net Profit ($, Net) | $2234.37 | Σ(NetPLᵢ) |
| Gross Profit ($) | $2626.25 | Σ(GrossPLᵢ) |
| Total Return (%) | 89.37% | (Net Profit ÷ Initial Capital) × 100 |
| Total Return (Gross %) | 105.05% | (Gross Profit ÷ Initial Capital) × 100 |
| Average Monthly Return | 9.50% | mean(Monthly equity % change) |
| CAGR | 145.94% | (Ending Equity ÷ Initial Capital)^(365 ÷ Days) − 1 |
| Profit Factor (Net) | 1.34 | (Σ profits) ÷ |Σ losses| |
| Profit Factor (Gross) | 1.41 | (Σ profits) ÷ |Σ losses| |
| Win Rate (%) | 49.48% | (# wins ÷ total trades) × 100 |
| Avg Win ($, Net) | $183.12 | mean(NetPL | NetPL>0) |
| Avg Loss ($, Net) | $-133.78 | mean(NetPL | NetPL<0) |
| Avg Win ($, Gross) | $187.16 | mean(GrossPL | GrossPL>0) |
| Avg Loss ($, Gross) | $-129.74 | mean(GrossPL | GrossPL<0) |
| **Avg Win ÷ Avg Loss** | 1.37 | |Avg Win| ÷ |Avg Loss| (Net) |
| **Largest Win ÷ Largest Loss** | 3.99 | |Largest Win| ÷ |Largest Loss| (Net) |
| Avg Win (pts / contract) | 36.62 | NetPL ÷ (point_value × |Qty|) |
| Avg Loss (pts / contract) | -26.76 | NetPL ÷ (point_value × |Qty|) |
| Expectancy per Trade ($, Net) | $23.03 | mean(NetPLᵢ) |
| Expectancy per Trade ($, Gross) | $27.07 | mean(GrossPLᵢ) |

---

## Risk (drawdowns & risk-adjusted)
| Metric | Result | Definition / Formula |
|---|---:|---|
| Max Drawdown ($) | $2097.45 | max(peak − Equity) |
| Max Drawdown (%) | 31.29% | min(Equity ÷ peak − 1) × 100 |
| Average Drawdown (%) | -12.98% | mean(Drawdownₜ) × 100 |
| Recovery Factor | 1.07 | Net Profit ÷ Max DD ($) |
| Sharpe (per-trade proxy) | 0.10 | mean(rᵢ) ÷ stdev(rᵢ) |
| Sortino (per-trade proxy) | 0.29 | mean(rᵢ) ÷ stdev(min(rᵢ,0)) |
| Largest Winning Trade ($) | $1293.46 | max(NetPLᵢ) |
| Largest Losing Trade ($) | $-324.04 | min(NetPLᵢ) |
| Volatility of Trade Returns | 0.09 | stdev(per-trade returns) |

## Exit Method Breakdown (RTH)
| Exit | Trades | Avg NetPL ($) | Profit Factor |
|---|---:|---:|---:|
| Close | 97 | 23.03 | 1.34 |

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
| 2025-04-30 | 3314.02 | 132.56 |
| 2025-05-31 | -1252.14 | -50.09 |
| 2025-06-30 | -639.73 | -25.59 |
| 2025-07-31 | 179.89 | 7.20 |
| 2025-08-31 | -46.65 | -1.87 |
| 2025-09-30 | -210.58 | -8.42 |
