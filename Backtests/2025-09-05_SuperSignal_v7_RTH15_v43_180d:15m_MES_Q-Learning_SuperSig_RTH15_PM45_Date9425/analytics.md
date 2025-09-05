
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
- **Net Profit (after commissions):** $4510.66
- **Gross Profit (before commissions):** $5302.50
- **Total Return (Net):** 180.43%
- **Total Return (Gross):** 212.10%
- **Win Rate:** 50.00%
- **Profit Factor (Net):** 1.39
- **Max Drawdown:** $2521.98 (27.72%)
- **CAGR:** 327.66%
- **Sharpe (annualized):** 1.70
- **Sortino (annualized):** 4.60

---

## Performance (returns & consistency)
| Metric | Result | Definition / Formula |
|---|---:|---|
| Net Profit ($, Net) | $4510.66 | Σ(NetPLᵢ) |
| Gross Profit ($) | $5302.50 | Σ(GrossPLᵢ) |
| Total Return (%) | 180.43% | (Net Profit ÷ Initial Capital) × 100 |
| Total Return (Gross %) | 212.10% | (Gross Profit ÷ Initial Capital) × 100 |
| Average Monthly Return | 14.99% | mean(Monthly equity % change) |
| CAGR | 327.66% | (Ending Equity ÷ Initial Capital)^(365 ÷ Days) − 1 |
| Profit Factor (Net) | 1.39 | (Σ profits) ÷ |Σ losses| |
| Profit Factor (Gross) | 1.47 | (Σ profits) ÷ |Σ losses| |
| Win Rate (%) | 50.00% | (# wins ÷ total trades) × 100 |
| Avg Win ($, Net) | $163.97 | mean(NetPL | NetPL>0) |
| Avg Loss ($, Net) | $-117.94 | mean(NetPL | NetPL<0) |
| Avg Win ($, Gross) | $166.34 | mean(GrossPL | GrossPL>0) |
| Avg Loss ($, Gross) | $-115.10 | mean(GrossPL | GrossPL<0) |
| **Avg Win ÷ Avg Loss** | 1.39 | |Avg Win| ÷ |Avg Loss| (Net) |
| **Largest Win ÷ Largest Loss** | 3.99 | |Largest Win| ÷ |Largest Loss| (Net) |
| Avg Win (pts / contract) | 32.79 | NetPL ÷ (point_value × |Qty|) |
| Avg Loss (pts / contract) | -23.59 | NetPL ÷ (point_value × |Qty|) |
| Expectancy per Trade ($, Net) | $23.01 | mean(NetPLᵢ) |
| Expectancy per Trade ($, Gross) | $27.05 | mean(GrossPLᵢ) |

---

## Risk (drawdowns & risk-adjusted)
| Metric | Result | Definition / Formula |
|---|---:|---|
| Max Drawdown ($) | $2521.98 | max(peak − Equity) |
| Max Drawdown (%) | 27.72% | min(Equity ÷ peak − 1) × 100 |
| Average Drawdown (%) | -11.33% | mean(Drawdownₜ) × 100 |
| Recovery Factor | 1.79 | Net Profit ÷ Max DD ($) |
| Sharpe (per-trade proxy) | 0.12 | mean(rᵢ) ÷ stdev(rᵢ) |
| Sortino (per-trade proxy) | 0.33 | mean(rᵢ) ÷ stdev(min(rᵢ,0)) |
| Largest Winning Trade ($) | $1293.46 | max(NetPLᵢ) |
| Largest Losing Trade ($) | $-324.04 | min(NetPLᵢ) |
| Volatility of Trade Returns | 0.07 | stdev(per-trade returns) |

## Exit Method Breakdown (RTH)
| Exit | Trades | Avg NetPL ($) | Profit Factor |
|---|---:|---:|---:|
| Close | 196 | 23.01 | 1.39 |

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
| 2025-04-30 | 4349.58 | 173.98 |
| 2025-05-31 | -2021.00 | -80.84 |
| 2025-06-30 | -168.97 | -6.76 |
| 2025-07-31 | -3.59 | -0.14 |
| 2025-08-31 | 388.33 | 15.53 |
| 2025-09-30 | -282.70 | -11.31 |
