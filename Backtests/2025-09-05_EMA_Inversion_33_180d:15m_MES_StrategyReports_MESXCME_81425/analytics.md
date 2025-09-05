
# Strategy One-Sheet (Trade Data)

**Strategy:** EMA_Inversion_33  
**Instrument:** Unknown  
**Timeframe:** 180d:15m  
**Run Date:** 2025-09-05  
**Session Basis:** New York time (ET). **Metrics Scope:** RTH (09:30–16:00)  
**Initial Capital (float):** $2500  
**Commission (RT / contract):** $4.04  
**Point Value:** $5.00 per point per contract

---

## Key Performance Indicators (KPI's)
- **Net Profit (after commissions):** $6175.23
- **Gross Profit (before commissions):** $7338.75
- **Total Return (Net):** 247.01%
- **Total Return (Gross):** 293.55%
- **Win Rate:** 46.53%
- **Profit Factor (Net):** 1.54
- **Max Drawdown:** $959.70 (23.42%)
- **CAGR:** 477.41%
- **Sharpe (annualized):** 2.60
- **Sortino (annualized):** 5.77

---

## Performance (returns & consistency)
| Metric | Result | Definition / Formula |
|---|---:|---|
| Net Profit ($, Net) | $6175.23 | Σ(NetPLᵢ) |
| Gross Profit ($) | $7338.75 | Σ(GrossPLᵢ) |
| Total Return (%) | 247.01% | (Net Profit ÷ Initial Capital) × 100 |
| Total Return (Gross %) | 293.55% | (Gross Profit ÷ Initial Capital) × 100 |
| Average Monthly Return | 16.06% | mean(Monthly equity % change) |
| CAGR | 477.41% | (Ending Equity ÷ Initial Capital)^(365 ÷ Days) − 1 |
| Profit Factor (Net) | 1.54 | (Σ profits) ÷ |Σ losses| |
| Profit Factor (Gross) | 1.68 | (Σ profits) ÷ |Σ losses| |
| Win Rate (%) | 46.53% | (# wins ÷ total trades) × 100 |
| Avg Win ($, Net) | $131.40 | mean(NetPL | NetPL>0) |
| Avg Loss ($, Net) | $-74.23 | mean(NetPL | NetPL<0) |
| Avg Win ($, Gross) | $134.45 | mean(GrossPL | GrossPL>0) |
| Avg Loss ($, Gross) | $-72.57 | mean(GrossPL | GrossPL<0) |
| **Avg Win ÷ Avg Loss** | 1.77 | |Avg Win| ÷ |Avg Loss| (Net) |
| **Largest Win ÷ Largest Loss** | 1.61 | |Largest Win| ÷ |Largest Loss| (Net) |
| Avg Win (pts / contract) | 26.28 | NetPL ÷ (point_value × |Qty|) |
| Avg Loss (pts / contract) | -14.85 | NetPL ÷ (point_value × |Qty|) |
| Expectancy per Trade ($, Net) | $21.44 | mean(NetPLᵢ) |
| Expectancy per Trade ($, Gross) | $25.48 | mean(GrossPLᵢ) |

---

## Risk (drawdowns & risk-adjusted)
| Metric | Result | Definition / Formula |
|---|---:|---|
| Max Drawdown ($) | $959.70 | max(peak − Equity) |
| Max Drawdown (%) | 23.42% | min(Equity ÷ peak − 1) × 100 |
| Average Drawdown (%) | -4.93% | mean(Drawdownₜ) × 100 |
| Recovery Factor | 6.43 | Net Profit ÷ Max DD ($) |
| Sharpe (per-trade proxy) | 0.16 | mean(rᵢ) ÷ stdev(rᵢ) |
| Sortino (per-trade proxy) | 0.34 | mean(rᵢ) ÷ stdev(min(rᵢ,0)) |
| Largest Winning Trade ($) | $672.21 | max(NetPLᵢ) |
| Largest Losing Trade ($) | $-417.79 | min(NetPLᵢ) |
| Volatility of Trade Returns | 0.06 | stdev(per-trade returns) |

## Exit Method Breakdown (RTH)
| Exit | Trades | Avg NetPL ($) | Profit Factor |
|---|---:|---:|---:|
| Close | 288 | 21.44 | 1.54 |

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
| 2025-03-31 | 2013.60 | 80.54 |
| 2025-04-30 | 2804.27 | 112.17 |
| 2025-05-31 | 563.89 | 22.56 |
| 2025-06-30 | -285.42 | -11.42 |
| 2025-07-31 | -30.44 | -1.22 |
| 2025-08-31 | -188.97 | -7.56 |
