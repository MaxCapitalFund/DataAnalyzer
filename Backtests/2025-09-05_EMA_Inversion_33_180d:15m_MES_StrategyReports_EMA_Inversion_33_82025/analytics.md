
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
- **Net Profit (after commissions):** $-4840.90
- **Gross Profit (before commissions):** $-4497.50
- **Total Return (Net):** -193.64%
- **Total Return (Gross):** -179.90%
- **Win Rate:** 18.82%
- **Profit Factor (Net):** 0.11
- **Max Drawdown:** $4709.36 (198.84%)
- **CAGR:** n/a
- **Sharpe (annualized):** -6.13
- **Sortino (annualized):** -6.91

---

## Performance (returns & consistency)
| Metric | Result | Definition / Formula |
|---|---:|---|
| Net Profit ($, Net) | $-4840.90 | Σ(NetPLᵢ) |
| Gross Profit ($) | $-4497.50 | Σ(GrossPLᵢ) |
| Total Return (%) | -193.64% | (Net Profit ÷ Initial Capital) × 100 |
| Total Return (Gross %) | -179.90% | (Gross Profit ÷ Initial Capital) × 100 |
| Average Monthly Return | 62.43% | mean(Monthly equity % change) |
| CAGR | n/a | (Ending Equity ÷ Initial Capital)^(365 ÷ Days) − 1 |
| Profit Factor (Net) | 0.11 | (Σ profits) ÷ |Σ losses| |
| Profit Factor (Gross) | 0.13 | (Σ profits) ÷ |Σ losses| |
| Win Rate (%) | 18.82% | (# wins ÷ total trades) × 100 |
| Avg Win ($, Net) | $38.77 | mean(NetPL | NetPL>0) |
| Avg Loss ($, Net) | $-79.15 | mean(NetPL | NetPL<0) |
| Avg Win ($, Gross) | $42.81 | mean(GrossPL | GrossPL>0) |
| Avg Loss ($, Gross) | $-76.21 | mean(GrossPL | GrossPL<0) |
| **Avg Win ÷ Avg Loss** | 0.49 | |Avg Win| ÷ |Avg Loss| (Net) |
| **Largest Win ÷ Largest Loss** | 0.52 | |Largest Win| ÷ |Largest Loss| (Net) |
| Avg Win (pts / contract) | 7.75 | NetPL ÷ (point_value × |Qty|) |
| Avg Loss (pts / contract) | -15.83 | NetPL ÷ (point_value × |Qty|) |
| Expectancy per Trade ($, Net) | $-56.95 | mean(NetPLᵢ) |
| Expectancy per Trade ($, Gross) | $-52.91 | mean(GrossPLᵢ) |

---

## Risk (drawdowns & risk-adjusted)
| Metric | Result | Definition / Formula |
|---|---:|---|
| Max Drawdown ($) | $4709.36 | max(peak − Equity) |
| Max Drawdown (%) | 198.84% | min(Equity ÷ peak − 1) × 100 |
| Average Drawdown (%) | -115.11% | mean(Drawdownₜ) × 100 |
| Recovery Factor | -1.03 | Net Profit ÷ Max DD ($) |
| Sharpe (per-trade proxy) | -0.67 | mean(rᵢ) ÷ stdev(rᵢ) |
| Sortino (per-trade proxy) | -0.76 | mean(rᵢ) ÷ stdev(min(rᵢ,0)) |
| Largest Winning Trade ($) | $192.21 | max(NetPLᵢ) |
| Largest Losing Trade ($) | $-366.54 | min(NetPLᵢ) |
| Volatility of Trade Returns | 0.03 | stdev(per-trade returns) |

## Exit Method Breakdown (RTH)
| Exit | Trades | Avg NetPL ($) | Profit Factor |
|---|---:|---:|---:|
| Close | 85 | -56.95 | 0.11 |

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
| 2025-03-31 | -723.77 | -28.95 |
| 2025-04-30 | -820.69 | -32.83 |
| 2025-05-31 | -438.57 | -17.54 |
| 2025-06-30 | -345.11 | -13.80 |
| 2025-07-31 | -112.90 | -4.52 |
| 2025-08-31 | -485.78 | -19.43 |
