
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
- **Net Profit (after commissions):** $1032.00
- **Gross Profit (before commissions):** $1335.00
- **Total Return (Net):** 41.28%
- **Total Return (Gross):** 53.40%
- **Win Rate:** 56.00%
- **Profit Factor (Net):** 1.23
- **Max Drawdown:** $1311.29 (28.66%)
- **CAGR:** 64.96%
- **Sharpe (annualized):** 0.74
- **Sortino (annualized):** 1.31

---

## Performance (returns & consistency)
| Metric | Result | Definition / Formula |
|---|---:|---|
| Net Profit ($, Net) | $1032.00 | Σ(NetPLᵢ) |
| Gross Profit ($) | $1335.00 | Σ(GrossPLᵢ) |
| Total Return (%) | 41.28% | (Net Profit ÷ Initial Capital) × 100 |
| Total Return (Gross %) | 53.40% | (Gross Profit ÷ Initial Capital) × 100 |
| Average Monthly Return | 4.40% | mean(Monthly equity % change) |
| CAGR | 64.96% | (Ending Equity ÷ Initial Capital)^(365 ÷ Days) − 1 |
| Profit Factor (Net) | 1.23 | (Σ profits) ÷ |Σ losses| |
| Profit Factor (Gross) | 1.30 | (Σ profits) ÷ |Σ losses| |
| Win Rate (%) | 56.00% | (# wins ÷ total trades) × 100 |
| Avg Win ($, Net) | $133.22 | mean(NetPL | NetPL>0) |
| Avg Loss ($, Net) | $-138.28 | mean(NetPL | NetPL<0) |
| Avg Win ($, Gross) | $137.26 | mean(GrossPL | GrossPL>0) |
| Avg Loss ($, Gross) | $-134.24 | mean(GrossPL | GrossPL<0) |
| **Avg Win ÷ Avg Loss** | 0.96 | |Avg Win| ÷ |Avg Loss| (Net) |
| **Largest Win ÷ Largest Loss** | 0.79 | |Largest Win| ÷ |Largest Loss| (Net) |
| Avg Win (pts / contract) | 26.64 | NetPL ÷ (point_value × |Qty|) |
| Avg Loss (pts / contract) | -27.66 | NetPL ÷ (point_value × |Qty|) |
| Expectancy per Trade ($, Net) | $13.76 | mean(NetPLᵢ) |
| Expectancy per Trade ($, Gross) | $17.80 | mean(GrossPLᵢ) |

---

## Risk (drawdowns & risk-adjusted)
| Metric | Result | Definition / Formula |
|---|---:|---|
| Max Drawdown ($) | $1311.29 | max(peak − Equity) |
| Max Drawdown (%) | 28.66% | min(Equity ÷ peak − 1) × 100 |
| Average Drawdown (%) | -10.12% | mean(Drawdownₜ) × 100 |
| Recovery Factor | 0.79 | Net Profit ÷ Max DD ($) |
| Sharpe (per-trade proxy) | 0.09 | mean(rᵢ) ÷ stdev(rᵢ) |
| Sortino (per-trade proxy) | 0.15 | mean(rᵢ) ÷ stdev(min(rᵢ,0)) |
| Largest Winning Trade ($) | $449.71 | max(NetPLᵢ) |
| Largest Losing Trade ($) | $-570.29 | min(NetPLᵢ) |
| Volatility of Trade Returns | 0.06 | stdev(per-trade returns) |

## Exit Method Breakdown (RTH)
| Exit | Trades | Avg NetPL ($) | Profit Factor |
|---|---:|---:|---:|
| Close | 75 | 13.76 | 1.23 |

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
| 2025-03-31 | 553.26 | 22.13 |
| 2025-04-30 | 837.68 | 33.51 |
| 2025-05-31 | -485.69 | -19.43 |
| 2025-06-30 | -493.48 | -19.74 |
| 2025-07-31 | -164.15 | -6.57 |
| 2025-08-31 | 100.47 | 4.02 |
