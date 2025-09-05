
# Strategy One-Sheet (Trade Data)

**Strategy:** SuperSignal_v7_PreMarket45_43  
**Instrument:** Unknown  
**Timeframe:** 180d:15m  
**Run Date:** 2025-09-05  
**Session Basis:** New York time (ET). **Metrics Scope:** RTH (09:30–16:00)  
**Initial Capital (float):** $2500  
**Commission (RT / contract):** $4.04  
**Point Value:** $5.00 per point per contract

---

## Key Performance Indicators (KPI's)
- **Net Profit (after commissions):** $1155.02
- **Gross Profit (before commissions):** $1607.50
- **Total Return (Net):** 46.20%
- **Total Return (Gross):** 64.30%
- **Win Rate:** 49.11%
- **Profit Factor (Net):** 1.15
- **Max Drawdown:** $2657.05 (50.89%)
- **CAGR:** 72.97%
- **Sharpe (annualized):** 0.52
- **Sortino (annualized):** 1.00

---

## Performance (returns & consistency)
| Metric | Result | Definition / Formula |
|---|---:|---|
| Net Profit ($, Net) | $1155.02 | Σ(NetPLᵢ) |
| Gross Profit ($) | $1607.50 | Σ(GrossPLᵢ) |
| Total Return (%) | 46.20% | (Net Profit ÷ Initial Capital) × 100 |
| Total Return (Gross %) | 64.30% | (Gross Profit ÷ Initial Capital) × 100 |
| Average Monthly Return | 11.93% | mean(Monthly equity % change) |
| CAGR | 72.97% | (Ending Equity ÷ Initial Capital)^(365 ÷ Days) − 1 |
| Profit Factor (Net) | 1.15 | (Σ profits) ÷ |Σ losses| |
| Profit Factor (Gross) | 1.22 | (Σ profits) ÷ |Σ losses| |
| Win Rate (%) | 49.11% | (# wins ÷ total trades) × 100 |
| Avg Win ($, Net) | $158.57 | mean(NetPL | NetPL>0) |
| Avg Loss ($, Net) | $-132.75 | mean(NetPL | NetPL<0) |
| Avg Win ($, Gross) | $162.61 | mean(GrossPL | GrossPL>0) |
| Avg Loss ($, Gross) | $-128.71 | mean(GrossPL | GrossPL<0) |
| **Avg Win ÷ Avg Loss** | 1.19 | |Avg Win| ÷ |Avg Loss| (Net) |
| **Largest Win ÷ Largest Loss** | 1.48 | |Largest Win| ÷ |Largest Loss| (Net) |
| Avg Win (pts / contract) | 31.71 | NetPL ÷ (point_value × |Qty|) |
| Avg Loss (pts / contract) | -26.55 | NetPL ÷ (point_value × |Qty|) |
| Expectancy per Trade ($, Net) | $10.31 | mean(NetPLᵢ) |
| Expectancy per Trade ($, Gross) | $14.35 | mean(GrossPLᵢ) |

---

## Risk (drawdowns & risk-adjusted)
| Metric | Result | Definition / Formula |
|---|---:|---|
| Max Drawdown ($) | $2657.05 | max(peak − Equity) |
| Max Drawdown (%) | 50.89% | min(Equity ÷ peak − 1) × 100 |
| Average Drawdown (%) | -27.53% | mean(Drawdownₜ) × 100 |
| Recovery Factor | 0.43 | Net Profit ÷ Max DD ($) |
| Sharpe (per-trade proxy) | 0.05 | mean(rᵢ) ÷ stdev(rᵢ) |
| Sortino (per-trade proxy) | 0.10 | mean(rᵢ) ÷ stdev(min(rᵢ,0)) |
| Largest Winning Trade ($) | $1244.71 | max(NetPLᵢ) |
| Largest Losing Trade ($) | $-839.04 | min(NetPLᵢ) |
| Volatility of Trade Returns | 0.08 | stdev(per-trade returns) |

## Exit Method Breakdown (RTH)
| Exit | Trades | Avg NetPL ($) | Profit Factor |
|---|---:|---:|---:|
| Close | 112 | 10.31 | 1.15 |

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
| 2025-03-31 | 1472.86 | 58.91 |
| 2025-04-30 | 1215.65 | 48.63 |
| 2025-05-31 | -1097.61 | -43.90 |
| 2025-06-30 | 387.77 | 15.51 |
| 2025-07-31 | -85.98 | -3.44 |
| 2025-08-31 | 397.86 | 15.91 |
