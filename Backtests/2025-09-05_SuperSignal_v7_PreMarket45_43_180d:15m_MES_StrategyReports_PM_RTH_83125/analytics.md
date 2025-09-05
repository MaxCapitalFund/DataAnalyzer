
# Strategy One-Sheet (Trade Data)

**Strategy:** SuperSignal_v7_PreMarket45_43  
**Instrument:** S&P500 Micro Mini Futures  
**Timeframe:** 180d:15m  
**Run Date:** 2025-09-05  
**Session Basis:** New York time (ET). **Metrics Scope:** RTH (09:30–16:00)  
**Initial Capital (float):** $2500  
**Commission (RT / contract):** $4.04  
**Point Value:** $5.00 per point per contract

---

## Key Performance Indicators (KPI's)
- **Net Profit (after commissions):** $1593.29
- **Gross Profit (before commissions):** $2296.25
- **Total Return (Net):** 63.73%
- **Total Return (Gross):** 91.85%
- **Win Rate:** 50.57%
- **Profit Factor (Net):** 1.14
- **Max Drawdown:** $3445.97 (48.49%)
- **CAGR:** 103.10%
- **Sharpe (annualized):** 0.63
- **Sortino (annualized):** 1.16

---

## Performance (returns & consistency)
| Metric | Result | Definition / Formula |
|---|---:|---|
| Net Profit ($, Net) | $1593.29 | Σ(NetPLᵢ) |
| Gross Profit ($) | $2296.25 | Σ(GrossPLᵢ) |
| Total Return (%) | 63.73% | (Net Profit ÷ Initial Capital) × 100 |
| Total Return (Gross %) | 91.85% | (Gross Profit ÷ Initial Capital) × 100 |
| Average Monthly Return | 11.52% | mean(Monthly equity % change) |
| CAGR | 103.10% | (Ending Equity ÷ Initial Capital)^(365 ÷ Days) − 1 |
| Profit Factor (Net) | 1.14 | (Σ profits) ÷ |Σ losses| |
| Profit Factor (Gross) | 1.21 | (Σ profits) ÷ |Σ losses| |
| Win Rate (%) | 50.57% | (# wins ÷ total trades) × 100 |
| Avg Win ($, Net) | $147.32 | mean(NetPL | NetPL>0) |
| Avg Loss ($, Net) | $-132.22 | mean(NetPL | NetPL<0) |
| Avg Win ($, Gross) | $151.36 | mean(GrossPL | GrossPL>0) |
| Avg Loss ($, Gross) | $-128.18 | mean(GrossPL | GrossPL<0) |
| **Avg Win ÷ Avg Loss** | 1.11 | |Avg Win| ÷ |Avg Loss| (Net) |
| **Largest Win ÷ Largest Loss** | 1.48 | |Largest Win| ÷ |Largest Loss| (Net) |
| Avg Win (pts / contract) | 29.46 | NetPL ÷ (point_value × |Qty|) |
| Avg Loss (pts / contract) | -26.44 | NetPL ÷ (point_value × |Qty|) |
| Expectancy per Trade ($, Net) | $9.16 | mean(NetPLᵢ) |
| Expectancy per Trade ($, Gross) | $13.20 | mean(GrossPLᵢ) |

---

## Risk (drawdowns & risk-adjusted)
| Metric | Result | Definition / Formula |
|---|---:|---|
| Max Drawdown ($) | $3445.97 | max(peak − Equity) |
| Max Drawdown (%) | 48.49% | min(Equity ÷ peak − 1) × 100 |
| Average Drawdown (%) | -27.24% | mean(Drawdownₜ) × 100 |
| Recovery Factor | 0.46 | Net Profit ÷ Max DD ($) |
| Sharpe (per-trade proxy) | 0.05 | mean(rᵢ) ÷ stdev(rᵢ) |
| Sortino (per-trade proxy) | 0.09 | mean(rᵢ) ÷ stdev(min(rᵢ,0)) |
| Largest Winning Trade ($) | $1244.71 | max(NetPLᵢ) |
| Largest Losing Trade ($) | $-839.04 | min(NetPLᵢ) |
| Volatility of Trade Returns | 0.08 | stdev(per-trade returns) |

## Exit Method Breakdown (RTH)
| Exit | Trades | Avg NetPL ($) | Profit Factor |
|---|---:|---:|---:|
| Close | 174 | 9.16 | 1.14 |

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
| 2025-03-31 | 2026.12 | 81.04 |
| 2025-04-30 | 1269.20 | 50.77 |
| 2025-05-31 | -1213.30 | -48.53 |
| 2025-06-30 | 192.37 | 7.69 |
| 2025-07-31 | -267.34 | -10.69 |
| 2025-08-31 | 236.70 | 9.47 |
