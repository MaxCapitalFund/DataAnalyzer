
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
- **Net Profit (after commissions):** $3286.16
- **Gross Profit (before commissions):** $3775.00
- **Total Return (Net):** 131.45%
- **Total Return (Gross):** 151.00%
- **Win Rate:** 52.89%
- **Profit Factor (Net):** 1.52
- **Max Drawdown:** $1284.35 (21.16%)
- **CAGR:** 226.29%
- **Sharpe (annualized):** 1.89
- **Sortino (annualized):** 4.81

---

## Performance (returns & consistency)
| Metric | Result | Definition / Formula |
|---|---:|---|
| Net Profit ($, Net) | $3286.16 | Σ(NetPLᵢ) |
| Gross Profit ($) | $3775.00 | Σ(GrossPLᵢ) |
| Total Return (%) | 131.45% | (Net Profit ÷ Initial Capital) × 100 |
| Total Return (Gross %) | 151.00% | (Gross Profit ÷ Initial Capital) × 100 |
| Average Monthly Return | 13.20% | mean(Monthly equity % change) |
| CAGR | 226.29% | (Ending Equity ÷ Initial Capital)^(365 ÷ Days) − 1 |
| Profit Factor (Net) | 1.52 | (Σ profits) ÷ |Σ losses| |
| Profit Factor (Gross) | 1.62 | (Σ profits) ÷ |Σ losses| |
| Win Rate (%) | 52.89% | (# wins ÷ total trades) × 100 |
| Avg Win ($, Net) | $150.49 | mean(NetPL | NetPL>0) |
| Avg Loss ($, Net) | $-111.32 | mean(NetPL | NetPL<0) |
| Avg Win ($, Gross) | $154.53 | mean(GrossPL | GrossPL>0) |
| Avg Loss ($, Gross) | $-107.28 | mean(GrossPL | GrossPL<0) |
| **Avg Win ÷ Avg Loss** | 1.35 | |Avg Win| ÷ |Avg Loss| (Net) |
| **Largest Win ÷ Largest Loss** | 3.79 | |Largest Win| ÷ |Largest Loss| (Net) |
| Avg Win (pts / contract) | 30.10 | NetPL ÷ (point_value × |Qty|) |
| Avg Loss (pts / contract) | -22.26 | NetPL ÷ (point_value × |Qty|) |
| Expectancy per Trade ($, Net) | $27.16 | mean(NetPLᵢ) |
| Expectancy per Trade ($, Gross) | $31.20 | mean(GrossPLᵢ) |

---

## Risk (drawdowns & risk-adjusted)
| Metric | Result | Definition / Formula |
|---|---:|---|
| Max Drawdown ($) | $1284.35 | max(peak − Equity) |
| Max Drawdown (%) | 21.16% | min(Equity ÷ peak − 1) × 100 |
| Average Drawdown (%) | -7.10% | mean(Drawdownₜ) × 100 |
| Recovery Factor | 2.56 | Net Profit ÷ Max DD ($) |
| Sharpe (per-trade proxy) | 0.17 | mean(rᵢ) ÷ stdev(rᵢ) |
| Sortino (per-trade proxy) | 0.44 | mean(rᵢ) ÷ stdev(min(rᵢ,0)) |
| Largest Winning Trade ($) | $734.71 | max(NetPLᵢ) |
| Largest Losing Trade ($) | $-194.04 | min(NetPLᵢ) |
| Volatility of Trade Returns | 0.06 | stdev(per-trade returns) |

## Exit Method Breakdown (RTH)
| Exit | Trades | Avg NetPL ($) | Profit Factor |
|---|---:|---:|---:|
| Close | 121 | 27.16 | 1.52 |

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
| 2025-04-30 | 1895.94 | 75.84 |
| 2025-05-31 | -1075.02 | -43.00 |
| 2025-06-30 | 397.68 | 15.91 |
| 2025-07-31 | -3.10 | -0.12 |
| 2025-08-31 | 661.32 | 26.45 |
| 2025-09-30 | -159.33 | -6.37 |
