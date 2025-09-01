
# Strategy One-Sheet (Trade Data)

**Strategy:** SuperSignal_v7  
**Timeframe:** 15m  
**Run Date:** 2025-08-31  
**Session Basis:** New York time (ET) RTH 09:30–16:00  
**Initial Capital (float):** $2500  
**Commission (RT / contract):** $4.04  
**Point Value:** $5.00 per point per contract

---

## Key Performance Indicators (KPI's)
- **Net Profit:** $-225.55
- **Total Return:** -9.02%
- **Win Rate:** 45.76%
- **Profit Factor:** 0.99
- **Max Drawdown:** $4883.37 (72.38%)
- **CAGR:** -12.61%

---

## Performance (returns & consistency)
| Metric | Result | Definition / Formula |
|---|---:|---|
| Net Profit ($) | $-225.55 | Σ(NetPLᵢ) |
| Total Return (%) | -9.02% | (Net Profit ÷ Initial Capital) × 100 |
| Average Monthly Return | 11.26% | mean(Monthly equity % change) |
| CAGR | -12.61% | (Ending Equity ÷ Initial Capital)^(365 ÷ Days) − 1 |
| Profit Factor | 0.99 | (Σ profits) ÷ |Σ losses| |
| Win Rate (%) | 45.76% | (# wins ÷ total trades) × 100 |
| Avg Win ($) | $128.34 | mean(NetPL | NetPL>0) |
| Avg Loss ($) | $-109.70 | mean(NetPL | NetPL<0) |
| Avg Win (pts / contract) | 25.67 | NetPL ÷ (point_value × |Qty|) |
| Avg Loss (pts / contract) | -21.94 | NetPL ÷ (point_value × |Qty|) |
| Expectancy per Trade ($) | $-0.76 | mean(NetPLᵢ) |

---

## Risk (drawdowns & risk-adjusted)
| Metric | Result | Definition / Formula |
|---|---:|---|
| Max Drawdown ($) | $4883.37 | max(peak − Equity) |
| Max Drawdown (%) | 72.38% | min(Equity ÷ peak − 1) × 100 |
| Average Drawdown (%) | -40.81% | mean(Drawdownₜ) × 100 |
| Recovery Factor | -0.05 | Net Profit ÷ Max DD ($) |
| Sharpe (per-trade proxy) | -0.00 | mean(rᵢ) ÷ stdev(rᵢ) |
| Sortino (per-trade proxy) | -0.01 | mean(rᵢ) ÷ stdev(min(rᵢ,0)) |
| Largest Winning Trade ($) | $1244.71 | max(NetPLᵢ) |
| Largest Losing Trade ($) | $-839.04 | min(NetPLᵢ) |
| Volatility of Trade Returns | 0.06 | stdev(per-trade returns) |

## Exit Method Breakdown
| Reason | Trades | Avg NetPL ($) | Profit Factor |
|---|---:|---:|---:|
| Close | 295 | -0.76 | 0.99 |


---

## Trade Analytics (behavior & cadence)
| Metric | Result |
|---|---:|
| Number of Trades | 295 |
| Long Trades | 0 |
| Short Trades | 0 |
| Avg Holding Time (minutes) | n/a |
| Session Tags (ET) | Overnight / Pre / Open / Midday / Late / Post |

> For session-level slicing, use `trades_enriched.csv` (column **Session**).

---

## Visuals & Tables (investor-friendly)
- **Equity Curve (last 180 days):** `equity_curve_180d.png`
- **Drawdown Curve:** `drawdown_curve.png`
- **Trade P/L Histogram:** `pl_histogram.png`
- **Monthly Performance Table:** `monthly_performance.csv`

### Monthly Performance Preview (last 6)
| Month | NetPL ($) | Return (%) |
|---|---:|---:|
| 2025-03-31 | 1693.31 | 67.73 |
| 2025-04-30 | 1605.72 | 64.23 |
| 2025-05-31 | -2247.56 | -89.90 |
| 2025-06-30 | 316.68 | 12.67 |
| 2025-07-31 | -567.94 | -22.72 |
| 2025-08-31 | 273.89 | 10.96 |
