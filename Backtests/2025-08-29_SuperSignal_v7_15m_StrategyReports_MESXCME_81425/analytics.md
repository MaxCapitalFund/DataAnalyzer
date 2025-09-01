
# Strategy One-Sheet (Trade Data)

**Strategy:** SuperSignal_v7  
**Timeframe:** 15m  
**Run Date:** 2025-08-29  
**Session Basis:** New York time (ET) RTH 09:30–16:00  
**Initial Capital (float):** $2500  
**Commission (RT / contract):** $4.04  
**Point Value:** $5.00 per point per contract

---

## Key Performance Indicators (KPI's)
- **Net Profit:** $3868.04
- **Total Return:** 154.72%
- **Win Rate:** 58.53%
- **Profit Factor:** 1.14
- **Max Drawdown:** $1546.70 (41.89%)
- **CAGR:** 273.47%

---

## Performance (returns & consistency)
| Metric | Result | Definition / Formula |
|---|---:|---|
| Net Profit ($) | $3868.04 | Σ(NetPLᵢ) |
| Total Return (%) | 154.72% | (Net Profit ÷ Initial Capital) × 100 |
| Average Monthly Return | 14.91% | mean(Monthly equity % change) |
| CAGR | 273.47% | (Ending Equity ÷ Initial Capital)^(365 ÷ Days) − 1 |
| Profit Factor | 1.14 | (Σ profits) ÷ |Σ losses| |
| Win Rate (%) | 58.53% | (# wins ÷ total trades) × 100 |
| Avg Win ($) | $71.43 | mean(NetPL | NetPL>0) |
| Avg Loss ($) | $-88.75 | mean(NetPL | NetPL<0) |
| Avg Win (pts / contract) | 14.29 | NetPL ÷ (point_value × |Qty|) |
| Avg Loss (pts / contract) | -17.75 | NetPL ÷ (point_value × |Qty|) |
| Expectancy per Trade ($) | $5.00 | mean(NetPLᵢ) |

---

## Risk (drawdowns & risk-adjusted)
| Metric | Result | Definition / Formula |
|---|---:|---|
| Max Drawdown ($) | $1546.70 | max(peak − Equity) |
| Max Drawdown (%) | 41.89% | min(Equity ÷ peak − 1) × 100 |
| Average Drawdown (%) | -11.08% | mean(Drawdownₜ) × 100 |
| Recovery Factor | 2.50 | Net Profit ÷ Max DD ($) |
| Sharpe (per-trade proxy) | 0.04 | mean(rᵢ) ÷ stdev(rᵢ) |
| Sortino (per-trade proxy) | 0.06 | mean(rᵢ) ÷ stdev(min(rᵢ,0)) |
| Largest Winning Trade ($) | $672.21 | max(NetPLᵢ) |
| Largest Losing Trade ($) | $-687.79 | min(NetPLᵢ) |
| Volatility of Trade Returns | 0.05 | stdev(per-trade returns) |

## Exit Method Breakdown
| Reason | Trades | Avg NetPL ($) | Profit Factor |
|---|---:|---:|---:|
| Close | 774 | 5.00 | 1.14 |


---

## Trade Analytics (behavior & cadence)
| Metric | Result |
|---|---:|
| Number of Trades | 774 |
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
| 2025-03-31 | 181.56 | 7.26 |
| 2025-04-30 | 3525.05 | 141.00 |
| 2025-05-31 | 204.19 | 8.17 |
| 2025-06-30 | -28.96 | -1.16 |
| 2025-07-31 | -86.66 | -3.47 |
| 2025-08-31 | -142.94 | -5.72 |
