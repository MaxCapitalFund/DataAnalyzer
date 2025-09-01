
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
- **Net Profit:** $919.51
- **Total Return:** 36.78%
- **Win Rate:** 49.62%
- **Profit Factor:** 1.14
- **Max Drawdown:** $1354.21 (30.48%)
- **CAGR:** 56.57%

---

## Performance (returns & consistency)
| Metric | Result | Definition / Formula |
|---|---:|---|
| Net Profit ($) | $919.51 | Σ(NetPLᵢ) |
| Total Return (%) | 36.78% | (Net Profit ÷ Initial Capital) × 100 |
| Average Monthly Return | 5.21% | mean(Monthly equity % change) |
| CAGR | 56.57% | (Ending Equity ÷ Initial Capital)^(365 ÷ Days) − 1 |
| Profit Factor | 1.14 | (Σ profits) ÷ |Σ losses| |
| Win Rate (%) | 49.62% | (# wins ÷ total trades) × 100 |
| Avg Win ($) | $111.79 | mean(NetPL | NetPL>0) |
| Avg Loss ($) | $-96.16 | mean(NetPL | NetPL<0) |
| Avg Win (pts / contract) | 22.36 | NetPL ÷ (point_value × |Qty|) |
| Avg Loss (pts / contract) | -19.23 | NetPL ÷ (point_value × |Qty|) |
| Expectancy per Trade ($) | $7.02 | mean(NetPLᵢ) |

---

## Risk (drawdowns & risk-adjusted)
| Metric | Result | Definition / Formula |
|---|---:|---|
| Max Drawdown ($) | $1354.21 | max(peak − Equity) |
| Max Drawdown (%) | 30.48% | min(Equity ÷ peak − 1) × 100 |
| Average Drawdown (%) | -11.17% | mean(Drawdownₜ) × 100 |
| Recovery Factor | 0.68 | Net Profit ÷ Max DD ($) |
| Sharpe (per-trade proxy) | 0.05 | mean(rᵢ) ÷ stdev(rᵢ) |
| Sortino (per-trade proxy) | 0.09 | mean(rᵢ) ÷ stdev(min(rᵢ,0)) |
| Largest Winning Trade ($) | $449.71 | max(NetPLᵢ) |
| Largest Losing Trade ($) | $-570.29 | min(NetPLᵢ) |
| Volatility of Trade Returns | 0.05 | stdev(per-trade returns) |

## Exit Method Breakdown
| Reason | Trades | Avg NetPL ($) | Profit Factor |
|---|---:|---:|---:|
| Close | 131 | 7.02 | 1.14 |


---

## Trade Analytics (behavior & cadence)
| Metric | Result |
|---|---:|
| Number of Trades | 131 |
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
| 2025-03-31 | 228.44 | 9.14 |
| 2025-04-30 | 1159.40 | 46.38 |
| 2025-05-31 | -547.43 | -21.90 |
| 2025-06-30 | -447.72 | -17.91 |
| 2025-07-31 | -314.35 | -12.57 |
| 2025-08-31 | 285.94 | 11.44 |
