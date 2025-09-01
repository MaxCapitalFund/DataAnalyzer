
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
- **Net Profit:** $-610.19
- **Total Return:** -24.41%
- **Win Rate:** 44.62%
- **Profit Factor:** 0.95
- **Max Drawdown:** $4032.56 (74.79%)
- **CAGR:** -32.90%

---

## Performance (returns & consistency)
| Metric | Result | Definition / Formula |
|---|---:|---|
| Net Profit ($) | $-610.19 | Σ(NetPLᵢ) |
| Total Return (%) | -24.41% | (Net Profit ÷ Initial Capital) × 100 |
| Average Monthly Return | 13.84% | mean(Monthly equity % change) |
| CAGR | -32.90% | (Ending Equity ÷ Initial Capital)^(365 ÷ Days) − 1 |
| Profit Factor | 0.95 | (Σ profits) ÷ |Σ losses| |
| Win Rate (%) | 44.62% | (# wins ÷ total trades) × 100 |
| Avg Win ($) | $139.47 | mean(NetPL | NetPL>0) |
| Avg Loss ($) | $-118.31 | mean(NetPL | NetPL<0) |
| Avg Win (pts / contract) | 27.89 | NetPL ÷ (point_value × |Qty|) |
| Avg Loss (pts / contract) | -23.66 | NetPL ÷ (point_value × |Qty|) |
| Expectancy per Trade ($) | $-3.28 | mean(NetPLᵢ) |

---

## Risk (drawdowns & risk-adjusted)
| Metric | Result | Definition / Formula |
|---|---:|---|
| Max Drawdown ($) | $4032.56 | max(peak − Equity) |
| Max Drawdown (%) | 74.79% | min(Equity ÷ peak − 1) × 100 |
| Average Drawdown (%) | -44.52% | mean(Drawdownₜ) × 100 |
| Recovery Factor | -0.15 | Net Profit ÷ Max DD ($) |
| Sharpe (per-trade proxy) | -0.02 | mean(rᵢ) ÷ stdev(rᵢ) |
| Sortino (per-trade proxy) | -0.04 | mean(rᵢ) ÷ stdev(min(rᵢ,0)) |
| Largest Winning Trade ($) | $1244.71 | max(NetPLᵢ) |
| Largest Losing Trade ($) | $-839.04 | min(NetPLᵢ) |
| Volatility of Trade Returns | 0.07 | stdev(per-trade returns) |

## Exit Method Breakdown
| Reason | Trades | Avg NetPL ($) | Profit Factor |
|---|---:|---:|---:|
| Close | 186 | -3.28 | 0.95 |


---

## Trade Analytics (behavior & cadence)
| Metric | Result |
|---|---:|
| Number of Trades | 186 |
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
| 2025-03-31 | 1464.58 | 58.58 |
| 2025-04-30 | 1174.87 | 46.99 |
| 2025-05-31 | -2012.92 | -80.52 |
| 2025-06-30 | 390.07 | 15.60 |
| 2025-07-31 | -220.42 | -8.82 |
| 2025-08-31 | 249.58 | 9.98 |
