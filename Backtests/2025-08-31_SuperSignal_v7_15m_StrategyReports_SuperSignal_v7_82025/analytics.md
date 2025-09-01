
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
- **Net Profit:** $4764.84
- **Total Return:** 190.59%
- **Win Rate:** 53.95%
- **Profit Factor:** 1.35
- **Max Drawdown:** $1387.73 (17.13%)
- **CAGR:** 349.67%

---

## Performance (returns & consistency)
| Metric | Result | Definition / Formula |
|---|---:|---|
| Net Profit ($) | $4764.84 | Σ(NetPLᵢ) |
| Total Return (%) | 190.59% | (Net Profit ÷ Initial Capital) × 100 |
| Average Monthly Return | 15.75% | mean(Monthly equity % change) |
| CAGR | 349.67% | (Ending Equity ÷ Initial Capital)^(365 ÷ Days) − 1 |
| Profit Factor | 1.35 | (Σ profits) ÷ |Σ losses| |
| Win Rate (%) | 53.95% | (# wins ÷ total trades) × 100 |
| Avg Win ($) | $96.65 | mean(NetPL | NetPL>0) |
| Avg Loss ($) | $-84.02 | mean(NetPL | NetPL<0) |
| Avg Win (pts / contract) | 19.33 | NetPL ÷ (point_value × |Qty|) |
| Avg Loss (pts / contract) | -16.80 | NetPL ÷ (point_value × |Qty|) |
| Expectancy per Trade ($) | $13.46 | mean(NetPLᵢ) |

---

## Risk (drawdowns & risk-adjusted)
| Metric | Result | Definition / Formula |
|---|---:|---|
| Max Drawdown ($) | $1387.73 | max(peak − Equity) |
| Max Drawdown (%) | 17.13% | min(Equity ÷ peak − 1) × 100 |
| Average Drawdown (%) | -6.41% | mean(Drawdownₜ) × 100 |
| Recovery Factor | 3.43 | Net Profit ÷ Max DD ($) |
| Sharpe (per-trade proxy) | 0.12 | mean(rᵢ) ÷ stdev(rᵢ) |
| Sortino (per-trade proxy) | 0.24 | mean(rᵢ) ÷ stdev(min(rᵢ,0)) |
| Largest Winning Trade ($) | $525.96 | max(NetPLᵢ) |
| Largest Losing Trade ($) | $-429.04 | min(NetPLᵢ) |
| Volatility of Trade Returns | 0.04 | stdev(per-trade returns) |

## Exit Method Breakdown
| Reason | Trades | Avg NetPL ($) | Profit Factor |
|---|---:|---:|---:|
| Close | 354 | 13.46 | 1.35 |


---

## Trade Analytics (behavior & cadence)
| Metric | Result |
|---|---:|
| Number of Trades | 354 |
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
| 2025-03-31 | 1987.82 | 79.51 |
| 2025-04-30 | 2822.15 | 112.89 |
| 2025-05-31 | -221.80 | -8.87 |
| 2025-06-30 | -630.93 | -25.24 |
| 2025-07-31 | -327.85 | -13.11 |
| 2025-08-31 | 422.84 | 16.91 |
