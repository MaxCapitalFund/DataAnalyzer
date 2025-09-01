
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
- **Net Profit:** $296.00
- **Total Return:** 11.84%
- **Win Rate:** 58.17%
- **Profit Factor:** 1.01
- **Max Drawdown:** $4991.11 (82.73%)
- **CAGR:** 17.01%

---

## Performance (returns & consistency)
| Metric | Result | Definition / Formula |
|---|---:|---|
| Net Profit ($) | $296.00 | Σ(NetPLᵢ) |
| Total Return (%) | 11.84% | (Net Profit ÷ Initial Capital) × 100 |
| Average Monthly Return | 7.09% | mean(Monthly equity % change) |
| CAGR | 17.01% | (Ending Equity ÷ Initial Capital)^(365 ÷ Days) − 1 |
| Profit Factor | 1.01 | (Σ profits) ÷ |Σ losses| |
| Win Rate (%) | 58.17% | (# wins ÷ total trades) × 100 |
| Avg Win ($) | $92.18 | mean(NetPL | NetPL>0) |
| Avg Loss ($) | $-126.99 | mean(NetPL | NetPL<0) |
| Avg Win (pts / contract) | 18.44 | NetPL ÷ (point_value × |Qty|) |
| Avg Loss (pts / contract) | -25.40 | NetPL ÷ (point_value × |Qty|) |
| Expectancy per Trade ($) | $0.49 | mean(NetPLᵢ) |

---

## Risk (drawdowns & risk-adjusted)
| Metric | Result | Definition / Formula |
|---|---:|---|
| Max Drawdown ($) | $4991.11 | max(peak − Equity) |
| Max Drawdown (%) | 82.73% | min(Equity ÷ peak − 1) × 100 |
| Average Drawdown (%) | -37.91% | mean(Drawdownₜ) × 100 |
| Recovery Factor | 0.06 | Net Profit ÷ Max DD ($) |
| Sharpe (per-trade proxy) | 0.00 | mean(rᵢ) ÷ stdev(rᵢ) |
| Sortino (per-trade proxy) | 0.00 | mean(rᵢ) ÷ stdev(min(rᵢ,0)) |
| Largest Winning Trade ($) | $2149.71 | max(NetPLᵢ) |
| Largest Losing Trade ($) | $-2302.79 | min(NetPLᵢ) |
| Volatility of Trade Returns | 0.09 | stdev(per-trade returns) |

## Exit Method Breakdown
| Reason | Trades | Avg NetPL ($) | Profit Factor |
|---|---:|---:|---:|
| Close | 600 | 0.49 | 1.01 |


---

## Trade Analytics (behavior & cadence)
| Metric | Result |
|---|---:|
| Number of Trades | 600 |
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
| 2025-03-31 | -426.30 | -17.05 |
| 2025-04-30 | 1164.98 | 46.60 |
| 2025-05-31 | -246.55 | -9.86 |
| 2025-06-30 | -746.15 | -29.85 |
| 2025-07-31 | 117.04 | 4.68 |
| 2025-08-31 | 514.76 | 20.59 |
