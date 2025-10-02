# -*- coding: utf-8 -*-
# Backtester_vercel.py v1.4
# /MES only, stoploss capped at 20 points ($100), multi-strategy supported
# Weekly + Monthly performance tables included, PDF-style markdown output

import os
import io
import re
import json
from dataclasses import dataclass, asdict
from datetime import datetime, time
from pathlib import Path
import numpy as np
import pandas as pd

# =========================
# Config
# =========================

@dataclass
class BacktestConfig:
    strategy_name: str = ""
    instrument: str = "/MES"
    timeframe: str = "180d:15m"
    initial_capital: float = 2500.0
    commission_per_round_trip: float = 4.04
    point_value: float = 5.0
    version: str = "1.4"

    def outdir(self, csv_stem: str, strategy_label: str) -> str:
        temp_dir = Path('/tmp')
        day = datetime.now().strftime("%Y-%m-%d")
        timestamp = datetime.now().strftime("%H%M%S_%f")[:-3]
        safe_strategy = strategy_label.replace(" ", "_")
        return str(temp_dir / f"Backtests_{day}_{safe_strategy}_{self.timeframe}_{csv_stem}_{timestamp}")

# =========================
# Helpers
# =========================

def _to_float(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.replace(r"[\$,]", "", regex=True)
    s = s.str.replace(r"\(([^()]*)\)", r"-\1", regex=True)
    s = s.replace("", np.nan)
    return pd.to_numeric(s, errors='coerce')

def _parse_datetime(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, format="%m/%d/%y %I:%M %p", errors='coerce')

PRE_START, PRE_END   = time(3, 0),  time(9, 15)
RTH_START, RTH_END   = time(9, 30), time(16, 0)
AUTO_CLOSE           = time(16, 45)

def _tag_session(dt: pd.Timestamp) -> str:
    if pd.isna(dt): return "Unknown"
    t = dt.time()
    if PRE_START <= t <= PRE_END: return "PRE"
    if RTH_START <= t <= RTH_END: return "RTH"
    if t <= AUTO_CLOSE: return "RTH-Close"
    return "Other"

def _max_drawdown(equity_curve: pd.Series) -> float:
    running_max = equity_curve.cummax()
    drawdown = equity_curve / running_max - 1.0
    return float(drawdown.min())

def _profit_factor(pl: pd.Series) -> float:
    s = pl.dropna()
    gp = s[s > 0].sum()
    gl = -s[s < 0].sum()
    if gl == 0:
        return float('inf') if gp > 0 else 0.0
    return float(gp / gl)

# =========================
# Load & Clean CSV
# =========================

def load_tos_strategy_report(file_path: str) -> pd.DataFrame:
    with open(file_path, 'r', errors='replace') as f:
        lines = f.readlines()

    start_idx = None
    for i, line in enumerate(lines):
        if line.lstrip().startswith("Id;Strategy;"):
            start_idx = i
            break
    if start_idx is None:
        raise ValueError("No trade table header found in file.")

    table_str = "".join(lines[start_idx:])
    df = pd.read_csv(io.StringIO(table_str), sep=';')

    # Date parsing
    if 'Date/Time' in df.columns:
        df['Date'] = _parse_datetime(df['Date/Time'])
    elif 'Date' in df.columns and 'Time' in df.columns:
        dt_str = df['Date'].astype(str).str.strip() + ' ' + df['Time'].astype(str).str.strip()
        df['Date'] = pd.to_datetime(dt_str, errors='coerce')
    elif 'Date' in df.columns:
        df['Date'] = _parse_datetime(df['Date'])
    else:
        raise ValueError("Could not find date column.")

    # Money fields
    if 'Trade P/L' in df.columns:
        df['TradePL'] = _to_float(df['Trade P/L']).fillna(0.0)
    else:
        df['TradePL'] = 0.0
    df['CumPL'] = _to_float(df['P/L']) if 'P/L' in df.columns else np.nan

    # Strategy split
    if 'Strategy' in df.columns:
        df['BaseStrategy'] = df['Strategy'].astype(str).str.split('(').str[0].str.strip()
        df['ParsedTag'] = df['Strategy'].astype(str).str.extract(r"\(([^()]*)\)", expand=False).fillna("")
    else:
        df['BaseStrategy'] = "Unknown"
        df['ParsedTag'] = ""

    # Side
    df['Side'] = df['Side'].astype(str) if 'Side' in df.columns else ""

    # Quantity
    if 'Quantity' in df.columns:
        df['Qty'] = pd.to_numeric(df['Quantity'], errors='coerce')
    elif 'Qty' in df.columns:
        df['Qty'] = pd.to_numeric(df['Qty'], errors='coerce')
    else:
        df['Qty'] = 1

    if 'Price' not in df.columns:
        df['Price'] = np.nan

    if 'Symbol' not in df.columns:
        df['Symbol'] = "/MES"

    df = df.dropna(subset=['Date']).sort_values('Date').reset_index(drop=True)
    return df

# =========================
# Build Trades
# =========================

def build_trades(df: pd.DataFrame, cfg: BacktestConfig) -> pd.DataFrame:
    trades = []

    def _safe_num(x):
        return pd.to_numeric(x, errors='coerce')

    for tid, grp in df.groupby('Id', sort=False):
        g = grp.sort_values('Date').copy()
        entry = g.iloc[0]
        exit_ = g.iloc[-1]

        entry_qty = _safe_num(entry.get('Qty', 1))
        qty_abs = abs(entry_qty) if pd.notna(entry_qty) else 1.0

        direction = "Long" if "BTO" in entry.get('ParsedTag','').upper() else \
                    "Short" if "STO" in entry.get('ParsedTag','').upper() else "Unknown"

        parsed_exit = str(exit_.get('ParsedTag','')).upper()
        if "TARGET" in parsed_exit or "PROFIT" in parsed_exit or "TP" in parsed_exit:
            exit_reason = "Target"
        elif "STOP" in parsed_exit or "SL" in parsed_exit:
            exit_reason = "Stop"
        else:
            exit_reason = "Close"

        gross_pl = _safe_num(exit_.get('TradePL'))
        points = gross_pl / (cfg.point_value * qty_abs) if pd.notna(gross_pl) else 0

        # Stoploss enforcement
        if points < -20:
            adj_gross = -20 * cfg.point_value * qty_abs
            exit_reason = "Stop (Capped)"
        else:
            adj_gross = gross_pl

        commission = cfg.commission_per_round_trip * qty_abs
        net_pl = adj_gross - commission

        sess = _tag_session(entry['Date'])
        if sess == "PRE" and exit_['Date'].time() > PRE_END:
            exit_reason = "Premarket Close (09:15)"
        if sess.startswith("RTH") and exit_['Date'].time() > AUTO_CLOSE:
            exit_reason = "RTH Close (16:45)"

        trades.append({
            "Id": tid,
            "BaseStrategy": entry.get('BaseStrategy','Unknown'),
            "ParsedTag": entry.get('ParsedTag',''),
            "EntryTime": entry['Date'],
            "ExitTime": exit_['Date'],
            "QtyAbs": qty_abs,
            "GrossPL": gross_pl,
            "AdjGrossPL": adj_gross,
            "Commission": commission,
            "NetPL": net_pl,
            "Direction": direction,
            "ExitReason": exit_reason,
            "Session": sess,
            "Symbol": entry.get('Symbol', cfg.instrument)
        })

    return pd.DataFrame(trades)

# =========================
# Metrics
# =========================

def compute_metrics(trades: pd.DataFrame, cfg: BacktestConfig, csv_file: str) -> dict:
    def _metrics(df: pd.DataFrame) -> dict:
        pl_net = df['NetPL'].fillna(0.0)
        equity = cfg.initial_capital + pl_net.cumsum()
        total_net = float(pl_net.sum())
        total_return_pct = (total_net / cfg.initial_capital) * 100.0 if cfg.initial_capital else np.nan
        avg_win = float(pl_net[pl_net > 0].mean()) if (pl_net > 0).any() else np.nan
        avg_loss = float(pl_net[pl_net < 0].mean()) if (pl_net < 0).any() else np.nan
        max_dd_pct = abs(_max_drawdown(equity)) * 100.0
        max_dd_dollars = float((equity.cummax() - equity).max())
        recovery_factor = float(total_net / max_dd_dollars) if max_dd_dollars else np.nan
        expectancy_dollars = float(pl_net.mean()) if len(pl_net) else np.nan
        largest_win = float(pl_net.max()) if len(pl_net) else np.nan
        largest_loss = float(pl_net.min()) if len(pl_net) else np.nan
        return {
            "net_profit": total_net,
            "total_return_pct": total_return_pct,
            "profit_factor": _profit_factor(pl_net),
            "win_rate_pct": float((pl_net > 0).mean() * 100.0),
            "avg_win_dollars": avg_win,
            "avg_loss_dollars": avg_loss,
            "expectancy_per_trade_dollars": expectancy_dollars,
            "max_drawdown_pct": max_dd_pct,
            "max_drawdown_dollars": max_dd_dollars,
            "recovery_factor": recovery_factor,
            "largest_winning_trade": largest_win,
            "largest_losing_trade": largest_loss,
            "total_commissions": float(df['Commission'].sum()),
            "num_trades": int(len(df))
        }

    results = {"csv_file": csv_file, "strategies": {}}
    results["combined"] = _metrics(trades)

    seen = []
    for strat in trades['BaseStrategy']:
        if strat not in seen:
            seen.append(strat)

    for strat in seen:
        df_strat = trades[trades['BaseStrategy'] == strat]
        results["strategies"][strat] = _metrics(df_strat)

    return results

# =========================
# Markdown Report
# =========================

def generate_analytics_md(metrics: dict, cfg: BacktestConfig, outdir: str, weekly: pd.DataFrame, monthly: pd.DataFrame) -> None:
    os.makedirs(outdir, exist_ok=True)

    def _fmt(x, p=2, pct=False):
        try:
            if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
                return "n/a"
            return (f"{x:.{p}f}%" if pct else f"{x:.{p}f}")
        except Exception:
            return str(x)

    m_all = metrics["combined"]
    strat_names = list(metrics["strategies"].keys())

    md = f"""
# Strategy Analysis Report

**CSV File:** {metrics.get('csv_file','')}  
**Strategies Found:** {", ".join(strat_names)}  
**Instrument:** {cfg.instrument}  
**Timeframe:** {cfg.timeframe}  

Run Date: {datetime.now().strftime('%Y-%m-%d')}  
Session Basis: New York time (ET). Metrics Scope: RTH (09:30–16:00)  
Initial Capital: ${_fmt(cfg.initial_capital,0)}  
Commission (RT / contract): ${_fmt(cfg.commission_per_round_trip,2)}  
Total Commissions Paid: ${_fmt(m_all.get('total_commissions'),2)}  

---

## Key Performance Indicators (All Strategies Combined)
• **Net Profit (adjusted):** ${_fmt(m_all.get('net_profit'))}  
• **Total Return:** {_fmt(m_all.get('total_return_pct'), pct=True)}  
• **Win Rate:** {_fmt(m_all.get('win_rate_pct'), pct=True)}  
• **Profit Factor:** {_fmt(m_all.get('profit_factor'))}  
• **Max Drawdown:** ${_fmt(m_all.get('max_drawdown_dollars'))} ({_fmt(m_all.get('max_drawdown_pct'), pct=True)})  
• **Total Trades:** {int(m_all.get('num_trades',0))}  

---

## Performance Details (All Strategies Combined)
• **Average Win:** ${_fmt(m_all.get('avg_win_dollars'))}  
• **Average Loss:** ${_fmt(m_all.get('avg_loss_dollars'))}  
• **Expectancy per Trade:** ${_fmt(m_all.get('expectancy_per_trade_dollars'))}  
• **Largest Win:** ${_fmt(m_all.get('largest_winning_trade'))}  
• **Largest Loss:** ${_fmt(m_all.get('largest_losing_trade'))}  
• **Recovery Factor:** {_fmt(m_all.get('recovery_factor'))}  

---
"""

    for strat, vals in metrics["strategies"].items():
        md += f"""
## {strat} (Individual Performance)

### Key Performance Indicators
• **Net Profit (adjusted):** ${_fmt(vals.get('net_profit'))}  
• **Total Return:** {_fmt(vals.get('total_return_pct'), pct=True)}  
• **Win Rate:** {_fmt(vals.get('win_rate_pct'), pct=True)}  
• **Profit Factor:** {_fmt(vals.get('profit_factor'))}  
• **Max Drawdown:** ${_fmt(vals.get('max_drawdown_dollars'))} ({_fmt(vals.get('max_drawdown_pct'), pct=True)})  
• **Total Trades:** {int(vals.get('num_trades',0))}  

### Performance Details
• **Average Win:** ${_fmt(vals.get('avg_win_dollars'))}  
• **Average Loss:** ${_fmt(vals.get('avg_loss_dollars'))}  
• **Expectancy per Trade:** ${_fmt(vals.get('expectancy_per_trade_dollars'))}  
• **Largest Win:** ${_fmt(vals.get('largest_winning_trade'))}  
• **Largest Loss:** ${_fmt(vals.get('largest_losing_trade'))}  
• **Recovery Factor:** {_fmt(vals.get('recovery_factor'))}  

---
"""

    md += "## Weekly Performance\nExitTime     NetPL     ReturnPct\n"
    for idx, row in weekly.iterrows():
        md += f"{idx.date()}   {row['NetPL']:.2f}   {row['ReturnPct']:.2f}%\n"

    md += "\n## Monthly Performance\nExitTime     NetPL     ReturnPct\n"
    for idx, row in monthly.iterrows():
        md += f"{idx.date()}   {row['NetPL']:.2f}   {row['ReturnPct']:.2f}%\n"

    md += f"""

---

*Report generated by Backtester_vercel.py v{cfg.version}*
"""
    with open(os.path.join(outdir, "analytics.md"), "w", encoding="utf-8") as f:
        f.write(md)

# =========================
# Runner
# =========================

def run_backtest(tos_csv_path: str, cfg: BacktestConfig):
    csv_file = Path(tos_csv_path).name
    csv_stem = Path(tos_csv_path).stem.replace(' ', '_')
    raw = load_tos_strategy_report(tos_csv_path)
    trades = build_trades(raw, cfg)

    outdir = cfg.outdir(csv_stem, trades['BaseStrategy'].iloc[0] if len(trades) else "Unknown")
    os.makedirs(outdir, exist_ok=True)
    trades.to_csv(os.path.join(outdir, "trades_enriched.csv"), index=False)

    # Weekly & Monthly
    dt = pd.to_datetime(trades['ExitTime'], errors='coerce')
    weekly = pd.DataFrame({'NetPL': trades['NetPL'].values}, index=dt)
    weekly = weekly.dropna().resample('W-SUN').sum()
    weekly['ReturnPct'] = weekly['NetPL'] / cfg.initial_capital * 100.0
    weekly.to_csv(os.path.join(outdir, "weekly_performance.csv"))

    monthly = pd.DataFrame({'NetPL': trades['NetPL'].values}, index=dt)
    monthly = monthly.dropna().resample('M').sum()
    monthly['ReturnPct'] = monthly['NetPL'] / cfg.initial_capital * 100.0
    monthly.to_csv(os.path.join(outdir, "monthly_performance.csv"))

    metrics = compute_metrics(trades, cfg, csv_file)
    with open(os.path.join(outdir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    generate_analytics_md(metrics, cfg, outdir, weekly, monthly)
    with open(os.path.join(outdir, "config.json"), "w") as f:
        json.dump(asdict(cfg), f, indent=2)

    return trades, metrics, outdir

if __name__ == "__main__":
    import argparse, glob, sys

    parser = argparse.ArgumentParser(description="Backtester for TOS Strategy Report CSVs.")
    parser.add_argument("--csv", nargs="+", required=True, help="Path(s) to TOS Strategy Report CSV file(s).")
    parser.add_argument("--capital", type=float, default=2500.0, help="Initial capital.")
    parser.add_argument("--commission", type=float, default=4.04, help="Commission per contract round trip.")
    parser.add_argument("--point_value", type=float, default=5.0, help="Point value for /MES (default $5).")
    parser.add_argument("--timeframe", type=str, default="180d:15m", help="Label for timeframe.")

    args = parser.parse_args()

    cfg = BacktestConfig(
        timeframe=args.timeframe,
        initial_capital=args.capital,
        commission_per_round_trip=args.commission,
        point_value=args.point_value
    )

    # resolve file paths
    resolved = []
    for item in args.csv:
        matches = glob.glob(item)
        if matches:
            resolved.extend(matches)
        else:
            resolved.append(item)

    csv_paths = sorted({str(Path(p)) for p in resolved if Path(p).exists()})
    if not csv_paths:
        print(f"[ERROR] No CSV files matched: {args.csv}", file=sys.stderr)
        sys.exit(1)

    for csv_path in csv_paths:
        print(f"\n[RUN] CSV: {csv_path}")
        trades, metrics, outdir = run_backtest(csv_path, cfg)
        print(json.dumps(metrics, indent=2))
        print(f"Results saved to {outdir}")
