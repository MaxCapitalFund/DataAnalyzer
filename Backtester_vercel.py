# -*- coding: utf-8 -*-
# Backtester_vercel.py
# Hybrid Backtester v1.4.2 (Vercel-friendly, accepts --timeframe but ignores it)

import os, io, re, json, glob, sys
from dataclasses import dataclass, asdict
from datetime import datetime, time
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # serverless backend
import matplotlib.pyplot as plt

# =========================
# Config
# =========================

@dataclass
class BacktestConfig:
    strategy_name: str = ""
    instruments: Tuple[str, ...] = ("/MES",)
    timeframe: str = "180d:15m"   # fixed default
    initial_capital: float = 2500.0
    commission_per_round_trip: float = 4.04
    point_value: float = 5.0
    version: str = "1.4.2"

    def outdir(self, csv_stem: str, instrument: str, strategy_label: str) -> str:
        temp_dir = Path('/tmp')
        day = datetime.now().strftime("%Y-%m-%d")
        timestamp = datetime.now().strftime("%H%M%S_%f")[:-3]
        safe_strategy = (strategy_label or "Unknown").replace(" ", "_")
        safe_instr = (instrument or "UNK").replace("/", "")
        return str(temp_dir / f"Backtests_{day}_{safe_strategy}_{self.timeframe}_{safe_instr}_{csv_stem}_{timestamp}")

# =========================
# Helpers
# =========================

def _to_float(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.replace(r"[\$,]", "", regex=True)
    s = s.str.replace(r"\(([^()]*)\)", r"-\1", regex=True)
    s = s.replace("", np.nan)
    return pd.to_numeric(s, errors='coerce')

def _parse_datetime(series: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(series, format="%m/%d/%y %I:%M %p", errors='coerce')
    if parsed.isna().all():
        parsed = pd.to_datetime(series, errors='coerce')
    return parsed

def _max_drawdown(equity_curve: pd.Series) -> float:
    running_max = equity_curve.cummax()
    drawdown = equity_curve / running_max - 1.0
    return float(drawdown.min())

def _profit_factor(pl: pd.Series) -> float:
    s = pl.dropna()
    gp = s[s > 0].sum()
    gl = -s[s < 0].sum()
    if gl == 0:
        return float("inf") if gp > 0 else 0.0
    return float(gp / gl)

def _exit_reason(text: str) -> str:
    s = str(text).upper()
    if any(w in s for w in ["TARGET", "TP", "PROFIT"]): return "Target"
    if any(w in s for w in ["STOP", "SL", "STOPPED"]): return "Stop"
    if any(w in s for w in ["TIME", "TIME EXIT", "TIMEOUT"]): return "Time"
    if any(w in s for w in ["MANUAL", "FLATTEN", "MKT CLOSE"]): return "Manual"
    return "Close"

# =========================
# Symbol normalization
# =========================

ROOT_RE = re.compile(r"^/?([A-Za-z]{1,3})(?:[FGHJKMNQUVXZ]\d{1,2})?$")

def normalize_symbol(sym: str) -> str:
    s = str(sym).strip()
    if not s: return "/UNK"
    has_slash = s.startswith("/")
    core = s[1:] if has_slash else s
    m = ROOT_RE.match(core.upper())
    if m: return f"/{m.group(1).upper()}"
    m2 = re.search(r"/([A-Za-z]{1,3})", s.upper())
    if m2: return f"/{m2.group(1)}"
    m3 = re.search(r"\b([A-Za-z]{1,3})\b", s.upper())
    return f"/{m3.group(1)}" if m3 else "/UNK"

# =========================
# Load CSV
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

    if 'Date/Time' in df.columns:
        df['Date'] = _parse_datetime(df['Date/Time'])
    elif 'Date' in df.columns and 'Time' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'].astype(str) + ' ' + df['Time'].astype(str), errors='coerce')
    elif 'Date' in df.columns:
        df['Date'] = _parse_datetime(df['Date'])
    else:
        raise ValueError("No Date column found.")

    df['TradePL'] = _to_float(df.get('Trade P/L', pd.Series([]))).fillna(0.0)
    df['CumPL'] = _to_float(df['P/L']) if 'P/L' in df.columns else np.nan
    df['BaseStrategy'] = df['Strategy'].astype(str).str.split('(').str[0].str.strip() if 'Strategy' in df.columns else "Unknown"
    side_col = next((c for c in ['Side','Action','Order','Type'] if c in df.columns), None)
    df['Side'] = df[side_col].astype(str) if side_col else ""
    if 'Price' not in df.columns: df['Price'] = np.nan
    df['Qty'] = pd.to_numeric(df.get('Quantity', df.get('Qty', np.nan)), errors='coerce')
    df['Symbol'] = df.get('Symbol', df.get('Instrument', df['Strategy'])).astype(str).map(normalize_symbol)
    return df.dropna(subset=['Date']).sort_values('Date').reset_index(drop=True)

# =========================
# Build Trades
# =========================

def build_trades(df: pd.DataFrame, commission_rt: float) -> pd.DataFrame:
    trades = []
    OPEN_RX  = r"\b(BTO|BUY TO OPEN|STO|SELL TO OPEN|SELL SHORT)\b"
    CLOSE_RX = r"\b(STC|SELL TO CLOSE|BTC|BUY TO CLOSE|CLOSE)\b"

    if "Id" in df.columns:
        for tid, grp in df.groupby("Id", sort=False):
            g = grp.sort_values("Date").copy()
            side_up = g["Side"].astype(str).str.upper()
            entry_rows = g[side_up.str.contains(OPEN_RX, regex=True, na=False)]
            close_rows = g[side_up.str.contains(CLOSE_RX, regex=True, na=False)]
            if len(entry_rows) and len(close_rows):
                entry, exit_ = entry_rows.iloc[0], close_rows.iloc[0]
                qty_abs = abs(entry.get("Qty", 1)) or 1
                direction = "Long" if "BTO" in str(entry["Side"]).upper() else "Short"
                trade_pl = pd.to_numeric(exit_.get("TradePL"), errors="coerce")
                commission = commission_rt * qty_abs
                net_pl = (trade_pl or 0.0) - commission
                trades.append({
                    "Id": tid,
                    "EntryTime": entry["Date"],
                    "ExitTime": exit_["Date"],
                    "EntryPrice": entry.get("Price"),
                    "ExitPrice": exit_.get("Price"),
                    "QtyAbs": qty_abs,
                    "TradePL": trade_pl,
                    "GrossPL": trade_pl,
                    "Commission": commission,
                    "NetPL": net_pl,
                    "BaseStrategy": entry.get("BaseStrategy", "Unknown"),
                    "Symbol": entry.get("Symbol", ""),
                    "ExitReason": _exit_reason(exit_.get("Side")),
                    "Direction": direction,
                })
    return pd.DataFrame(trades)

# =========================
# Stop-loss corrections
# =========================

def apply_stoploss_corrections(trades: pd.DataFrame, point_value: float) -> pd.DataFrame:
    df = trades.copy()
    df["SLBreached"] = df["NetPL"] < -100.0
    df["AdjustedNetPL"] = np.where(df["SLBreached"], -100.0, df["NetPL"])
    df["PointsPerContract"] = (df["AdjustedNetPL"] + df["Commission"]) / (point_value * df["QtyAbs"])
    return df

# =========================
# Metrics
# =========================

def compute_metrics(trades_df: pd.DataFrame, cfg: BacktestConfig) -> dict:
    df = trades_df.copy()
    pl_net = df["AdjustedNetPL"].fillna(0.0)
    equity = cfg.initial_capital + pl_net.cumsum()
    total_net = float(pl_net.sum())
    return {
        "strategy_name": cfg.strategy_name,
        "version": cfg.version,
        "timeframe": cfg.timeframe,
        "initial_capital": cfg.initial_capital,
        "num_trades": int(len(df)),
        "net_profit": total_net,
        "total_return_pct": (total_net / cfg.initial_capital) * 100.0,
        "profit_factor": _profit_factor(pl_net),
        "win_rate_pct": float((pl_net > 0).mean() * 100.0),
        "max_drawdown_pct": abs(_max_drawdown(equity)) * 100.0,
        "max_drawdown_dollars": float((equity.cummax() - equity).max()),
        "largest_winning_trade": float(pl_net.max() if len(pl_net) else 0),
        "largest_losing_trade": float(pl_net.min() if len(pl_net) else 0),
    }

# =========================
# Visuals + Markdown
# =========================

def save_visuals(trades_df: pd.DataFrame, cfg: BacktestConfig, outdir: str):
    pl = trades_df["AdjustedNetPL"].fillna(0.0)
    equity = cfg.initial_capital + pl.cumsum()
    dd = equity / equity.cummax() - 1.0

    plt.figure(figsize=(9,4.5)); plt.plot(equity.index, equity.values)
    plt.title("Equity Curve"); plt.savefig(os.path.join(outdir,"equity_curve_all.png")); plt.close()

    plt.figure(figsize=(9,4.5)); plt.plot(dd.index, dd.values*100)
    plt.title("Drawdown Curve"); plt.savefig(os.path.join(outdir,"drawdown_curve_all.png")); plt.close()

    plt.figure(figsize=(9,4.5)); plt.hist(trades_df["AdjustedNetPL"].dropna().values,bins=30)
    plt.title("Trade P/L Distribution"); plt.savefig(os.path.join(outdir,"pl_histogram_all.png")); plt.close()

def generate_analytics_md(metrics: dict, cfg: BacktestConfig, outdir: str):
    md = f"""# Strategy Analysis Report

**Strategy:** {metrics['strategy_name']}  
**Instrument:** {metrics.get('instrument','/UNK')}  
**Timeframe:** {metrics['timeframe']}  
**Run Date:** {datetime.now().strftime('%Y-%m-%d')}  

---

## Key Performance Indicators
- Net Profit: ${metrics['net_profit']:.2f}
- Return: {metrics['total_return_pct']:.2f}%
- Win Rate: {metrics['win_rate_pct']:.2f}%
- Profit Factor: {metrics['profit_factor']:.2f}
- Max Drawdown: ${metrics['max_drawdown_dollars']:.2f} ({metrics['max_drawdown_pct']:.2f}%)
- Trades: {metrics['num_trades']}

---

## Charts
![Equity](equity_curve_all.png)  
![Drawdown](drawdown_curve_all.png)  
![P/L Histogram](pl_histogram_all.png)  
"""
    with open(os.path.join(outdir,"analytics.md"),"w") as f: f.write(md)

# =========================
# Main runner
# =========================

def run_backtest(tos_csv_path: str, cfg: BacktestConfig):
    csv_stem = Path(tos_csv_path).stem.replace(" ","_")
    raw = load_tos_strategy_report(tos_csv_path)
    symbols = raw["Symbol"].dropna().unique().tolist() or ["/MES"]

    results = []
    for instr in symbols:
        trades = build_trades(raw, cfg.commission_per_round_trip)
        trades = apply_stoploss_corrections(trades, cfg.point_value)
        metrics = compute_metrics(trades, cfg)
        metrics["instrument"] = instr
        outdir = cfg.outdir(csv_stem, instr, cfg.strategy_name or "Unknown")
        os.makedirs(outdir, exist_ok=True)
        trades.to_csv(os.path.join(outdir,"trades_enriched.csv"), index=False)
        with open(os.path.join(outdir,"metrics.json"),"w") as f: json.dump(metrics,f,indent=2)
        save_visuals(trades,cfg,outdir)
        generate_analytics_md(metrics,cfg,outdir)
        results.append({"instrument":instr,"metrics":metrics,"outdir":outdir})
    return results

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Backtester with analytics (Vercel-friendly)")
    parser.add_argument("--csv", nargs="+", required=True, help="Path(s) to TOS CSV(s)")
    parser.add_argument("--timeframe", type=str, default="180d:15m", help="(ignored, kept for compatibility)")
    parser.add_argument("--capital", type=float, default=2500.0, help="Initial capital")
    parser.add_argument("--commission", type=float, default=4.04, help="Commission per round trip")
    parser.add_argument("--point_value", type=float, default=5.0, help="Point value ($ per point per contract)")
    args = parser.parse_args()

    cfg = BacktestConfig(
        initial_capital=args.capital,
        commission_per_round_trip=args.commission,
        point_value=args.point_value,
    )

    for item in args.csv:
        for csv_path in glob.glob(item):
            print(f"\n[RUN] {csv_path}")
            results = run_backtest(csv_path, cfg)
            for r in results:
                print(json.dumps(r["metrics"], indent=2))
