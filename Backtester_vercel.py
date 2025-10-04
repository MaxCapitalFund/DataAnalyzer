# -*- coding: utf-8 -*-
# Backtester_vercel.py — Clean, minimal working version
# Author: GPT-5 (for Larry Poe & Harrison)
# Optimized for Vercel / serverless (no chart output)

import os
import io
import re
import json
from dataclasses import dataclass, asdict
from datetime import datetime, time
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import pandas as pd

# =========================
# Configuration
# =========================

@dataclass
class BacktestConfig:
    strategy_name: str = ""
    instruments: Tuple[str, ...] = ("/MES",)
    timeframe: str = "180d:15m"
    session_hours_rth: Tuple[str, str] = ("09:30", "16:00")
    initial_capital: float = 2500.0
    commission_per_round_trip: float = 4.04
    point_value: float = 5.0
    version: str = "1.3.2"

    def outdir(self, csv_stem: str, instrument: str, strategy_label: str) -> str:
        temp_dir = Path("/tmp")
        day = datetime.now().strftime("%Y-%m-%d")
        timestamp = datetime.now().strftime("%H%M%S_%f")[:-3]
        safe_strategy = strategy_label.replace(" ", "_")
        safe_instr = (instrument or "UNK").replace("/", "")
        return str(temp_dir / f"Backtests_{day}_{safe_strategy}_{self.timeframe}_{safe_instr}_{csv_stem}_{timestamp}")

# =========================
# Helpers
# =========================

def _to_float(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.replace(r"[\$,]", "", regex=True)
    s = s.str.replace(r"\(([^()]*)\)", r"-\1", regex=True)
    s = s.replace("", np.nan)
    return pd.to_numeric(s, errors="coerce")

def _parse_datetime(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, format="%m/%d/%y %I:%M %p", errors="coerce")

PRE_START, PRE_END = time(3, 0), time(9, 15)
RTH_START, RTH_END = time(9, 30), time(16, 45)

def _tag_session(dt: pd.Timestamp) -> str:
    if pd.isna(dt):
        return "Unknown"
    t = dt.time()
    if PRE_START <= t <= PRE_END:
        return "PRE"
    if RTH_START <= t <= RTH_END:
        return "RTH"
    return "OFF"

def _in_rth(dt: pd.Timestamp) -> bool:
    if pd.isna(dt):
        return False
    t = dt.time()
    return RTH_START <= t <= RTH_END

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
    if any(w in s for w in ["STOP", "SL", "STOPPED"]):  return "Stop"
    if any(w in s for w in ["TIME", "TIME EXIT", "TIMED"]): return "Time"
    return "Close"

# =========================
# Load & Clean (TOS Strategy Report)
# =========================

def load_tos_strategy_report(file_path: str) -> pd.DataFrame:
    with open(file_path, "r", errors="replace") as f:
        lines = f.readlines()

    start_idx = None
    for i, line in enumerate(lines):
        if line.lstrip().startswith("Id;Strategy;"):
            start_idx = i
            break
    if start_idx is None:
        raise ValueError("No trade table header found in file.")

    df = pd.read_csv(io.StringIO("".join(lines[start_idx:])), sep=";")

    # --- Date/Time
    if "Date/Time" in df.columns:
        df["Date"] = _parse_datetime(df["Date/Time"])
    elif "Date" in df.columns and "Time" in df.columns:
        dt_str = df["Date"].astype(str) + " " + df["Time"].astype(str)
        df["Date"] = pd.to_datetime(dt_str, errors="coerce")

    # --- Strategy clean/tag
    df["BaseStrategy"] = df["Strategy"].astype(str).str.split("(").str[0].str.strip()
    df["Tag"] = df["Strategy"].astype(str).str.extract(r"\(([^()]*)\)", expand=False).fillna("")

    # --- Normalize side
    def normalize_side(v):
        if not isinstance(v, str): return ""
        v = v.upper()
        if "BTO" in v or "BUY TO OPEN" in v: return "BTO"
        if "STC" in v or "SELL TO CLOSE" in v: return "STC"
        if "STO" in v or "SELL TO OPEN" in v: return "STO"
        if "BTC" in v or "BUY TO CLOSE" in v: return "BTC"
        return v.strip()

    df["SideNorm"] = df["Side"].map(normalize_side)

    # --- Parse numerics
    df["TradePL"] = _to_float(df.get("Trade P/L", 0.0))
    df["CumPL"] = _to_float(df.get("P/L", 0.0))
    df["Qty"] = pd.to_numeric(df.get("Quantity", np.nan), errors="coerce")

    df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)
    return df

# =========================
# Trade pairing
# =========================

def build_trades(df: pd.DataFrame, commission_rt: float) -> pd.DataFrame:
    trades = []
    for tid, grp in df.groupby("Id", sort=False):
        g = grp.sort_values("Date").copy()
        entries = g[g["SideNorm"].isin(["BTO", "STO"])]
        exits = g[g["SideNorm"].isin(["STC", "BTC"])]

        if not len(entries) or not len(exits):
            continue

        entry = entries.iloc[0]
        exit_ = exits[exits["Date"] >= entry["Date"]].iloc[0] if len(exits[exits["Date"] >= entry["Date"]]) else exits.iloc[-1]

        qty = abs(entry.get("Qty", 1) or 1)
        trade_pl = float(exit_.get("TradePL", 0.0))
        commission = commission_rt * qty
        net_pl = trade_pl - commission

        trades.append({
            "Id": tid,
            "EntryTime": entry["Date"],
            "ExitTime": exit_["Date"],
            "EntrySide": entry["SideNorm"],
            "ExitSide": exit_["SideNorm"],
            "Direction": "Long" if entry["SideNorm"] == "BTO" else "Short",
            "TradePL": trade_pl,
            "Commission": commission,
            "NetPL": net_pl,
            "BaseStrategy": entry["BaseStrategy"],
            "Tag": entry["Tag"],
            "Session": _tag_session(entry["Date"])
        })

    return pd.DataFrame(trades)

# =========================
# Stop-loss correction
# =========================

def apply_stoploss_cap(trades: pd.DataFrame, point_value: float) -> pd.DataFrame:
    """Cap all trades at -$100 or -20 points."""
    df = trades.copy()
    stop_cap = -100.0
    df["SLBreached"] = df["NetPL"] < stop_cap
    df["AdjustedNetPL"] = np.where(df["SLBreached"], stop_cap, df["NetPL"])
    df["PointsPerContract"] = df["AdjustedNetPL"] / point_value
    return df

# =========================
# Compute metrics
# =========================

def compute_metrics(trades: pd.DataFrame, cfg: BacktestConfig) -> dict:
    if trades.empty:
        return {}

    pl = trades["AdjustedNetPL"].fillna(0.0)
    equity = cfg.initial_capital + pl.cumsum()

    total_net = float(pl.sum())
    total_return_pct = (total_net / cfg.initial_capital) * 100.0 if cfg.initial_capital else 0
    avg_win = float(pl[pl > 0].mean()) if any(pl > 0) else 0
    avg_loss = float(pl[pl < 0].mean()) if any(pl < 0) else 0

    max_dd = abs(_max_drawdown(equity)) * 100.0
    max_dd_dollars = float((equity.cummax() - equity).max())
    recovery_factor = total_net / max_dd_dollars if max_dd_dollars else np.nan

    metrics = {
        "strategy": cfg.strategy_name,
        "version": cfg.version,
        "num_trades": len(trades),
        "net_profit": total_net,
        "return_pct": total_return_pct,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "profit_factor": _profit_factor(pl),
        "max_drawdown_pct": max_dd,
        "max_drawdown_dollars": max_dd_dollars,
        "recovery_factor": recovery_factor,
        "win_rate_pct": float((pl > 0).mean() * 100.0)
    }
    return metrics

# =========================
# Analytics Markdown
# =========================

def save_analytics_md(trades: pd.DataFrame, metrics: dict, cfg: BacktestConfig, outdir: str):
    os.makedirs(outdir, exist_ok=True)
    m = metrics

    md = f"""# Strategy Analysis Report

**Strategy:** {m.get('strategy')}  
**Instrument:** /MES  
**Timeframe:** {cfg.timeframe}  
**Run Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**Initial Capital:** ${cfg.initial_capital:,.2f}  
**Commission (RT):** ${cfg.commission_per_round_trip:.2f}  
**Stop Cap:** $100 / 20 pts  

---

## Key Metrics
- **Total Trades:** {m.get('num_trades', 0)}
- **Net Profit:** ${m.get('net_profit', 0):,.2f}
- **Total Return:** {m.get('return_pct', 0):.2f}%
- **Win Rate:** {m.get('win_rate_pct', 0):.2f}%
- **Profit Factor:** {m.get('profit_factor', 0):.2f}
- **Max Drawdown:** ${m.get('max_drawdown_dollars', 0):,.2f} ({m.get('max_drawdown_pct', 0):.2f}%)
- **Recovery Factor:** {m.get('recovery_factor', 0):.2f}

---

## Averages
- **Average Win:** ${m.get('avg_win', 0):,.2f}
- **Average Loss:** ${m.get('avg_loss', 0):,.2f}

---

*Report generated by Backtester_vercel.py v{cfg.version}*
"""
    with open(os.path.join(outdir, "analytics.md"), "w", encoding="utf-8") as f:
        f.write(md)

# =========================
# Runner
# =========================

def run_backtest(tos_csv_path: str, cfg: BacktestConfig):
    csv_stem = Path(tos_csv_path).stem.replace(" ", "_")
    df = load_tos_strategy_report(tos_csv_path)
    cfg.strategy_name = df["BaseStrategy"].iloc[0]

    trades = build_trades(df, cfg.commission_per_round_trip)
    trades = apply_stoploss_cap(trades, cfg.point_value)
    metrics = compute_metrics(trades, cfg)

    outdir = cfg.outdir(csv_stem, "/MES", cfg.strategy_name)
    trades.to_csv(os.path.join(outdir, "trades_enriched.csv"), index=False)
    save_analytics_md(trades, metrics, cfg, outdir)

    print(f"✅ Backtest complete — {len(trades)} trades | Net P/L ${metrics['net_profit']:.2f}")
    print(f"📁 Output directory: {outdir}")
    return metrics

if __name__ == "__main__":
    import argparse, glob, sys

    parser = argparse.ArgumentParser(description="Minimal TOS Strategy Report Backtester for /MES")
    parser.add_argument("--csv", required=True, help="Path to TOS Strategy Report CSV")
    parser.add_argument("--capital", type=float, default=2500.0, help="Initial capital")
    parser.add_argument("--commission", type=float, default=4.04, help="Commission per RT")
    args = parser.parse_args()

    cfg = BacktestConfig(initial_capital=args.capital, commission_per_round_trip=args.commission)
    run_backtest(args.csv, cfg)
