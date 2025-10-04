# -*- coding: utf-8 -*-
# Backtester_vercel.py — CLEAN ERROR-FREE VERSION (v1.3.10)
# Author: GPT-5 for Larry Poe
# Description:
# Parses ThinkorSwim Strategy Report CSV (Trade P/L, P/L)
# Applies $100 stop-loss cap (20 pts for /MES)
# Deducts commissions post-cap
# Generates enriched CSV + markdown analytics
# Serverless-safe for Vercel (no charts)
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
# CONFIGURATION
# =========================
@dataclass
class BacktestConfig:
    strategy_name: str = "Unknown"
    instruments: Tuple[str, ...] = ("/MES",)
    timeframe: str = "180d:15m"
    initial_capital: float = 2500.0
    commission_per_round_trip: float = 4.04
    point_value: float = 5.0
    version: str = "1.3.10"
    def outdir(self, csv_stem: str, instrument: str, strategy_label: str) -> str:
        temp_dir = Path("/tmp")
        day = datetime.now().strftime("%Y-%m-%d")
        timestamp = datetime.now().strftime("%H%M%S_%f")[:-3]
        safe_strategy = strategy_label.replace(" ", "_")
        safe_instr = (instrument or "UNK").replace("/", "")
        safe_timeframe = self.timeframe.replace(":", "_").replace(" ", "_")
        return str(temp_dir / f"Backtests_{day}_{safe_strategy}_{safe_timeframe}_{safe_instr}_{csv_stem}_{timestamp}")
# =========================
# HELPERS
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
    if PRE_START <= t <= PRE_END: return "PRE"
    if RTH_START <= t <= RTH_END: return "RTH"
    return "OFF"
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
# =========================
# LOAD & CLEAN
# =========================
def load_tos_strategy_report(file_path: str) -> pd.DataFrame:
    try:
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
        print(f"DEBUG: Loaded {len(lines)} lines from {file_path}")
    except FileNotFoundError:
        print(f"ERROR: Input CSV file not found: {file_path}")
        raise RuntimeError(f"Input CSV file not found: {file_path}")
    except Exception as e:
        print(f"ERROR: Failed to read CSV file {file_path}: {e}")
        raise RuntimeError(f"Failed to read CSV file {file_path}: {e}")
    
    if not lines:
        print(f"ERROR: CSV file {file_path} is empty")
        raise RuntimeError(f"CSV file {file_path} is empty")
    
    start_idx = None
    for i, line in enumerate(lines):
        if line.lstrip().startswith("Id;Strategy;"):
            start_idx = i
            break
    if start_idx is None:
        print(f"ERROR: No trade table header found in {file_path}")
        raise RuntimeError(f"No trade table header found in {file_path}")
    
    try:
        df = pd.read_csv(io.StringIO("".join(lines[start_idx:])), sep=";")
        print(f"DEBUG: DataFrame shape after loading: {df.shape}")
    except Exception as e:
        print(f"ERROR: Failed to parse CSV data: {e}")
        raise RuntimeError(f"Failed to parse CSV data: {e}")
    
    if df.empty:
        print(f"WARNING: Parsed DataFrame is empty for {file_path}")
        return df
    
    # Normalize column names
    df.rename(columns={"Trade P/L": "TradePL", "P/L": "CumPL"}, inplace=True)
    # Parse Date/Time
    df["Date"] = _parse_datetime(df["Date/Time"])
    # Strategy cleanup
    df["BaseStrategy"] = df["Strategy"].astype(str).str.split("(").str[0].str.strip()
    df["Tag"] = df["Strategy"].astype(str).str.extract(r"\(([^()]*)\)", expand=False).fillna("")
    # Normalize side
    def normalize_side(v):
        if not isinstance(v, str): return ""
        v = v.upper()
        if "BTO" in v or "BUY TO OPEN" in v: return "BTO"
        if "STC" in v or "SELL TO CLOSE" in v: return "STC"
        if "STO" in v or "SELL TO OPEN" in v: return "STO"
        if "BTC" in v or "BUY TO CLOSE" in v: return "BTC"
        return v.strip()
    df["SideNorm"] = df["Side"].map(normalize_side)
    df["TradePL"] = _to_float(df["TradePL"]) # gross
    df["CumPL"] = _to_float(df["CumPL"]) # cumulative
    df["Qty"] = pd.to_numeric(df["Quantity"], errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)
    print(f"DEBUG: DataFrame shape after cleaning: {df.shape}")
    return df
# =========================
# BUILD TRADES
# =========================
def build_trades(df: pd.DataFrame, commission_rt: float) -> pd.DataFrame:
    if df.empty:
        print("DEBUG: Input DataFrame is empty, no trades to build")
        return pd.DataFrame(columns=[
            "Id", "EntryTime", "ExitTime", "EntrySide", "ExitSide",
            "Direction", "TradePL", "Commission", "NetPL",
            "BaseStrategy", "Tag", "Session"
        ])
    
    trades = []
    unique_ids = df["Id"].unique()
    print(f"DEBUG: Grouping by {len(unique_ids)} unique trade IDs")
    for tid, grp in df.groupby("Id", sort=False):
        g = grp.sort_values("Date").copy()
        print(f"DEBUG: Processing trade ID {tid}, group size: {len(g)}")
        entries = g[g["SideNorm"].isin(["BTO", "STO"])]
        exits = g[g["SideNorm"].isin(["STC", "BTC"])]
        if not len(entries):
            print(f"DEBUG: Skipping trade ID {tid}: no valid entries (BTO/STO)")
            continue
        if not len(exits):
            print(f"DEBUG: Skipping trade ID {tid}: no valid exits (STC/BTC)")
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
    print(f"DEBUG: Built {len(trades)} trades")
    if not trades:
        return pd.DataFrame(columns=[
            "Id", "EntryTime", "ExitTime", "EntrySide", "ExitSide",
            "Direction", "TradePL", "Commission", "NetPL",
            "BaseStrategy", "Tag", "Session"
        ])
    return pd.DataFrame(trades)
# =========================
# STOP-LOSS CAP
# =========================
def apply_stoploss_cap(trades: pd.DataFrame, point_value: float) -> pd.DataFrame:
    """Apply $100 (20-point) stop cap to gross TradePL before commission."""
    if trades.empty:
        print("DEBUG: Trades DataFrame is empty, skipping stop-loss cap")
        return trades
    df = trades.copy()
    stop_cap = -100.0 # $100 = 20 pts
    df["SLBreached"] = df["TradePL"] < stop_cap
    df["AdjustedGrossPL"] = np.where(df["SLBreached"], stop_cap, df["TradePL"])
    df["AdjustedNetPL"] = df["AdjustedGrossPL"] - df["Commission"]
    df["PointsPerContract"] = df["AdjustedNetPL"] / point_value
    print(f"DEBUG: Applied stop-loss cap, DataFrame shape: {df.shape}")
    return df
# =========================
# METRICS
# =========================
def compute_metrics(trades: pd.DataFrame, cfg: BacktestConfig) -> dict:
    metrics = {
        "status": "success",
        "message": "Metrics computed",
        "strategy": cfg.strategy_name,
        "version": cfg.version,
        "num_trades": 0,
        "net_profit": 0.0,
        "return_pct": 0.0,
        "avg_win": 0.0,
        "avg_loss": 0.0,
        "profit_factor": 0.0,
        "max_drawdown_pct": 0.0,
        "max_drawdown_dollars": 0.0,
        "recovery_factor": 0.0,
        "win_rate_pct": 0.0
    }
    if trades.empty:
        print("DEBUG: No trades to compute metrics, returning default metrics")
        metrics.update({"message": "No trades to compute metrics"})
        return metrics
    pl = trades["AdjustedNetPL"].fillna(0.0)
    equity = cfg.initial_capital + pl.cumsum()
    total_net = float(pl.sum())
    total_return_pct = (total_net / cfg.initial_capital) * 100.0
    avg_win = float(pl[pl > 0].mean()) if any(pl > 0) else 0.0
    avg_loss = float(pl[pl < 0].mean()) if any(pl < 0) else 0.0
    max_dd = abs(_max_drawdown(equity)) * 100.0
    max_dd_dollars = float((equity.cummax() - equity).max())
    recovery_factor = total_net / max_dd_dollars if max_dd_dollars else 0.0
    metrics.update({
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
    })
    print(f"DEBUG: Computed metrics: {metrics}")
    return metrics
# =========================
# MARKDOWN REPORT
# =========================
def save_analytics_md(trades: pd.DataFrame, metrics: dict, cfg: BacktestConfig, outdir: str):
    try:
        os.makedirs(outdir, exist_ok=True)
        print(f"DEBUG: Created/verified directory for markdown: {outdir}")
    except OSError as e:
        print(f"ERROR: Failed to create directory {outdir}: {e}")
        raise RuntimeError(f"Failed to create directory {outdir}: {e}")
    
    m = metrics or {}
    md = f"""# Strategy Analysis Report
**Strategy:** {m.get('strategy', 'Unknown')}
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
    try:
        with open(os.path.join(outdir, "analytics.md"), "w", encoding="utf-8") as f:
            f.write(md)
        print(f"DEBUG: Successfully wrote analytics.md to {outdir}")
    except OSError as e:
        print(f"ERROR: Failed to write analytics.md to {outdir}: {e}")
        raise RuntimeError(f"Failed to write analytics.md to {outdir}: {e}")
# =========================
# RUNNER
# =========================
def run_backtest(tos_csv_path: str, cfg: BacktestConfig):
    metrics = {
        "status": "failed",
        "message": "Backtest not started",
        "strategy": cfg.strategy_name,
        "version": cfg.version,
        "num_trades": 0,
        "net_profit": 0.0
    }
    outdir = None
    try:
        csv_stem = Path(tos_csv_path).stem.replace(" ", "_")
        print(f"DEBUG: Starting backtest with CSV: {tos_csv_path}, stem: {csv_stem}")
        df = load_tos_strategy_report(tos_csv_path)
        cfg.strategy_name = df["BaseStrategy"].iloc[0] if "BaseStrategy" in df.columns and not df.empty else cfg.strategy_name
        print(f"DEBUG: Strategy name set to: {cfg.strategy_name}")
        trades = build_trades(df, cfg.commission_per_round_trip)
        print(f"DEBUG: Trades DataFrame shape: {trades.shape}")
        if trades.empty:
            print("WARNING: No trades generated, returning default metrics")
            metrics.update({"message": "No trades generated from CSV"})
            outdir = cfg.outdir(csv_stem, "/MES", cfg.strategy_name)
            try:
                os.makedirs(outdir, exist_ok=True)
                print(f"DEBUG: Created/verified directory: {outdir}")
                with open(os.path.join(outdir, "analytics.md"), "w", encoding="utf-8") as f:
                    f.write("# Strategy Analysis Report\nNo trades generated from CSV.")
                print(f"DEBUG: Successfully wrote analytics.md to {outdir}")
            except OSError as e:
                print(f"ERROR: Failed to write analytics.md to {outdir}: {e}")
                metrics.update({"message": f"No trades generated; failed to write analytics.md: {e}"})
            return metrics
        trades = apply_stoploss_cap(trades, cfg.point_value)
        metrics = compute_metrics(trades, cfg)
        
        outdir = cfg.outdir(csv_stem, "/MES", cfg.strategy_name)
        try:
            os.makedirs(outdir, exist_ok=True)
            print(f"DEBUG: Created/verified directory: {outdir}")
        except OSError as e:
            print(f"ERROR: Failed to create directory {outdir}: {e}")
            metrics.update({"message": f"Failed to create directory {outdir}: {e}"})
            return metrics
        
        try:
            trades.to_csv(os.path.join(outdir, "trades_enriched.csv"), index=False)
            print(f"DEBUG: Successfully wrote trades_enriched.csv to {outdir}")
        except OSError as e:
            print(f"ERROR: Failed to write trades_enriched.csv to {outdir}: {e}")
            metrics.update({"message": f"Failed to write trades_enriched.csv: {e}"})
            return metrics
        
        save_analytics_md(trades, metrics, cfg, outdir)
        metrics.update({"status": "success", "message": "Backtest completed"})
        print(f"✅ Backtest complete — {len(trades)} trades | Net P/L ${metrics.get('net_profit', 0):.2f}")
        print(f"📁 Output directory: {outdir}")
        return metrics
    except Exception as e:
        print(f"ERROR: Backtest failed: {str(e)}")
        metrics.update({"message": f"Backtest failed: {str(e)}"})
        return metrics
# =========================
# MAIN
# =========================
if __name__ == "__main__":
    import argparse, sys
    parser = argparse.ArgumentParser(description="ThinkorSwim Strategy Report Backtester for /MES")
    parser.add_argument("--csv", required=True, help="Path to TOS Strategy Report CSV")
    parser.add_argument("--timeframe", type=str, default="180d:15m", help="Display label for timeframe")
    parser.add_argument("--capital", type=float, default=2500.0, help="Initial capital")
    parser.add_argument("--commission", type=float, default=4.04, help="Commission per RT")
    args = parser.parse_args()
    cfg = BacktestConfig(
        initial_capital=args.capital,
        commission_per_round_trip=args.commission,
        timeframe=args.timeframe
    )
    result = run_backtest(args.csv, cfg)
    if result.get("status") == "failed":
        print(f"ERROR: {result.get('message')}")
        sys.exit(1)
