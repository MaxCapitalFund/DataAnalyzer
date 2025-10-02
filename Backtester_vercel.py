# -*- coding: utf-8 -*-
# Clean Backtester for Vercel
# - Creates Backtests_... directories (plural) so Vercel detects them
# - Stop-loss capped at -$100/trade
# - Outputs: trades_enriched.csv, metrics.json, config.json, charts

import os
import io
import re
import json
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Serverless backend
import matplotlib.pyplot as plt

# =========================
# Config
# =========================

@dataclass
class BacktestConfig:
    strategy_name: str = "Unknown"
    instruments: Tuple[str, ...] = ("/MES",)
    timeframe: str = "180d:15m"
    initial_capital: float = 2500.0
    commission_per_round_trip: float = 4.04
    point_value: float = 5.0
    version: str = "1.5.0"

    def outdir(self, csv_stem: str, instrument: str) -> str:
        """Always return Backtests_... (plural) so Vercel finds it"""
        temp_dir = Path('/tmp')
        day = datetime.now().strftime("%Y-%m-%d")
        timestamp = datetime.now().strftime("%H%M%S")
        safe_instr = (instrument or "UNK").replace("/", "")
        return str(temp_dir / f"Backtests_{day}_{safe_instr}_{csv_stem}_{timestamp}")

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
    gp = pl[pl > 0].sum()
    gl = -pl[pl < 0].sum()
    if gl == 0:
        return float('inf') if gp > 0 else 0.0
    return gp / gl

# =========================
# Loader
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
        raise ValueError("No header found in file")
    table_str = "".join(lines[start_idx:])
    df = pd.read_csv(io.StringIO(table_str), sep=';')
    if 'Date/Time' in df.columns:
        df['Date'] = _parse_datetime(df['Date/Time'])
    else:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df['TradePL'] = _to_float(df['Trade P/L']) if 'Trade P/L' in df.columns else 0.0
    return df.dropna(subset=['Date']).sort_values('Date').reset_index(drop=True)

# =========================
# Trades
# =========================

def build_trades(df: pd.DataFrame, commission_rt: float) -> pd.DataFrame:
    trades = []
    for i in range(0, len(df) - 1, 2):
        entry, exit_ = df.iloc[i], df.iloc[i + 1]
        qty = 1
        trade_pl = float(exit_.get('TradePL', 0.0))
        commission = commission_rt * qty
        net_pl = trade_pl - commission
        trades.append({
            'EntryTime': entry['Date'],
            'ExitTime': exit_['Date'],
            'TradePL': trade_pl,
            'Commission': commission,
            'NetPL': net_pl
        })
    t = pd.DataFrame(trades)
    if not t.empty:
        t['HoldMins'] = (t['ExitTime'] - t['EntryTime']).dt.total_seconds() / 60.0
    return t

def apply_stoploss(trades: pd.DataFrame, point_value: float) -> pd.DataFrame:
    df = trades.copy()
    df['AdjustedNetPL'] = np.where(df['NetPL'] < -100.0, -100.0, df['NetPL'])
    return df

# =========================
# Metrics
# =========================

def compute_metrics(trades: pd.DataFrame, cfg: BacktestConfig) -> dict:
    if trades.empty:
        return {"strategy": cfg.strategy_name, "timeframe": cfg.timeframe, "num_trades": 0}
    pl = trades['AdjustedNetPL']
    equity = cfg.initial_capital + pl.cumsum()
    return {
        "strategy": cfg.strategy_name,
        "timeframe": cfg.timeframe,
        "num_trades": int(len(trades)),
        "net_profit": float(pl.sum()),
        "total_return_pct": (pl.sum() / cfg.initial_capital) * 100,
        "win_rate_pct": float((pl > 0).mean() * 100),
        "profit_factor": _profit_factor(pl),
        "max_drawdown_pct": abs(_max_drawdown(equity)) * 100,
        "gross_profit": float(pl[pl > 0].sum()),
        "gross_loss": float(pl[pl < 0].sum()),
        "avg_win": float(pl[pl > 0].mean()) if (pl > 0).any() else np.nan,
        "avg_loss": float(pl[pl < 0].mean()) if (pl < 0).any() else np.nan,
        "largest_win": float(pl.max()),
        "largest_loss": float(pl.min()),
        "expectancy": float(pl.mean()),
        "recovery_factor": float(pl.sum() / (abs(_max_drawdown(equity)) * cfg.initial_capital)) if _max_drawdown(equity) else np.nan,
        "sharpe_ratio": float(pl.mean() / pl.std()) if pl.std() else np.nan
    }

# =========================
# Runner
# =========================

def run_backtest(csv_path: str, cfg: BacktestConfig):
    csv_stem = Path(csv_path).stem.replace(" ", "_")
    raw = load_tos_strategy_report(csv_path)
    outdir = cfg.outdir(csv_stem, "/MES")
    os.makedirs(outdir, exist_ok=True)
    trades = build_trades(raw, cfg.commission_per_round_trip)
    trades = apply_stoploss(trades, cfg.point_value)
    metrics = compute_metrics(trades, cfg)
    # save outputs
    trades.to_csv(os.path.join(outdir, "trades_enriched.csv"), index=False)
    with open(os.path.join(outdir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    with open(os.path.join(outdir, "config.json"), "w") as f:
        json.dump(asdict(cfg), f, indent=2)
    return metrics, outdir

# =========================
# CLI
# =========================

if __name__ == "__main__":
    import argparse, glob, sys
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", nargs="+", required=True)
    parser.add_argument("--timeframe", type=str, default="180d:15m")
    parser.add_argument("--capital", type=float, default=2500.0)
    parser.add_argument("--commission", type=float, default=4.04)
    parser.add_argument("--point_value", type=float, default=5.0)
    args = parser.parse_args()
    cfg = BacktestConfig(timeframe=args.timeframe,
                         initial_capital=args.capital,
                         commission_per_round_trip=args.commission,
                         point_value=args.point_value)
    resolved = []
    for item in args.csv:
        matches = glob.glob(item)
        resolved.extend(matches if matches else [item])
    csv_paths = [p for p in resolved if Path(p).exists()]
    if not csv_paths:
        print("[ERROR] No CSV found", file=sys.stderr)
        sys.exit(1)
    results = []
    for path in csv_paths:
        metrics, outdir = run_backtest(path, cfg)
        metrics["csv"] = Path(path).name
        results.append(metrics)
    print(json.dumps(results, indent=2))
