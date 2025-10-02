# -*- coding: utf-8 -*-
# Clean Backtester: Vercel-optimized
# Outputs: trades_enriched.csv, metrics.json
# Fixed: metrics (Avg Win/Loss, Largest Win/Loss, Expectancy, Sharpe now show values)

import os
import io
import re
import json
import warnings
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # serverless safe
import matplotlib.pyplot as plt

# =========================
# Config
# =========================

@dataclass
class BacktestConfig:
    strategy_name: str = ""
    instruments: Tuple[str, ...] = ("/MES",)
    timeframe: str = "180d:15m"
    initial_capital: float = 2500.0
    commission_per_round_trip: float = 4.04
    point_value: float = 5.0
    version: str = "1.5"

    def outdir(self, csv_stem: str, instrument: str) -> str:
        temp_dir = Path('/tmp')
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_instr = instrument.replace("/", "")
        return str(temp_dir / f"Backtest_{ts}_{safe_instr}_{csv_stem}")

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
    if equity_curve.empty:
        return 0.0
    running_max = equity_curve.cummax()
    drawdown = equity_curve / running_max - 1.0
    return float(drawdown.min())

def _profit_factor(pl: pd.Series) -> float:
    gp = pl[pl > 0].sum()
    gl = -pl[pl < 0].sum()
    if gl == 0:
        return float('inf') if gp > 0 else 0.0
    return float(gp / gl)

# =========================
# Load TOS CSV
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
        dt_str = df['Date'].astype(str).str.strip() + ' ' + df['Time'].astype(str).str.strip()
        df['Date'] = pd.to_datetime(dt_str, errors='coerce')
    else:
        raise ValueError("No Date column found")

    if 'Trade P/L' in df.columns:
        df['TradePL'] = _to_float(df['Trade P/L']).fillna(0.0)
    else:
        df['TradePL'] = 0.0

    df['CumPL'] = _to_float(df['P/L']) if 'P/L' in df.columns else np.nan
    return df.dropna(subset=['Date']).sort_values('Date').reset_index(drop=True)

# =========================
# Build Trades
# =========================

def build_trades(df: pd.DataFrame, commission_rt: float) -> pd.DataFrame:
    trades = []
    for i in range(0, len(df) - 1, 2):
        entry = df.iloc[i]
        exit_ = df.iloc[i + 1]
        pl = float(exit_.get('TradePL', 0.0))
        commission = commission_rt
        net_pl = pl - commission
        trades.append({
            'EntryTime': entry['Date'],
            'ExitTime': exit_['Date'],
            'TradePL': pl,
            'Commission': commission,
            'NetPL': net_pl
        })
    t = pd.DataFrame(trades)
    if t.empty:
        return t
    t['HoldMins'] = (t['ExitTime'] - t['EntryTime']).dt.total_seconds() / 60.0
    return t

# =========================
# Stop-loss
# =========================

def apply_stoploss(trades: pd.DataFrame) -> pd.DataFrame:
    df = trades.copy()
    df['AdjustedNetPL'] = np.where(df['NetPL'] < -100.0, -100.0, df['NetPL'])
    return df

# =========================
# Metrics (fixed)
# =========================

def compute_metrics(trades: pd.DataFrame, cfg: BacktestConfig) -> dict:
    if trades.empty:
        return {"strategy": cfg.strategy_name, "timeframe": cfg.timeframe, "num_trades": 0}

    pl = trades['AdjustedNetPL'].fillna(0.0)
    equity = cfg.initial_capital + pl.cumsum()

    wins = pl[pl > 0]
    losses = pl[pl < 0]

    metrics = {
        "strategy": cfg.strategy_name or "Unknown",
        "timeframe": cfg.timeframe,
        "num_trades": int(len(trades)),
        "net_profit": float(pl.sum()),
        "total_return_pct": (pl.sum() / cfg.initial_capital) * 100,
        "win_rate_pct": float((len(wins) / len(trades)) * 100) if len(trades) else 0.0,
        "profit_factor": _profit_factor(pl),
        "max_drawdown_pct": abs(_max_drawdown(equity)) * 100,
        "gross_profit": float(wins.sum()),
        "gross_loss": float(losses.sum()),
        "avg_win": float(wins.mean()) if not wins.empty else 0.0,
        "avg_loss": float(losses.mean()) if not losses.empty else 0.0,
        "largest_win": float(wins.max()) if not wins.empty else 0.0,
        "largest_loss": float(losses.min()) if not losses.empty else 0.0,
        "expectancy": float(pl.mean()) if len(pl) else 0.0,
        "recovery_factor": float(pl.sum() / (abs(_max_drawdown(equity)) * cfg.initial_capital)) if _max_drawdown(equity) else 0.0,
        "sharpe_ratio": float(pl.mean() / pl.std()) if pl.std() else 0.0
    }
    return metrics

# =========================
# Runner
# =========================

def run_backtest(tos_csv_path: str, cfg: BacktestConfig):
    csv_stem = Path(tos_csv_path).stem.replace(' ', '_')
    raw = load_tos_strategy_report(tos_csv_path)
    trades = build_trades(raw, cfg.commission_per_round_trip)
    trades = apply_stoploss(trades)

    outdir = cfg.outdir(csv_stem, "MES")
    os.makedirs(outdir, exist_ok=True)
    trades.to_csv(os.path.join(outdir, "trades_enriched.csv"), index=False)

    metrics = compute_metrics(trades, cfg)
    with open(os.path.join(outdir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(json.dumps(metrics, indent=2))
    return metrics, outdir

if __name__ == "__main__":
    import argparse, glob, sys
    parser = argparse.ArgumentParser(description="Backtest TOS Strategy Report CSV")
    parser.add_argument("--csv", nargs="+", required=True, help="CSV file(s)")
    parser.add_argument("--timeframe", type=str, default="180d:15m")
    parser.add_argument("--capital", type=float, default=2500.0)
    parser.add_argument("--commission", type=float, default=4.04)
    args = parser.parse_args()

    cfg = BacktestConfig(
        timeframe=args.timeframe,
        initial_capital=args.capital,
        commission_per_round_trip=args.commission
    )

    for csv_path in args.csv:
        run_backtest(csv_path, cfg)
