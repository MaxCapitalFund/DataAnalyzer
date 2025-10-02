# -*- coding: utf-8 -*-
# Simplified backtester for Vercel deployment
# Robust + compatible with extra CLI args (parse_known_args)

import os
import io
import re
import json
import warnings
from dataclasses import dataclass, asdict
from datetime import datetime, time
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for serverless
import matplotlib.pyplot as plt


# =========================
# Configuration
# =========================

@dataclass
class BacktestConfig:
    strategy_name: str = ""
    instruments: Tuple[str, ...] = ("/MES",)
    timeframe: str = "180d:15m"  # default only for labeling
    initial_capital: float = 2500.0
    commission_per_round_trip: float = 4.04
    point_value: float = 5.0
    version: str = "1.4.3"  # patched version

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
        return float('inf') if gp > 0 else 0.0
    return float(gp / gl)


# =========================
# Load & Build Trades
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

    # Parse date/time
    if 'Date/Time' in df.columns:
        df['Date'] = _parse_datetime(df['Date/Time'])
    elif 'Date' in df.columns and 'Time' in df.columns:
        dt_str = df['Date'].astype(str).str.strip() + ' ' + df['Time'].astype(str).str.strip()
        df['Date'] = pd.to_datetime(dt_str, errors='coerce')
    elif 'Date' in df.columns:
        df['Date'] = _parse_datetime(df['Date'])
    else:
        raise ValueError("Could not find 'Date/Time' or 'Date' column.")

    # Money fields
    if 'Trade P/L' in df.columns:
        df['TradePL'] = _to_float(df['Trade P/L']).fillna(0.0)
    elif 'TradePL' in df.columns:
        df['TradePL'] = _to_float(df['TradePL']).fillna(0.0)
    else:
        df['TradePL'] = 0.0

    df['CumPL'] = _to_float(df['P/L']) if 'P/L' in df.columns else np.nan

    return df


def build_trades(df: pd.DataFrame, commission_rt: float) -> pd.DataFrame:
    trades = []
    for i in range(0, len(df) - 1, 2):
        entry = df.iloc[i]
        exit_ = df.iloc[i + 1]

        qty_abs = 1.0
        trade_pl = entry.get('TradePL', 0.0)
        commission = commission_rt * qty_abs
        net_pl = (trade_pl if pd.notna(trade_pl) else 0.0) - commission

        trades.append({
            'EntryTime': entry['Date'],
            'ExitTime': exit_['Date'],
            'TradePL': trade_pl,
            'Commission': commission,
            'NetPL': net_pl,
        })

    t = pd.DataFrame(trades)
    if not t.empty:
        t['HoldMins'] = (t['ExitTime'] - t['EntryTime']).dt.total_seconds() / 60.0
    return t


# =========================
# Stoploss & Metrics
# =========================

def apply_stoploss_corrections(trades: pd.DataFrame, point_value: float) -> pd.DataFrame:
    df = trades.copy()
    df['SLBreached'] = df['NetPL'] < -100.0
    df['AdjustedNetPL'] = np.where(df['SLBreached'], -100.0, df['NetPL'])
    return df

def compute_metrics(trades_df: pd.DataFrame, cfg: BacktestConfig) -> dict:
    df = trades_df.copy()
    if 'AdjustedNetPL' not in df.columns:
        raise RuntimeError("Call apply_stoploss_corrections first")

    pl_net = df['AdjustedNetPL'].fillna(0.0)
    equity = cfg.initial_capital + pl_net.cumsum()

    total_net = float(pl_net.sum())
    metrics = {
        "net_profit": total_net,
        "num_trades": int(len(df)),
        "profit_factor": _profit_factor(pl_net),
        "max_drawdown": _max_drawdown(equity),
    }
    return metrics


# =========================
# Main Runner
# =========================

def run_backtest(tos_csv_path: str, cfg: BacktestConfig):
    raw = load_tos_strategy_report(tos_csv_path)
    trades = build_trades(raw, cfg.commission_per_round_trip)
    trades = apply_stoploss_corrections(trades, cfg.point_value)
    metrics = compute_metrics(trades, cfg)
    return trades, metrics


if __name__ == "__main__":
    import argparse, sys, glob

    parser = argparse.ArgumentParser(description="Lean backtester (Vercel compatible)")
    parser.add_argument("--csv", nargs="+", required=True, help="Path(s) to TOS CSV(s)")
    parser.add_argument("--capital", type=float, default=2500.0, help="Initial capital")
    parser.add_argument("--commission", type=float, default=4.04, help="Commission per round trip")
    parser.add_argument("--point_value", type=float, default=5.0, help="Point value per contract")

    # 👇 IMPORTANT: tolerate unknown args like --timeframe
    args, unknown = parser.parse_known_args()
    if unknown:
        print(f"[WARN] Ignoring unknown arguments: {unknown}")

    cfg = BacktestConfig(
        initial_capital=args.capital,
        commission_per_round_trip=args.commission,
        point_value=args.point_value,
    )

    all_metrics = []
    for item in args.csv:
        matches = glob.glob(item)
        for csv_path in matches:
            print(f"[RUN] CSV: {csv_path}")
            trades, metrics = run_backtest(csv_path, cfg)
            metrics["csv"] = os.path.basename(csv_path)
            all_metrics.append(metrics)

    print(json.dumps(all_metrics, indent=2))
