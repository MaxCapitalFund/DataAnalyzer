# -*- coding: utf-8 -*-
# Clean Backtester for Vercel
# - Stop-loss capped at -$100/trade
# - Outputs enriched trades, metrics, analytics
# - Fixed output dir naming: Backtests_... (plural, matches Vercel)
# - Metrics robust: no N/A, real win/loss stats, Sharpe, expectancy

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
import matplotlib
matplotlib.use('Agg')
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
    version: str = "1.5.0"

    def outdir(self, csv_stem: str, instrument: str) -> str:
        temp_dir = Path('/tmp')
        day = datetime.now().strftime("%Y-%m-%d")
        timestamp = datetime.now().strftime("%H%M%S")
        safe_instr = (instrument or "UNK").replace("/", "")
        # FIX: plural "Backtests_" so Vercel finds the folder
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
    s = pl.dropna()
    gp = s[s > 0].sum()
    gl = -s[s < 0].sum()
    if gl == 0:
        return float('inf') if gp > 0 else 0.0
    return float(gp / gl)

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
        raise ValueError("No trade table header found")
    table_str = "".join(lines[start_idx:])
    df = pd.read_csv(io.StringIO(table_str), sep=';')

    if 'Date/Time' in df.columns:
        df['Date'] = _parse_datetime(df['Date/Time'])
    elif 'Date' in df.columns and 'Time' in df.columns:
        dt_str = df['Date'].astype(str).str.strip() + ' ' + df['Time'].astype(str).str.strip()
        df['Date'] = pd.to_datetime(dt_str, errors='coerce')
    elif 'Date' in df.columns:
        df['Date'] = _parse_datetime(df['Date'])

    if 'Trade P/L' in df.columns:
        df['TradePL'] = _to_float(df['Trade P/L']).fillna(0.0)
    elif 'TradePL' in df.columns:
        df['TradePL'] = _to_float(df['TradePL']).fillna(0.0)
    else:
        df['TradePL'] = 0.0

    df = df.dropna(subset=['Date']).sort_values('Date').reset_index(drop=True)
    return df

# =========================
# Trade Builder
# =========================

def build_trades(df: pd.DataFrame, commission_rt: float) -> pd.DataFrame:
    trades = []
    def _safe_num(x): return pd.to_numeric(x, errors='coerce')

    OPEN_RX  = r"(BTO|BUY TO OPEN|BOT|STO|SELL TO OPEN|SELL SHORT|OPEN)"
    CLOSE_RX = r"(STC|SELL TO CLOSE|SLD|BTC|BUY TO CLOSE|CLOSE)"

    i = 0
    while i < len(df) - 1:
        entry = df.iloc[i]
        exit_ = df.iloc[i + 1]
        side_entry = str(entry.get('Side', '')).upper()
        side_exit = str(exit_.get('Side', '')).upper()

        if re.search(OPEN_RX, side_entry) and re.search(CLOSE_RX, side_exit):
            entry_qty = _safe_num(entry.get('Qty'))
            qty_abs = abs(entry_qty) if pd.notna(entry_qty) and entry_qty != 0 else 1.0
            direction = 'Long' if 'BTO' in side_entry or 'BUY' in side_entry else 'Short'
            trade_pl = _safe_num(exit_.get('TradePL'))
            commission = commission_rt * qty_abs
            net_pl = (trade_pl if pd.notna(trade_pl) else 0.0) - commission

            trades.append({
                'EntryTime': entry['Date'],
                'ExitTime': exit_['Date'],
                'EntryPrice': _safe_num(entry.get('Price')),
                'ExitPrice': _safe_num(exit_.get('Price')),
                'QtyAbs': qty_abs,
                'TradePL': trade_pl,
                'GrossPL': trade_pl,
                'Commission': commission,
                'NetPL': net_pl,
                'Direction': direction,
            })
            i += 2
        else:
            i += 1

    t = pd.DataFrame(trades)
    if t.empty:
        return pd.DataFrame(columns=['EntryTime','ExitTime','NetPL'])
    t['HoldMins'] = (t['ExitTime'] - t['EntryTime']).dt.total_seconds() / 60.0
    return t

# =========================
# Stoploss
# =========================

def apply_stoploss_corrections(trades: pd.DataFrame, point_value: float) -> pd.DataFrame:
    df = trades.copy()
    df['SLBreached'] = df['NetPL'] < -100.0
    df['AdjustedNetPL'] = np.where(df['SLBreached'], -100.0, df['NetPL'])
    qty_abs = pd.to_numeric(df['QtyAbs'], errors='coerce').replace(0, np.nan)
    gross_adjusted = np.where(df['SLBreached'], -100.0 + df['Commission'], df['NetPL'] + df['Commission'])
    df['PointsPerContract'] = gross_adjusted / (point_value * qty_abs)
    return df

# =========================
# Metrics
# =========================

def compute_metrics(trades: pd.DataFrame, cfg: BacktestConfig) -> dict:
    if trades.empty:
        return {"strategy": "Unknown", "num_trades": 0}

    pl_net = trades['AdjustedNetPL'].fillna(0.0)
    wins = pl_net[pl_net > 0]
    losses = pl_net[pl_net < 0]

    total_net = float(pl_net.sum())
    gross_profit = float(wins.sum()) if len(wins) else 0.0
    gross_loss = float(losses.sum()) if len(losses) else 0.0
    win_rate = (len(wins) / len(pl_net) * 100.0) if len(pl_net) else 0.0
    avg_win = float(wins.mean()) if len(wins) else 0.0
    avg_loss = float(losses.mean()) if len(losses) else 0.0
    largest_win = float(wins.max()) if len(wins) else 0.0
    largest_loss = float(losses.min()) if len(losses) else 0.0
    expectancy = float(pl_net.mean()) if len(pl_net) else 0.0

    equity = cfg.initial_capital + pl_net.cumsum()
    max_dd_pct = abs(_max_drawdown(equity)) * 100.0
    recovery_factor = total_net / (equity.cummax() - equity).max() if not equity.empty else 0.0

    returns = pl_net / cfg.initial_capital
    sharpe = (returns.mean() / returns.std(ddof=1) * np.sqrt(len(returns))) if returns.std(ddof=1) > 0 else 0.0

    return {
        "strategy": cfg.strategy_name or "Unknown",
        "timeframe": cfg.timeframe,
        "num_trades": len(pl_net),
        "net_profit": total_net,
        "total_return_pct": (total_net / cfg.initial_capital) * 100.0,
        "win_rate_pct": win_rate,
        "profit_factor": _profit_factor(pl_net),
        "max_drawdown_pct": max_dd_pct,
        "gross_profit": gross_profit,
        "gross_loss": gross_loss,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "largest_win": largest_win,
        "largest_loss": largest_loss,
        "expectancy": expectancy,
        "recovery_factor": recovery_factor,
        "sharpe_ratio": sharpe,
    }

# =========================
# Runner
# =========================

def run_backtest(csv_path: str, cfg: BacktestConfig):
    csv_stem = Path(csv_path).stem.replace(" ", "_")
    raw = load_tos_strategy_report(csv_path)
    trades = build_trades(raw, cfg.commission_per_round_trip)
    trades = apply_stoploss_corrections(trades, cfg.point_value)

    outdir = cfg.outdir(csv_stem, "/MES")
    os.makedirs(outdir, exist_ok=True)
    trades.to_csv(os.path.join(outdir, "trades_enriched.csv"), index=False)

    metrics = compute_metrics(trades, cfg)
    with open(os.path.join(outdir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    return metrics, outdir

if __name__ == "__main__":
    import argparse, glob, sys
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", nargs="+", required=True)
    parser.add_argument("--timeframe", type=str, default="180d:15m")
    parser.add_argument("--capital", type=float, default=2500.0)
    parser.add_argument("--commission", type=float, default=4.04)
    parser.add_argument("--point_value", type=float, default=5.0)
    args = parser.parse_args()

    cfg = BacktestConfig(
        timeframe=args.timeframe,
        initial_capital=args.capital,
        commission_per_round_trip=args.commission,
        point_value=args.point_value,
    )

    resolved = []
    for item in args.csv:
        matches = glob.glob(item)
        resolved.extend(matches if matches else [item])

    for csv_file in resolved:
        print(f"[RUN] {csv_file}")
        metrics, outdir = run_backtest(csv_file, cfg)
        print(json.dumps(metrics, indent=2))
