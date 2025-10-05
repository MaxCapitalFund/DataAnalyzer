# ============================================================
#  FILE: Backtester_vercel_v1.3.2.py
#  CREATED: 2025-10-04
#  PURPOSE:
#     Simplified yet comprehensive ThinkorSwim Strategy Report analyzer.
#     Designed for serverless deployment (e.g., Vercel, AWS Lambda).
#     Parses TOS CSVs, constructs trades, applies capped stop loss (-$100),
#     and generates enriched performance analytics + markdown summary.
#
#  CHANGELOG:
#     v1.3.2 — Adds ensure_dir_ready() safety, 09:15 premarket cutoff,
#              and robust I/O handling for /tmp persistence.
# ============================================================

import os
import io
import re
import json
import time
from dataclasses import dataclass, asdict
from datetime import datetime, time as dtime
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ============================================================
#  CONFIGURATION
# ============================================================

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
        temp_dir = Path('/tmp')
        day = datetime.now().strftime("%Y-%m-%d")
        timestamp = datetime.now().strftime("%H%M%S_%f")[:-3]
        safe_strategy = (strategy_label or "Unknown").replace(" ", "_")
        safe_instr = (instrument or "UNK").replace("/", "")
        return str(temp_dir / f"Backtests_{day}_{safe_strategy}_{self.timeframe}_{safe_instr}_{csv_stem}_{timestamp}")

# ============================================================
#  HELPERS
# ============================================================

def ensure_dir_ready(path: str, retries: int = 5, delay: float = 0.1):
    """Ensure /tmp subdirectory exists before file operations."""
    for _ in range(retries):
        if os.path.exists(path):
            return True
        os.makedirs(path, exist_ok=True)
        time.sleep(delay)
    raise OSError(f"Cannot create or access directory: {path}")

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

# ---- Sessions ----
PRE_START, PRE_END = dtime(3, 0), dtime(9, 15)
GAP_START, GAP_END = dtime(9, 15), dtime(9, 30)
OPEN_START, OPEN_END = dtime(9, 30), dtime(11, 30)
LUNCH_START, LUNCH_END = dtime(11, 30), dtime(14, 0)
CLOSE_START, CLOSE_END = dtime(14, 0), dtime(16, 0)

def _tag_session(dt: pd.Timestamp) -> str:
    if pd.isna(dt): return "Unknown"
    t = dt.time()
    if PRE_START <= t < PRE_END: return "PRE"
    if GAP_START <= t < GAP_END: return "GAP"
    if OPEN_START <= t <= OPEN_END: return "OPEN"
    if LUNCH_START <= t <= LUNCH_END: return "LUNCH"
    if CLOSE_START <= t <= CLOSE_END: return "CLOSING"
    return "OTHER"

def _in_rth(dt: pd.Timestamp) -> bool:
    if pd.isna(dt): return False
    t = dt.time()
    return OPEN_START <= t <= CLOSE_END

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

def _exit_reason(text: str) -> str:
    s = str(text).upper()
    if any(w in s for w in ["TARGET", "TGT", "TP", "PROFIT"]): return "Target"
    if any(w in s for w in ["STOP", "SL", "STOPPED"]): return "Stop"
    if any(w in s for w in ["TIME", "TIME EXIT", "TIMED", "TIMEOUT"]): return "Time"
    if any(w in s for w in ["MANUAL", "FLATTEN", "MKT CLOSE", "DISCRETIONARY"]): return "Manual"
    return "Close"

# ============================================================
#  LOAD & CLEAN (ToS STRATEGY REPORT)
# ============================================================

def load_tos_strategy_report(file_path: str) -> pd.DataFrame:
    with open(file_path, 'r', errors='replace') as f:
        lines = f.readlines()
    start_idx = None
    for i, line in enumerate(lines):
        if line.lstrip().startswith("Id;Strategy;"):
            start_idx = i
            break
    if start_idx is None:
        raise ValueError("No trade table header found (expected 'Id;Strategy;').")

    table_str = "".join(lines[start_idx:])
    df = pd.read_csv(io.StringIO(table_str), sep=';')

    # Parse datetime
    if 'Date/Time' in df.columns:
        df['Date'] = _parse_datetime(df['Date/Time'])
    else:
        raise ValueError("Missing 'Date/Time' column in CSV.")

    # Trade P/L → numeric
    if 'Trade P/L' in df.columns:
        df['TradePL'] = _to_float(df['Trade P/L']).fillna(0.0)
    else:
        df['TradePL'] = 0.0

    # Strategy cleanup
    if 'Strategy' in df.columns:
        df['BaseStrategy'] = df['Strategy'].astype(str).str.split('(').str[0].str.strip()
    else:
        df['BaseStrategy'] = "Unknown"

    # Side
    side_col = next((c for c in ['Side', 'Action', 'Order', 'Type'] if c in df.columns), None)
    df['Side'] = df[side_col].astype(str) if side_col else ""

    # Quantity
    qty_col = 'Quantity' if 'Quantity' in df.columns else ('Qty' if 'Qty' in df.columns else None)
    df['Qty'] = pd.to_numeric(df[qty_col], errors='coerce') if qty_col else np.nan

    df = df.dropna(subset=['Date']).sort_values('Date').reset_index(drop=True)
    return df

# ============================================================
#  BUILD TRADES
# ============================================================

def build_trades(df: pd.DataFrame, commission_rt: float) -> pd.DataFrame:
    trades = []
    OPEN_RX = r"\b(?:BTO|BUY TO OPEN|STO|SELL TO OPEN)\b"
    CLOSE_RX = r"\b(?:STC|SELL TO CLOSE|BTC|BUY TO CLOSE)\b"

    def _safe_num(x):
        return pd.to_numeric(x, errors='coerce')

    i = 0
    while i < len(df) - 1:
        entry, exit_ = df.iloc[i], df.iloc[i + 1]
        if re.search(OPEN_RX, str(entry['Side']).upper()) and re.search(CLOSE_RX, str(exit_['Side']).upper()):
            entry_qty = _safe_num(entry.get('Qty'))
            qty_abs = abs(entry_qty) if pd.notna(entry_qty) and entry_qty != 0 else 1.0
            direction = 'Long' if 'BTO' in str(entry['Side']).upper() else 'Short'
            trade_pl = _safe_num(exit_.get('TradePL'))
            commission = commission_rt * qty_abs
            net_pl = (trade_pl if pd.notna(trade_pl) else 0.0) - commission

            trades.append({
                'Id': entry.get('Id', np.nan),
                'EntryTime': entry['Date'],
                'ExitTime': exit_['Date'],
                'TradePL': trade_pl,
                'Commission': commission,
                'NetPL': net_pl,
                'BaseStrategy': entry.get('BaseStrategy', 'Unknown'),
                'EntrySide': entry.get('Side', ''),
                'ExitSide': exit_.get('Side', ''),
                'Direction': direction
            })
            i += 2
        else:
            i += 1

    t = pd.DataFrame(trades)
    if t.empty:
        return t.assign(HoldMins=np.nan)
    t['HoldMins'] = (t['ExitTime'] - t['EntryTime']).dt.total_seconds() / 60.0
    return t

# ============================================================
#  STOP-LOSS CORRECTION
# ============================================================

def apply_stoploss_corrections(trades: pd.DataFrame, point_value: float) -> pd.DataFrame:
    df = trades.copy()
    df['SLBreached'] = df['NetPL'] < -100.0
    df['AdjustedNetPL'] = np.where(df['SLBreached'], -100.0, df['NetPL'])
    qty_abs = pd.to_numeric(df.get('QtyAbs', 1.0), errors='coerce').replace(0, np.nan)
    gross_adjusted = np.where(df['SLBreached'], -100.0 + df['Commission'], df['NetPL'] + df['Commission'])
    df['PointsPerContract'] = gross_adjusted / (point_value * qty_abs)
    return df

# ============================================================
#  METRICS
# ============================================================

def compute_metrics(trades_df: pd.DataFrame, cfg: BacktestConfig, scope_label: str) -> dict:
    df = trades_df.copy()
    pl_net = df['AdjustedNetPL'].fillna(0.0)
    equity = cfg.initial_capital + pl_net.cumsum()
    total_net = float(pl_net.sum())
    total_return_pct = (total_net / cfg.initial_capital) * 100.0
    metrics = {
        "scope": scope_label,
        "strategy_name": cfg.strategy_name,
        "net_profit": total_net,
        "total_return_pct": total_return_pct,
        "win_rate_pct": float((pl_net > 0).mean() * 100),
        "profit_factor": _profit_factor(pl_net),
        "max_drawdown_pct": abs(_max_drawdown(equity)) * 100.0,
        "num_trades": int(len(df))
    }
    return metrics

# ============================================================
#  MAIN RUNNER
# ============================================================

def run_backtest(tos_csv_path: str, cfg: BacktestConfig):
    csv_stem = Path(tos_csv_path).stem.replace(" ", "_")
    df = load_tos_strategy_report(tos_csv_path)
    cfg.strategy_name = df['BaseStrategy'].iloc[0] if len(df['BaseStrategy'].dropna()) else "Unknown"

    trades = build_trades(df, cfg.commission_per_round_trip)
    trades = apply_stoploss_corrections(trades, cfg.point_value)

    outdir = cfg.outdir(csv_stem, "/MES", cfg.strategy_name)
    ensure_dir_ready(outdir)

    trades.to_csv(os.path.join(outdir, "trades_enriched.csv"), index=False)
    metrics = compute_metrics(trades, cfg, "ALL")

    with open(os.path.join(outdir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n✅ Backtest complete — {len(trades)} trades")
    print(f"📊 Net P/L: ${metrics['net_profit']:.2f}")
    print(f"📁 Output directory: {outdir}")
    return metrics

# ============================================================
#  ENTRY POINT
# ============================================================

if __name__ == "__main__":
    import argparse, glob, sys
    parser = argparse.ArgumentParser(description="Backtester for TOS Strategy Report CSV.")
    parser.add_argument("--csv", required=True, help="Path to CSV file.")
    parser.add_argument("--capital", type=float, default=2500.0)
    parser.add_argument("--commission", type=float, default=4.04)
    parser.add_argument("--point_value", type=float, default=5.0)
    args = parser.parse_args()

    cfg = BacktestConfig(
        initial_capital=args.capital,
        commission_per_round_trip=args.commission,
        point_value=args.point_value
    )

    run_backtest(args.csv, cfg)

# ============================================================
#  END OF FILE — Backtester_vercel_v1.3.2.py
# ============================================================
