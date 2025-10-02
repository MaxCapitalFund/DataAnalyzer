# -*- coding: utf-8 -*-
# Clean Hybrid Backtester for Vercel
# - Pairs TOS trades
# - Applies $100 stop-loss cap
# - Computes robust metrics
# - Outputs JSON + CSV
# - Lightweight (no charts/markdown in this version)

import os, io, re, json, warnings, glob, sys
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import pandas as pd

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
        return str(temp_dir / f"Backtest_{day}_{safe_instr}_{csv_stem}_{timestamp}")


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


# =========================
# Load TOS Strategy Report
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
        raise ValueError("No trade table header found (expected 'Id;Strategy;').")
    table_str = "".join(lines[start_idx:])
    df = pd.read_csv(io.StringIO(table_str), sep=';')

    if 'Date/Time' in df.columns:
        df['Date'] = _parse_datetime(df['Date/Time'])
    elif 'Date' in df.columns and 'Time' in df.columns:
        dt_str = df['Date'].astype(str).str.strip() + ' ' + df['Time'].astype(str).str.strip()
        df['Date'] = pd.to_datetime(dt_str, errors='coerce')
    else:
        df['Date'] = _parse_datetime(df['Date'])

    df['TradePL'] = _to_float(df['Trade P/L']) if 'Trade P/L' in df.columns else 0.0
    df['CumPL'] = _to_float(df['P/L']) if 'P/L' in df.columns else np.nan

    if 'Strategy' in df.columns:
        base = df['Strategy'].astype(str).str.split('(').str[0].str.strip()
        df['BaseStrategy'] = base
    else:
        df['BaseStrategy'] = "Unknown"

    side_col = None
    for cand in ['Side', 'Action', 'Order', 'Type']:
        if cand in df.columns:
            side_col = cand
            break
    df['Side'] = df[side_col].astype(str) if side_col else ""

    if 'Price' not in df.columns:
        df['Price'] = np.nan
    if 'Quantity' in df.columns:
        df['Qty'] = pd.to_numeric(df['Quantity'], errors='coerce')
    elif 'Qty' not in df.columns:
        df['Qty'] = np.nan

    df = df.dropna(subset=['Date']).sort_values('Date').reset_index(drop=True)
    return df


# =========================
# Build Trades
# =========================

def build_trades(df: pd.DataFrame, commission_rt: float) -> pd.DataFrame:
    trades = []
    def _safe_num(x):
        return pd.to_numeric(x, errors='coerce')

    OPEN_RX  = r"(BTO|BUY TO OPEN|BOT|STO|SELL TO OPEN|SELL SHORT|OPEN)"
    CLOSE_RX = r"(STC|SELL TO CLOSE|SLD|BTC|BUY TO CLOSE|CLOSE)"

    i = 0
    while i < len(df) - 1:
        entry = df.iloc[i]
        exit_ = df.iloc[i + 1]
        side_entry = str(entry['Side']).upper()
        side_exit = str(exit_['Side']).upper()

        if re.search(OPEN_RX, side_entry) and re.search(CLOSE_RX, side_exit):
            entry_qty = _safe_num(entry.get('Qty'))
            qty_abs = abs(entry_qty) if pd.notna(entry_qty) and entry_qty != 0 else 1.0

            direction = 'Long' if 'BTO' in side_entry or 'BUY' in side_entry or 'BOT' in side_entry else 'Short'
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

    t = t.sort_values('ExitTime').reset_index(drop=True)
    t['HoldMins'] = (t['ExitTime'] - t['EntryTime']).dt.total_seconds() / 60.0
    return t


# =========================
# Stop-Loss Cap
# =========================

def apply_stoploss_corrections(trades: pd.DataFrame) -> pd.DataFrame:
    df = trades.copy()
    df['SLBreached'] = df['NetPL'] < -100.0
    df['AdjustedNetPL'] = np.where(df['SLBreached'], -100.0, df['NetPL'])
    return df


# =========================
# Metrics
# =========================

def compute_metrics(trades_df: pd.DataFrame, cfg: BacktestConfig) -> dict:
    if trades_df.empty:
        return {"num_trades": 0, "net_profit": 0.0}

    df = trades_df.copy()
    pl_net = df['AdjustedNetPL'].fillna(0.0)

    # Win/Loss masks
    wins = pl_net[pl_net > 0]
    losses = pl_net[pl_net < 0]

    num_trades = len(pl_net)
    num_wins, num_losses = len(wins), len(losses)

    gross_profit = float(wins.sum())
    gross_loss = float(losses.sum())
    avg_win = float(wins.mean()) if num_wins > 0 else 0.0
    avg_loss = float(losses.mean()) if num_losses > 0 else 0.0
    largest_win = float(wins.max()) if num_wins > 0 else 0.0
    largest_loss = float(losses.min()) if num_losses > 0 else 0.0
    win_rate = (num_wins / num_trades * 100.0) if num_trades > 0 else 0.0
    expectancy = float(pl_net.mean()) if num_trades > 0 else 0.0

    equity = cfg.initial_capital + pl_net.cumsum()
    dd = equity / equity.cummax() - 1.0
    max_dd_pct = abs(dd.min() * 100.0)
    max_dd_dollars = float((equity.cummax() - equity).max())
    recovery = (pl_net.sum() / max_dd_dollars) if max_dd_dollars > 0 else 0.0

    # Sharpe ratio
    returns = pl_net / cfg.initial_capital
    if returns.std(ddof=1) > 0:
        sharpe = returns.mean() / returns.std(ddof=1) * np.sqrt(len(returns))
    else:
        sharpe = 0.0

    metrics = {
        "strategy": cfg.strategy_name or "Unknown",
        "timeframe": cfg.timeframe,
        "num_trades": num_trades,
        "net_profit": float(pl_net.sum()),
        "total_return_pct": (pl_net.sum() / cfg.initial_capital * 100.0),
        "win_rate_pct": win_rate,
        "profit_factor": (gross_profit / abs(gross_loss)) if gross_loss != 0 else np.inf,
        "max_drawdown_pct": max_dd_pct,
        "gross_profit": gross_profit,
        "gross_loss": gross_loss,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "largest_win": largest_win,
        "largest_loss": largest_loss,
        "expectancy": expectancy,
        "recovery_factor": recovery,
        "sharpe_ratio": sharpe,
    }
    return metrics


# =========================
# Runner
# =========================

def run_backtest(tos_csv_path: str, cfg: BacktestConfig):
    csv_stem = Path(tos_csv_path).stem.replace(' ', '_')
    raw = load_tos_strategy_report(tos_csv_path)
    trades = build_trades(raw, cfg.commission_per_round_trip)
    trades = apply_stoploss_corrections(trades)
    metrics = compute_metrics(trades, cfg)

    outdir = cfg.outdir(csv_stem, "/MES")
    os.makedirs(outdir, exist_ok=True)
    trades.to_csv(os.path.join(outdir, "trades_enriched.csv"), index=False)
    with open(os.path.join(outdir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics


# =========================
# CLI
# =========================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", nargs="+", required=True, help="Path(s) to CSV")
    parser.add_argument("--capital", type=float, default=2500.0)
    parser.add_argument("--commission", type=float, default=4.04)
    parser.add_argument("--point_value", type=float, default=5.0)
    args, _ = parser.parse_known_args()   # ignores unknown args (e.g. --timeframe)

    cfg = BacktestConfig(
        initial_capital=args.capital,
        commission_per_round_trip=args.commission,
        point_value=args.point_value,
    )

    all_metrics = []
    for path in args.csv:
        if Path(path).exists():
            m = run_backtest(path, cfg)
            m["csv"] = Path(path).name
            all_metrics.append(m)

    print(json.dumps(all_metrics, indent=2))
