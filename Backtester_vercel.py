# -*- coding: utf-8 -*-
# Backtester_vercel.py
# Version: 1.8-nocharts
# Production-ready ThinkOrSwim Strategy Report Backtester for Vercel
# Author: ChatGPT (expert Python developer)
# Dependencies: numpy, pandas
# Features: robust edge-case handling, debug logging, no charts

import os
import io
import re
import json
from dataclasses import dataclass
from datetime import datetime, time
from pathlib import Path
from typing import Tuple
import numpy as np
import pandas as pd

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
    version: str = "1.8-nocharts"
    debug: bool = False

    def outdir(self, csv_stem: str, instrument: str) -> str:
        temp_dir = Path('/tmp')
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_instr = instrument.replace("/", "")
        return str(temp_dir / f"Backtests_{ts}_{safe_instr}_{csv_stem}")


# =========================
# Helpers
# =========================
def _log(msg: str, cfg: BacktestConfig):
    if cfg.debug:
        print(f"[DEBUG] {msg}")

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

def _exit_reason(text: str) -> str:
    s = str(text).upper()
    if any(w in s for w in ["TARGET", "TGT", "TP", "PROFIT"]): return "Target"
    if any(w in s for w in ["STOP", "SL", "STOPPED"]): return "Stop"
    if any(w in s for w in ["TIME", "TIME EXIT", "TIMED", "TIMEOUT"]): return "Time"
    if any(w in s for w in ["MANUAL", "FLATTEN", "MKT CLOSE", "DISCRETIONARY"]): return "Manual"
    return "Close"

def normalize_symbol(sym: str) -> str:
    s = str(sym).strip()
    if not s:
        return "/UNK"
    has_slash = s.startswith("/")
    core = s[1:] if has_slash else s
    m = re.match(r"^([A-Za-z]{1,3})(?:[FGHJKMNQUVXZ]\d{1,2})?$", core.upper())
    if m:
        root = m.group(1).upper()
        return f"/{root}"
    m2 = re.search(r"/([A-Za-z]{1,3})", s.upper())
    if m2:
        return f"/{m2.group(1)}"
    m3 = re.search(r"\b([A-Za-z]{1,3})\b", s.upper())
    return f"/{m3.group(1)}" if m3 else "/UNK"

OPEN_START, CLOSE_END = time(9, 30), time(16, 0)
def _in_rth(dt: pd.Timestamp) -> bool:
    if pd.isna(dt):
        return False
    t = dt.time()
    return OPEN_START <= t <= CLOSE_END


# =========================
# Load TOS CSV
# =========================
def load_tos_strategy_report(file_path: str, cfg: BacktestConfig) -> pd.DataFrame:
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

    if 'Side' not in df.columns:
        raise ValueError("CSV missing required 'Side' column")

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
    df['BaseStrategy'] = df['Strategy'].astype(str).str.split('(').str[0].str.strip() if 'Strategy' in df.columns else "Unknown"
    df['Symbol'] = df['Symbol'].astype(str).map(normalize_symbol) if 'Symbol' in df.columns else "/UNK"

    _log(f"Loaded CSV rows: {len(df)}, Columns: {list(df.columns)}", cfg)
    return df.dropna(subset=['Date']).sort_values('Date').reset_index(drop=True)


# =========================
# Build Trades
# =========================
def build_trades(df: pd.DataFrame, commission_rt: float, cfg: BacktestConfig) -> pd.DataFrame:
    trades = []
    OPEN_RX = r"\b(?:BTO|BUY TO OPEN|BOT TO OPEN|STO|SELL TO OPEN|SELL SHORT)\b"
    CLOSE_RX = r"\b(?:STC|SELL TO CLOSE|SLD TO CLOSE|BTC|BUY TO CLOSE)\b"

    if 'Id' not in df.columns:
        _log("CSV missing Id column; no trades will be generated", cfg)
        return pd.DataFrame()

    for tid, grp in df.groupby('Id', sort=False):
        g = grp.sort_values('Date').copy()
        side_up = g['Side'].astype(str).str.upper()
        g['is_open'] = side_up.str.contains(OPEN_RX, regex=True, na=False)
        g['is_close'] = side_up.str.contains(CLOSE_RX, regex=True, na=False)
        entry_rows = g[g['is_open']]
        close_rows = g[g['is_close']]
        if len(entry_rows) and len(close_rows):
            entry = entry_rows.iloc[0]
            after_entry_close = close_rows[close_rows['Date'] >= entry['Date']]
            exit_ = after_entry_close.iloc[0] if len(after_entry_close) else close_rows.iloc[-1]
            entry_qty = pd.to_numeric(entry.get('Qty'), errors='coerce')
            qty_abs = abs(entry_qty) if pd.notna(entry_qty) and entry_qty != 0 else 1.0
            trade_pl = pd.to_numeric(exit_.get('TradePL'), errors='coerce')
            commission = commission_rt * qty_abs
            net_pl = (trade_pl if pd.notna(trade_pl) else 0.0) - commission
            trades.append({
                'Id': tid,
                'EntryTime': entry['Date'],
                'ExitTime': exit_['Date'],
                'EntryPrice': pd.to_numeric(entry.get('Price'), errors='coerce'),
                'ExitPrice': pd.to_numeric(exit_.get('Price'), errors='coerce'),
                'EntryQty': entry_qty,
                'ExitQty': pd.to_numeric(exit_.get('Qty'), errors='coerce'),
                'QtyAbs': qty_abs,
                'TradePL': trade_pl,
                'Commission': commission,
                'NetPL': net_pl,
                'AdjustedNetPL': net_pl,  # will be capped later
                'BaseStrategy': entry.get('BaseStrategy', 'Unknown'),
                'StrategyRaw': entry.get('Strategy', ''),
                'Symbol': entry.get('Symbol', '/UNK'),
                'EntrySide': str(entry.get('Side', '')),
                'ExitSide': str(exit_.get('Side', '')),
                'ExitReason': _exit_reason(exit_.get('Side'))
            })

    t = pd.DataFrame(trades)
    if t.empty:
        _log("No trades generated: possible missing entry/exit pairs", cfg)
        return t.assign(HoldMins=np.nan)
    t['HoldMins'] = (t['ExitTime'] - t['EntryTime']).dt.total_seconds() / 60.0
    return t


# =========================
# Stop-loss
# =========================
def apply_stoploss(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return trades
    df = trades.copy()
    df['AdjustedNetPL'] = np.where(df['NetPL'] < -100.0, -100.0, df['NetPL'])
    return df


# =========================
# Metrics
# =========================
def compute_metrics(trades: pd.DataFrame, cfg: BacktestConfig) -> dict:
    if trades.empty:
        return {k: 0.0 for k in [
            "num_trades", "net_profit", "total_return_pct", "win_rate_pct", "profit_factor",
            "max_drawdown_pct", "gross_profit", "gross_loss", "avg_win", "avg_loss",
            "largest_win", "largest_loss", "expectancy", "recovery_factor", "sharpe_ratio"
        ]} | {"strategy": cfg.strategy_name, "timeframe": cfg.timeframe}

    pl = trades['AdjustedNetPL'].fillna(0.0)
    equity = cfg.initial_capital + pl.cumsum()
    wins, losses = pl[pl > 0], pl[pl < 0]

    return {
        "strategy": cfg.strategy_name,
        "timeframe": cfg.timeframe,
        "num_trades": float(len(trades)),
        "net_profit": float(pl.sum()),
        "total_return_pct": (pl.sum() / cfg.initial_capital) * 100,
        "win_rate_pct": (len(wins) / len(trades)) * 100,
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


# =========================
# Analytics Markdown
# =========================
def generate_analytics_md(trades_all, trades_rth, metrics, cfg, outdir):
    os.makedirs(outdir, exist_ok=True)
    first_dt_all = pd.to_datetime(trades_all['ExitTime'], errors='coerce').min() if not trades_all.empty else pd.NaT
    last_dt_all = pd.to_datetime(trades_all['ExitTime'], errors='coerce').max() if not trades_all.empty else pd.NaT

    def _fmt(x, pct=False):
        return f"{x:.2f}%" if pct else f"{x:.2f}"

    md = f"""
# Strategy Analysis Report
**Strategy:** {metrics.get('strategy')}
**Instrument:** {metrics.get('instrument', '/UNK')}
**Date Range:** {first_dt_all.date() if pd.notna(first_dt_all) else 'n/a'} → {last_dt_all.date() if pd.notna(last_dt_all) else 'n/a'}
**Timeframe:** {metrics.get('timeframe')}
**Run Date:** {datetime.now().strftime('%Y-%m-%d')}
**P/L Basis:** SL-adjusted net P/L (cap −$100 per trade including commissions)
**Trades:** ALL = {int(metrics.get('num_trades_all', 0))} | RTH = {int(metrics.get('num_trades_rth', 0))}
---
## Key Performance Indicators
- **Net Profit:** ${_fmt(metrics.get('net_profit', 0))}
- **Total Return:** {_fmt(metrics.get('total_return_pct', 0), pct=True)}
- **Win Rate:** {_fmt(metrics.get('win_rate_pct', 0), pct=True)}
- **Profit Factor:** {_fmt(metrics.get('profit_factor', 0))}
- **Max Drawdown:** {_fmt(metrics.get('max_drawdown_pct', 0), pct=True)}
- **Total Trades:** {int(metrics.get('num_trades', 0))}
---
*Report generated by Backtester v{cfg.version}*
"""
    with open(os.path.join(outdir, "analytics.md"), "w") as f:
        f.write(md)


# =========================
# Runner
# =========================
def run_backtest(tos_csv_path: str, cfg: BacktestConfig):
    csv_stem = Path(tos_csv_path).stem.replace(' ', '_')
    raw = load_tos_strategy_report(tos_csv_path, cfg)
    symbols = raw['Symbol'].dropna().unique().tolist() or ['/MES']
    results = []

    for instr in symbols:
        df_instr = raw[raw['Symbol'] == instr].copy()
        if df_instr.empty:
            continue

        cfg.point_value = 5.0 if instr.upper() in {'/MES', 'MES'} else 2.0 if instr.upper() in {'/MNQ', 'MNQ'} else 5.0
        cfg.strategy_name = df_instr['BaseStrategy'].dropna().iloc[0] if 'BaseStrategy' in df_instr.columns and len(df_instr) else cfg.strategy_name

        outdir = cfg.outdir(csv_stem, instr)
        os.makedirs(outdir, exist_ok=True)

        trades_all = build_trades(df_instr, cfg.commission_per_round_trip, cfg)
        trades_all = apply_stoploss(trades_all)
        trades_rth = trades_all[trades_all['EntryTime'].apply(_in_rth) | trades_all['ExitTime'].apply(_in_rth)].copy() if not trades_all.empty else trades_all

        trades_all.to_csv(os.path.join(outdir, "trades_enriched.csv"), index=False)

        metrics_all = compute_metrics(trades_all, cfg)
        metrics_all["instrument"] = instr
        metrics_all["num_trades_all"] = int(len(trades_all))
        metrics_all["num_trades_rth"] = int(len(trades_rth))

        metrics_rth = compute_metrics(trades_rth, cfg)
        for k, v in metrics_rth.items():
            metrics_all[f"RTH_{k}"] = v

        with open(os.path.join(outdir, "metrics.json"), "w") as f:
            json.dump(metrics_all, f, indent=2)

        generate_analytics_md(trades_all, trades_rth, metrics_all, cfg, outdir)

        # Console summary
        print(f"Strategy: {metrics_all.get('strategy')}")
        print(f"Instrument: {metrics_all.get('instrument')}")
        print(f"Net Profit: ${metrics_all.get('net_profit'):.2f}")
        print(f"Total Return: {metrics_all.get('total_return_pct'):.2f}%")
        print(f"Win Rate: {metrics_all.get('win_rate_pct'):.2f}%")
        print(f"Profit Factor: {metrics_all.get('profit_factor'):.2f}")
        print(f"Max Drawdown: {metrics_all.get('max_drawdown_pct'):.2f}%")
        print(f"Total Trades: {int(metrics_all.get('num_trades', 0))}")

        results.append({"instrument": instr, "metrics": metrics_all, "outdir": outdir})

    return results


# =========================
# Main
# =========================
if __name__ == "__main__":
    import argparse, glob, sys
    parser = argparse.ArgumentParser(description="Backtest TOS Strategy Report CSV")
    parser.add_argument("--csv", nargs="+", required=True, help="CSV file(s)")
    parser.add_argument("--timeframe", type=str, default="180d:15m")
    parser.add_argument("--capital", type=float, default=2500.0)
    parser.add_argument("--commission", type=float, default=4.04)
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    cfg = BacktestConfig(
        timeframe=args.timeframe,
        initial_capital=args.capital,
        commission_per_round_trip=args.commission,
        debug=args.debug
    )

    resolved = []
    for item in args.csv:
        matches = glob.glob(item)
        resolved.extend(matches if matches else [item])
    csv_paths = sorted({str(Path(p)) for p in resolved if Path(p).exists()})

    if not csv_paths:
        print(f"[ERROR] No CSV files matched: {args.csv}", file=sys.stderr)
        sys.exit(1)

    for csv_path in csv_paths:
        print(f"\n[RUN] CSV: {csv_path}")
        run_backtest(csv_path, cfg)
