# -*- coding: utf-8 -*-
# Simplified backtester for Vercel deployment
# Optimized for serverless environment with reduced dependencies

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
matplotlib.use('Agg')  # Use non-interactive backend for serverless
import matplotlib.pyplot as plt

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
    version: str = "1.3.3"

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
    s = s.str.replace(r"\(([^()]*)\)", r"-\1", regex=True)  # (123) -> -123
    s = s.replace("", np.nan)
    return pd.to_numeric(s, errors='coerce')

def _parse_datetime(series: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(series, format="%m/%d/%y %I:%M %p", errors='coerce')
    if parsed.isna().all():
        parsed = pd.to_datetime(series, errors='coerce')
    return parsed

# Session tags
PRE_START, PRE_END = time(3, 0), time(9, 29)
OPEN_START, OPEN_END = time(9, 30), time(11, 30)
LUNCH_START, LUNCH_END = time(11, 30), time(14, 0)
CLOSE_START, CLOSE_END = time(14, 0), time(16, 0)

def _tag_session(dt: pd.Timestamp) -> str:
    if pd.isna(dt):
        return "Unknown"
    t = dt.time()
    if PRE_START <= t <= PRE_END: return "PRE"
    if OPEN_START <= t <= OPEN_END: return "OPEN"
    if LUNCH_START <= t <= LUNCH_END: return "LUNCH"
    if CLOSE_START <= t <= CLOSE_END: return "CLOSING"
    return "OTHER"

def _in_rth(dt: pd.Timestamp) -> bool:
    if pd.isna(dt):
        return False
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
    if any(w in s for w in ["TIME", "TIME EXIT", "TIMED", "TIMEOUT", "DAILY"]): return "Time"
    if any(w in s for w in ["MANUAL", "FLATTEN", "MKT CLOSE", "DISCRETIONARY"]): return "Manual"
    return "Close"

# Symbol normalization
ROOT_RE = re.compile(r"^/?([A-Za-z]{1,3})(?:[FGHJKMNQUVXZ]\d{1,2})?$")

def normalize_symbol(sym: str) -> str:
    s = str(sym).strip()
    if not s:
        return "/UNK"
    has_slash = s.startswith("/")
    core = s[1:] if has_slash else s
    m = ROOT_RE.match(core.upper())
    if m:
        root = m.group(1).upper()
        return f"/{root}"
    m2 = re.search(r"/([A-Za-z]{1,3})", s.upper())
    if m2:
        return f"/{m2.group(1)}"
    m3 = re.search(r"\b([A-Za-z]{1,3})\b", s.upper())
    return f"/{m3.group(1)}" if m3 else "/UNK"


# =========================
# Load & Clean (TOS CSV)
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
    elif 'Date' in df.columns:
        df['Date'] = _parse_datetime(df['Date'])
    else:
        raise ValueError("No Date column found")
    if 'Trade P/L' in df.columns:
        df['TradePL'] = _to_float(df['Trade P/L']).fillna(0.0)
    elif 'TradePL' in df.columns:
        df['TradePL'] = _to_float(df['TradePL']).fillna(0.0)
    else:
        df['TradePL'] = 0.0
    df['BaseStrategy'] = df['Strategy'].astype(str).str.split('(').str[0].str.strip() if 'Strategy' in df.columns else "Unknown"
    side_col = next((c for c in ['Side','Action','Order','Type'] if c in df.columns), None)
    df['Side'] = df[side_col].astype(str) if side_col else ""
    if 'Price' not in df.columns: df['Price'] = np.nan
    if 'Qty' not in df.columns:
        qcol = 'Quantity' if 'Quantity' in df.columns else None
        df['Qty'] = pd.to_numeric(df[qcol], errors='coerce') if qcol else np.nan
    if 'Symbol' in df.columns:
        df['Symbol'] = df['Symbol'].astype(str).map(normalize_symbol)
    else:
        df['Symbol'] = "/UNK"
    df = df.dropna(subset=['Date']).sort_values('Date').reset_index(drop=True)
    return df


# =========================
# Trades + Stop Loss
# =========================

def build_trades(df: pd.DataFrame, commission_rt: float) -> pd.DataFrame:
    trades = []
    def _safe_num(x): return pd.to_numeric(x, errors='coerce')
    for i in range(0, len(df)-1, 2):
        entry, exit_ = df.iloc[i], df.iloc[i+1]
        qty = _safe_num(entry.get('Qty'))
        qty_abs = abs(qty) if pd.notna(qty) and qty != 0 else 1.0
        trade_pl = _safe_num(exit_.get('TradePL'))
        commission = commission_rt * qty_abs
        net_pl = (trade_pl if pd.notna(trade_pl) else 0.0) - commission
        trades.append({
            'EntryTime': entry['Date'],
            'ExitTime': exit_['Date'],
            'QtyAbs': qty_abs,
            'TradePL': trade_pl,
            'GrossPL': trade_pl,
            'Commission': commission,
            'NetPL': net_pl,
            'BaseStrategy': entry.get('BaseStrategy', 'Unknown'),
            'Symbol': entry.get('Symbol', ''),
            'ExitReason': _exit_reason(exit_.get('Side'))
        })
    t = pd.DataFrame(trades)
    if t.empty: return t.assign(HoldMins=np.nan)
    t['HoldMins'] = (t['ExitTime'] - t['EntryTime']).dt.total_seconds()/60.0
    return t

def apply_stoploss_corrections(trades: pd.DataFrame, point_value: float) -> pd.DataFrame:
    df = trades.copy()
    df['SLBreached'] = df['NetPL'] < -100.0
    df['AdjustedNetPL'] = np.where(df['SLBreached'], -100.0, df['NetPL'])
    qty_abs = pd.to_numeric(df['QtyAbs'], errors='coerce').replace(0, np.nan)
    gross_adj = np.where(df['SLBreached'], -100.0 + df['Commission'], df['NetPL']+df['Commission'])
    df['PointsPerContract'] = gross_adj/(point_value*qty_abs)
    return df


# =========================
# Metrics
# =========================

def compute_metrics(trades: pd.DataFrame, cfg: BacktestConfig, scope: str) -> dict:
    pl = trades['AdjustedNetPL'].fillna(0.0)
    equity = cfg.initial_capital + pl.cumsum()
    return {
        "scope": scope,
        "strategy_name": cfg.strategy_name,
        "version": cfg.version,
        "timeframe": cfg.timeframe,
        "num_trades": int(len(trades)),
        "net_profit": float(pl.sum()),
        "profit_factor": _profit_factor(pl),
        "win_rate_pct": float((pl>0).mean()*100),
        "max_drawdown_pct": abs(_max_drawdown(equity))*100.0
    }


# =========================
# Runner
# =========================

def run_backtest(csv_path: str, cfg: BacktestConfig):
    raw = load_tos_strategy_report(csv_path)
    trades = build_trades(raw, cfg.commission_per_round_trip)
    trades = apply_stoploss_corrections(trades, cfg.point_value)
    outdir = cfg.outdir(Path(csv_path).stem, "/MES", cfg.strategy_name or "Unknown")
    os.makedirs(outdir, exist_ok=True)
    trades.to_csv(os.path.join(outdir,"trades_enriched.csv"), index=False)
    metrics = compute_metrics(trades, cfg, "ALL")
    with open(os.path.join(outdir,"metrics.json"),"w") as f:
        json.dump(metrics,f,indent=2)
    return metrics,outdir


if __name__=="__main__":
    import argparse,glob,sys
    parser = argparse.ArgumentParser(description="Backtester (Vercel safe)")
    parser.add_argument("--csv",nargs="+",required=True,help="CSV file(s)")
    parser.add_argument("--capital",type=float,default=2500.0)
    parser.add_argument("--commission",type=float,default=4.04)
    parser.add_argument("--point_value",type=float,default=5.0)
    args,unknown = parser.parse_known_args()
    if unknown: print(f"[WARN] Ignoring unknown args {unknown}")
    cfg = BacktestConfig(initial_capital=args.capital,
                         commission_per_round_trip=args.commission,
                         point_value=args.point_value)
    all_metrics=[]
    for p in args.csv:
        for path in glob.glob(p):
            m,out=run_backtest(path,cfg)
            m["csv"]=os.path.basename(path)
            all_metrics.append(m)
    print(json.dumps(all_metrics,indent=2))
