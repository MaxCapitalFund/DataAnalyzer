# -*- coding: utf-8 -*-
# Clean Baseline Backtester for Vercel Deployment
# Stable foundation before adding analytics/charts

import os
import io
import re
import json
import glob
import argparse
from dataclasses import dataclass, asdict
from datetime import datetime
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
    initial_capital: float = 2500.0
    commission_per_round_trip: float = 4.04
    point_value: float = 5.0
    version: str = "1.6.5"

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
# Load & Clean (TOS Strategy Report)
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
        raise ValueError("No trade table header found in file (expected 'Id;Strategy;').")

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
        raise ValueError("No Date column found.")

    if 'Trade P/L' in df.columns:
        df['TradePL'] = _to_float(df['Trade P/L']).fillna(0.0)
    else:
        df['TradePL'] = 0.0

    df['CumPL'] = _to_float(df['P/L']) if 'P/L' in df.columns else np.nan

    if 'Strategy' in df.columns:
        df['BaseStrategy'] = df['Strategy'].astype(str).str.split('(').str[0].str.strip()
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

    qty_col = 'Quantity' if 'Quantity' in df.columns else ('Qty' if 'Qty' in df.columns else None)
    if qty_col and qty_col != 'Qty':
        df['Qty'] = pd.to_numeric(df[qty_col], errors='coerce')
    elif 'Qty' not in df.columns:
        df['Qty'] = np.nan

    if 'Symbol' in df.columns:
        df['Symbol'] = df['Symbol'].astype(str).map(normalize_symbol)
    elif 'Instrument' in df.columns:
        df['Symbol'] = df['Instrument'].astype(str).map(normalize_symbol)
    else:
        s = df['Strategy'].astype(str) if 'Strategy' in df.columns else pd.Series([], dtype=str)
        pat = re.compile(r"/([A-Z]{1,3})")
        sym_guess = s.str.extract(pat, expand=False).fillna("").map(lambda x: f"/{x}" if x else "/UNK")
        df['Symbol'] = sym_guess.map(normalize_symbol)

    df = df.dropna(subset=['Date']).sort_values('Date').reset_index(drop=True)
    return df

# =========================
# Build Trades
# =========================

def build_trades(df: pd.DataFrame, commission_rt: float) -> pd.DataFrame:
    trades = []

    OPEN_RX  = r"(BTO|BUY TO OPEN|BUY_TO_OPEN|BOT TO OPEN|STO|SELL TO OPEN|SELL_TO_OPEN|SELL SHORT)"
    CLOSE_RX = r"(STC|SELL TO CLOSE|SELL_TO_CLOSE|SLD TO CLOSE|BTC|BUY TO CLOSE|BUY_TO_CLOSE)"

    for _, grp in df.groupby('Id', sort=False):
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

            qty_abs = abs(pd.to_numeric(entry.get('Qty'), errors='coerce')) or 1.0
            trade_pl = pd.to_numeric(exit_.get('TradePL'), errors='coerce')
            commission = commission_rt * qty_abs
            net_pl = (trade_pl if pd.notna(trade_pl) else 0.0) - commission

            trades.append({
                'Id': entry.get('Id'),
                'EntryTime': entry['Date'],
                'ExitTime': exit_['Date'],
                'TradePL': trade_pl,
                'GrossPL': trade_pl,
                'Commission': commission,
                'NetPL': net_pl,
                'QtyAbs': qty_abs,
                'BaseStrategy': entry.get('BaseStrategy', 'Unknown'),
                'Symbol': entry.get('Symbol', ''),
            })

    t = pd.DataFrame(trades)
    if t.empty:
        return t.assign(HoldMins=np.nan)

    t['HoldMins'] = (t['ExitTime'] - t['EntryTime']).dt.total_seconds() / 60.0
    return t

# =========================
# Stop-Loss
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

def compute_metrics(trades_df: pd.DataFrame, cfg: BacktestConfig, scope_label: str) -> dict:
    df = trades_df.copy()
    if df.empty:
        return {"scope": scope_label, "net_profit": 0.0, "num_trades": 0}

    pl = df['AdjustedNetPL']
    equity = cfg.initial_capital + pl.cumsum()

    return {
        "scope": scope_label,
        "net_profit": float(pl.sum()),
        "num_trades": int(len(df)),
        "profit_factor": _profit_factor(pl),
        "win_rate_pct": float((pl > 0).mean() * 100.0),
        "avg_win": float(pl[pl > 0].mean()) if (pl > 0).any() else 0.0,
        "avg_loss": float(pl[pl < 0].mean()) if (pl < 0).any() else 0.0,
        "max_drawdown_pct": abs(_max_drawdown(equity)) * 100.0,
    }

# =========================
# Main Runner
# =========================

def run_backtest_for_instrument(df_raw: pd.DataFrame, instrument: Optional[str], cfg: BacktestConfig, csv_stem: str):
    strategy_label = df_raw['BaseStrategy'].dropna().iloc[0] if 'BaseStrategy' in df_raw.columns else cfg.strategy_name
    instr = normalize_symbol(instrument or '/UNK')
    cfg.strategy_name = strategy_label

    if instr.upper() in {'/MES', 'MES'}:
        cfg.point_value = 5.0
    elif instr.upper() in {'/MNQ', 'MNQ'}:
        cfg.point_value = 2.0

    outdir = cfg.outdir(csv_stem, instr, strategy_label)
    os.makedirs(outdir, exist_ok=True)

    trades_all = build_trades(df_raw, cfg.commission_per_round_trip)
    trades_all = apply_stoploss_corrections(trades_all, cfg.point_value)
    trades_all.to_csv(os.path.join(outdir, "trades_enriched.csv"), index=False)

    metrics = compute_metrics(trades_all, cfg, scope_label="ALL")
    metrics["strategy_name"] = strategy_label
    metrics["instrument"] = instr

    with open(os.path.join(outdir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    return trades_all, metrics, outdir

def run_backtest(tos_csv_path: str, cfg: BacktestConfig):
    csv_stem = Path(tos_csv_path).stem.replace(' ', '_')
    raw = load_tos_strategy_report(tos_csv_path)
    symbols = raw['Symbol'].dropna().unique().tolist() or ['/MES']

    results = []
    for instr in symbols:
        trades_all, metrics, outdir = run_backtest_for_instrument(raw, instr, cfg, csv_stem)
        results.append({"instrument": instr, "metrics": metrics, "outdir": outdir})
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Baseline Backtester (clean & stable).")
    parser.add_argument("--csv", nargs="+", required=True, help="Path(s) to TOS Strategy Report CSV(s).")
    parser.add_argument("--timeframe", type=str, default="180d:15m")
    parser.add_argument("--capital", type=float, default=2500.0)
    parser.add_argument("--commission", type=float, default=4.04)
    parser.add_argument("--point_value", type=float, default=5.0)

    args = parser.parse_args()

    resolved = []
    for item in args.csv:
        matches = glob.glob(item)
        if matches: resolved.extend(matches)
        else: resolved.append(item)

    csv_paths = sorted({str(Path(p)) for p in resolved if Path(p).exists()})
    if not csv_paths:
        print(f"[ERROR] No CSV files found.", file=sys.stderr)
        exit(1)

    cfg = BacktestConfig(
        timeframe=args.timeframe,
        initial_capital=args.capital,
        commission_per_round_trip=args.commission,
        point_value=args.point_value,
    )

    for csv_path in csv_paths:
        print(f"\n[RUN] {csv_path}")
        run_backtest(csv_path, cfg)
