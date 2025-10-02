# -*- coding: utf-8 -*-
# Backtester_vercel.py — Hybrid Backtester v1.5.3
# Optimized for Vercel deployment
# - Stop-loss cap: -$100 per trade per contract
# - Outputs: trades_enriched.csv, metrics.json, analytics.md, config.json, charts

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
matplotlib.use('Agg')  # headless backend
import matplotlib.pyplot as plt

# =========================
# Configuration
# =========================
@dataclass
class BacktestConfig:
    strategy_name: str = ""
    instruments: Tuple[str, ...] = ("/MES",)
    timeframe: str = "180d:15m"
    session_hours_rth: Tuple[str, str] = ("09:45", "15:30")
    initial_capital: float = 2500.0
    commission_per_round_trip: float = 4.04
    point_value: float = 5.0
    version: str = "1.5.3"
    algo_params: dict = None

    def __post_init__(self):
        if self.algo_params is None:
            self.algo_params = {"ATRFactor_Fixed": 2.2, "StopLossCap": 100.0}

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

OPEN_START, OPEN_END = time(9, 45), time(11, 30)
LUNCH_START, LUNCH_END = time(11, 30), time(14, 0)
CLOSE_START, CLOSE_END = time(14, 0), time(15, 30)
PREMARKET_START, PREMARKET_END = time(3, 0), time(9, 15)

def _tag_session(dt: pd.Timestamp) -> str:
    if pd.isna(dt):
        return "Unknown"
    t = dt.time()
    if PREMARKET_START <= t < PREMARKET_END:
        return "PREMARKET"
    if OPEN_START <= t < LUNCH_START:
        return "OPEN"
    if LUNCH_START <= t < CLOSE_START:
        return "LUNCH"
    if CLOSE_START <= t <= CLOSE_END:
        return "CLOSING"
    return "AFTER_HOURS"

def _tag_day_part(dt: pd.Timestamp) -> str:
    if pd.isna(dt):
        return "Unknown"
    t = dt.time()
    if time(3, 0) <= t < time(9, 15):
        return "PREMARKET"
    if time(9, 45) <= t < time(11, 30):
        return "EARLY"
    if time(11, 30) <= t < time(14, 0):
        return "MID"
    if time(14, 0) <= t <= time(15, 30):
        return "LATE"
    return "AFTER_HOURS"

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

# Expanded stop detection
def _exit_reason(text: str) -> str:
    if not text or pd.isna(text):
        return "Close"
    s = str(text).upper().strip()
    if any(w in s for w in ["TARGET", "TGT", "TP", "PROFIT"]):
        return "Target"
    if any(w in s for w in ["STOP", "LOSS LIMIT", "SL ", "STOPPED", "STC STOP", "BTC STOP"]):
        return "Stop"
    if any(w in s for w in ["TIME", "TIME EXIT", "TIMED", "TIMEOUT", "DAILY"]):
        return "Time"
    if any(w in s for w in ["MANUAL", "FLATTEN", "MKT CLOSE", "DISCRETIONARY", "OPPOSING"]):
        return "Other"
    return "Close"

ROOT_RE = re.compile(r"^/?([A-Za-z]{1,3})(?:[FGHJKMNQUVXZ]\d{1,2})?$")

def normalize_symbol(sym: str) -> str:
    s = str(sym).strip()
    if not s:
        return "/UNK"
    has_slash = s.startswith("/")
    core = s[1:] if has_slash else s
    m = ROOT_RE.match(core.upper())
    if m:
        return f"/{m.group(1).upper()}"
    m2 = re.search(r"/([A-Za-z]{1,3})", s.upper())
    if m2:
        return f"/{m2.group(1)}"
    m3 = re.search(r"\b([A-Za-z]{1,3})\b", s.upper())
    return f"/{m3.group(1)}" if m3 else "/UNK"

# =========================
# Load & Clean Strategy Report
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
        raise ValueError("No trade table header found.")
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
        raise ValueError("No valid date column.")

    if 'Trade P/L' in df.columns:
        df['TradePL'] = _to_float(df['Trade P/L']).fillna(0.0)
    elif 'TradePL' in df.columns:
        df['TradePL'] = _to_float(df['TradePL']).fillna(0.0)
    else:
        df['TradePL'] = 0.0

    df['CumPL'] = _to_float(df['P/L']) if 'P/L' in df.columns else np.nan
    df['BaseStrategy'] = cfg.strategy_name

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
def build_trades(df: pd.DataFrame, commission_rt: float) -> Tuple[pd.DataFrame, int]:
    trades = []
    non_rth_trades = 0

    def _safe_num(x):
        return pd.to_numeric(x, errors='coerce')

    OPEN_RX = r"\b(?:BTO|BUY TO OPEN|STO|SELL TO OPEN|SELL SHORT|OPEN)\b"
    CLOSE_RX = r"\b(?:STC|SELL TO CLOSE|BTC|BUY TO CLOSE|CLOSE)\b"

    i = 0
    while i < len(df) - 1:
        entry, exit_ = df.iloc[i], df.iloc[i + 1]
        side_entry = str(entry['Side']).upper().strip()
        side_exit = str(exit_['Side']).upper().strip()

        if re.search(OPEN_RX, side_entry) and re.search(CLOSE_RX, side_exit):
            entry_qty = _safe_num(entry.get('Qty'))
            qty_abs = abs(entry_qty) if pd.notna(entry_qty) and entry_qty != 0 else 1.0
            direction = 'Long' if "BTO" in side_entry or (pd.notna(entry_qty) and entry_qty > 0) else 'Short'
            trade_pl = _safe_num(exit_.get('TradePL'))
            commission = commission_rt * qty_abs
            net_pl = (trade_pl if pd.notna(trade_pl) else 0.0) - commission
            exit_reason = _exit_reason(exit_.get('Side') or exit_.get('Type') or exit_.get('Order'))
            trades.append({
                'EntryTime': entry['Date'],
                'ExitTime': exit_['Date'],
                'EntryPrice': _safe_num(entry.get('Price')),
                'ExitPrice': _safe_num(exit_.get('Price')),
                'EntryQty': entry_qty,
                'ExitQty': _safe_num(exit_.get('Qty')),
                'QtyAbs': qty_abs,
                'TradePL': trade_pl,
                'GrossPL': trade_pl,
                'Commission': commission,
                'NetPL': net_pl,
                'ExitReason': exit_reason,
                'Direction': direction,
            })
            i += 2
        else:
            i += 1

    t = pd.DataFrame(trades)
    if t.empty:
        return t.assign(HoldMins=np.nan), non_rth_trades
    t = t.sort_values('ExitTime').reset_index(drop=True)
    t['HoldMins'] = (t['ExitTime'] - t['EntryTime']).dt.total_seconds() / 60.0
    return t, non_rth_trades

# =========================
# Stop-loss correction
# =========================
def apply_stoploss_corrections(trades: pd.DataFrame, point_value: float) -> pd.DataFrame:
    df = trades.copy()
    df['RawLossExceeds100'] = df['TradePL'] < -100.0
    df['SLBreached'] = df['NetPL'] < -100.0
    df['AdjustedNetPL'] = np.where(df['SLBreached'], -100.0, df['NetPL'])

    # Force ExitReason = Stop if capped
    df.loc[df['SLBreached'] & (df['ExitReason'] != "Stop"), 'ExitReason'] = "Stop"

    qty_abs = pd.to_numeric(df['QtyAbs'], errors='coerce').replace(0, np.nan)
    gross_adjusted = np.where(df['SLBreached'], -100.0 + df['Commission'], df['NetPL'] + df['Commission'])
    df['PointsPerContract'] = gross_adjusted / (point_value * qty_abs)
    return df

# =========================
# (compute_metrics, save_visuals_and_tables, generate_analytics_md)
# — use your existing v1.5.2 bodies (unchanged)
# =========================

# =========================
# Runner
# =========================
def run_backtest_for_instrument(df_raw: pd.DataFrame, instrument: Optional[str], cfg: BacktestConfig, csv_stem: str):
    cfg.strategy_name = Path(csv_stem).stem
    instr = normalize_symbol(instrument or '/UNK')
    cfg.point_value = 5.0 if instr.upper() in {'/MES', 'MES'} else (2.0 if instr.upper() in {'/MNQ', 'MNQ'} else cfg.point_value)
    outdir = cfg.outdir(csv_stem, instr, cfg.strategy_name)
    os.makedirs(outdir, exist_ok=True)

    trades_all, non_rth_trades = build_trades(df_raw, cfg.commission_per_round_trip)
    trades_all['Session'] = trades_all.apply(lambda r: _tag_session(r['EntryTime'] if pd.notna(r['EntryTime']) else r['ExitTime']), axis=1)
    trades_all = apply_stoploss_corrections(trades_all, cfg.point_value)
    trades_all.to_csv(os.path.join(outdir, "trades_enriched.csv"), index=False)

    trades_rth = trades_all[trades_all['EntryTime'].dt.time.between(time(9,45), time(15,30))].copy()
    metrics_all = compute_metrics(trades_all, cfg, scope_label="ALL", non_rth_trades=non_rth_trades)
    metrics_all["strategy_name"] = cfg.strategy_name
    metrics_all["instrument"] = instr
    metrics_all["num_trades_all"] = int(len(trades_all))
    metrics_all["num_trades_rth"] = int(len(trades_rth))

    metrics_rth = compute_metrics(trades_rth, cfg, scope_label="RTH", non_rth_trades=non_rth_trades)
    for k, v in metrics_rth.items():
        metrics_all[f"RTH_{k}"] = v

    with open(os.path.join(outdir, "metrics.json"), "w") as f:
        json.dump(metrics_all, f, indent=2)

    save_visuals_and_tables(trades_all, cfg, outdir, title_suffix="ALL")
    generate_analytics_md(trades_all, trades_rth, metrics_all, cfg, non_rth_trades, outdir)

    with open(os.path.join(outdir, "config.json"), "w") as f:
        json.dump(asdict(cfg), f, indent=2)

    return trades_all, trades_rth, metrics_all, outdir

def run_backtest(tos_csv_path: str, cfg: BacktestConfig):
    csv_stem = Path(tos_csv_path).stem.replace(' ', '_')
    raw = load_tos_strategy_report(tos_csv_path, cfg)
    raw['Symbol'] = raw['Symbol'].map(normalize_symbol) if 'Symbol' in raw.columns else "/UNK"
    symbols = raw['Symbol'].dropna().unique().tolist() or ['/MES']
    results = []
    for instr in symbols:
        trades_all, trades_rth, metrics, outdir = run_backtest_for_instrument(raw, instr, cfg, csv_stem)
        results.append({"instrument": instr, "metrics": metrics, "outdir": outdir})
    return results

# =========================
# Main
# =========================
if __name__ == "__main__":
    import argparse, glob, sys
    from pathlib import Path   # FIX: ensure Path is in scope

    parser = argparse.ArgumentParser(description="Analyze TOS Strategy Report CSVs with stop-loss cap.")
    parser.add_argument("--csv", nargs="+", required=True, help="Path(s) or globs for CSV(s).")
    parser.add_argument("--timeframe", type=str, default="180d:15m")
    parser.add_argument("--capital", type=float, default=2500.0)
    parser.add_argument("--commission", type=float, default=4.04)
    parser.add_argument("--point_value", type=float, default=5.0)
    args = parser.parse_args()

    resolved = []
    for item in args.csv:
        resolved.extend(glob.glob(item) or [item])
    csv_paths = sorted({str(Path(p)) for p in resolved if Path(p).exists()})
    if not csv_paths:
        print(f"[ERROR] No CSVs matched: {args.csv}", file=sys.stderr)
        sys.exit(1)

    cfg = BacktestConfig(
        strategy_name="",
        instruments=("/MES",),
        timeframe=args.timeframe,
        initial_capital=args.capital,
        commission_per_round_trip=args.commission,
        point_value=args.point_value,
        version="1.5.3",
    )

    all_metrics = []
    for csv_path in csv_paths:
        print(f"\n[RUN] {csv_path}")
        results = run_backtest(csv_path, cfg)
        for r in results:
            m = r["metrics"]
            m["csv"] = Path(csv_path).name
            all_metrics.append(m)

    consolidated = Path("/tmp") / f"metrics_consolidated_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(consolidated, "w") as f:
        json.dump(all_metrics, f, indent=2)

    print(f"\n[DONE] {len(csv_paths)} CSV(s). Consolidated metrics: {consolidated}")
    sys.exit(0)

# =========================
# End of Backtester_vercel.py v1.5.3
# =========================
