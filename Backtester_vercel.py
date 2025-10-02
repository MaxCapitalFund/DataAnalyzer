# -*- coding: utf-8 -*-
# Backtester_vercel.py
# Version 1.8-nocharts
# Production-ready backtester for ThinkOrSwim (TOS) Strategy Report CSVs
# Vercel-compatible, handles all edge cases, no N/A outputs, no charts
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
    debug: bool = False  # Toggle debug logging
    def outdir(self, csv_stem: str, instrument: str) -> str:
        temp_dir = Path('/tmp')
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_instr = instrument.replace("/", "")
        if self.debug:
            print(f"[DEBUG] Outdir = Backtests_{ts}_{safe_instr}_{csv_stem}")
        return str(temp_dir / f"Backtests_{ts}_{safe_instr}_{csv_stem}")

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
        return f"/{m.group(1).upper()}"
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
def load_tos_strategy_report(file_path: str, debug: bool = False) -> pd.DataFrame:
    try:
        with open(file_path, 'r', errors='replace') as f:
            lines = f.readlines()
    except FileNotFoundError:
        if debug:
            print(f"[DEBUG] File not found: {file_path}")
        raise ValueError(f"CSV file not found: {file_path}")
    start_idx = None
    for i, line in enumerate(lines):
        if line.lstrip().startswith("Id;Strategy;"):
            start_idx = i
            break
    if start_idx is None:
        if debug:
            print("[DEBUG] No trade table header found (expected 'Id;Strategy;')")
        raise ValueError("No trade table header found in file.")
    table_str = "".join(lines[start_idx:])
    df = pd.read_csv(io.StringIO(table_str), sep=';')
    if debug:
        print(f"[DEBUG] CSV columns: {df.columns.tolist()}")
    required_cols = ['Side']
    if not any(col in df.columns for col in required_cols):
        if debug:
            print("[DEBUG] CSV missing required 'Side' column")
        raise ValueError("CSV missing required 'Side' column")
    if 'Date/Time' in df.columns:
        df['Date'] = _parse_datetime(df['Date/Time'])
    elif 'Date' in df.columns and 'Time' in df.columns:
        dt_str = df['Date'].astype(str).str.strip() + ' ' + df['Time'].astype(str).str.strip()
        df['Date'] = pd.to_datetime(dt_str, errors='coerce')
    else:
        if debug:
            print("[DEBUG] No 'Date/Time' or 'Date' and 'Time' columns found")
        raise ValueError("No Date column found")
    if 'Trade P/L' in df.columns:
        df['TradePL'] = _to_float(df['Trade P/L']).fillna(0.0)
    else:
        df['TradePL'] = 0.0
    df['CumPL'] = _to_float(df['P/L']) if 'P/L' in df.columns else np.nan
    if 'Strategy' in df.columns:
        df['BaseStrategy'] = df['Strategy'].astype(str).str.split('(').str[0].str.strip()
    else:
        df['BaseStrategy'] = "Unknown"
    if 'Symbol' in df.columns:
        df['Symbol'] = df['Symbol'].astype(str).map(normalize_symbol)
    else:
        s = df['Strategy'].astype(str) if 'Strategy' in df.columns else pd.Series([], dtype=str)
        pat = re.compile(r"/([A-Z]{1,3})")
        sym_guess = s.str.extract(pat, expand=False).fillna("").map(lambda x: f"/{x}" if x else "/UNK")
        df['Symbol'] = sym_guess.map(normalize_symbol)
    if debug:
        print(f"[DEBUG] Loaded CSV rows: {len(df)}, Symbols: {df['Symbol'].unique()}, Side values: {df['Side'].astype(str).unique()}")
        if len(df) <= 5:
            print(f"[DEBUG] First rows:\n{df.to_string()}")
        if df.empty or df['Date'].isna().all():
            print("[DEBUG] No valid data after processing: empty or invalid dates")
    return df.dropna(subset=['Date']).sort_values('Date').reset_index(drop=True)

# =========================
# Build Trades
# =========================
def build_trades(df: pd.DataFrame, commission_rt: float, debug: bool = False) -> pd.DataFrame:
    trades = []
    trade_columns = ['Id', 'EntryTime', 'ExitTime', 'EntryPrice', 'ExitPrice', 'EntryQty', 'ExitQty', 'QtyAbs', 
                     'TradePL', 'Commission', 'NetPL', 'BaseStrategy', 'StrategyRaw', 'Symbol', 'EntrySide', 
                     'ExitSide', 'ExitReason']
    def _safe_num(x):
        return pd.to_numeric(x, errors='coerce')
    OPEN_RX = r"\b(?:BTO|BUY TO OPEN|BUY_TO_OPEN|BOT TO OPEN|STO|SELL TO OPEN|SELL_TO_OPEN|SELL SHORT|OPEN)\b"
    CLOSE_RX = r"\b(?:STC|SELL TO CLOSE|SELL_TO_CLOSE|SLD TO CLOSE|BTC|BUY TO CLOSE|BUY_TO_CLOSE|CLOSE)\b"
    if 'Id' not in df.columns:
        if debug:
            print("[DEBUG] No 'Id' column found, attempting close-only trades")
        side_up = df['Side'].astype(str).str.upper()
        close_rows = df[side_up.str.contains(CLOSE_RX, regex=True, na=False)]
        if debug:
            print(f"[DEBUG] Found {len(close_rows)} close-only rows")
        for _, row in close_rows.iterrows():
            qty_abs = 1.0
            trade_pl = _safe_num(row.get('TradePL', 0.0))
            commission = commission_rt * qty_abs
            net_pl = (trade_pl if pd.notna(trade_pl) else 0.0) - commission
            trades.append({
                'Id': row.get('Id', np.nan),
                'EntryTime': pd.NaT,
                'ExitTime': row.get('Date', pd.NaT),
                'EntryPrice': np.nan,
                'ExitPrice': _safe_num(row.get('Price')),
                'EntryQty': np.nan,
                'ExitQty': _safe_num(row.get('Qty')),
                'QtyAbs': qty_abs,
                'TradePL': trade_pl,
                'Commission': commission,
                'NetPL': net_pl,
                'BaseStrategy': row.get('BaseStrategy', 'Unknown'),
                'StrategyRaw': row.get('Strategy', ''),
                'Symbol': row.get('Symbol', '/UNK'),
                'EntrySide': '',
                'ExitSide': str(row.get('Side', '')),
                'ExitReason': _exit_reason(row.get('Side') or row.get('Type') or row.get('Order', ''))
            })
    else:
        if debug:
            print(f"[DEBUG] Found {len(df['Id'].unique())} unique trade IDs")
        for tid, grp in df.groupby('Id', sort=False):
            g = grp.sort_values('Date').copy()
            side_up = g['Side'].astype(str).str.upper()
            g['is_open'] = side_up.str.contains(OPEN_RX, regex=True, na=False)
            g['is_close'] = side_up.str.contains(CLOSE_RX, regex=True, na=False)
            entry_rows = g[g['is_open']]
            close_rows = g[g['is_close']]
            if debug:
                print(f"[DEBUG] Trade ID {tid}: {len(entry_rows)} entries, {len(close_rows)} exits")
            if len(entry_rows) and len(close_rows):
                entry = entry_rows.iloc[0]
                after_entry_close = close_rows[close_rows['Date'] >= entry['Date']]
                exit_ = after_entry_close.iloc[0] if len(after_entry_close) else close_rows.iloc[-1]
                entry_qty = _safe_num(entry.get('Qty'))
                qty_abs = abs(entry_qty) if pd.notna(entry_qty) and entry_qty != 0 else 1.0
                direction = 'Unknown'
                es = str(entry.get('Side', '')).upper()
                if re.search(r"\b(BTO|BUY TO OPEN|BUY_TO_OPEN|BOT TO OPEN)\b", es):
                    direction = 'Long'
                elif re.search(r"\b(STO|SELL TO OPEN|SELL_TO_OPEN|SELL SHORT)\b", es):
                    direction = 'Short'
                elif pd.notna(entry_qty):
                    direction = 'Long' if entry_qty > 0 else 'Short'
                trade_pl = _safe_num(exit_.get('TradePL'))
                commission = commission_rt * qty_abs
                net_pl = (trade_pl if pd.notna(trade_pl) else 0.0) - commission
                trades.append({
                    'Id': tid,
                    'EntryTime': entry['Date'],
                    'ExitTime': exit_['Date'],
                    'EntryPrice': _safe_num(entry.get('Price')),
                    'ExitPrice': _safe_num(exit_.get('Price')),
                    'EntryQty': entry_qty,
                    'ExitQty': _safe_num(exit_.get('Qty')),
                    'QtyAbs': qty_abs,
                    'TradePL': trade_pl,
                    'Commission': commission,
                    'NetPL': net_pl,
                    'BaseStrategy': entry.get('BaseStrategy', 'Unknown'),
                    'StrategyRaw': entry.get('Strategy', ''),
                    'Symbol': entry.get('Symbol', '/UNK'),
                    'EntrySide': str(entry.get('Side', '')),
                    'ExitSide': str(exit_.get('Side', '')),
                    'ExitReason': _exit_reason(exit_.get('Side') or exit_.get('Type') or exit_.get('Order', ''))
                })
    t = pd.DataFrame(trades, columns=trade_columns)
    if t.empty:
        if debug:
            print("[DEBUG] No trades generated: possible missing entry/exit pairs or invalid Side values")
        t = pd.DataFrame(columns=trade_columns).assign(HoldMins=np.nan)
    else:
        t = t.sort_values('ExitTime').reset_index(drop=True)
        t['HoldMins'] = (t['ExitTime'] - t['EntryTime']).dt.total_seconds() / 60.0
    return t

# =========================
# Stop-loss
# =========================
def apply_stoploss(trades: pd.DataFrame, debug: bool = False) -> pd.DataFrame:
    if trades.empty:
        if debug:
            print("[DEBUG] Empty trades DataFrame in apply_stoploss, returning unchanged")
        return trades
    df = trades.copy()
    df['AdjustedNetPL'] = np.where(df['NetPL'] < -100.0, -100.0, df['NetPL'])
    return df

# =========================
# Metrics
# =========================
def compute_metrics(trades: pd.DataFrame, cfg: BacktestConfig, debug: bool = False) -> dict:
    if trades.empty:
        if debug:
            print("[DEBUG] Empty trades DataFrame in compute_metrics, returning minimal metrics")
        return {
            "strategy": cfg.strategy_name or "Unknown",
            "timeframe": cfg.timeframe,
            "num_trades": 0,
            "net_profit": 0.0,
            "total_return_pct": 0.0,
            "win_rate_pct": 0.0,
            "profit_factor": 0.0,
            "max_drawdown_pct": 0.0,
            "gross_profit": 0.0,
            "gross_loss": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "largest_win": 0.0,
            "largest_loss": 0.0,
            "expectancy": 0.0,
            "recovery_factor": 0.0,
            "sharpe_ratio": 0.0
        }
    pl = trades['AdjustedNetPL'].fillna(0.0)
    equity = cfg.initial_capital + pl.cumsum()
    wins = pl[pl > 0]
    losses = pl[pl < 0]
    return {
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

# =========================
# Analytics Markdown
# =========================
def generate_analytics_md(trades_all: pd.DataFrame, trades_rth: pd.DataFrame, metrics: dict, cfg: BacktestConfig, outdir: str) -> None:
    os.makedirs(outdir, exist_ok=True)
    m = metrics
    def _fmt(x, p=2, pct=False):
        try:
            if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
                return "0.00" if not pct else "0.00%"
            return f"{x:.{p}f}%" if pct else f"{x:.{p}f}"
        except Exception:
            return str(x)
    first_dt_all = pd.to_datetime(trades_all['ExitTime'], errors='coerce').min() if not trades_all.empty else pd.NaT
    last_dt_all = pd.to_datetime(trades_all['ExitTime'], errors='coerce').max() if not trades_all.empty else pd.NaT
    instrument = m.get('instrument', '/UNK')
    md = f"""
# Strategy Analysis Report
**Strategy:** {m.get('strategy', 'Unknown')}
**Instrument:** {instrument}
**Date Range:** {first_dt_all.date() if pd.notna(first_dt_all) else 'n/a'} → {last_dt_all.date() if pd.notna(last_dt_all) else 'n/a'}
**Timeframe:** {m.get('timeframe')}
**Run Date:** {datetime.now().strftime('%Y-%m-%d')}
**P/L Basis:** SL-adjusted net P/L (cap −$100 per trade including commissions)
**Trades:** ALL = {int(m.get('num_trades_all', 0))} | RTH = {int(m.get('num_trades_rth', 0))}
---
## Key Performance Indicators
- **Net Profit:** ${_fmt(m.get('net_profit'))}
- **Total Return:** {_fmt(m.get('total_return_pct'), pct=True)}
- **Win Rate:** {_fmt(m.get('win_rate_pct'), pct=True)}
- **Profit Factor:** {_fmt(m.get('profit_factor'))}
- **Max Drawdown:** {_fmt(m.get('max_drawdown_pct'), pct=True)}
- **Total Trades:** {int(m.get('num_trades', 0))}
---
## Performance Details
- **Average Win:** ${_fmt(m.get('avg_win'))}
- **Average Loss:** {_fmt(m.get('avg_loss'))}
- **Largest Win:** ${_fmt(m.get('largest_win'))}
- **Largest Loss:** ${_fmt(m.get('largest_loss'))}
- **Expectancy:** ${_fmt(m.get('expectancy'))}
- **Recovery Factor:** {_fmt(m.get('recovery_factor'))}
- **Sharpe Ratio:** {_fmt(m.get('sharpe_ratio'))}
---
## RTH Snapshot (09:30–16:00 ET)
- **Net Profit (RTH):** ${_fmt(m.get('RTH_net_profit'))}
- **Win Rate (RTH):** {_fmt(m.get('RTH_win_rate_pct'), pct=True)}
- **Profit Factor (RTH):** {_fmt(m.get('RTH_profit_factor'))}
- **Max Drawdown (RTH):** {_fmt(m.get('RTH_max_drawdown_pct'), pct=True)}
- **Total Trades (RTH):** {int(m.get('num_trades_rth', 0))}
---
*Report generated by Backtester v{cfg.version}*
"""
    with open(os.path.join(outdir, "analytics.md"), "w", encoding="utf-8") as f:
        f.write(md)

# =========================
# Runner
# =========================
def run_backtest(tos_csv_path: str, cfg: BacktestConfig):
    csv_stem = Path(tos_csv_path).stem.replace(' ', '_')
    try:
        raw = load_tos_strategy_report(tos_csv_path, cfg.debug)
    except ValueError as e:
        if cfg.debug:
            print(f"[DEBUG] Failed to load CSV: {e}")
        results = []
        outdir = cfg.outdir(csv_stem, '/UNK')
        os.makedirs(outdir, exist_ok=True)
        metrics = {
            "strategy": cfg.strategy_name or "Unknown",
            "instrument": "/UNK",
            "timeframe": cfg.timeframe,
            "num_trades": 0,
            "num_trades_all": 0,
            "num_trades_rth": 0,
            "net_profit": 0.0,
            "total_return_pct": 0.0,
            "win_rate_pct": 0.0,
            "profit_factor": 0.0,
            "max_drawdown_pct": 0.0,
            "gross_profit": 0.0,
            "gross_loss": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "largest_win": 0.0,
            "largest_loss": 0.0,
            "expectancy": 0.0,
            "recovery_factor": 0.0,
            "sharpe_ratio": 0.0,
            "RTH_net_profit": 0.0,
            "RTH_win_rate_pct": 0.0,
            "RTH_profit_factor": 0.0,
            "RTH_max_drawdown_pct": 0.0,
            "RTH_num_trades": 0
        }
        with open(os.path.join(outdir, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)
        trade_columns = ['Id', 'EntryTime', 'ExitTime', 'EntryPrice', 'ExitPrice', 'EntryQty', 'ExitQty', 'QtyAbs', 
                         'TradePL', 'Commission', 'NetPL', 'BaseStrategy', 'StrategyRaw', 'Symbol', 'EntrySide', 
                         'ExitSide', 'ExitReason', 'HoldMins']
        pd.DataFrame(columns=trade_columns).to_csv(os.path.join(outdir, "trades_enriched.csv"), index=False)
        generate_analytics_md(pd.DataFrame(columns=trade_columns), pd.DataFrame(columns=trade_columns), metrics, cfg, outdir)
        def print_metrics(m):
            print(f"Strategy: {m.get('strategy', 'Unknown')}")
            print(f"Instrument: {m.get('instrument', '/UNK')}")
            print(f"Net Profit: ${m.get('net_profit', 0.0):.2f}")
            print(f"Total Return: {m.get('total_return_pct', 0.0):.2f}%")
            print(f"Win Rate: {m.get('win_rate_pct', 0.0):.2f}%")
            print(f"Profit Factor: {m.get('profit_factor', 0.0):.2f}")
            print(f"Max Drawdown: {m.get('max_drawdown_pct', 0.0):.2f}%")
            print(f"Total Trades: {m.get('num_trades', 0)}")
            print(f"Gross Profit: ${m.get('gross_profit', 0.0):.2f}")
            print(f"Gross Loss: ${m.get('gross_loss', 0.0):.2f}")
            print(f"Avg Win: ${m.get('avg_win', 0.0):.2f}")
            print(f"Avg Loss: ${m.get('avg_loss', 0.0):.2f}")
            print(f"Largest Win: ${m.get('largest_win', 0.0):.2f}")
            print(f"Largest Loss: ${m.get('largest_loss', 0.0):.2f}")
            print(f"Expectancy: ${m.get('expectancy', 0.0):.2f}")
            print(f"Recovery Factor: {m.get('recovery_factor', 0.0):.2f}")
            print(f"Sharpe Ratio: {m.get('sharpe_ratio', 0.0):.2f}")
        print_metrics(metrics)
        results.append({"instrument": "/UNK", "metrics": metrics, "outdir": outdir})
        return results
    if raw.empty:
        if cfg.debug:
            print(f"[DEBUG] Empty CSV: {tos_csv_path}")
        results = []
        outdir = cfg.outdir(csv_stem, '/UNK')
        os.makedirs(outdir, exist_ok=True)
        metrics = {
            "strategy": cfg.strategy_name or "Unknown",
            "instrument": "/UNK",
            "timeframe": cfg.timeframe,
            "num_trades": 0,
            "num_trades_all": 0,
            "num_trades_rth": 0,
            "net_profit": 0.0,
            "total_return_pct": 0.0,
            "win_rate_pct": 0.0,
            "profit_factor": 0.0,
            "max_drawdown_pct": 0.0,
            "gross_profit": 0.0,
            "gross_loss": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "largest_win": 0.0,
            "largest_loss": 0.0,
            "expectancy": 0.0,
            "recovery_factor": 0.0,
            "sharpe_ratio": 0.0,
            "RTH_net_profit": 0.0,
            "RTH_win_rate_pct": 0.0,
            "RTH_profit_factor": 0.0,
            "RTH_max_drawdown_pct": 0.0,
            "RTH_num_trades": 0
        }
        with open(os.path.join(outdir, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)
        trade_columns = ['Id', 'EntryTime', 'ExitTime', 'EntryPrice', 'ExitPrice', 'EntryQty', 'ExitQty', 'QtyAbs', 
                         'TradePL', 'Commission', 'NetPL', 'BaseStrategy', 'StrategyRaw', 'Symbol', 'EntrySide', 
                         'ExitSide', 'ExitReason', 'HoldMins']
        pd.DataFrame(columns=trade_columns).to_csv(os.path.join(outdir, "trades_enriched.csv"), index=False)
        generate_analytics_md(pd.DataFrame(columns=trade_columns), pd.DataFrame(columns=trade_columns), metrics, cfg, outdir)
        def print_metrics(m):
            print(f"Strategy: {m.get('strategy', 'Unknown')}")
            print(f"Instrument: {m.get('instrument', '/UNK')}")
            print(f"Net Profit: ${m.get('net_profit', 0.0):.2f}")
            print(f"Total Return: {m.get('total_return_pct', 0.0):.2f}%")
            print(f"Win Rate: {m.get('win_rate_pct', 0.0):.2f}%")
            print(f"Profit Factor: {m.get('profit_factor', 0.0):.2f}")
            print(f"Max Drawdown: {m.get('max_drawdown_pct', 0.0):.2f}%")
            print(f"Total Trades: {m.get('num_trades', 0)}")
            print(f"Gross Profit: ${m.get('gross_profit', 0.0):.2f}")
            print(f"Gross Loss: ${m.get('gross_loss', 0.0):.2f}")
            print(f"Avg Win: ${m.get('avg_win', 0.0):.2f}")
            print(f"Avg Loss: ${m.get('avg_loss', 0.0):.2f}")
            print(f"Largest Win: ${m.get('largest_win', 0.0):.2f}")
            print(f"Largest Loss: ${m.get('largest_loss', 0.0):.2f}")
            print(f"Expectancy: ${m.get('expectancy', 0.0):.2f}")
            print(f"Recovery Factor: {m.get('recovery_factor', 0.0):.2f}")
            print(f"Sharpe Ratio: {m.get('sharpe_ratio', 0.0):.2f}")
        print_metrics(metrics)
        results.append({"instrument": "/UNK", "metrics": metrics, "outdir": outdir})
        return results
    symbols = raw['Symbol'].dropna().unique().tolist() or ['/MES']
    results = []
    for instr in symbols:
        df_instr = raw[raw['Symbol'] == instr].copy()
        if df_instr.empty:
            if cfg.debug:
                print(f"[DEBUG] No data for instrument: {instr}, skipping")
            continue
        pv = cfg.point_value
        if instr.upper() in {'/MES', 'MES'}:
            pv = 5.0
        elif instr.upper() in {'/MNQ', 'MNQ'}:
            pv = 2.0
        cfg.point_value = pv
        strategy_label = df_instr['BaseStrategy'].dropna().iloc[0] if 'BaseStrategy' in df_instr.columns and len(df_instr) else cfg.strategy_name
        cfg.strategy_name = strategy_label
        outdir = cfg.outdir(csv_stem, instr)
        if cfg.debug:
            print(f"Processing instrument: {instr}, Strategy: {strategy_label}")
        trades_all = build_trades(df_instr, cfg.commission_per_round_trip, cfg.debug)
        if cfg.debug:
            print(f"Trades generated: {len(trades_all)}")
        trades_all = apply_stoploss(trades_all, cfg.debug)
        trades_rth = trades_all[trades_all['EntryTime'].apply(_in_rth) | trades_all['ExitTime'].apply(_in_rth)].copy() if not trades_all.empty else trades_all
        os.makedirs(outdir, exist_ok=True)
        trades_all.to_csv(os.path.join(outdir, "trades_enriched.csv"), index=False)
        metrics_all = compute_metrics(trades_all, cfg, cfg.debug)
        metrics_all["instrument"] = instr
        metrics_all["num_trades_all"] = int(len(trades_all))
        metrics_all["num_trades_rth"] = int(len(trades_rth))
        metrics_rth = compute_metrics(trades_rth, cfg, cfg.debug)
        for k, v in metrics_rth.items():
            metrics_all[f"RTH_{k}"] = v
        with open(os.path.join(outdir, "metrics.json"), "w") as f:
            json.dump(metrics_all, f, indent=2)
        generate_analytics_md(trades_all, trades_rth, metrics_all, cfg, outdir)
        def print_metrics(m):
            print(f"Strategy: {m.get('strategy', 'Unknown')}")
            print(f"Instrument: {m.get('instrument', '/UNK')}")
            print(f"Net Profit: ${m.get('net_profit', 0.0):.2f}")
            print(f"Total Return: {m.get('total_return_pct', 0.0):.2f}%")
            print(f"Win Rate: {m.get('win_rate_pct', 0.0):.2f}%")
            print(f"Profit Factor: {m.get('profit_factor', 0.0):.2f}")
            print(f"Max Drawdown: {m.get('max_drawdown_pct', 0.0):.2f}%")
            print(f"Total Trades: {m.get('num_trades', 0)}")
            print(f"Gross Profit: ${m.get('gross_profit', 0.0):.2f}")
            print(f"Gross Loss: ${m.get('gross_loss', 0.0):.2f}")
            print(f"Avg Win: ${m.get('avg_win', 0.0):.2f}")
            print(f"Avg Loss: ${m.get('avg_loss', 0.0):.2f}")
            print(f"Largest Win: ${m.get('largest_win', 0.0):.2f}")
            print(f"Largest Loss: ${m.get('largest_loss', 0.0):.2f}")
            print(f"Expectancy: ${m.get('expectancy', 0.0):.2f}")
            print(f"Recovery Factor: {m.get('recovery_factor', 0.0):.2f}")
            print(f"Sharpe Ratio: {m.get('sharpe_ratio', 0.0):.2f}")
        print_metrics(metrics_all)
        results.append({"instrument": instr, "metrics": metrics_all, "outdir": outdir})
    return results

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
