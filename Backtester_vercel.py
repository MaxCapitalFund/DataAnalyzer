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
    version: str = "1.3.1"

    def outdir(self, csv_stem: str, instrument: str, strategy_label: str) -> str:
        # Use /tmp for Vercel/serverless environments
        temp_dir = Path('/tmp')
        day = datetime.now().strftime("%Y-%m-%d")
        timestamp = datetime.now().strftime("%H%M%S_%f")[:-3]  # Include milliseconds for uniqueness
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
    # Try TOS default first, then fall back to generic parser
    parsed = pd.to_datetime(series, format="%m/%d/%y %I:%M %p", errors='coerce')
    if parsed.isna().all():
        parsed = pd.to_datetime(series, errors='coerce')
    return parsed

# Session tags per spec (ET)
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

def _tag_day_part(dt: pd.Timestamp) -> str:
    if pd.isna(dt):
        return "Unknown"
    t = dt.time()
    if time(9, 30) <= t <= time(11, 30): return "EARLY"
    if time(11, 30) < t <= time(14, 0): return "MID"
    if time(14, 0) < t <= time(16, 0): return "LATE"
    return "OTHER"

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

# ===== Symbol normalization =====
ROOT_RE = re.compile(r"^/?([A-Za-z]{1,3})(?:[FGHJKMNQUVXZ]\d{1,2})?$")  # e.g., /MESU25 or MESZ4

def normalize_symbol(sym: str) -> str:
    """Collapse variations like /MESU25 → /MES, MESZ4 → /MES; keep a leading slash."""
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
        raise ValueError("No trade table header found in file (expected line starting with 'Id;Strategy;').")
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
    # Strategy base name
    if 'Strategy' in df.columns:
        base = df['Strategy'].astype(str).str.split('(').str[0].str.strip()
        df['BaseStrategy'] = base
    else:
        df['BaseStrategy'] = "Unknown"
    # Side / Action
    side_col = None
    for cand in ['Side', 'Action', 'Order', 'Type']:
        if cand in df.columns:
            side_col = cand
            break
    df['Side'] = df[side_col].astype(str) if side_col else ""
    # Price
    if 'Price' not in df.columns:
        df['Price'] = np.nan
    # Quantity
    qty_col = 'Quantity' if 'Quantity' in df.columns else ('Qty' if 'Qty' in df.columns else None)
    if qty_col and qty_col != 'Qty':
        df['Qty'] = pd.to_numeric(df[qty_col], errors='coerce')
    elif 'Qty' not in df.columns:
        df['Qty'] = np.nan
    # Symbol/instrument normalization
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
# Build Trades (pair entries/exits)
# =========================

def build_trades(df: pd.DataFrame, commission_rt: float) -> pd.DataFrame:
    id_col = 'Id' if 'Id' in df.columns else None
    trades = []
    def _safe_num(x):
        return pd.to_numeric(x, errors='coerce')
    OPEN_RX = r"\b(?:BTO|BUY TO OPEN|BUY_TO_OPEN|BOT TO OPEN|STO|SELL TO OPEN|SELL_TO_OPEN|SELL SHORT|OPEN)\b"
    CLOSE_RX = r"\b(?:STC|SELL TO CLOSE|SELL_TO_CLOSE|SLD TO CLOSE|BTC|BUY TO CLOSE|BUY_TO_CLOSE|CLOSE)\b"
    if id_col:
        for tid, grp in df.groupby(id_col, sort=False):
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
                    'GrossPL': trade_pl,  # Gross before commissions
                    'Commission': commission,
                    'NetPL': net_pl,  # After commissions
                    'BaseStrategy': entry.get('BaseStrategy', 'Unknown'),
                    'StrategyRaw': entry.get('Strategy', ''),
                    'Symbol': entry.get('Symbol', ''),
                    'EntrySide': str(entry.get('Side', '')),
                    'ExitSide': str(exit_.get('Side', '')),
                    'ExitReason': _exit_reason(exit_.get('Side') or exit_.get('Type') or exit_.get('Order')),
                    'Direction': direction,
                })
    # Fallback: treat each close as a trade row
    if not trades:
        g = df.sort_values('Date').copy()
        side_up = g['Side'].astype(str).str.upper()
        close_rows = g[side_up.str.contains(CLOSE_RX, regex=True, na=False)]
        for _, row in close_rows.iterrows():
            entry_qty = np.nan
            qty_abs = 1.0
            trade_pl = _safe_num(row.get('TradePL'))
            commission = commission_rt * qty_abs
            net_pl = (trade_pl if pd.notna(trade_pl) else 0.0) - commission
            trades.append({
                'Id': row.get('Id', np.nan),
                'EntryTime': pd.NaT,
                'ExitTime': row['Date'],
                'EntryPrice': np.nan,
                'ExitPrice': _safe_num(row.get('Price')),
                'EntryQty': entry_qty,
                'ExitQty': _safe_num(row.get('Qty')),
                'QtyAbs': qty_abs,
                'TradePL': trade_pl,
                'GrossPL': trade_pl,
                'Commission': commission,
                'NetPL': net_pl,
                'BaseStrategy': row.get('BaseStrategy', 'Unknown'),
                'StrategyRaw': row.get('Strategy', ''),
                'Symbol': row.get('Symbol', ''),
                'EntrySide': str(row.get('Side', '')),
                'ExitSide': str(row.get('Side', '')),
                'ExitReason': _exit_reason(row.get('Side') or row.get('Type') or row.get('Order')),
                'Direction': 'Unknown',
            })
    t = pd.DataFrame(trades)
    if t.empty:
        return t.assign(HoldMins=np.nan)
    t = t.sort_values('ExitTime').reset_index(drop=True)
    t['HoldMins'] = (t['ExitTime'] - t['EntryTime']).dt.total_seconds() / 60.0
    return t

# =========================
# Stop-loss correction
# =========================

def apply_stoploss_corrections(trades: pd.DataFrame, point_value: float) -> pd.DataFrame:
    """
    Apply SL normalization:
      - Cap AdjustedNetPL at -$100 per trade (including commissions).
      - Compute PointsPerContract off gross (price move) to avoid commission drag.
    """
    df = trades.copy()
    df['SLBreached'] = df['NetPL'] < -100.0
    # Adjusted Net P/L (after commissions), capped at -100
    df['AdjustedNetPL'] = np.where(df['SLBreached'], -100.0, df['NetPL'])
    # Recompute points per contract from "gross-adjusted" P/L (exclude commissions for interpretability)
    qty_abs = pd.to_numeric(df['QtyAbs'], errors='coerce').replace(0, np.nan)
    gross_adjusted = np.where(df['SLBreached'], -100.0 + df['Commission'], df['NetPL'] + df['Commission'])
    df['PointsPerContract'] = gross_adjusted / (point_value * qty_abs)
    return df

# =========================
# Metrics
# =========================

def compute_metrics(trades_df: pd.DataFrame, cfg: BacktestConfig, scope_label: str) -> dict:
    df = trades_df.copy()
    if 'AdjustedNetPL' not in df.columns:
        raise RuntimeError("AdjustedNetPL missing; call apply_stoploss_corrections() before compute_metrics().")
    # Gross vs Net P/L series
    pl_net = df['AdjustedNetPL'].fillna(0.0)
    pl_gross = df['GrossPL'].fillna(0.0) if 'GrossPL' in df.columns else pl_net.copy()
    # Equity from NET P/L (Adjusted)
    equity = cfg.initial_capital + pl_net.cumsum()
    # Totals & returns
    total_net = float(pl_net.sum())
    total_gross = float(pl_gross.sum())
    total_return_pct = (total_net / cfg.initial_capital) * 100.0 if cfg.initial_capital else np.nan
    # Win/loss (Adjusted)
    win_mask = pl_net > 0
    loss_mask = pl_net < 0
    avg_win = float(pl_net[win_mask].mean()) if win_mask.any() else np.nan
    avg_loss = float(pl_net[loss_mask].mean()) if loss_mask.any() else np.nan
    # Drawdowns (Adjusted net equity)
    max_dd_pct = abs(_max_drawdown(equity)) * 100.0
    max_dd_dollars = float((equity.cummax() - equity).max())
    recovery_factor = float(total_net / max_dd_dollars) if max_dd_dollars else np.nan
    # Expectancy
    expectancy_dollars = float(pl_net.mean()) if len(pl_net) else np.nan
    # Risk-adjusted (per-trade returns vs initial capital; quick proxy)
    trade_rets = pl_net / cfg.initial_capital if cfg.initial_capital else pd.Series(np.nan, index=pl_net.index)
    std = trade_rets.std(ddof=1)
    per_trade_sharpe = float(trade_rets.mean() / std) if std and std > 0 else np.nan
    first_dt = pd.to_datetime(df['ExitTime']).min()
    last_dt = pd.to_datetime(df['ExitTime']).max()
    days = max((last_dt - first_dt).days, 1) if pd.notna(first_dt) and pd.notna(last_dt) else 1
    trades_per_year = (len(df) / days * 252.0) if days and days > 0 else np.nan
    sharpe_annualized = float(np.sqrt(trades_per_year) * per_trade_sharpe) if trades_per_year and per_trade_sharpe == per_trade_sharpe else np.nan
    largest_win = float(pl_net.max()) if len(pl_net) else np.nan
    largest_loss = float(pl_net.min()) if len(pl_net) else np.nan
    # Holding Time Analysis
    hold_mins = df['HoldMins'].dropna()
    winners = df[win_mask]['HoldMins'].dropna()
    losers = df[loss_mask]['HoldMins'].dropna()
    avg_hold_winners = float(winners.mean()) if len(winners) else np.nan
    avg_hold_losers = float(losers.mean()) if len(losers) else np.nan
    median_hold = float(hold_mins.median()) if len(hold_mins) else np.nan
    longest_hold = float(hold_mins.max()) if len(hold_mins) else np.nan
    shortest_hold = float(hold_mins.min()) if len(hold_mins) else np.nan
    hold_bins = [0, 15, 60, 180, float('inf')]
    hold_labels = ['0-15', '15-60', '60-180', '>180']
    df['HoldBin'] = pd.cut(df['HoldMins'], bins=hold_bins, labels=hold_labels, include_lowest=True)
    hold_dist = df['HoldBin'].value_counts().reindex(hold_labels).fillna(0).to_dict()
    # Trade Frequency by Session
    df['Session'] = df.apply(lambda r: _tag_session(r['EntryTime'] if pd.notna(r['EntryTime']) else r['ExitTime']), axis=1)
    session_counts = df['Session'].value_counts().to_dict()
    session_expectancy = df.groupby('Session')['AdjustedNetPL'].mean().to_dict()
    # Day of Week Performance
    df['DayOfWeek'] = df['ExitTime'].dt.day_name()
    dow_metrics = {}
    for day in ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']:
        day_df = df[df['DayOfWeek'] == day]
        dow_metrics[day] = {
            'win_rate': float((day_df['AdjustedNetPL'] > 0).mean() * 100) if len(day_df) else np.nan,
            'net_pl': float(day_df['AdjustedNetPL'].sum()) if len(day_df) else np.nan,
            'expectancy': float(day_df['AdjustedNetPL'].mean()) if len(day_df) else np.nan
        }
    # Expectancy by Trade Length
    expectancy_by_hold = df.groupby('HoldBin')['AdjustedNetPL'].mean().reindex(hold_labels).fillna(np.nan).to_dict()
    # Capital Efficiency
    avg_capital_per_trade = df['EntryPrice'] * df['QtyAbs'] * cfg.point_value
    avg_hold_mins = df['HoldMins'].mean() if len(df['HoldMins'].dropna()) else 1.0
    capital_efficiency = total_net / (avg_capital_per_trade.mean() * avg_hold_mins) if avg_capital_per_trade.mean() else np.nan
    # Concentration Risk
    top_10_pct_count = max(1, int(len(pl_net) * 0.1))
    top_trades = pl_net.nlargest(top_10_pct_count).sum()
    concentration_risk = (top_trades / total_net * 100) if total_net != 0 else np.nan
    # Trade Outcome by Day Part
    df['DayPart'] = df['ExitTime'].apply(_tag_day_part)
    day_part_expectancy = df.groupby('DayPart')['AdjustedNetPL'].mean().reindex(['EARLY', 'MID', 'LATE', 'OTHER']).fillna(np.nan).to_dict()
    metrics = {
        "scope": scope_label,
        "strategy_name": cfg.strategy_name,
        "version": cfg.version,
        "timeframe": cfg.timeframe,
        "initial_capital": cfg.initial_capital,
        "point_value": cfg.point_value,
        "num_trades": int(len(df)),
        "net_profit": total_net,
        "gross_profit": total_gross,
        "total_return_pct": total_return_pct,
        "profit_factor": _profit_factor(pl_net),
        "win_rate_pct": float((pl_net > 0).mean() * 100.0),
        "avg_win_dollars": avg_win,
        "avg_loss_dollars": avg_loss,
        "expectancy_per_trade_dollars": expectancy_dollars,
        "max_drawdown_pct": max_dd_pct,
        "max_drawdown_dollars": max_dd_dollars,
        "recovery_factor": recovery_factor,
        "sharpe_annualized": sharpe_annualized,
        "largest_winning_trade": largest_win,
        "largest_losing_trade": largest_loss,
        "avg_hold_winners_mins": avg_hold_winners,
        "avg_hold_losers_mins": avg_hold_losers,
        "median_hold_mins": median_hold,
        "longest_hold_mins": longest_hold,
        "shortest_hold_mins": shortest_hold,
        "hold_distribution": hold_dist,
        "session_counts": session_counts,
        "session_expectancy": session_expectancy,
        "dow_metrics": dow_metrics,
        "expectancy_by_hold": expectancy_by_hold,
        "capital_efficiency": capital_efficiency,
        "concentration_risk_pct": concentration_risk,
        "day_part_expectancy": day_part_expectancy
    }
    return metrics

# =========================
# Visuals (ALL sessions by default)
# =========================

def save_visuals_and_tables(trades_df: pd.DataFrame, cfg: BacktestConfig, outdir: str, title_suffix: str = "ALL") -> None:
    os.makedirs(outdir, exist_ok=True)
    pl = trades_df['AdjustedNetPL'].fillna(0.0)
    equity = cfg.initial_capital + pl.cumsum()
    # Equity curve
    plt.figure(figsize=(9, 4.5))
    plt.plot(equity.index, equity.values)
    plt.ylabel("Equity ($)")
    plt.xlabel("Trade #")
    plt.title(f"Equity Curve — {cfg.strategy_name} ({cfg.timeframe}) [{title_suffix}, SL-adjusted]")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"equity_curve_{title_suffix.lower()}.png"), dpi=160, bbox_inches='tight')
    plt.close()
    # Drawdown curve
    dd = equity / equity.cummax() - 1.0
    plt.figure(figsize=(9, 4.5))
    plt.plot(dd.index, dd.values * 100.0)
    plt.ylabel("Drawdown (%)")
    plt.xlabel("Trade #")
    plt.title(f"Drawdown Curve [{title_suffix}, SL-adjusted]")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"drawdown_curve_{title_suffix.lower()}.png"), dpi=160, bbox_inches='tight')
    plt.close()
    # Histogram of trade P/L
    plt.figure(figsize=(9, 4.5))
    plt.hist(trades_df['AdjustedNetPL'].dropna().values, bins=30)
    plt.xlabel("Net P/L per Trade ($) — SL-adjusted")
    plt.ylabel("Count")
    plt.title(f"Trade P/L Distribution [{title_suffix}]")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"pl_histogram_{title_suffix.lower()}.png"), dpi=160, bbox_inches='tight')
    plt.close()
    # Histogram of hold times
    plt.figure(figsize=(9, 4.5))
    plt.hist(trades_df['HoldMins'].dropna().values, bins=30)
    plt.xlabel("Hold Time (Minutes)")
    plt.ylabel("Count")
    plt.title(f"Hold Time Distribution [{title_suffix}]")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"hold_time_histogram_{title_suffix.lower()}.png"), dpi=160, bbox_inches='tight')
    plt.close()
    # Monthly performance table
    dt = pd.to_datetime(trades_df['ExitTime'], errors='coerce')
    monthly = pd.DataFrame({'NetPL': trades_df['AdjustedNetPL'].values}, index=dt)
    monthly = monthly.dropna().resample('ME').sum()
    monthly['ReturnPct'] = monthly['NetPL'] / cfg.initial_capital * 100.0
    monthly.to_csv(os.path.join(outdir, f"monthly_performance_{title_suffix.lower()}.csv"))

# =========================
# Main runner
# =========================

def run_backtest_for_instrument(df_raw: pd.DataFrame, instrument: Optional[str], cfg: BacktestConfig, csv_stem: str):
    # Strategy label pulled from CSV base strategy
    strategy_label = df_raw['BaseStrategy'].dropna().iloc[0] if 'BaseStrategy' in df_raw.columns and len(df_raw.dropna(subset=['BaseStrategy'])) else (cfg.strategy_name or 'Unknown')
    # Normalize instrument and set point value
    instr = normalize_symbol(instrument or '/UNK')
    cfg.strategy_name = strategy_label
    # Point value mapping
    pv = cfg.point_value
    if instr.upper() in {'/MES', 'MES'}:
        pv = 5.0
    elif instr.upper() in {'/MNQ', 'MNQ'}:
        pv = 2.0
    cfg.point_value = pv
    outdir = cfg.outdir(csv_stem, instr, strategy_label)
    os.makedirs(outdir, exist_ok=True)
    # Build trades (ALL trades first)
    trades_all = build_trades(df_raw, cfg.commission_per_round_trip)
    # Session tags — EntryTime, fallback to ExitTime
    def _session_dt(row):
        et = row.get('EntryTime')
        return et if pd.notna(et) else row.get('ExitTime')
    trades_all['Session'] = trades_all.apply(lambda r: _tag_session(_session_dt(r)), axis=1)
    # Apply stop-loss normalization
    trades_all = apply_stoploss_corrections(trades_all, cfg.point_value)
    # Save enriched trades (ALL)
    trades_out = os.path.join(outdir, "trades_enriched.csv")
    trades_all.to_csv(trades_out, index=False)
    # RTH subset
    def _rth_row(row):
        t = row['EntryTime'] if pd.notna(row['EntryTime']) else row.get('ExitTime')
        return _in_rth(t)
    trades_rth = trades_all[trades_all.apply(_rth_row, axis=1)].copy()
    # Metrics (ALL primary, RTH snapshot)
    metrics_all = compute_metrics(trades_all, cfg, scope_label="ALL")
    metrics_all["strategy_name"] = strategy_label
    metrics_all["instrument"] = instr
    metrics_all["num_trades_all"] = int(len(trades_all))
    metrics_all["num_trades_rth"] = int(len(trades_rth))
    metrics_rth = compute_metrics(trades_rth, cfg, scope_label="RTH")
    # Prefix RTH keys for secondary display
    for k, v in metrics_rth.items():
        metrics_all[f"RTH_{k}"] = v
    metrics = metrics_all
    # Save metrics
    with open(os.path.join(outdir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    # Visuals & tables (ALL as primary)
    save_visuals_and_tables(trades_all, cfg, outdir, title_suffix="ALL")
    # Generate analytics markdown (ALL primary with RTH snapshot; charts at bottom)
    generate_analytics_md(trades_all, trades_rth, metrics, cfg, outdir)
    # Save config
    with open(os.path.join(outdir, "config.json"), "w") as f:
        json.dump(asdict(cfg), f, indent=2)
    return trades_all, trades_rth, metrics, outdir

def run_backtest(tos_csv_path: str, cfg: BacktestConfig):
    csv_stem = Path(tos_csv_path).stem.replace(' ', '_')
    raw = load_tos_strategy_report(tos_csv_path)
    # Ensure normalized symbols
    if 'Symbol' in raw.columns:
        raw['Symbol'] = raw['Symbol'].map(normalize_symbol)
    else:
        raw['Symbol'] = "/UNK"
    # Detect instruments present (normalized roots)
    symbols = raw['Symbol'].dropna().unique().tolist()
    if not symbols:
        # Attempt extraction from Strategy text, normalize
        s = raw['Strategy'].astype(str) if 'Strategy' in df.columns else pd.Series([], dtype=str)
        pat = re.compile(r"/([A-Z]{1,3})")
        symbols = s.str.extract(pat, expand=False).dropna().map(lambda x: f"/{x}").map(normalize_symbol).unique().tolist()
    if not symbols:
        symbols = ['/MES']  # fallback default
    results = []
    for instr in symbols:
        trades_all, trades_rth, metrics, outdir = run_backtest_for_instrument(raw, instr, cfg, csv_stem)
        results.append({"instrument": instr, "metrics": metrics, "outdir": outdir})
    return results

def generate_analytics_md(trades_all: pd.DataFrame, trades_rth: pd.DataFrame, metrics: dict, cfg: BacktestConfig, outdir: str) -> None:
    """Generate analytics markdown:
       - Primary KPIs: ALL sessions
       - Snapshot section: RTH
       - Efficiency and Effectiveness Metrics
       - Charts at bottom (ALL)
    """
    os.makedirs(outdir, exist_ok=True)
    m = metrics
    def g(key, default=np.nan):
        return m.get(key, default)
    def _fmt(x, p=2, pct=False):
        try:
            if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
                return "n/a"
            return (f"{x:.{p}f}%" if pct else f"{x:.{p}f}")
        except Exception:
            return str(x)
    first_dt_all = pd.to_datetime(trades_all['ExitTime'], errors='coerce').min()
    last_dt_all = pd.to_datetime(trades_all['ExitTime'], errors='coerce').max()
    instrument = g('instrument', '/UNK')
    md = f"""
# Strategy Analysis Report
**Strategy:** {g('strategy_name')}
**Instrument:** {instrument}
**Date Range (ALL):** {first_dt_all.date() if pd.notna(first_dt_all) else 'n/a'} → {last_dt_all.date() if pd.notna(last_dt_all) else 'n/a'}
**Timeframe:** {g('timeframe')}
**Run Date:** {datetime.now().strftime('%Y-%m-%d')}
**Session Basis:** New York time (ET)
**P/L Basis:** *All KPIs computed on **SL-adjusted net P/L** (cap −$100 per trade including commissions).*
**Trades:** ALL = {int(g('num_trades_all', 0))} | RTH = {int(g('num_trades_rth', 0))}
---
## Key Performance Indicators — ALL Sessions
- **Net Profit (ALL):** ${_fmt(g('net_profit'))}
- **Total Return (ALL):** {_fmt(g('total_return_pct'), pct=True)}
- **Win Rate (ALL):** {_fmt(g('win_rate_pct'), pct=True)}
- **Profit Factor (ALL):** {_fmt(g('profit_factor'))}
- **Max Drawdown (ALL):** ${_fmt(g('max_drawdown_dollars'))} ({_fmt(g('max_drawdown_pct'), pct=True)})
- **Sharpe (annualized, ALL):** {_fmt(g('sharpe_annualized'))}
- **Total Trades (ALL):** {int(g('num_trades', 0))}
---
## RTH Snapshot (09:30–16:00 ET)
- **Net Profit (RTH):** ${_fmt(g('RTH_net_profit'))}
- **Win Rate (RTH):** {_fmt(g('RTH_win_rate_pct'), pct=True)}
- **Profit Factor (RTH):** {_fmt(g('RTH_profit_factor'))}
- **Max Drawdown (RTH):** ${_fmt(g('RTH_max_drawdown_dollars'))} ({_fmt(g('RTH_max_drawdown_pct'), pct=True)})
- **Sharpe (annualized, RTH):** {_fmt(g('RTH_sharpe_annualized'))}
- **Total Trades (RTH):** {int(g('num_trades_rth', 0))}
---
## Efficiency Metrics
### Holding Time Analysis
- **Average Hold Time (Winners):** {_fmt(g('avg_hold_winners_mins'))} minutes
- **Average Hold Time (Losers):** {_fmt(g('avg_hold_losers_mins'))} minutes
- **Median Hold Time:** {_fmt(g('median_hold_mins'))} minutes
- **Longest Hold Time:** {_fmt(g('longest_hold_mins'))} minutes
- **Shortest Hold Time:** {_fmt(g('shortest_hold_mins'))} minutes
- **Hold Time Distribution:**
  - 0–15 min: {int(g('hold_distribution').get('0-15', 0))} trades
  - 15–60 min: {int(g('hold_distribution').get('15-60', 0))} trades
  - 60–180 min: {int(g('hold_distribution').get('60-180', 0))} trades
  - >180 min: {int(g('hold_distribution').get('>180', 0))} trades
### Trade Frequency by Session
- **Trade Counts by Session:**
  - PRE: {int(g('session_counts').get('PRE', 0))} trades
  - OPEN: {int(g('session_counts').get('OPEN', 0))} trades
  - LUNCH: {int(g('session_counts').get('LUNCH', 0))} trades
  - CLOSING: {int(g('session_counts').get('CLOSING', 0))} trades
  - OTHER: {int(g('session_counts').get('OTHER', 0))} trades
- **Expectancy by Session:**
  - PRE: ${_fmt(g('session_expectancy').get('PRE', np.nan))}
  - OPEN: ${_fmt(g('session_expectancy').get('OPEN', np.nan))}
  - LUNCH: ${_fmt(g('session_expectancy').get('LUNCH', np.nan))}
  - CLOSING: ${_fmt(g('session_expectancy').get('CLOSING', np.nan))}
  - OTHER: ${_fmt(g('session_expectancy').get('OTHER', np.nan))}
### Day of Week Performance
- **Monday:**
  - Win Rate: {_fmt(g('dow_metrics').get('Monday', {}).get('win_rate', np.nan), pct=True)}
  - Net P/L: ${_fmt(g('dow_metrics').get('Monday', {}).get('net_pl', np.nan))}
  - Expectancy: ${_fmt(g('dow_metrics').get('Monday', {}).get('expectancy', np.nan))}
- **Tuesday:**
  - Win Rate: {_fmt(g('dow_metrics').get('Tuesday', {}).get('win_rate', np.nan), pct=True)}
  - Net P/L: ${_fmt(g('dow_metrics').get('Tuesday', {}).get('net_pl', np.nan))}
  - Expectancy: ${_fmt(g('dow_metrics').get('Tuesday', {}).get('expectancy', np.nan))}
- **Wednesday:**
  - Win Rate: {_fmt(g('dow_metrics').get('Wednesday', {}).get('win_rate', np.nan), pct=True)}
  - Net P/L: ${_fmt(g('dow_metrics').get('Wednesday', {}).get('net_pl', np.nan))}
  - Expectancy: ${_fmt(g('dow_metrics').get('Wednesday', {}).get('expectancy', np.nan))}
- **Thursday:**
  - Win Rate: {_fmt(g('dow_metrics').get('Thursday', {}).get('win_rate', np.nan), pct=True)}
  - Net P/L: ${_fmt(g('dow_metrics').get('Thursday', {}).get('net_pl', np.nan))}
  - Expectancy: ${_fmt(g('dow_metrics').get('Thursday', {}).get('expectancy', np.nan))}
- **Friday:**
  - Win Rate: {_fmt(g('dow_metrics').get('Friday', {}).get('win_rate', np.nan), pct=True)}
  - Net P/L: ${_fmt(g('dow_metrics').get('Friday', {}).get('net_pl', np.nan))}
  - Expectancy: ${_fmt(g('dow_metrics').get('Friday', {}).get('expectancy', np.nan))}
---
## Effectiveness Metrics
### Expectancy by Trade Length
- 0–15 min: ${_fmt(g('expectancy_by_hold').get('0-15', np.nan))}
- 15–60 min: ${_fmt(g('expectancy_by_hold').get('15-60', np.nan))}
- 60–180 min: ${_fmt(g('expectancy_by_hold').get('60-180', np.nan))}
- >180 min: ${_fmt(g('expectancy_by_hold').get('>180', np.nan))}
### Capital Efficiency
- **Net Profit per Trade-Minute of Capital:** ${_fmt(g('capital_efficiency'), 4)} per dollar-minute
### Concentration Risk
- **% of Net Profit from Top 10% of Trades:** {_fmt(g('concentration_risk_pct'), pct=True)}
### Trade Outcome by Day Part
- **Early (09:30–11:30):** ${_fmt(g('day_part_expectancy').get('EARLY', np.nan))}
- **Mid (11:30–14:00):** ${_fmt(g('day_part_expectancy').get('MID', np.nan))}
- **Late (14:00–16:00):** ${_fmt(g('day_part_expectancy').get('LATE', np.nan))}
- **Other:** ${_fmt(g('day_part_expectancy').get('OTHER', np.nan))}
---
## Performance Details — ALL
- **Average Win:** ${_fmt(g('avg_win_dollars'))}
- **Average Loss:** ${_fmt(g('avg_loss_dollars'))}
- **Expectancy per Trade:** ${_fmt(g('expectancy_per_trade_dollars'))}
- **Largest Win:** ${_fmt(g('largest_winning_trade'))}
- **Largest Loss:** ${_fmt(g('largest_losing_trade'))}
- **Recovery Factor:** {_fmt(g('recovery_factor'))}
---
## Charts
![Equity Curve (ALL)](equity_curve_all.png)
![Drawdown Curve (ALL)](drawdown_curve_all.png)
![Trade P/L Distribution (ALL)](pl_histogram_all.png)
![Hold Time Distribution (ALL)](hold_time_histogram_all.png)
*Report generated by DataAnalyzer v{cfg.version}*
"""
    with open(os.path.join(outdir, "analytics.md"), "w", encoding="utf-8") as f:
        f.write(md)

if __name__ == "__main__":
    import argparse, glob, sys
    parser = argparse.ArgumentParser(description="Lean analysis of TOS Strategy Report CSV (trade data only). ALL sessions primary + RTH snapshot.")
    parser.add_argument(
        "--csv",
        nargs="+",
        required=True,
        help="Path(s) or globs for TOS Strategy Report CSV(s). Examples: file.csv StrategyReports/*.csv 'StrategyReports/*.csv'"
    )
    parser.add_argument("--timeframe", type=str, default="180d:15m", help="Timeframe label for outputs (display only).")
    parser.add_argument("--capital", type=float, default=2500.0, help="Initial capital (1-contract float).")
    parser.add_argument("--commission", type=float, default=4.04, help="Commission per contract round trip.")
    parser.add_argument("--point_value", type=float, default=5.0, help="Default dollars per point per contract (used if instrument unknown).")
    args = parser.parse_args()
    # Resolve CSVs
    resolved = []
    for item in args.csv:
        matches = glob.glob(item)
        if matches:
            resolved.extend(matches)
        else:
            resolved.append(item)
    csv_paths = sorted({str(Path(p)) for p in resolved if Path(p).exists()})
    if not csv_paths:
        print(f"[ERROR] No CSV files matched any of: {args.csv}", file=sys.stderr)
        sys.exit(1)
    # global config
    cfg_global = BacktestConfig(
        strategy_name="",  # will be set from CSV
        instruments=("/MES",),
        timeframe=args.timeframe,
        initial_capital=args.capital,
        commission_per_round_trip=args.commission,
        point_value=args.point_value,
        version="1.3.1",
    )
    all_metrics = []
    for csv_path in csv_paths:
        print(f"\n[RUN] CSV: {csv_path}")
        results = run_backtest(csv_path, cfg_global)
        for r in results:
            m = r["metrics"]
            m["csv"] = str(Path(csv_path).name)
            all_metrics.append(m)

# End of Code
