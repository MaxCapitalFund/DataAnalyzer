# -*- coding: utf-8 -*-
# trading_report_analyzer_lean_v1.3.1.py
# Purpose: Lean, investor-friendly analysis of ThinkorSwim Strategy Report CSVs
# Scope: TRADE DATA ONLY (no EMA/VWAP/ATR). Focus on P/L, risk, trade analytics, visuals.
# Session basis: **New York time (ET)**. Metrics computed for **RTH only (09:30–16:00 ET)**.
# Capital float: $2,500 (1 contract). Commission: $4.04 round trip per contract. Point value: $5.00/pt.
# Outputs (per run):
#   Backtests/<YYYY-MM-DD>_<Strategy>_<Timeframe>_<Instrument>_<CSVStem>/
#     - trades_enriched.csv (all trades, with PRE/OPEN/LUNCH/CLOSING tags)
#     - metrics.json (RTH-only metrics; scope="RTH")
#     - monthly_performance.csv (RTH-only aggregation)
#     - equity_curve_180d.png (RTH-only equity)
#     - drawdown_curve.png (RTH-only)
#     - pl_histogram.png (RTH-only)
#     - analytics.md (RTH-only KPIs + notes)
#     - dow_kpis.csv, hold_kpis.csv, session_kpis.csv
#     - heatmap_dow_hour_count.png
#     - max_loss_streak_trades.csv, max_win_streak_trades.csv
#     - top_worst_trades.csv, top_best_trades.csv
#     - config.json
#
# ---- CHANGELOG ----
# v1.3.1 (2025-09-09)
# - Added top_best_trades.csv (top 10 winners by AdjustedNetPL) and reference in analytics.md.
#
# v1.3.0 (2025-09-09)
# - **Stop-loss normalization**: Any trade with NetPL < -100 is capped at -100 and the overage is added back as a positive correction.
#   All metrics/plots now use AdjustedNetPL. New columns: AdjustedNetPL, SLCorrection, SLBreached.
# - **Streaks**: Compute largest losing and winning streaks (using AdjustedNetPL). Export CSVs for both streaks.
# - **Largest win/loss in points (per contract)** added to metrics/one-sheet.
# - **Directional breakdown** extended with wins/losses counts, totals, averages, points, and simple return%.
# - **Top losers**: Save top 10 worst trades (AdjustedNetPL) to CSV and reference in the one-sheet.
# - **Exit method × StrategyBucket (Premarket45/RTH15/Other)** support based on strategy name.
# - **Heatmap** counts computed over AdjustedNetPL index.
# - **Timeframe label** remains dynamic (from CLI arg) and echoed in analytics.md.
# -------------------

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
import matplotlib.pyplot as plt

# =========================
# Configuration
# =========================

@dataclass
class BacktestConfig:
    strategy_name: str = ""  # filled from CSV Strategy base name
    instruments: Tuple[str, ...] = ("MES",)
    timeframe: str = "180d:15m"
    # RTH session in **New York time** (ET)
    session_hours_rth: Tuple[str, str] = ("09:30", "16:00")
    # Capital float = cost to run exactly **1 contract**
    initial_capital: float = 2500.0
    # Round-trip commission per **contract**
    commission_per_round_trip: float = 4.04
    # Default point value (used if instrument unknown)
    point_value: float = 5.0
    version: str = "1.3.1"

    def outdir(self, csv_stem: str, instrument: str, strategy_label: str) -> str:
        day = datetime.now().strftime("%Y-%m-%d")
        safe_strategy = strategy_label.replace(" ", "_")
        safe_instr = (instrument or "UNK").replace("/", "")
        return os.path.join(os.getcwd(), f"Backtests/{day}_{safe_strategy}_{self.timeframe}_{safe_instr}_{csv_stem}")


# =========================
# Helpers
# =========================

def _to_float(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.replace(r"[\$,]", "", regex=True)
    s = s.str.replace(r"\(([^()]*)\)", r"-\1", regex=True)  # (123) -> -123
    s = s.replace("", np.nan)
    return pd.to_numeric(s, errors='coerce')


def _parse_datetime(series: pd.Series) -> pd.Series:
    # Typical TOS: "11/27/24 3:45 AM"
    return pd.to_datetime(series, format="%m/%d/%y %I:%M %p", errors='coerce')


# Session tags per spec
PRE_START,  PRE_END  = time(3, 0),  time(9, 29)
OPEN_START, OPEN_END = time(9, 30), time(11, 30)
LUNCH_START,LUNCH_END= time(11, 30),time(14, 0)
CLOSE_START,CLOSE_END= time(14, 0), time(16, 0)

def _tag_session(dt: pd.Timestamp) -> str:
    if pd.isna(dt):
        return "Unknown"
    t = dt.time()
    if PRE_START <= t <= PRE_END:     return "PRE"
    if OPEN_START <= t <= OPEN_END:   return "OPEN"
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
    return float(drawdown.min())  # negative value

def drawdown_series(equity: pd.Series) -> pd.Series:
    peak = equity.cummax()
    return equity / peak - 1.0

def max_drawdown_dollars(equity: pd.Series) -> float:
    peak = equity.cummax()
    return float((peak - equity).max())

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
    if any(w in s for w in ["STOP", "SL", "STOPPED"]):          return "Stop"
    if any(w in s for w in ["TIME", "TIME EXIT", "TIMED", "TIMEOUT"]): return "Time"
    if any(w in s for w in ["MANUAL", "FLATTEN", "MKT CLOSE", "DISCRETIONARY"]): return "Manual"
    return "Close"

# ---- Segmentation helpers ----
WEEKDAY_ORDER = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
MIN_SAMPLE_PER_SEGMENT = 30  # doc: "Ensure ≥30 trades per segment"
TOP_WORST_N = 10

def _with_dow(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out['DOW'] = pd.to_datetime(out['ExitTime'], errors='coerce').dt.day_name()
    out['DOW'] = pd.Categorical(out['DOW'], categories=WEEKDAY_ORDER, ordered=True)
    return out

def _with_hold_buckets(df: pd.DataFrame) -> pd.DataFrame:
    bins   = [0, 5, 15, 30, 60, 120, np.inf]
    labels = ['<=5m','5–15m','15–30m','30–60m','60–120m','>120m']
    out = df.copy()
    out['HoldBucket'] = pd.cut(out['HoldMins'], bins=bins, labels=labels, right=True, include_lowest=True)
    return out

def _kpi_table(series_pl: pd.Series) -> dict:
    """Return basic KPIs for a slice (sum, count, mean, win rate, profit factor)."""
    s = series_pl.dropna()
    if s.empty:
        return {"count": 0, "sum": 0.0, "mean": np.nan, "win_rate_pct": np.nan, "profit_factor": np.nan}
    return {
        "count": int(s.size),
        "sum": float(s.sum()),
        "mean": float(s.mean()),
        "win_rate_pct": float((s > 0).mean() * 100.0),
        "profit_factor": _profit_factor(s),
    }

def _infer_strategy_bucket(text: str) -> str:
    t = (text or "").upper()
    if "RTH15" in t: return "RTH15"
    if "PRE45" in t or "PRE" in t: return "Premarket45"
    return "Other"

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

    # Keep original column set for diagnostics
    original_cols = set(df.columns)

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

    # Strategy base name (before '(' if any)
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

    # Quantity (signed, preserve sign)
    qty_col = 'Quantity' if 'Quantity' in df.columns else ('Qty' if 'Qty' in df.columns else None)
    if qty_col and qty_col != 'Qty':
        df['Qty'] = pd.to_numeric(df[qty_col], errors='coerce')
    elif 'Qty' not in df.columns:
        df['Qty'] = np.nan

    # Optional symbol/instrument
    if 'Symbol' in df.columns:
        df['Symbol'] = df['Symbol'].astype(str)
    elif 'Instrument' in df.columns:
        df['Symbol'] = df['Instrument'].astype(str)
    else:
        s = df['Strategy'].astype(str) if 'Strategy' in df.columns else pd.Series([], dtype=str)
        pat = re.compile(r"/([A-Z]{2,5})")
        df['Symbol'] = s.str.extract(pat, expand=False)

    # Preserve optional fields if present
    for opt in ['Amount', 'Position']:
        if opt not in df.columns:
            df[opt] = np.nan

    # Warn if key columns missing
    required_any = ['Id','Strategy','Side','Price','Qty','TradePL']
    missing = [c for c in required_any if c not in original_cols]
    if missing:
        print(f"[WARN] Missing expected columns: {missing}. Attempting to proceed.")

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

    OPEN_RX  = r"\b(?:BTO|BUY TO OPEN|BUY_TO_OPEN|BOT TO OPEN|STO|SELL TO OPEN|SELL_TO_OPEN|SELL SHORT|OPEN)\b"
    CLOSE_RX = r"\b(?:STC|SELL TO CLOSE|SELL_TO_CLOSE|SLD TO CLOSE|BTC|BUY TO CLOSE|BUY_TO_CLOSE|CLOSE)\b"

    if id_col:
        for tid, grp in df.groupby(id_col, sort=False):
            g = grp.sort_values('Date').copy()
            side_up = g['Side'].astype(str).str.upper()
            g['is_open']  = side_up.str.contains(OPEN_RX,  regex=True, na=False)
            g['is_close'] = side_up.str.contains(CLOSE_RX, regex=True, na=False)

            entry_rows = g[g['is_open']]
            close_rows = g[g['is_close']]

            if len(entry_rows) and len(close_rows):
                entry = entry_rows.iloc[0]
                after_entry_close = close_rows[close_rows['Date'] >= entry['Date']]
                exit_ = after_entry_close.iloc[0] if len(after_entry_close) else close_rows.iloc[-1]

                entry_qty = _safe_num(entry.get('Qty'))
                qty_abs = abs(entry_qty) if pd.notna(entry_qty) else 1.0

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

                point_value = getattr(globals().get('cfg_global', object()), 'point_value', 5.0)
                denom = point_value * qty_abs if (point_value and qty_abs) else np.nan
                points_per_contract = float(net_pl / denom) if (denom and pd.notna(net_pl)) else np.nan  # will recompute after SL fix

                trades.append({
                    'Id': tid,
                    'EntryTime': entry['Date'],
                    'ExitTime': exit_['Date'],
                    'EntryPrice': _safe_num(entry.get('Price')),
                    'ExitPrice': _safe_num(exit_.get('Price')),
                    'EntryQty': entry_qty,
                    'ExitQty': _safe_num(exit_.get('Qty')),
                    'QtyAbs': qty_abs,
                    'TradePL': _safe_num(exit_.get('TradePL')),
                    'GrossPL': _safe_num(exit_.get('TradePL')),
                    'Commission': commission,
                    'NetPL': net_pl,
                    'PointsPerContract': points_per_contract,
                    'GrossAmount': trade_pl if pd.notna(trade_pl) else np.nan,
                    'NetAmount': net_pl,
                    'AmountExit': _safe_num(exit_.get('Amount')),
                    'PositionAfterExit': _safe_num(exit_.get('Position')),
                    'BaseStrategy': entry.get('BaseStrategy', 'Unknown'),
                    'StrategyRaw': entry.get('Strategy', ''),
                    'Symbol': entry.get('Symbol', ''),
                    'EntrySide': str(entry.get('Side', '')),
                    'ExitSide':  str(exit_.get('Side', '')),
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
            point_value = getattr(globals().get('cfg_global', object()), 'point_value', 5.0)
            denom = point_value * qty_abs if (point_value and qty_abs) else np.nan
            points_per_contract = float(net_pl / denom) if (denom and pd.notna(net_pl)) else np.nan
            trades.append({
                'Id': row.get('Id', np.nan),
                'EntryTime': pd.NaT,
                'ExitTime': row['Date'],
                'EntryPrice': np.nan,
                'ExitPrice': _safe_num(row.get('Price')),
                'EntryQty': entry_qty,
                'ExitQty': _safe_num(row.get('Qty')),
                'QtyAbs': qty_abs,
                'TradePL': _safe_num(row.get('TradePL')),
                'GrossPL': _safe_num(row.get('TradePL')),
                'Commission': commission,
                'NetPL': net_pl,
                'PointsPerContract': points_per_contract,
                'GrossAmount': trade_pl if pd.notna(trade_pl) else np.nan,
                'NetAmount': net_pl,
                'AmountExit': _safe_num(row.get('Amount')),
                'PositionAfterExit': _safe_num(row.get('Position')),
                'BaseStrategy': row.get('BaseStrategy', 'Unknown'),
                'StrategyRaw': row.get('Strategy', ''),
                'Symbol': row.get('Symbol', ''),
                'EntrySide': str(row.get('Side', '')),
                'ExitSide':  str(row.get('Side', '')),
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
# Stop-loss correction & streaks
# =========================

def apply_stoploss_corrections(trades: pd.DataFrame, point_value: float) -> pd.DataFrame:
    """
    If NetPL < -100, cap to -100 and add overage back as positive correction.
    All per-trade metrics downstream should use AdjustedNetPL.
    """
    df = trades.copy()
    df['SLBreached'] = df['NetPL'] < -100.0
    df['SLCorrection'] = np.where(df['SLBreached'], (-df['NetPL']) - 100.0, 0.0)
    df['AdjustedNetPL'] = np.where(df['SLBreached'], -100.0, df['NetPL'])

    # Recompute points per contract on AdjustedNetPL
    qty_abs = pd.to_numeric(df['QtyAbs'], errors='coerce').replace(0, np.nan)
    df['PointsPerContract'] = df['AdjustedNetPL'] / (point_value * qty_abs)

    return df

def _streak_lengths(sign_series: pd.Series) -> Tuple[int, int, Tuple[int,int], Tuple[int,int]]:
    """
    sign_series: boolean Series (True=win, False=loss)
    Returns:
      max_win_len, max_loss_len, (win_start_index, win_end_index), (loss_start_index, loss_end_index)
    """
    max_win = max_loss = 0
    max_win_range = (None, None)
    max_loss_range = (None, None)

    curr = 0
    curr_start = 0
    curr_type = None  # 'win' or 'loss'

    for i, is_win in enumerate(sign_series):
        t = 'win' if is_win else 'loss'
        if curr_type is None or t != curr_type:
            curr_type = t
            curr = 1
            curr_start = i
        else:
            curr += 1

        if t == 'win' and curr > max_win:
            max_win = curr
            max_win_range = (curr_start, i)
        if t == 'loss' and curr > max_loss:
            max_loss = curr
            max_loss_range = (curr_start, i)

    return max_win, max_loss, max_win_range, max_loss_range


# =========================
# Metrics (RTH only)
# =========================

def _safe_days(first_dt, last_dt):
    if pd.isna(first_dt) or pd.isna(last_dt):
        return np.nan
    return max((last_dt - first_dt).days, 1)


def compute_metrics(trades_rth: pd.DataFrame, cfg: BacktestConfig, scope_label: str = "RTH") -> dict:
    df = trades_rth.copy()

    if 'AdjustedNetPL' not in df.columns:
        raise RuntimeError("AdjustedNetPL missing; call apply_stoploss_corrections() before compute_metrics().")

    # --- Gross vs Net P/L series ---
    pl_net = df['AdjustedNetPL'].fillna(0.0)
    pl_gross = df['GrossPL'].fillna(0.0) if 'GrossPL' in df.columns else pl_net.copy()

    # Equity from NET P/L (Adjusted)
    equity = cfg.initial_capital + pl_net.cumsum()

    # Totals & returns
    total_net = float(pl_net.sum())
    total_gross = float(pl_gross.sum())
    total_return_pct = (total_net / cfg.initial_capital) * 100.0 if cfg.initial_capital else np.nan
    total_return_pct_gross = (total_gross / cfg.initial_capital) * 100.0 if cfg.initial_capital else np.nan

    # Win/loss (Adjusted)
    win_mask = pl_net > 0
    loss_mask = pl_net < 0
    avg_win = float(pl_net[win_mask].mean()) if win_mask.any() else np.nan
    avg_loss = float(pl_net[loss_mask].mean()) if loss_mask.any() else np.nan

    # Points per trade (Adjusted, per contract)
    qty_abs = pd.to_numeric(df['QtyAbs'], errors='coerce').replace(0, np.nan)
    pts_per_trade = pl_net / (cfg.point_value * qty_abs)
    avg_win_pts  = float(pts_per_trade[win_mask].mean()) if win_mask.any() else np.nan
    avg_loss_pts = float(pts_per_trade[loss_mask].mean()) if loss_mask.any() else np.nan

    # Drawdowns (Adjusted net equity)
    max_dd_pct = abs(_max_drawdown(equity)) * 100.0
    dd_series = drawdown_series(equity)
    avg_dd_pct = float(dd_series.mean() * 100.0) if len(dd_series) else np.nan
    max_dd_dollars = max_drawdown_dollars(equity)
    recovery_factor = float(total_net / max_dd_dollars) if max_dd_dollars else np.nan

    # Expectancy
    expectancy_dollars = float(pl_net.mean()) if len(pl_net) else np.nan
    expectancy_dollars_gross = float(pl_gross.mean()) if len(pl_gross) else np.nan

    # Risk-adjusted (per-trade; then annualize using trades-per-year)
    trade_rets = pl_net / cfg.initial_capital if cfg.initial_capital else pd.Series(np.nan, index=pl_net.index)
    per_trade_sharpe = float(trade_rets.mean() / trade_rets.std(ddof=1)) if trade_rets.std(ddof=1) > 0 else np.nan

    downside = trade_rets.copy()
    downside[downside > 0] = 0
    down_stdev = downside.std(ddof=1)
    per_trade_sortino = float(trade_rets.mean() / abs(down_stdev)) if down_stdev and down_stdev > 0 else np.nan

    first_dt = pd.to_datetime(df['ExitTime']).min()
    last_dt  = pd.to_datetime(df['ExitTime']).max()
    days = _safe_days(first_dt, last_dt)
    trades_per_year = (len(df) / days * 252.0) if days and days > 0 else np.nan

    sharpe_annualized = float(np.sqrt(trades_per_year) * per_trade_sharpe) if trades_per_year and per_trade_sharpe == per_trade_sharpe else np.nan
    sortino_annualized = float(np.sqrt(trades_per_year) * per_trade_sortino) if trades_per_year and per_trade_sortino == per_trade_sortino else np.nan

    largest_win = float(pl_net.max()) if len(pl_net) else np.nan
    largest_loss = float(pl_net.min()) if len(pl_net) else np.nan
    largest_win_pts = float(pts_per_trade.max()) if len(pts_per_trade) else np.nan
    largest_loss_pts = float(pts_per_trade.min()) if len(pts_per_trade) else np.nan
    vol_of_trade_returns = float(trade_rets.std(ddof=1)) if len(trade_rets) > 1 else np.nan

    # Monthly & CAGR (Adjusted net equity)
    dt = pd.to_datetime(df['ExitTime'], errors='coerce')
    eq_df = pd.DataFrame({'dt': dt, 'equity': equity})
    eq_month = eq_df.dropna(subset=['dt']).set_index('dt').resample('ME').last()
    monthly_ret = eq_month['equity'].pct_change()
    avg_monthly_return = float(monthly_ret.mean()) if monthly_ret.notna().any() else np.nan

    ending_equity = float(equity.iloc[-1]) if len(equity) else cfg.initial_capital
    CAGR = np.nan
    if days and days > 0 and cfg.initial_capital and cfg.initial_capital > 0:
        ratio = ending_equity / cfg.initial_capital
        if ratio > 0:
            CAGR = float(np.power(ratio, 365.0 / days) - 1.0)

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
        "total_return_pct_gross": total_return_pct_gross,
        "avg_monthly_return": avg_monthly_return,
        "CAGR": CAGR,
        "profit_factor": _profit_factor(pl_net),
        "profit_factor_gross": _profit_factor(pl_gross),
        "win_rate_pct": float((pl_net > 0).mean() * 100.0),
        "avg_win_dollars": avg_win,
        "avg_loss_dollars": avg_loss,
        "avg_win_points_per_contract": avg_win_pts,
        "avg_loss_points_per_contract": avg_loss_pts,
        "largest_winning_trade_points_per_contract": largest_win_pts,
        "largest_losing_trade_points_per_contract": largest_loss_pts,
        "expectancy_per_trade_dollars": expectancy_dollars,
        "expectancy_per_trade_dollars_gross": expectancy_dollars_gross,
        "max_drawdown_pct": max_dd_pct,
        "max_drawdown_dollars": max_dd_dollars,
        "avg_drawdown_pct": avg_dd_pct,
        "recovery_factor": recovery_factor,
        "per_trade_sharpe_proxy": per_trade_sharpe,
        "per_trade_sortino_proxy": per_trade_sortino,
        "sharpe_annualized": sharpe_annualized,
        "sortino_annualized": sortino_annualized,
        "largest_winning_trade": largest_win,
        "largest_losing_trade": largest_loss,
        "vol_of_trade_returns": vol_of_trade_returns,
    }

    try:
        metrics["avg_win_over_avg_loss"] = float(avg_win / abs(avg_loss)) if (isinstance(avg_loss, float) and avg_loss < 0) else np.nan
    except Exception:
        metrics["avg_win_over_avg_loss"] = np.nan

    try:
        metrics["largest_win_over_largest_loss"] = float(largest_win / abs(largest_loss)) if (isinstance(largest_loss, float) and largest_loss < 0) else np.nan
    except Exception:
        metrics["largest_win_over_largest_loss"] = np.nan

    # Direction splits (RTH)
    if 'Direction' in df.columns:
        for dir_label in ['Long', 'Short']:
            mask = (df['Direction'] == dir_label)
            if mask.any():
                pl_dir = pl_net[mask]
                pts_dir = pts_per_trade[mask]
                wins = pl_dir[pl_dir > 0]
                losses = pl_dir[pl_dir < 0]
                metrics.update({
                    f"{dir_label.lower()}_trades": int(mask.sum()),
                    f"{dir_label.lower()}_win_rate_pct": float((pl_dir > 0).mean() * 100.0),
                    f"{dir_label.lower()}_avg_netpl": float(pl_dir.mean()),
                    f"{dir_label.lower()}_profit_factor": _profit_factor(pl_dir),
                    f"{dir_label.lower()}_wins_count": int(wins.size),
                    f"{dir_label.lower()}_losses_count": int(losses.size),
                    f"{dir_label.lower()}_total_win_dollars": float(wins.sum()) if wins.size else 0.0,
                    f"{dir_label.lower()}_total_loss_dollars": float(losses.sum()) if losses.size else 0.0,
                    f"{dir_label.lower()}_avg_win_dollars": float(wins.mean()) if wins.size else np.nan,
                    f"{dir_label.lower()}_avg_loss_dollars": float(losses.mean()) if losses.size else np.nan,
                    f"{dir_label.lower()}_largest_win_points_per_contract": float(pts_dir.max()) if len(pts_dir) else np.nan,
                    f"{dir_label.lower()}_largest_loss_points_per_contract": float(pts_dir.min()) if len(pts_dir) else np.nan,
                    f"{dir_label.lower()}_return_pct": float(pl_dir.sum() / cfg.initial_capital * 100.0) if cfg.initial_capital else np.nan,
                })

    # Exit method breakdown (Adjusted)
    if 'ExitReason' in df.columns:
        reason_counts = df['ExitReason'].value_counts(dropna=False).to_dict()
        reason_avg_adj = df.groupby('ExitReason')['AdjustedNetPL'].mean().to_dict()
        reason_pf = {r: _profit_factor(df.loc[df['ExitReason'] == r, 'AdjustedNetPL'])
                     for r in df['ExitReason'].dropna().unique()}
        metrics.update({
            "exit_reason_counts": reason_counts,
            "exit_reason_avg_netpl": reason_avg_adj,
            "exit_reason_profit_factor": reason_pf,
        })

    # Optional: Exit method × StrategyBucket
    if 'StrategyBucket' in df.columns and 'ExitReason' in df.columns:
        xb = (df.groupby(['StrategyBucket', 'ExitReason'])['AdjustedNetPL']
              .agg(['count','mean','sum']).reset_index())
        metrics['exit_reason_by_strategy_bucket'] = {}
        for _, row in xb.iterrows():
            sb = row['StrategyBucket']; er = row['ExitReason']
            d = metrics['exit_reason_by_strategy_bucket'].setdefault(sb, {})
            d[er] = {"count": int(row['count']), "avg_adj_netpl": float(row['mean']), "sum_adj_netpl": float(row['sum'])}

    # SL correction summary
    if 'SLCorrection' in df.columns and 'SLBreached' in df.columns:
        metrics['sl_total_corrections'] = float(df['SLCorrection'].sum())
        metrics['sl_num_breaches'] = int(df['SLBreached'].sum())

    # Streaks (Adjusted)
    sign = pl_net > 0
    max_win, max_loss, win_rng, loss_rng = _streak_lengths(sign)
    metrics.update({
        "max_win_streak": int(max_win),
        "max_loss_streak": int(max_loss),
        "max_win_streak_range": [int(win_rng[0]) if win_rng[0] is not None else -1,
                                  int(win_rng[1]) if win_rng[1] is not None else -1],
        "max_loss_streak_range": [int(loss_rng[0]) if loss_rng[0] is not None else -1,
                                   int(loss_rng[1]) if loss_rng[1] is not None else -1],
    })

    return metrics


# =========================
# Visuals & Tables (RTH-only)
# =========================

def save_visuals_and_tables(trades_rth: pd.DataFrame, cfg: BacktestConfig, outdir: str) -> None:
    os.makedirs(outdir, exist_ok=True)
    pl = trades_rth['AdjustedNetPL'].fillna(0.0)
    equity = cfg.initial_capital + pl.cumsum()

    # Equity (last 180 calendar days)
    eq_df_daily = pd.DataFrame({
        'dt': pd.to_datetime(trades_rth['ExitTime'], errors='coerce'),
        'equity': equity
    }).dropna(subset=['dt']).set_index('dt').resample('D').last()
    if len(eq_df_daily) > 0:
        eq_last_180 = eq_df_daily.tail(180)
        plt.figure(figsize=(9, 4.5))
        plt.plot(eq_last_180.index, eq_last_180['equity'].values)
        plt.ylabel("Equity ($)")
        plt.xlabel("Date")
        plt.title(f"Equity Curve (last 180 days) — {cfg.strategy_name} ({cfg.timeframe}) [RTH]")
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, "equity_curve_180d.png"), dpi=160)
        plt.close()
    else:
        plt.figure(figsize=(9, 4.5))
        plt.plot(equity.index, equity.values)
        plt.ylabel("Equity ($)")
        plt.xlabel("Trade #")
        plt.title(f"Equity Curve — {cfg.strategy_name} ({cfg.timeframe}) [RTH]")
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, "equity_curve_180d.png"), dpi=160)
        plt.close()

    # Drawdown curve (per-trade index)
    dd = drawdown_series(equity)
    plt.figure(figsize=(9, 4.5))
    plt.plot(dd.index, dd.values * 100.0)
    plt.ylabel("Drawdown (%)")
    plt.xlabel("Trade #")
    plt.title("Drawdown Curve [RTH]")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "drawdown_curve.png"), dpi=160)
    plt.close()

    # Histogram of trade P/L
    plt.figure(figsize=(9, 4.5))
    plt.hist(trades_rth['AdjustedNetPL'].dropna().values, bins=30)
    plt.xlabel("Net P/L per Trade ($) — Adjusted")
    plt.ylabel("Count")
    plt.title("Trade P/L Distribution [RTH]")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "pl_histogram.png"), dpi=160)
    plt.close()

    # Monthly performance table (RTH only)
    dt = pd.to_datetime(trades_rth['ExitTime'], errors='coerce')
    monthly = pd.DataFrame({'NetPL': trades_rth['AdjustedNetPL'].values}, index=dt)
    monthly = monthly.dropna().resample('ME').sum()  # month-end
    monthly['ReturnPct'] = monthly['NetPL'] / cfg.initial_capital * 100.0
    monthly.to_csv(os.path.join(outdir, "monthly_performance.csv"))


def save_segment_tables_and_heatmap(trades_rth: pd.DataFrame, cfg: BacktestConfig, outdir: str) -> None:
    """
    Saves:
      - dow_kpis.csv
      - hold_kpis.csv
      - session_kpis.csv
      - heatmap_dow_hour_count.png (trade distribution)
    Flags low sample sizes (N < MIN_SAMPLE_PER_SEGMENT).
    """
    os.makedirs(outdir, exist_ok=True)
    df = trades_rth.copy()
    if df.empty:
        return
    
    # Day-of-week KPIs
    d_dow = _with_dow(df)
    if 'DOW' in d_dow.columns:
        g = d_dow.groupby('DOW', observed=True)['AdjustedNetPL']
        dow_tbl = pd.DataFrame({
            'count': g.size(),
            'sum': g.sum(min_count=1),
            'mean': g.mean(),
            'win_rate_pct': g.apply(lambda s: (s > 0).mean() * 100.0 if len(s) else np.nan),
            'profit_factor': g.apply(_profit_factor),
        })
        dow_tbl = dow_tbl.reindex(WEEKDAY_ORDER)
        dow_tbl['low_sample_flag'] = dow_tbl['count'].fillna(0) < MIN_SAMPLE_PER_SEGMENT
        dow_tbl.to_csv(os.path.join(outdir, "dow_kpis.csv"))

    # Hold-time buckets KPIs
    d_hold = _with_hold_buckets(df)
    if 'HoldBucket' in d_hold.columns:
        g = d_hold.groupby('HoldBucket', observed=True)['AdjustedNetPL']
        hold_tbl = pd.DataFrame({
            'count': g.size(),
            'sum': g.sum(min_count=1),
            'mean': g.mean(),
            'win_rate_pct': g.apply(lambda s: (s > 0).mean() * 100.0 if len(s) else np.nan),
            'profit_factor': g.apply(_profit_factor),
        })
        hold_tbl = hold_tbl.reindex(d_hold['HoldBucket'].cat.categories)
        hold_tbl['low_sample_flag'] = hold_tbl['count'].fillna(0) < MIN_SAMPLE_PER_SEGMENT
        hold_tbl.to_csv(os.path.join(outdir, "hold_kpis.csv"))

    # Session KPIs
    if 'Session' in df.columns:
        g = df.groupby('Session')['AdjustedNetPL']
        sess_tbl = pd.DataFrame({
            'count': g.size(),
            'sum': g.sum(min_count=1),
            'mean': g.mean(),
            'win_rate_pct': g.apply(lambda s: (s > 0).mean() * 100.0 if len(s) else np.nan),
            'profit_factor': g.apply(_profit_factor),
        })
        sess_tbl['low_sample_flag'] = sess_tbl['count'].fillna(0) < MIN_SAMPLE_PER_SEGMENT
        sess_tbl.to_csv(os.path.join(outdir, "session_kpis.csv"))

    # Heatmap: trade count by (DOW x ExitHour)
    dt = pd.to_datetime(df['ExitTime'], errors='coerce')
    df['ExitHour'] = dt.dt.hour
    d_heat = _with_dow(df.dropna(subset=['ExitHour']))
    if not d_heat.empty:
        pivot_counts = d_heat.pivot_table(index='DOW', columns='ExitHour',
                                          values='AdjustedNetPL', aggfunc='size', fill_value=0)
        pivot_counts = pivot_counts.reindex(WEEKDAY_ORDER)
        plt.figure(figsize=(10, 4.5))
        plt.imshow(pivot_counts.values, aspect='auto', interpolation='nearest')
        plt.colorbar(label='Trade Count')
        plt.yticks(range(len(pivot_counts.index)), list(pivot_counts.index))
        plt.xticks(range(len(pivot_counts.columns)), list(pivot_counts.columns), rotation=0)
        plt.xlabel("Hour of Day (ET)")
        plt.title("Trade Distribution Heatmap (Count) — DOW × Hour [RTH]")
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, "heatmap_dow_hour_count.png"), dpi=160)
        plt.close()


def _fmt(x, p=2, pct=False):
    try:
        if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
            return "n/a"
        return (f"{x:.{p}f}%" if pct else f"{x:.{p}f}")
    except Exception:
        return str(x)


def generate_analytics_md(trades_all: pd.DataFrame, trades_rth: pd.DataFrame, metrics: dict, cfg: BacktestConfig, outdir: str) -> None:
    os.makedirs(outdir, exist_ok=True)
    m = metrics

    def g(key, default=np.nan):
        return m.get(key, default)

    # Monthly preview (RTH)
    monthly_path = os.path.join(outdir, "monthly_performance.csv")
    monthly_preview = ""
    if os.path.exists(monthly_path):
        try:
            dfm = pd.read_csv(monthly_path)
            if 'Unnamed: 0' in dfm.columns:
                dfm.rename(columns={'Unnamed: 0': 'Month'}, inplace=True)
            if 'Month' not in dfm.columns:
                dfm.insert(0, 'Month', dfm.iloc[:,0])
            dfm = dfm[['Month','NetPL','ReturnPct']].tail(6)
            lines = ["| Month | NetPL ($) | Return (%) |", "|---|---:|---:|"]
            for _, r in dfm.iterrows():
                lines.append(f"| {r['Month']} | {_fmt(r['NetPL'])} | {_fmt(r['ReturnPct'], p=2)} |")
            monthly_preview = "\n".join(lines)
        except Exception:
            monthly_preview = "(Monthly table could not be parsed.)"

    md = f"""
# Strategy One-Sheet (Trade Data)

**Strategy:** {g('strategy_name')}  
**Instrument:** {(trades_all['Symbol'].dropna().iloc[0] if 'Symbol' in trades_all.columns and len(trades_all.dropna(subset=['Symbol'])) else 'Unknown')}  
**Timeframe:** {g('timeframe')}  
**Run Date:** {datetime.now().strftime('%Y-%m-%d')}  
**Session Basis:** New York time (ET). **Metrics Scope:** {g('scope', 'RTH')} (09:30–16:00)  
**Initial Capital (float):** ${_fmt(g('initial_capital'), 0)}  
**Commission (RT / contract):** ${_fmt(cfg.commission_per_round_trip, 2)}  
**Point Value:** ${_fmt(g('point_value'), 2)} per point per contract

> **Note:** Stop-loss normalization is enabled: losses below **−$100** are capped at −$100 and the overage is added back as a positive correction (TOS discrepancy fix). All metrics use **AdjustedNetPL**.

---

## Key Performance Indicators (KPI's)
- **Net Profit (after commissions, adjusted):** ${_fmt(g('net_profit'))}
- **Gross Profit (before commissions):** ${_fmt(g('gross_profit'))}
- **Total Return (Net, adjusted):** {_fmt(g('total_return_pct'), pct=True)}
- **Total Return (Gross):** {_fmt(g('total_return_pct_gross'), pct=True)}
- **Win Rate:** {_fmt(g('win_rate_pct'), pct=True)}
- **Profit Factor (Adjusted):** {_fmt(g('profit_factor'))}
- **Max Drawdown:** ${_fmt(g('max_drawdown_dollars'))} ({_fmt(g('max_drawdown_pct'), pct=True)})
- **CAGR:** {_fmt(g('CAGR')*100, pct=True)}
- **Sharpe (annualized):** {_fmt(g('sharpe_annualized'))}
- **Sortino (annualized):** {_fmt(g('sortino_annualized'))}

---

## Performance (returns & consistency)
| Metric | Result | Definition / Formula |
|---|---:|---|
| Net Profit ($, Adjusted) | ${_fmt(g('net_profit'))} | Σ(AdjustedNetPLᵢ) |
| Gross Profit ($) | ${_fmt(g('gross_profit'))} | Σ(GrossPLᵢ) |
| Total Return (%) | {_fmt(g('total_return_pct'), pct=True)} | (Net Profit ÷ Initial Capital) × 100 |
| Total Return (Gross %) | {_fmt(g('total_return_pct_gross'), pct=True)} | (Gross Profit ÷ Initial Capital) × 100 |
| Average Monthly Return | {_fmt(g('avg_monthly_return')*100, pct=True)} | mean(Monthly equity % change) |
| CAGR | {_fmt(g('CAGR')*100, pct=True)} | (Ending Equity ÷ Initial Capital)^(365 ÷ Days) − 1 |
| Profit Factor (Adjusted) | {_fmt(g('profit_factor'))} | (Σ profits) ÷ |Σ losses| |
| Win Rate (%) | {_fmt(g('win_rate_pct'), pct=True)} | (# wins ÷ total trades) × 100 |
| Avg Win ($, Adjusted) | ${_fmt(g('avg_win_dollars'))} | mean(AdjustedNetPL | >0) |
| Avg Loss ($, Adjusted) | ${_fmt(g('avg_loss_dollars'))} | mean(AdjustedNetPL | <0) |
| **Avg Win ÷ Avg Loss** | {_fmt(g('avg_win_over_avg_loss'))} | |Avg Win| ÷ |Avg Loss| |
| **Largest Win ÷ Largest Loss** | {_fmt(g('largest_win_over_largest_loss'))} | |Largest Win| ÷ |Largest Loss| |
| Avg Win (pts / contract) | {_fmt(g('avg_win_points_per_contract'))} | AdjustedNetPL ÷ (point_value × |Qty|) |
| Avg Loss (pts / contract) | {_fmt(g('avg_loss_points_per_contract'))} | AdjustedNetPL ÷ (point_value × |Qty|) |
| Largest Win (pts/contract) | {_fmt(g('largest_winning_trade_points_per_contract'))} | max over trades |
| Largest Loss (pts/contract) | {_fmt(g('largest_losing_trade_points_per_contract'))} | min over trades |
| Expectancy per Trade ($, Adjusted) | ${_fmt(g('expectancy_per_trade_dollars'))} | mean(AdjustedNetPLᵢ) |

---

## Risk (drawdowns & risk-adjusted)
| Metric | Result | Definition / Formula |
|---|---:|---|
| Max Drawdown ($) | ${_fmt(g('max_drawdown_dollars'))} | max(peak − Equity) |
| Max Drawdown (%) | {_fmt(g('max_drawdown_pct'), pct=True)} | min(Equity ÷ peak − 1) × 100 |
| Average Drawdown (%) | {_fmt(g('avg_drawdown_pct'), pct=True)} | mean(Drawdownₜ) × 100 |
| Recovery Factor | {_fmt(g('recovery_factor'))} | Net Profit ÷ Max DD ($) |
| Sharpe (per-trade proxy) | {_fmt(g('per_trade_sharpe_proxy'))} | mean(rᵢ) ÷ stdev(rᵢ) |
| Sortino (per-trade proxy) | {_fmt(g('per_trade_sortino_proxy'))} | mean(rᵢ) ÷ stdev(min(rᵢ,0)) |
| Largest Winning Trade ($) | ${_fmt(g('largest_winning_trade'))} | max(AdjustedNetPLᵢ) |
| Largest Losing Trade ($) | ${_fmt(g('largest_losing_trade'))} | min(AdjustedNetPLᵢ) |
| Volatility of Trade Returns | {_fmt(g('vol_of_trade_returns'))} | stdev(per-trade returns) |

## Exit Method Breakdown (RTH)
| Exit | Trades | Avg NetPL ($, Adj.) | Profit Factor (Adj.) |
|---|---:|---:|---:|
"""
    if 'exit_reason_counts' in m:
        for r, n in m['exit_reason_counts'].items():
            avg = m.get('exit_reason_avg_netpl', {}).get(r, np.nan)
            pf  = m.get('exit_reason_profit_factor', {}).get(r, np.nan)
            md += f"| {r} | {int(n)} | {_fmt(avg)} | {_fmt(pf)} |\n"

    # Strategy bucket split (if present)
    if 'exit_reason_by_strategy_bucket' in m:
        md += "\n### Exit Method × Strategy Bucket\n| Bucket | Exit | Trades | Avg Adj. NetPL | Sum Adj. NetPL |\n|---|---|---:|---:|---:|\n"
        for bucket, d in m['exit_reason_by_strategy_bucket'].items():
            for exit_, vals in d.items():
                md += f"| {bucket} | {exit_} | {vals['count']} | {_fmt(vals['avg_adj_netpl'])} | {_fmt(vals['sum_adj_netpl'])} |\n"

    md += f"""
---

## Long vs Short Breakdown (RTH)
| Direction | Trades | Win Rate | Avg NetPL ($) | Profit Factor |
|---|---:|---:|---:|---:|
| Long | {int(g('long_trades', 0))} | {_fmt(g('long_win_rate_pct'), pct=True)} | {_fmt(g('long_avg_netpl'))} | {_fmt(g('long_profit_factor'))} |
| Short | {int(g('short_trades', 0))} | {_fmt(g('short_win_rate_pct'), pct=True)} | {_fmt(g('short_avg_netpl'))} | {_fmt(g('short_profit_factor'))} |

### Extra (by direction)
**Longs:** wins={int(g('long_wins_count',0))}, losses={int(g('long_losses_count',0))}, total win=${_fmt(g('long_total_win_dollars',0))}, total loss=${_fmt(g('long_total_loss_dollars',0))}, avg win=${_fmt(g('long_avg_win_dollars'))}, avg loss=${_fmt(g('long_avg_loss_dollars'))}, largest win (pts/contract)={_fmt(g('long_largest_win_points_per_contract'))}, largest loss (pts/contract)={_fmt(g('long_largest_loss_points_per_contract'))}, return={_fmt(g('long_return_pct'), pct=True)}

**Shorts:** wins={int(g('short_wins_count',0))}, losses={int(g('short_losses_count',0))}, total win=${_fmt(g('short_total_win_dollars',0))}, total loss=${_fmt(g('short_total_loss_dollars',0))}, avg win=${_fmt(g('short_avg_win_dollars'))}, avg loss=${_fmt(g('short_avg_loss_dollars'))}, largest win (pts/contract)={_fmt(g('short_largest_win_points_per_contract'))}, largest loss (pts/contract)={_fmt(g('short_largest_loss_points_per_contract'))}, return={_fmt(g('short_return_pct'), pct=True)}

---

## Streaks (Adjusted)
- **Max Winning Streak:** {int(g('max_win_streak',0))} trades (index range {g('max_win_streak_range',[None,None])})
- **Max Losing Streak:** {int(g('max_loss_streak',0))} trades (index range {g('max_loss_streak_range',[None,None])})
- See files: `max_win_streak_trades.csv`, `max_loss_streak_trades.csv`.

---

## Visuals & Tables (investor-friendly)
- **Equity Curve (last 180 days):** `equity_curve_180d.png`
- **Drawdown Curve:** `drawdown_curve.png`
- **Trade P/L Histogram:** `pl_histogram.png`
- **Monthly Performance Table:** `monthly_performance.csv`
- **DOW KPIs:** `dow_kpis.csv` (flags rows with low sample size < {MIN_SAMPLE_PER_SEGMENT})
- **Hold-Time KPIs:** `hold_kpis.csv` (flags rows with low sample size < {MIN_SAMPLE_PER_SEGMENT})
- **Session KPIs:** `session_kpis.csv` (flags rows with low sample size < {MIN_SAMPLE_PER_SEGMENT})
- **Trade Distribution Heatmap (Count):** `heatmap_dow_hour_count.png`
- **Top Worst Trades:** `top_worst_trades.csv` (N={TOP_WORST_N})
- **Top Best Trades:** `top_best_trades.csv` (N={TOP_WORST_N})

### Monthly Performance Preview (last 6)
{monthly_preview}
"""
    with open(os.path.join(outdir, "analytics.md"), "w", encoding="utf-8") as f:
        f.write(md)


# =========================
# Runner (per-instrument split)
# =========================

def run_backtest_for_instrument(df_raw: pd.DataFrame, instrument: Optional[str], cfg: BacktestConfig, csv_stem: str):
    # Strategy label pulled from CSV base strategy
    strategy_label = df_raw['BaseStrategy'].dropna().iloc[0] if 'BaseStrategy' in df_raw.columns and len(df_raw.dropna(subset=['BaseStrategy'])) else (cfg.strategy_name or 'Unknown')

    # Set instrument label and point value
    instr = (instrument or '/UNK')
    cfg.strategy_name = strategy_label

    # Point value mapping (extend as needed)
    pv = cfg.point_value
    if instr.upper() in {'/MES', 'MES'}:
        pv = 5.0
    elif instr.upper() in {'/MNQ', 'MNQ'}:
        pv = 2.0
    elif instr.upper() in {'/MYM', 'MYM'}:
        pv = 0.5  # example mapping
    cfg.point_value = pv

    outdir = cfg.outdir(csv_stem, instr, strategy_label)
    os.makedirs(outdir, exist_ok=True)

    # Build trades (all trades first)
    trades_all = build_trades(df_raw, cfg.commission_per_round_trip)

    # Strategy bucket from strategy name
    bucket = _infer_strategy_bucket(strategy_label)
    trades_all['StrategyBucket'] = bucket

    # Session tags
    trades_all['Session'] = trades_all['EntryTime'].apply(_tag_session)

    # Apply stop-loss normalization & recompute points
    trades_all = apply_stoploss_corrections(trades_all, cfg.point_value)

    # Save enriched trades (all)
    trades_out = os.path.join(outdir, "trades_enriched.csv")
    trades_all.to_csv(trades_out, index=False)

    # RTH subset (EntryTime within 09:30–16:00 ET), fall back to ExitTime if EntryTime is NaT
    def _rth_row(row):
        t = row['EntryTime']
        if pd.isna(t):
            t = row.get('ExitTime')
        return _in_rth(t)

    trades_rth = trades_all[trades_all.apply(_rth_row, axis=1)].copy()

    # Metrics (RTH)
    metrics = compute_metrics(trades_rth, cfg, scope_label="RTH")
    metrics["strategy_name"] = strategy_label
    metrics["instrument"] = instr

    # Save metrics
    with open(os.path.join(outdir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # Visuals & tables (RTH)
    save_visuals_and_tables(trades_rth, cfg, outdir)

    # Segment tables & heatmap (RTH)
    save_segment_tables_and_heatmap(trades_rth, cfg, outdir)

    # Streak CSVs (Adjusted, RTH order)
    pl = trades_rth['AdjustedNetPL'].fillna(0.0).reset_index(drop=True)
    sign = pl > 0
    max_win, max_loss, win_rng, loss_rng = _streak_lengths(sign)

    def _slice_to_csv(rng, fname):
        if rng[0] is None or rng[1] is None or rng[0] < 0 or rng[1] < 0:
            pd.DataFrame([]).to_csv(os.path.join(outdir, fname), index=False)
            return
        sub = trades_rth.iloc[rng[0]:rng[1]+1].copy()
        sub.to_csv(os.path.join(outdir, fname), index=False)

    _slice_to_csv(win_rng, "max_win_streak_trades.csv")
    _slice_to_csv(loss_rng, "max_loss_streak_trades.csv")

    # Top worst & top best trades (Adjusted)
    worst = trades_rth.sort_values('AdjustedNetPL').head(TOP_WORST_N)
    worst.to_csv(os.path.join(outdir, "top_worst_trades.csv"), index=False)

    best = trades_rth.sort_values('AdjustedNetPL', ascending=False).head(TOP_WORST_N)
    best.to_csv(os.path.join(outdir, "top_best_trades.csv"), index=False)

    # One-sheet
    generate_analytics_md(trades_all, trades_rth, metrics, cfg, outdir)

    # Save config
    with open(os.path.join(outdir, "config.json"), "w") as f:
        json.dump(asdict(cfg), f, indent=2)

    return trades_all, trades_rth, metrics, outdir


def run_backtest(tos_csv_path: str, cfg: BacktestConfig):
    csv_stem = Path(tos_csv_path).stem.replace(' ', '_')

    raw = load_tos_strategy_report(tos_csv_path)

    # Detect instruments present
    symbols = raw['Symbol'].dropna().unique().tolist() if 'Symbol' in raw.columns else []
    if not symbols:
        s = raw['Strategy'].astype(str) if 'Strategy' in raw.columns else pd.Series([], dtype=str)
        pat = re.compile(r"/([A-Z]{2,5})")
        symbols = s.str.extract(pat, expand=False).dropna().unique().tolist()
    if not symbols:
        symbols = ['/MES']  # fallback default

    results = []
    for instr in symbols:
        trades_all, trades_rth, metrics, outdir = run_backtest_for_instrument(raw, instr, cfg, csv_stem)
        results.append({"instrument": instr, "metrics": metrics, "outdir": outdir})
    return results


# =========================
# CLI
# =========================
if __name__ == "__main__":
    import argparse, glob, sys

    parser = argparse.ArgumentParser(description="Lean analysis of TOS Strategy Report CSV (trade data only). RTH metrics + session tags.")
    parser.add_argument(
        "--csv",
        nargs="+",
        required=True,
        help="Path(s) or globs for TOS Strategy Report CSV(s). Examples: file.csv  StrategyReports/*.csv  'StrategyReports/*.csv'"
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

    # global config (read by build_trades)
    global cfg_global
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
            print(json.dumps(m, indent=2))
            print(f"Saved outputs to: {r['outdir']}")
            all_metrics.append(m)

    # Optional: summary CSV at repo root
    try:
        if all_metrics:
            pd.DataFrame(all_metrics).to_csv("Backtests_summary_metrics.csv", index=False)
            print("\n[OK] Wrote Backtests_summary_metrics.csv")
    except Exception as e:
        print(f"[WARN] Could not write summary metrics: {e}")
