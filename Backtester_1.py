# -*- coding: utf-8 -*-
# trading_report_analyzer_lean_v1.2.py
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
#     - config.json
#
# ---- CHANGELOG ----
# v1.2.0 (2025-09-01)
# - Timeframe default set to 180d:15m.
# - Metrics: added GROSS (pre-commission) alongside NET (post-commission) everywhere.
# - Risk ratios: annualized Sharpe/Sortino (trade-level -> trades-per-year scaler).
# - One-Sheet: Exit Method Breakdown and Long vs Short Breakdown sections added.
# - Preserves signed Qty; uses QtyAbs for commissions/normalization; adds AmountExit & PositionAfterExit.
# - Session buckets: PRE(03:00–09:29), OPEN(09:30–11:30), LUNCH(11:30–14:00), CLOSING(14:00–16:00).
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
    instruments: Tuple[str, ...] = ("/MES",)
    timeframe: str = "180d:15m"
    # RTH session in **New York time** (ET)
    session_hours_rth: Tuple[str, str] = ("09:30", "16:00")
    # Capital float = cost to run exactly **1 contract**
    initial_capital: float = 2500.0
    # Round-trip commission per **contract**
    commission_per_round_trip: float = 4.04
    # Default point value (used if instrument unknown)
    point_value: float = 5.0
    version: str = "1.2.0"

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
    gp = pl[pl > 0].sum()
    gl = -pl[pl < 0].sum()
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
        # Combine if provided separately
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
        # Try to infer from Strategy text, e.g., contains "/MES"
        s = df['Strategy'].astype(str) if 'Strategy' in df.columns else pd.Series([], dtype=str)
        pat = re.compile(r"/(?:[A-Z]{2,5})")
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

                # Derived normalizations
                point_value = getattr(globals().get('cfg_global', object()), 'point_value', 5.0)
                denom = point_value * qty_abs if (point_value and qty_abs) else np.nan
                points_per_contract = float(trade_pl / denom) if (denom and pd.notna(trade_pl)) else np.nan

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
            points_per_contract = float(trade_pl / denom) if (denom and pd.notna(trade_pl)) else np.nan
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
# Metrics (RTH only)
# =========================

def _safe_days(first_dt, last_dt):
    if pd.isna(first_dt) or pd.isna(last_dt):
        return np.nan
    return max((last_dt - first_dt).days, 1)


def compute_metrics(trades_rth: pd.DataFrame, cfg: BacktestConfig, scope_label: str = "RTH") -> dict:
    df = trades_rth.copy()

    # --- Gross vs Net P/L series ---
    pl_net = df['NetPL'].fillna(0.0)
    pl_gross = df['GrossPL'].fillna(0.0) if 'GrossPL' in df.columns else pl_net.copy()

    # Equity from NET P/L
    equity = cfg.initial_capital + pl_net.cumsum()

    # Totals & returns
    total_net = float(pl_net.sum())
    total_gross = float(pl_gross.sum())
    total_return_pct = (total_net / cfg.initial_capital) * 100.0 if cfg.initial_capital else np.nan
    total_return_pct_gross = (total_gross / cfg.initial_capital) * 100.0 if cfg.initial_capital else np.nan

    # Win/loss (Net)
    win_mask = pl_net > 0
    loss_mask = pl_net < 0
    avg_win = float(pl_net[win_mask].mean()) if win_mask.any() else np.nan
    avg_loss = float(pl_net[loss_mask].mean()) if loss_mask.any() else np.nan

    # Win/loss (Gross)
    win_mask_g = pl_gross > 0
    loss_mask_g = pl_gross < 0
    avg_win_gross = float(pl_gross[win_mask_g].mean()) if win_mask_g.any() else np.nan
    avg_loss_gross = float(pl_gross[loss_mask_g].mean()) if loss_mask_g.any() else np.nan

    # Points per trade (Net, per contract)
    qty_abs = pd.to_numeric(df['QtyAbs'], errors='coerce').replace(0, np.nan)
    pts_per_trade = pl_net / (cfg.point_value * qty_abs)
    avg_win_pts  = float(pts_per_trade[win_mask].mean()) if win_mask.any() else np.nan
    avg_loss_pts = float(pts_per_trade[loss_mask].mean()) if loss_mask.any() else np.nan

    # Drawdowns (Net equity)
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
    vol_of_trade_returns = float(trade_rets.std(ddof=1)) if len(trade_rets) > 1 else np.nan

    # Monthly & CAGR (Net equity)
    dt = pd.to_datetime(df['ExitTime'], errors='coerce')
    eq_df = pd.DataFrame({'dt': dt, 'equity': equity})
    eq_month = eq_df.dropna(subset=['dt']).set_index('dt').resample('M').last()
    monthly_ret = eq_month['equity'].pct_change()
    avg_monthly_return = float(monthly_ret.mean()) if monthly_ret.notna().any() else np.nan

    ending_equity = float(equity.iloc[-1]) if len(equity) else cfg.initial_capital
    CAGR = float((ending_equity / cfg.initial_capital) ** (365.0 / days) - 1.0) if days and days > 0 else np.nan

    # Direction counts
    num_longs  = int((df['Direction'] == 'Long').sum()) if 'Direction' in df.columns else 0
    num_shorts = int((df['Direction'] == 'Short').sum()) if 'Direction' in df.columns else 0

    metrics = {
        # identifiers
        "scope": scope_label,
        "strategy_name": cfg.strategy_name,
        "version": cfg.version,
        "timeframe": cfg.timeframe,
        "initial_capital": cfg.initial_capital,
        "point_value": cfg.point_value,

        # counts
        "num_trades": int(len(df)),

        # profitability (Net & Gross)
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
        "avg_win_dollars_gross": avg_win_gross,
        "avg_loss_dollars_gross": avg_loss_gross,
        "avg_win_points_per_contract": avg_win_pts,
        "avg_loss_points_per_contract": avg_loss_pts,
        "expectancy_per_trade_dollars": expectancy_dollars,
        "expectancy_per_trade_dollars_gross": expectancy_dollars_gross,

        # risk
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

        # trade analytics
        "num_longs": num_longs,
        "num_shorts": num_shorts,
        "avg_hold_minutes": float(df['HoldMins'].mean()) if 'HoldMins' in df.columns else np.nan,
    }

    # Direction splits (RTH)
    if 'Direction' in df.columns:
        for dir_label in ['Long', 'Short']:
            mask = (df['Direction'] == dir_label)
            if mask.any():
                pl_dir = pl_net[mask]
                metrics[f"{dir_label.lower()}_trades"] = int(mask.sum())
                metrics[f"{dir_label.lower()}_win_rate_pct"] = float((pl_dir > 0).mean() * 100.0)
                metrics[f"{dir_label.lower()}_avg_netpl"] = float(pl_dir.mean())
                metrics[f"{dir_label.lower()}_profit_factor"] = _profit_factor(pl_dir)

    # Exit method breakdown (RTH)
    if 'ExitReason' in df.columns:
        reason_counts = df['ExitReason'].value_counts(dropna=False).to_dict()
        reason_avg = df.groupby('ExitReason')['NetPL'].mean().to_dict()
        reason_pf = {r: _profit_factor(df.loc[df['ExitReason'] == r, 'NetPL'])
                     for r in df['ExitReason'].dropna().unique()}
        metrics.update({
            "exit_reason_counts": reason_counts,
            "exit_reason_avg_netpl": reason_avg,
            "exit_reason_profit_factor": reason_pf,
        })

    return metrics


# =========================
# Visuals & Tables (RTH-only)
# =========================

def save_visuals_and_tables(trades_rth: pd.DataFrame, cfg: BacktestConfig, outdir: str) -> None:
    os.makedirs(outdir, exist_ok=True)
    pl = trades_rth['NetPL'].fillna(0.0)
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
    plt.hist(trades_rth['NetPL'].dropna().values, bins=30)
    plt.xlabel("Net P/L per Trade ($)")
    plt.ylabel("Count")
    plt.title("Trade P/L Distribution [RTH]")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "pl_histogram.png"), dpi=160)
    plt.close()

    # Monthly performance table (RTH only)
    dt = pd.to_datetime(trades_rth['ExitTime'], errors='coerce')
    monthly = pd.DataFrame({'NetPL': trades_rth['NetPL'].values}, index=dt)
    monthly = monthly.dropna().resample('M').sum()
    monthly['ReturnPct'] = monthly['NetPL'] / cfg.initial_capital * 100.0
    monthly.to_csv(os.path.join(outdir, "monthly_performance.csv"))


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

---

## Key Performance Indicators (KPI's)
- **Net Profit (after commissions):** ${_fmt(g('net_profit'))}
- **Gross Profit (before commissions):** ${_fmt(g('gross_profit'))}
- **Total Return (Net):** {_fmt(g('total_return_pct'), pct=True)}
- **Total Return (Gross):** {_fmt(g('total_return_pct_gross'), pct=True)}
- **Win Rate:** {_fmt(g('win_rate_pct'), pct=True)}
- **Profit Factor (Net):** {_fmt(g('profit_factor'))}
- **Max Drawdown:** ${_fmt(g('max_drawdown_dollars'))} ({_fmt(g('max_drawdown_pct'), pct=True)})
- **CAGR:** {_fmt(g('CAGR')*100, pct=True)}
- **Sharpe (annualized):** {_fmt(g('sharpe_annualized'))}
- **Sortino (annualized):** {_fmt(g('sortino_annualized'))}

---

## Performance (returns & consistency)
| Metric | Result | Definition / Formula |
|---|---:|---|
| Net Profit ($, Net) | ${_fmt(g('net_profit'))} | Σ(NetPLᵢ) |
| Gross Profit ($) | ${_fmt(g('gross_profit'))} | Σ(GrossPLᵢ) |
| Total Return (%) | {_fmt(g('total_return_pct'), pct=True)} | (Net Profit ÷ Initial Capital) × 100 |
| Total Return (Gross %) | {_fmt(g('total_return_pct_gross'), pct=True)} | (Gross Profit ÷ Initial Capital) × 100 |
| Average Monthly Return | {_fmt(g('avg_monthly_return')*100, pct=True)} | mean(Monthly equity % change) |
| CAGR | {_fmt(g('CAGR')*100, pct=True)} | (Ending Equity ÷ Initial Capital)^(365 ÷ Days) − 1 |
| Profit Factor (Net) | {_fmt(g('profit_factor'))} | (Σ profits) ÷ |Σ losses| |
| Profit Factor (Gross) | {_fmt(g('profit_factor_gross'))} | (Σ profits) ÷ |Σ losses| |
| Win Rate (%) | {_fmt(g('win_rate_pct'), pct=True)} | (# wins ÷ total trades) × 100 |
| Avg Win ($, Net) | ${_fmt(g('avg_win_dollars'))} | mean(NetPL | NetPL>0) |
| Avg Loss ($, Net) | ${_fmt(g('avg_loss_dollars'))} | mean(NetPL | NetPL<0) |
| Avg Win ($, Gross) | ${_fmt(g('avg_win_dollars_gross'))} | mean(GrossPL | GrossPL>0) |
| Avg Loss ($, Gross) | ${_fmt(g('avg_loss_dollars_gross'))} | mean(GrossPL | GrossPL<0) |
| Avg Win (pts / contract) | {_fmt(g('avg_win_points_per_contract'))} | NetPL ÷ (point_value × |Qty|) |
| Avg Loss (pts / contract) | {_fmt(g('avg_loss_points_per_contract'))} | NetPL ÷ (point_value × |Qty|) |
| Expectancy per Trade ($, Net) | ${_fmt(g('expectancy_per_trade_dollars'))} | mean(NetPLᵢ) |
| Expectancy per Trade ($, Gross) | ${_fmt(g('expectancy_per_trade_dollars_gross'))} | mean(GrossPLᵢ) |

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
| Largest Winning Trade ($) | ${_fmt(g('largest_winning_trade'))} | max(NetPLᵢ) |
| Largest Losing Trade ($) | ${_fmt(g('largest_losing_trade'))} | min(NetPLᵢ) |
| Volatility of Trade Returns | {_fmt(g('vol_of_trade_returns'))} | stdev(per-trade returns) |

## Exit Method Breakdown (RTH)
| Exit | Trades | Avg NetPL ($) | Profit Factor |
|---|---:|---:|---:|
"""
    if 'exit_reason_counts' in m:
        for r, n in m['exit_reason_counts'].items():
            avg = m.get('exit_reason_avg_netpl', {}).get(r, np.nan)
            pf  = m.get('exit_reason_profit_factor', {}).get(r, np.nan)
            md += f"| {r} | {int(n)} | {_fmt(avg)} | {_fmt(pf)} |\n"

    md += f"""
---

## Long vs Short Breakdown (RTH)
| Direction | Trades | Win Rate | Avg NetPL ($) | Profit Factor |
|---|---:|---:|---:|---:|
| Long | {int(g('long_trades', 0))} | {_fmt(g('long_win_rate_pct'), pct=True)} | {_fmt(g('long_avg_netpl'))} | {_fmt(g('long_profit_factor'))} |
| Short | {int(g('short_trades', 0))} | {_fmt(g('short_win_rate_pct'), pct=True)} | {_fmt(g('short_avg_netpl'))} | {_fmt(g('short_profit_factor'))} |

---

## Trade Analytics (behavior & cadence)
| Metric | Result |
|---|---:|
| Number of Trades (RTH) | {int(g('num_trades', 0))} |
| Long Trades | {int(g('num_longs', 0))} |
| Short Trades | {int(g('num_shorts', 0))} |
| Avg Holding Time (minutes) | {_fmt(g('avg_hold_minutes'))} |
| Session Tags (in trades_enriched.csv) | PRE / OPEN / LUNCH / CLOSING |

---

## Visuals & Tables (investor-friendly)
- **Equity Curve (last 180 days):** `equity_curve_180d.png`
- **Drawdown Curve:** `drawdown_curve.png`
- **Trade P/L Histogram:** `pl_histogram.png`
- **Monthly Performance Table:** `monthly_performance.csv`

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
    # Session tags
    trades_all['Session'] = trades_all['EntryTime'].apply(_tag_session)

    # Save enriched trades (all)
    trades_out = os.path.join(outdir, "trades_enriched.csv")
    trades_all.to_csv(trades_out, index=False)

    # RTH subset (EntryTime within 09:30–16:00 ET)
    trades_rth = trades_all[trades_all['EntryTime'].apply(_in_rth)].copy()

    # Metrics (RTH)
    metrics = compute_metrics(trades_rth, cfg, scope_label="RTH")
    metrics["strategy_name"] = strategy_label
    metrics["instrument"] = instr

    with open(os.path.join(outdir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # Visuals & tables (RTH)
    save_visuals_and_tables(trades_rth, cfg, outdir)

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
        # try to parse from Strategy text
        s = raw['Strategy'].astype(str) if 'Strategy' in raw.columns else pd.Series([], dtype=str)
        pat = re.compile(r"/(?:[A-Z]{2,5})")
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
    parser.add_argument("--timeframe", type=str, default="180d:15m", help="Timeframe label for outputs.")
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
        version="1.2.0",
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
