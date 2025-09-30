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
    instruments: Tuple[str, ...] = ("MES",)
    timeframe: str = "180d:15m"
    session_hours_rth: Tuple[str, str] = ("09:30", "16:00")
    initial_capital: float = 2500.0
    commission_per_round_trip: float = 4.04
    point_value: float = 5.0
    version: str = "1.3.1"

    def outdir(self, csv_stem: str, instrument: str, strategy_label: str) -> str:
        # Use /tmp for Vercel serverless environment
        temp_dir = Path('/tmp')
        day = datetime.now().strftime("%Y-%m-%d")
        timestamp = datetime.now().strftime("%H%M%S_%f")[:-3]  # Include milliseconds for uniqueness
        safe_strategy = strategy_label.replace(" ", "_")
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

    # Symbol/instrument
    if 'Symbol' in df.columns:
        df['Symbol'] = df['Symbol'].astype(str)
    elif 'Instrument' in df.columns:
        df['Symbol'] = df['Instrument'].astype(str)
    else:
        s = df['Strategy'].astype(str) if 'Strategy' in df.columns else pd.Series([], dtype=str)
        pat = re.compile(r"/([A-Z]{2,5})")
        df['Symbol'] = s.str.extract(pat, expand=False)

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
# Stop-loss correction
# =========================

def apply_stoploss_corrections(trades: pd.DataFrame, point_value: float) -> pd.DataFrame:
    """Apply stop-loss normalization for Vercel compatibility"""
    df = trades.copy()
    df['SLBreached'] = df['NetPL'] < -100.0
    df['SLCorrection'] = np.where(df['SLBreached'], (-df['NetPL']) - 100.0, 0.0)
    df['AdjustedNetPL'] = np.where(df['SLBreached'], -100.0, df['NetPL'])

    # Recompute points per contract on AdjustedNetPL
    qty_abs = pd.to_numeric(df['QtyAbs'], errors='coerce').replace(0, np.nan)
    df['PointsPerContract'] = df['AdjustedNetPL'] / (point_value * qty_abs)

    return df

# =========================
# Metrics (RTH only)
# =========================

def compute_metrics(trades_rth: pd.DataFrame, cfg: BacktestConfig, scope_label: str = "RTH") -> dict:
    df = trades_rth.copy()

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

    # Risk-adjusted
    trade_rets = pl_net / cfg.initial_capital if cfg.initial_capital else pd.Series(np.nan, index=pl_net.index)
    per_trade_sharpe = float(trade_rets.mean() / trade_rets.std(ddof=1)) if trade_rets.std(ddof=1) > 0 else np.nan

    first_dt = pd.to_datetime(df['ExitTime']).min()
    last_dt  = pd.to_datetime(df['ExitTime']).max()
    days = max((last_dt - first_dt).days, 1) if pd.notna(first_dt) and pd.notna(last_dt) else 1
    trades_per_year = (len(df) / days * 252.0) if days and days > 0 else np.nan

    sharpe_annualized = float(np.sqrt(trades_per_year) * per_trade_sharpe) if trades_per_year and per_trade_sharpe == per_trade_sharpe else np.nan

    largest_win = float(pl_net.max()) if len(pl_net) else np.nan
    largest_loss = float(pl_net.min()) if len(pl_net) else np.nan

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
    }

    return metrics

# =========================
# Visuals (simplified for Vercel)
# =========================

def save_visuals_and_tables(trades_rth: pd.DataFrame, cfg: BacktestConfig, outdir: str) -> None:
    os.makedirs(outdir, exist_ok=True)
    pl = trades_rth['AdjustedNetPL'].fillna(0.0)
    equity = cfg.initial_capital + pl.cumsum()

    # Equity curve (simplified)
    plt.figure(figsize=(9, 4.5))
    plt.plot(equity.index, equity.values)
    plt.ylabel("Equity ($)")
    plt.xlabel("Trade #")
    plt.title(f"Equity Curve — {cfg.strategy_name} ({cfg.timeframe}) [RTH]")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "equity_curve_180d.png"), dpi=160, bbox_inches='tight')
    plt.close()

    # Drawdown curve
    dd = equity / equity.cummax() - 1.0
    plt.figure(figsize=(9, 4.5))
    plt.plot(dd.index, dd.values * 100.0)
    plt.ylabel("Drawdown (%)")
    plt.xlabel("Trade #")
    plt.title("Drawdown Curve [RTH]")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "drawdown_curve.png"), dpi=160, bbox_inches='tight')
    plt.close()

    # Histogram of trade P/L
    plt.figure(figsize=(9, 4.5))
    plt.hist(trades_rth['AdjustedNetPL'].dropna().values, bins=30)
    plt.xlabel("Net P/L per Trade ($) — Adjusted")
    plt.ylabel("Count")
    plt.title("Trade P/L Distribution [RTH]")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "pl_histogram.png"), dpi=160, bbox_inches='tight')
    plt.close()

    # Monthly performance table
    dt = pd.to_datetime(trades_rth['ExitTime'], errors='coerce')
    monthly = pd.DataFrame({'NetPL': trades_rth['AdjustedNetPL'].values}, index=dt)
    monthly = monthly.dropna().resample('ME').sum()
    monthly['ReturnPct'] = monthly['NetPL'] / cfg.initial_capital * 100.0
    monthly.to_csv(os.path.join(outdir, "monthly_performance.csv"))

# =========================
# Main runner
# =========================

def run_backtest_for_instrument(df_raw: pd.DataFrame, instrument: Optional[str], cfg: BacktestConfig, csv_stem: str):
    # Strategy label pulled from CSV base strategy
    strategy_label = df_raw['BaseStrategy'].dropna().iloc[0] if 'BaseStrategy' in df_raw.columns and len(df_raw.dropna(subset=['BaseStrategy'])) else (cfg.strategy_name or 'Unknown')

    # Set instrument label and point value
    instr = (instrument or '/UNK')
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

    # Build trades (all trades first)
    trades_all = build_trades(df_raw, cfg.commission_per_round_trip)

    # Session tags
    trades_all['Session'] = trades_all['EntryTime'].apply(_tag_session)

    # Apply stop-loss normalization
    trades_all = apply_stoploss_corrections(trades_all, cfg.point_value)

    # Save enriched trades (all)
    trades_out = os.path.join(outdir, "trades_enriched.csv")
    trades_all.to_csv(trades_out, index=False)

    # RTH subset
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

    # Generate analytics markdown
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

def generate_analytics_md(trades_all: pd.DataFrame, trades_rth: pd.DataFrame, metrics: dict, cfg: BacktestConfig, outdir: str) -> None:
    """Generate simplified analytics markdown for Vercel"""
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

    md = f"""
# Strategy Analysis Report

**Strategy:** {g('strategy_name')}  
**Timeframe:** {g('timeframe')}  
**Run Date:** {datetime.now().strftime('%Y-%m-%d')}  
**Session Basis:** New York time (ET). **Metrics Scope:** {g('scope', 'RTH')} (09:30–16:00)  
**Initial Capital:** ${_fmt(g('initial_capital'), 0)}  
**Commission (RT / contract):** ${_fmt(cfg.commission_per_round_trip, 2)}  

---

## Key Performance Indicators
- **Net Profit (adjusted):** ${_fmt(g('net_profit'))}
- **Total Return:** {_fmt(g('total_return_pct'), pct=True)}
- **Win Rate:** {_fmt(g('win_rate_pct'), pct=True)}
- **Profit Factor:** {_fmt(g('profit_factor'))}
- **Max Drawdown:** ${_fmt(g('max_drawdown_dollars'))} ({_fmt(g('max_drawdown_pct'), pct=True)})
- **Sharpe (annualized):** {_fmt(g('sharpe_annualized'))}
- **Total Trades:** {int(g('num_trades', 0))}

---

## Performance Details
- **Average Win:** ${_fmt(g('avg_win_dollars'))}
- **Average Loss:** ${_fmt(g('avg_loss_dollars'))}
- **Expectancy per Trade:** ${_fmt(g('expectancy_per_trade_dollars'))}
- **Largest Win:** ${_fmt(g('largest_winning_trade'))}
- **Largest Loss:** ${_fmt(g('largest_losing_trade'))}
- **Recovery Factor:** {_fmt(g('recovery_factor'))}

---

*Report generated by DataAnalyzer v{cfg.version}*
"""

    with open(os.path.join(outdir, "analytics.md"), "w", encoding="utf-8") as f:
        f.write(md)

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
            all_metrics.append(m)
