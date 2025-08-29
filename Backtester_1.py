# trading_report_analyzer_lean.py
# Purpose: Lean, investor-friendly analysis of ThinkorSwim Strategy Report CSVs
# Scope: TRADE DATA ONLY (no EMA/VWAP/ATR). Focus on P/L, risk, trade analytics, visuals.
# Session basis: **New York time (ET)** RTH 09:30–16:00.
# Capital float: $2,500 (1 contract). Commission: $4.04 round trip per contract. Point value: $5.00/pt.
# Outputs (per run):
#   Backtests/<YYYY-MM-DD>_<Strategy>_<Timeframe>_<CSVStem>/
#     - trades_enriched.csv
#     - metrics.json
#     - monthly_performance.csv
#     - equity_curve_180d.png
#     - drawdown_curve.png
#     - pl_histogram.png
#     - analytics.md
#
# ---- CHANGELOG ----
# v1.0.4 (2025-08-29)
# NEW
# - ExitReason tagging added (Target / Stop / Time / Manual / Close) derived from exit-row text.
# - New columns in trades_enriched.csv: ExitSide, ExitReason.
# - Exit-method analytics added to metrics.json: exit_reason_counts, exit_reason_avg_netpl, exit_reason_profit_factor.
# - "Exit Method Breakdown" section added to analytics.md.
# IMPROVEMENTS
# - Direction inference hardened: BTO => Long, STO => Short (supports BUY_TO_OPEN, SELL_TO_OPEN, SELL SHORT, etc.).
# - Percentage rendering fixed in analytics.md (CAGR & Avg Monthly as percentages).
# - Monthly preview generation fixed (proper newline join) and resilient CSV parsing.
# - Regex boundaries cleaned (uses \b), NA-safe matching, broader open/close variants.
# - CLI output tidy; version bumped to 1.0.4.
# -------------------
# trading_report_analyzer_lean.py
# Purpose: Lean, investor-friendly analysis of ThinkorSwim Strategy Report CSVs
# Scope: TRADE DATA ONLY (no EMA/VWAP/ATR). Focus on P/L, risk, trade analytics, visuals.
# Session basis: **New York time (ET)** RTH 09:30–16:00.
# Capital float: $2,500 (1 contract). Commission: $4.04 round trip per contract. Point value: $5.00/pt.
# Outputs (per run):
#   Backtests/<YYYY-MM-DD>_<Strategy>_<Timeframe>_<CSVStem>/
#     - trades_enriched.csv
#     - metrics.json
#     - monthly_performance.csv
#     - equity_curve_180d.png
#     - drawdown_curve.png
#     - pl_histogram.png
#     - analytics.md
# trading_report_analyzer_lean.py
# Purpose: Lean, investor-friendly analysis of ThinkorSwim Strategy Report CSVs
# Scope: TRADE DATA ONLY (no EMA/VWAP/ATR). Focus on P/L, risk, trade analytics, visuals.
# Session basis: **New York time (ET)** RTH 09:30–16:00.
# Capital float: $2,500 (1 contract). Commission: $4.04 round trip per contract. Point value: $5.00/pt.
# Outputs (per run):
#   Backtests/<YYYY-MM-DD>_<Strategy>_<Timeframe>_<CSVStem>/
#     - trades_enriched.csv
#     - metrics.json
#     - monthly_performance.csv
#     - equity_curve_180d.png
#     - drawdown_curve.png
#     - pl_histogram.png
#     - analytics.md

import os
import io
import json
from dataclasses import dataclass, asdict
from datetime import datetime, time
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# Configuration
# =========================

@dataclass
class BacktestConfig:
    strategy_name: str = "SuperSignal_v7"
    instruments: Tuple[str, ...] = ("/MES",)
    timeframe: str = "15m"
    # RTH session in **New York time** (ET): 09:30–16:00
    session_hours: Tuple[str, str] = ("09:30", "16:00")
    # Capital float = cost to run exactly **1 contract**
    initial_capital: float = 2500.0
    # Round-trip commission per **contract**
    commission_per_round_trip: float = 4.04
    # /MES: $5.00 per point per contract
    point_value: float = 5.0
    version: str = "1.0.4"

    def outdir(self, csv_stem: str) -> str:
        day = datetime.now().strftime("%Y-%m-%d")
        safe_strategy = self.strategy_name.replace(" ", "_")
        return os.path.join(os.getcwd(), f"Backtests/{day}_{safe_strategy}_{self.timeframe}_{csv_stem}")


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


def _tag_session(dt: pd.Timestamp) -> str:
    """Tag by **New York time** (ET). If your CSV timestamps are not ET, align upstream."""
    if pd.isna(dt):
        return "Unknown"
    t = dt.time()
    bands = {
        "Overnight": (time(20, 0), time(3, 59)),
        "Pre": (time(4, 0), time(9, 29)),
        "Open": (time(9, 30), time(10, 30)),
        "Midday": (time(10, 30), time(15, 0)),
        "Late": (time(15, 0), time(16, 0)),
        "Post": (time(16, 0), time(20, 0)),
    }
    # Overnight wraps midnight
    if t >= bands["Overnight"][0] or t <= bands["Overnight"][1]:
        return "Overnight"
    for name, (start, end) in bands.items():
        if name == "Overnight":
            continue
        if start <= t <= end:
            return name
    return "Other"


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
    """Map the exit row's text to a normalized reason (Target/Stop/Time/Manual/Close)."""
    s = str(text).upper()
    if any(w in s for w in ["TARGET", "TGT", "TP", "PROFIT"]):
        return "Target"
    if any(w in s for w in ["STOP", "SL", "STOPPED"]):
        return "Stop"
    if any(w in s for w in ["TIME", "TIME EXIT", "TIMED", "TIMEOUT"]):
        return "Time"
    if any(w in s for w in ["MANUAL", "FLATTEN", "MKT CLOSE", "DISCRETIONARY"]):
        return "Manual"
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

    if 'Date/Time' in df.columns:
        df['Date'] = _parse_datetime(df['Date/Time'])
    elif 'Date' in df.columns:
        df['Date'] = _parse_datetime(df['Date'])
    else:
        raise ValueError("Could not find 'Date/Time' or 'Date' column.")

    if 'Trade P/L' in df.columns:
        df['TradePL'] = _to_float(df['Trade P/L']).fillna(0.0)
    elif 'TradePL' in df.columns:
        df['TradePL'] = _to_float(df['TradePL']).fillna(0.0)
    else:
        df['TradePL'] = 0.0

    df['CumPL'] = _to_float(df['P/L']) if 'P/L' in df.columns else np.nan
    df['BaseStrategy'] = df['Strategy'].astype(str).str.split('(').str[0].str.strip() if 'Strategy' in df.columns else "Unknown"

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

    df = df.dropna(subset=['Date']).sort_values('Date').reset_index(drop=True)
    return df


# =========================
# Build Trades (pair entries/exits)
# =========================

def build_trades(df: pd.DataFrame) -> pd.DataFrame:
    id_col = 'Id' if 'Id' in df.columns else None
    trades = []

    def _safe_num(x):
        return pd.to_numeric(x, errors='coerce')

    if id_col:
        # Regex variants to capture typical TOS wording/underscores
        OPEN_RX  = r"\b(?:BTO|BUY TO OPEN|BUY_TO_OPEN|BOT TO OPEN|STO|SELL TO OPEN|SELL_TO_OPEN|SELL SHORT|OPEN)\b"
        CLOSE_RX = r"\b(?:STC|SELL TO CLOSE|SELL_TO_CLOSE|SLD TO CLOSE|BTC|BUY TO CLOSE|BUY_TO_CLOSE|CLOSE)\b"

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

                trades.append({
                    'Id': tid,
                    'EntryTime': entry['Date'],
                    'ExitTime': exit_['Date'],
                    'EntryPrice': _safe_num(entry.get('Price')),
                    'ExitPrice': _safe_num(exit_.get('Price')),
                    'Qty': _safe_num(entry.get('Qty')),
                    'TradePL': _safe_num(exit_.get('TradePL')),
                    'BaseStrategy': entry.get('BaseStrategy', 'Unknown'),
                    'EntrySide': str(entry.get('Side', '')),
                    'ExitSide':  str(exit_.get('Side', '')),
                    'ExitReason': _exit_reason(exit_.get('Side') or exit_.get('Type') or exit_.get('Order'))
                })

    # Fallback: if no Ids/pairs, treat each close as a completed trade row
    if not trades:
        g = df.sort_values('Date').copy()
        side_up = g['Side'].astype(str).str.upper()
        CLOSE_RX = r"\b(?:STC|SELL TO CLOSE|SELL_TO_CLOSE|SLD TO CLOSE|BTC|BUY TO CLOSE|BUY_TO_CLOSE|CLOSE)\b"
        close_rows = g[side_up.str.contains(CLOSE_RX, regex=True, na=False)]
        for _, row in close_rows.iterrows():
            trades.append({
                'Id': row.get('Id', np.nan),
                'EntryTime': pd.NaT,
                'ExitTime': row['Date'],
                'EntryPrice': np.nan,
                'ExitPrice': _safe_num(row.get('Price')),
                'Qty': _safe_num(row.get('Qty')),
                'TradePL': _safe_num(row.get('TradePL')),
                'BaseStrategy': row.get('BaseStrategy', 'Unknown'),
                'EntrySide': str(row.get('Side', '')),
                'ExitSide':  str(row.get('Side', '')),
                'ExitReason': _exit_reason(row.get('Side') or row.get('Type') or row.get('Order'))
            })

    t = pd.DataFrame(trades)
    if t.empty:
        # nothing matched — avoid downstream KeyErrors
        return t.assign(
            HoldMins=np.nan, Commission=np.nan, NetPL=np.nan, GrossPL=np.nan,
            Direction="", ExitReason=""
        )

    t = t.sort_values('ExitTime').reset_index(drop=True)
    t['HoldMins'] = (t['ExitTime'] - t['EntryTime']).dt.total_seconds() / 60.0

    qty = pd.to_numeric(t['Qty'], errors='coerce').fillna(1.0).abs()
    t['Commission'] = cfg_global.commission_per_round_trip * qty
    t['NetPL'] = pd.to_numeric(t['TradePL'], errors='coerce').fillna(0.0) - t['Commission']
    t['GrossPL'] = pd.to_numeric(t['TradePL'], errors='coerce').fillna(0.0)

    es = t['EntrySide'].astype(str).str.upper()
    t['Direction'] = np.where(
        es.str.contains(r"\b(BTO|BUY TO OPEN|BUY_TO_OPEN|BOT TO OPEN)\b", regex=True, na=False), 'Long',
        np.where(es.str.contains(r"\b(STO|SELL TO OPEN|SELL_TO_OPEN|SELL SHORT)\b", regex=True, na=False), 'Short', 'Unknown')
    )

    return t


# =========================
# Metrics
# =========================

def _safe_days(first_dt, last_dt):
    if pd.isna(first_dt) or pd.isna(last_dt):
        return np.nan
    return max((last_dt - first_dt).days, 1)


def compute_metrics(trades: pd.DataFrame, cfg: BacktestConfig) -> dict:
    df = trades.copy()
    pl = df['NetPL'].fillna(0.0)

    equity = cfg.initial_capital + pl.cumsum()

    # Core returns
    total_net = float(pl.sum())
    total_return_pct = (total_net / cfg.initial_capital) * 100.0 if cfg.initial_capital else np.nan

    # Win/Loss & averages
    win_mask = pl > 0
    loss_mask = pl < 0
    avg_win = float(pl[win_mask].mean()) if win_mask.any() else np.nan
    avg_loss = float(pl[loss_mask].mean()) if loss_mask.any() else np.nan

    # Points per trade (per contract)
    qty = pd.to_numeric(df['Qty'], errors='coerce').abs().replace(0, np.nan)
    pts_per_trade = pl / (cfg.point_value * qty)
    avg_win_pts  = float(pts_per_trade[win_mask].mean()) if win_mask.any() else np.nan
    avg_loss_pts = float(pts_per_trade[loss_mask].mean()) if loss_mask.any() else np.nan

    # Drawdowns
    max_dd_pct = abs(_max_drawdown(equity)) * 100.0
    dd_series = drawdown_series(equity)
    avg_dd_pct = float(dd_series.mean() * 100.0) if len(dd_series) else np.nan
    max_dd_dollars = max_drawdown_dollars(equity)
    recovery_factor = float(total_net / max_dd_dollars) if max_dd_dollars else np.nan

    # Expectancy & risk-adjusted
    expectancy_dollars = float(pl.mean()) if len(pl) else np.nan

    trade_rets = pl / cfg.initial_capital if cfg.initial_capital else pd.Series(np.nan, index=pl.index)
    sharpe_proxy = float(trade_rets.mean() / trade_rets.std(ddof=1)) if trade_rets.std(ddof=1) > 0 else np.nan

    downside = trade_rets.copy()
    downside[downside > 0] = 0
    down_stdev = downside.std(ddof=1)
    sortino_proxy = float(trade_rets.mean() / abs(down_stdev)) if down_stdev and down_stdev > 0 else np.nan

    largest_win = float(pl.max()) if len(pl) else np.nan
    largest_loss = float(pl.min()) if len(pl) else np.nan
    vol_of_trade_returns = float(trade_rets.std(ddof=1)) if len(trade_rets) > 1 else np.nan

    # Monthly returns & CAGR
    dt = pd.to_datetime(df['ExitTime'], errors='coerce')
    eq_df = pd.DataFrame({'dt': dt, 'equity': equity})
    eq_month = eq_df.dropna(subset=['dt']).set_index('dt').resample('M').last()
    monthly_ret = eq_month['equity'].pct_change()
    avg_monthly_return = float(monthly_ret.mean()) if monthly_ret.notna().any() else np.nan

    first_dt = pd.to_datetime(df['ExitTime']).min()
    last_dt  = pd.to_datetime(df['ExitTime']).max()
    days = _safe_days(first_dt, last_dt)
    ending_equity = float(equity.iloc[-1]) if len(equity) else cfg.initial_capital
    CAGR = float((ending_equity / cfg.initial_capital) ** (365.0 / days) - 1.0) if days and days > 0 else np.nan

    # Counts
    long_count  = int((df['Direction'] == 'Long').sum()) if 'Direction' in df.columns else 0
    short_count = int((df['Direction'] == 'Short').sum()) if 'Direction' in df.columns else 0

    metrics = {
        # identifiers
        "strategy_name": cfg.strategy_name,
        "version": cfg.version,
        "timeframe": cfg.timeframe,
        "initial_capital": cfg.initial_capital,
        "point_value": cfg.point_value,

        # performance
        "num_trades": int(len(df)),
        "net_profit": total_net,
        "total_return_pct": total_return_pct,
        "avg_monthly_return": avg_monthly_return,
        "CAGR": CAGR,
        "profit_factor": _profit_factor(pl),
        "win_rate_pct": float((pl > 0).mean() * 100.0),
        "avg_win_dollars": avg_win,
        "avg_loss_dollars": avg_loss,
        "avg_win_points_per_contract": avg_win_pts,
        "avg_loss_points_per_contract": avg_loss_pts,
        "expectancy_per_trade_dollars": expectancy_dollars,

        # risk
        "max_drawdown_pct": max_dd_pct,
        "max_drawdown_dollars": max_dd_dollars,
        "avg_drawdown_pct": avg_dd_pct,
        "recovery_factor": recovery_factor,
        "per_trade_sharpe_proxy": sharpe_proxy,
        "per_trade_sortino_proxy": sortino_proxy,
        "largest_winning_trade": largest_win,
        "largest_losing_trade": largest_loss,
        "vol_of_trade_returns": vol_of_trade_returns,

        # trade analytics
        "num_longs": long_count,
        "num_shorts": short_count,
        "avg_hold_minutes": float(df['HoldMins'].mean()) if 'HoldMins' in df.columns else np.nan,
    }

    # Exit method breakdown (if present)
    if 'ExitReason' in df.columns:
        reason_counts = df['ExitReason'].value_counts(dropna=False).to_dict()
        reason_avg = df.groupby('ExitReason')['NetPL'].mean().to_dict()
        reason_pf = {r: _profit_factor(df.loc[df['ExitReason'] == r, 'NetPL']) for r in df['ExitReason'].dropna().unique()}
        metrics.update({
            "exit_reason_counts": reason_counts,
            "exit_reason_avg_netpl": reason_avg,
            "exit_reason_profit_factor": reason_pf,
        })

    return metrics


# =========================
# Visuals & Tables
# =========================

def save_visuals_and_tables(trades: pd.DataFrame, cfg: BacktestConfig, outdir: str) -> None:
    os.makedirs(outdir, exist_ok=True)
    pl = trades['NetPL'].fillna(0.0)
    equity = cfg.initial_capital + pl.cumsum()

    # Equity (last ~180 calendar days if ExitTime present)
    eq_df_daily = pd.DataFrame({
        'dt': pd.to_datetime(trades['ExitTime'], errors='coerce'),
        'equity': equity
    }).dropna(subset=['dt']).set_index('dt').resample('D').last()
    if len(eq_df_daily) > 0:
        eq_last_180 = eq_df_daily.tail(180)
        plt.figure(figsize=(9, 4.5))
        plt.plot(eq_last_180.index, eq_last_180['equity'].values)
        plt.ylabel("Equity ($)")
        plt.xlabel("Date")
        plt.title(f"Equity Curve (last 180 days) — {cfg.strategy_name} ({cfg.timeframe})")
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, "equity_curve_180d.png"), dpi=160)
        plt.close()
    else:
        # Fallback to per-trade index curve
        plt.figure(figsize=(9, 4.5))
        plt.plot(equity.index, equity.values)
        plt.ylabel("Equity ($)")
        plt.xlabel("Trade #")
        plt.title(f"Equity Curve — {cfg.strategy_name} ({cfg.timeframe})")
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, "equity_curve_180d.png"), dpi=160)
        plt.close()

    # Drawdown curve (per-trade index)
    dd = drawdown_series(equity)
    plt.figure(figsize=(9, 4.5))
    plt.plot(dd.index, dd.values * 100.0)
    plt.ylabel("Drawdown (%)")
    plt.xlabel("Trade #")
    plt.title("Drawdown Curve")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "drawdown_curve.png"), dpi=160)
    plt.close()

    # Histogram of trade P/L
    plt.figure(figsize=(9, 4.5))
    plt.hist(trades['NetPL'].dropna().values, bins=30)
    plt.xlabel("Net P/L per Trade ($)")
    plt.ylabel("Count")
    plt.title("Trade P/L Distribution")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "pl_histogram.png"), dpi=160)
    plt.close()

    # Monthly performance table
    dt = pd.to_datetime(trades['ExitTime'], errors='coerce')
    monthly = pd.DataFrame({'NetPL': trades['NetPL'].values}, index=dt)
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


def generate_analytics_md(trades: pd.DataFrame, metrics: dict, cfg: BacktestConfig, outdir: str) -> None:
    """Create a one-sheet analytics.md with definitions and the run's actual numbers."""
    os.makedirs(outdir, exist_ok=True)

    m = metrics
    def g(key, default=np.nan):
        return m.get(key, default)

    # Optional monthly preview (last 6 rows)
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
# Strategy One‑Sheet (Trade Data)

**Strategy:** {g('strategy_name')}  
**Timeframe:** {g('timeframe')}  
**Run Date:** {datetime.now().strftime('%Y-%m-%d')}  
**Session Basis:** New York time (ET) RTH 09:30–16:00  
**Initial Capital (float):** ${_fmt(g('initial_capital'), 0)}  
**Commission (RT / contract):** ${_fmt(cfg.commission_per_round_trip, 2)}  
**Point Value:** ${_fmt(g('point_value'), 2)} per point per contract

---

## Headline KPIs
- **Net Profit:** ${_fmt(g('net_profit'))}
- **Total Return:** {_fmt(g('total_return_pct'), pct=True)}
- **Win Rate:** {_fmt(g('win_rate_pct'), pct=True)}
- **Profit Factor:** {_fmt(g('profit_factor'))}
- **Max Drawdown:** ${_fmt(g('max_drawdown_dollars'))} ({_fmt(g('max_drawdown_pct'), pct=True)})
- **CAGR:** {_fmt(g('CAGR')*100, pct=True)}

---

## Performance (returns & consistency)
| Metric | Result | Definition / Formula |
|---|---:|---|
| Net Profit ($) | ${_fmt(g('net_profit'))} | Σ(NetPLᵢ) |
| Total Return (%) | {_fmt(g('total_return_pct'), pct=True)} | (Net Profit ÷ Initial Capital) × 100 |
| Average Monthly Return | {_fmt(g('avg_monthly_return')*100, pct=True)} | mean(Monthly equity % change) |
| CAGR | {_fmt(g('CAGR')*100, pct=True)} | (Ending Equity ÷ Initial Capital)^(365 ÷ Days) − 1 |
| Profit Factor | {_fmt(g('profit_factor'))} | (Σ profits) ÷ |Σ losses| |
| Win Rate (%) | {_fmt(g('win_rate_pct'), pct=True)} | (# wins ÷ total trades) × 100 |
| Avg Win ($) | ${_fmt(g('avg_win_dollars'))} | mean(NetPL | NetPL>0) |
| Avg Loss ($) | ${_fmt(g('avg_loss_dollars'))} | mean(NetPL | NetPL<0) |
| Avg Win (pts / contract) | {_fmt(g('avg_win_points_per_contract'))} | NetPL ÷ (point_value × |Qty|) |
| Avg Loss (pts / contract) | {_fmt(g('avg_loss_points_per_contract'))} | NetPL ÷ (point_value × |Qty|) |
| Expectancy per Trade ($) | ${_fmt(g('expectancy_per_trade_dollars'))} | mean(NetPLᵢ) |

---

## Risk (drawdowns & risk‑adjusted)
| Metric | Result | Definition / Formula |
|---|---:|---|
| Max Drawdown ($) | ${_fmt(g('max_drawdown_dollars'))} | max(peak − Equity) |
| Max Drawdown (%) | {_fmt(g('max_drawdown_pct'), pct=True)} | min(Equity ÷ peak − 1) × 100 |
| Average Drawdown (%) | {_fmt(g('avg_drawdown_pct'), pct=True)} | mean(Drawdownₜ) × 100 |
| Recovery Factor | {_fmt(g('recovery_factor'))} | Net Profit ÷ Max DD ($) |
| Sharpe (per‑trade proxy) | {_fmt(g('per_trade_sharpe_proxy'))} | mean(rᵢ) ÷ stdev(rᵢ) |
| Sortino (per‑trade proxy) | {_fmt(g('per_trade_sortino_proxy'))} | mean(rᵢ) ÷ stdev(min(rᵢ,0)) |
| Largest Winning Trade ($) | ${_fmt(g('largest_winning_trade'))} | max(NetPLᵢ) |
| Largest Losing Trade ($) | ${_fmt(g('largest_losing_trade'))} | min(NetPLᵢ) |
| Volatility of Trade Returns | {_fmt(g('vol_of_trade_returns'))} | stdev(per‑trade returns) |

"""

    # Exit method breakdown section (if present)
    if 'exit_reason_counts' in m:
        lines = ["## Exit Method Breakdown", "| Reason | Trades | Avg NetPL ($) | Profit Factor |", "|---|---:|---:|---:|"]
        for r, n in m['exit_reason_counts'].items():
            avg = m.get('exit_reason_avg_netpl', {}).get(r, np.nan)
            pf  = m.get('exit_reason_profit_factor', {}).get(r, np.nan)
            lines.append(f"| {r} | {int(n)} | {_fmt(avg)} | {_fmt(pf)} |")
        md += "\n" + "\n".join(lines) + "\n\n"md += f"""
---

## Trade Analytics (behavior & cadence)
| Metric | Result |
|---|---:|
| Number of Trades | {int(g('num_trades', 0))} |
| Long Trades | {int(g('num_longs', 0))} |
| Short Trades | {int(g('num_shorts', 0))} |
| Avg Holding Time (minutes) | {_fmt(g('avg_hold_minutes'))} |
| Session Tags (ET) | Overnight / Pre / Open / Midday / Late / Post |

> For session‑level slicing, use `trades_enriched.csv` (column **Session**).

---

## Visuals & Tables (investor‑friendly)
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
# Runner
# =========================

def run_backtest(tos_csv_path: str, cfg: BacktestConfig):
    csv_stem = Path(tos_csv_path).stem.replace(' ', '_')
    outdir = cfg.outdir(csv_stem)
    os.makedirs(outdir, exist_ok=True)

    raw = load_tos_strategy_report(tos_csv_path)
    trades = build_trades(raw)

    # Session tags and descriptive fields (ET RTH basis)
    trades['Session'] = trades['EntryTime'].apply(_tag_session)

    # Save enriched trades
    trades_out = os.path.join(outdir, "trades_enriched.csv")
    trades.to_csv(trades_out, index=False)

    # Metrics
    metrics = compute_metrics(trades, cfg)
    with open(os.path.join(outdir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # Visuals & tables
    save_visuals_and_tables(trades, cfg, outdir)

    # One-sheet
    generate_analytics_md(trades, metrics, cfg, outdir)

    # Save config
    with open(os.path.join(outdir, "config.json"), "w") as f:
        json.dump(asdict(cfg), f, indent=2)

    return trades, metrics


# =========================
# CLI
# =========================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Lean analysis of TOS Strategy Report CSV (trade data only).")
    parser.add_argument("--csv", type=str, default="StrategyReports_MESXCME_81425.csv",
                        help="Path to a TOS Strategy Report CSV.")
    parser.add_argument("--strategy", type=str, default="SuperSignal_v7", help="Strategy name label.")
    parser.add_argument("--timeframe", type=str, default="15m", help="Timeframe label for outputs.")
    parser.add_argument("--capital", type=float, default=2500.0, help="Initial capital (1-contract float).")
    parser.add_argument("--commission", type=float, default=4.04, help="Commission per contract round trip.")
    parser.add_argument("--point_value", type=float, default=5.0, help="Dollars per index point per contract (e.g., /MES = 5.0)")

    args = parser.parse_args()

    # global config (used in build_trades for commission)
    global cfg_global
    cfg_global = BacktestConfig(
        strategy_name=args.strategy,
        instruments=("/MES",),
        timeframe=args.timeframe,
        initial_capital=args.capital,
        commission_per_round_trip=args.commission,
        point_value=args.point_value,
        version="1.0.4",
    )

    print(f"[RUN] CSV: {args.csv}")
    trades_df, metrics = run_backtest(args.csv, cfg_global)
    print(json.dumps(metrics, indent=2))
    print(f"Saved outputs to: {cfg_global.outdir(Path(args.csv).stem.replace(' ', '_'))}")
