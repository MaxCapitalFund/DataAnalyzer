# trading_report_analyzer.py
# Backtesting pipeline for ThinkorSwim Strategy Report (.csv)
# Reads your TOS report, cleans it, computes features/metrics, and saves outputs under:
#   ./Backtests/<YYYY-MM-DD>_<StrategyName>_<Timeframe>/

import os
import json
from dataclasses import dataclass, asdict
from datetime import datetime, time
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#CONFIGURATION

@dataclass
class BacktestConfig:
    strategy_name: str = "SuperSignal_v7"
    instruments: Tuple[str, ...] = ("/MES",)
    timeframe: str = "15m"
    session_hours: Tuple[str, str] = ("08:30", "15:00")  # RTH (CT). Adjust as needed.
    lookback_months: int = 12
    initial_capital: float = 10000.0
    commission_per_round_trip: float = 2.02  # /MES
    atr_period: int = 7
    ema_fast: int = 9
    ema_slow: int = 15
    vwap: bool = True  # compute VWAP if OHLCV provided
    version: str = "1.0.0"
    risk_free_rate_annual: float = 0.0  # assume 0 for simplicity

    def outdir(self) -> str:
        day = datetime.now().strftime("%Y-%m-%d")
        safe_strategy = self.strategy_name.replace(" ", "_")
        # Save under the current working directory (local machine-friendly)
        return os.path.join(os.getcwd(), f"Backtests/{day}_{safe_strategy}_{self.timeframe}")


#HELPER METHOD

def _to_float(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.replace(r'[\$,]', '', regex=True)
    s = s.str.replace(r'\(([^()]*)\)', r'-\1', regex=True)  # (123) -> -123
    s = s.replace('', np.nan)
    return pd.to_numeric(s, errors='coerce')


def _parse_datetime(series: pd.Series) -> pd.Series:
    # ThinkorSwim reports are commonly like "11/27/24 3:45 AM"
    return pd.to_datetime(series, format="%m/%d/%y %I:%M %p", errors='coerce')


def _tag_session(dt: pd.Timestamp) -> str:
    # Session tags (America/Chicago assumed). Adjust bands if needed.
    t = dt.time()
    bands = {
        "Globex Evening": (time(16, 0), time(17, 0)),
        "Overnight": (time(17, 0), time(8, 29)),
        "Open": (time(8, 30), time(9, 30)),
        "Midday": (time(9, 30), time(14, 0)),
        "Late": (time(14, 0), time(15, 0)),
        "Post": (time(15, 0), time(16, 0))
    }
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


def _profit_factor(pl: pd.Series) -> float:
    gp = pl[pl > 0].sum()
    gl = -pl[pl < 0].sum()
    if gl == 0:
        return np.inf if gp > 0 else 0.0
    return float(gp / gl)


def _avg_win_loss_ratio(pl: pd.Series) -> float:
    wins = pl[pl > 0]
    losses = -pl[pl < 0]
    if len(wins) == 0 or len(losses) == 0:
        return np.nan
    return float(wins.mean() / losses.mean())


def _expectancy_R(r_values: pd.Series) -> float:
    return float(np.nanmean(r_values.values)) if len(r_values) else np.nan


#DATA LOAD & CLEAN

def load_tos_strategy_report(file_path: str) -> pd.DataFrame:
    with open(file_path, 'r', errors='replace') as f:
        lines = f.readlines()

    start_idx = None
    for i, line in enumerate(lines):
        if line.startswith("Id;Strategy;"):
            start_idx = i
            break
    if start_idx is None:
        raise ValueError("No trade table header found in file (expected line starting with 'Id;Strategy;').")

    table_str = "".join(lines[start_idx:])
    df = pd.read_csv(pd.io.common.StringIO(table_str), sep=';')

    # Dates
    if 'Date/Time' in df.columns:
        df['Date'] = _parse_datetime(df['Date/Time'])
    elif 'Date' in df.columns:
        df['Date'] = _parse_datetime(df['Date'])
    else:
        raise ValueError("Could not find 'Date/Time' or 'Date' column.")

    # Common monetary fields
    if 'Trade P/L' in df.columns:
        df['TradePL'] = _to_float(df['Trade P/L']).fillna(0.0)
    elif 'TradePL' in df.columns:
        df['TradePL'] = _to_float(df['TradePL']).fillna(0.0)
    else:
        df['TradePL'] = 0.0

    if 'P/L' in df.columns:
        df['CumPL'] = _to_float(df['P/L'])
    else:
        df['CumPL'] = np.nan

    # Strategy base name
    if 'Strategy' in df.columns:
        df['BaseStrategy'] = df['Strategy'].astype(str).str.split('(').str[0].str.strip()
    else:
        df['BaseStrategy'] = "Unknown"

    # Side normalization for open/close tagging
    side_col = None
    for cand in ['Side', 'Action', 'Order', 'Type']:
        if cand in df.columns:
            side_col = cand
            break
    if side_col is None:
        df['Side'] = ""
    else:
        df['Side'] = df[side_col].astype(str)

    # Ensure price & qty exist
    if 'Price' not in df.columns:
        df['Price'] = np.nan
    # TOS often uses 'Quantity' for fill size
    qty_col = 'Quantity' if 'Quantity' in df.columns else ('Qty' if 'Qty' in df.columns else None)
    if qty_col and qty_col != 'Qty':
        df['Qty'] = pd.to_numeric(df[qty_col], errors='coerce')
    elif 'Qty' not in df.columns:
        df['Qty'] = np.nan

    # Remove empty/bad dates, sort
    df = df.dropna(subset=['Date']).sort_values('Date').reset_index(drop=True)
    return df


#TRADE TABLE CONSTRUCTION

def build_trades(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build per-trade table by pairing the first 'Open' with the first subsequent 'Close' for each Id.
    If pairing is not possible, fall back to close-only rows to preserve realized P/L.
    """
    id_col = 'Id' if 'Id' in df.columns else None
    trades = []

    def _safe_num(x):
        return pd.to_numeric(x, errors='coerce')

    if id_col:
        for tid, grp in df.groupby(id_col, sort=False):
            g = grp.sort_values('Date').copy()
            g['is_open'] = g['Side'].str.contains('Open', case=False, na=False)
            g['is_close'] = g['Side'].str.contains('Close', case=False, na=False)

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
                    'BaseStrategy': entry.get('BaseStrategy', 'Unknown')
                })

    # If no pairs were formed, use close-only rows (realized P/L still valid)
    if not trades:
        g = df.sort_values('Date')
        close_rows = g[g['Side'].str.contains('Close', case=False, na=False)]
        for _, row in close_rows.iterrows():
            trades.append({
                'Id': row.get('Id', np.nan),
                'EntryTime': pd.NaT,
                'ExitTime': row['Date'],
                'EntryPrice': np.nan,
                'ExitPrice': _safe_num(row.get('Price')),
                'Qty': _safe_num(row.get('Qty')),
                'TradePL': _safe_num(row.get('TradePL')),
                'BaseStrategy': row.get('BaseStrategy', 'Unknown')
            })

    trades_df = pd.DataFrame(trades).sort_values('ExitTime').reset_index(drop=True)
    # Holding time
    if 'EntryTime' in trades_df.columns and 'ExitTime' in trades_df.columns:
        trades_df['HoldMins'] = (trades_df['ExitTime'] - trades_df['EntryTime']).dt.total_seconds() / 60.0
    else:
        trades_df['HoldMins'] = np.nan
    return trades_df


#MERGE INDICATORS FROM OHLCV 

def compute_indicators(bars: pd.DataFrame, cfg: BacktestConfig) -> pd.DataFrame:
    """
    bars: DataFrame with columns ['Date','Open','High','Low','Close','Volume'].
    Returns DataFrame with EMA9, EMA15, ATR, VWAP.
    """
    b = bars.copy()
    if 'Date' in b.columns:
        b['Date'] = pd.to_datetime(b['Date'])
        b = b.sort_values('Date').set_index('Date')
    b = b.sort_index()

    # EMAs
    b['EMA9'] = b['Close'].ewm(span=cfg.ema_fast, adjust=False).mean()
    b['EMA15'] = b['Close'].ewm(span=cfg.ema_slow, adjust=False).mean()

    # ATR
    tr = pd.concat([
        (b['High'] - b['Low']),
        (b['High'] - b['Close'].shift()).abs(),
        (b['Low'] - b['Close'].shift()).abs()
    ], axis=1).max(axis=1)
    b['ATR'] = tr.rolling(cfg.atr_period, min_periods=1).mean()

    # VWAP (cumulative simple definition)
    if cfg.vwap:
        tp = (b['High'] + b['Low'] + b['Close']) / 3.0
        pv = tp * b['Volume']
        b['CumPV'] = pv.cumsum()
        b['CumVol'] = b['Volume'].cumsum()
        b['VWAP'] = b['CumPV'] / b['CumVol']
        b = b.drop(columns=['CumPV', 'CumVol'])
    else:
        b['VWAP'] = np.nan

    return b


def attach_indicators_to_trades(trades: pd.DataFrame, ind: pd.DataFrame) -> pd.DataFrame:
    # Align by nearest timestamp at/just before EntryTime
    t = trades.copy()
    ind = ind.copy().sort_index()

    t['EMA9_entry'] = np.nan
    t['EMA15_entry'] = np.nan
    t['VWAP_entry'] = np.nan
    t['ATR_entry'] = np.nan

    left = t[['EntryTime']].rename(columns={'EntryTime': 'Date'})
    # merge_asof on NaT yields NaN indicators, which is fine for close-only fallbacks
    merged = pd.merge_asof(
        left.sort_values('Date'),
        ind.reset_index().rename(columns={'index': 'Date'}).sort_values('Date'),
        on='Date',
        direction='backward'
    )
    for col_src, col_dst in [('EMA9', 'EMA9_entry'), ('EMA15', 'EMA15_entry'),
                             ('VWAP', 'VWAP_entry'), ('ATR', 'ATR_entry')]:
        if col_src in merged.columns:
            t.loc[merged.index, col_dst] = merged[col_src].values
    return t


#FEATURE ENGINEERING 

def feature_engineering(trades: pd.DataFrame, cfg: BacktestConfig) -> pd.DataFrame:
    t = trades.copy()

    # Session tags
    t['Session'] = t['EntryTime'].apply(lambda x: _tag_session(x) if pd.notna(x) else 'Unknown')

    # Trade classification (simple heuristic — customize as needed)
    def classify(row):
        p = row.get('EntryPrice', np.nan)
        e9 = row.get('EMA9_entry', np.nan)
        e15 = row.get('EMA15_entry', np.nan)
        if np.isnan(p) or np.isnan(e9) or np.isnan(e15):
            return "Unknown"
        if p > e9 > e15:
            return "TrendContinuation"
        if p < e9 < e15:
            return "DowntrendContinuation"
        if (p > e9 and p < e15) or (p < e9 and p > e15):
            return "Consolidation"
        if (p > e9 and e9 < e15) or (p < e9 and e9 > e15):
            return "Reversal"
        return "Other"

    t['TradeType'] = t.apply(classify, axis=1)

    # R-multiple: profit divided by initial risk.
    # Here risk ~ ATR at entry * point_value * Qty (approx). For /MES, $5/point.
    point_value = 5.0
    qty = pd.to_numeric(t['Qty'], errors='coerce').fillna(1.0)
    atr_points = pd.to_numeric(t.get('ATR_entry', pd.Series(index=t.index, dtype=float)), errors='coerce')\
                    .fillna(method='ffill').fillna(0.0)
    dollar_risk = (atr_points * point_value * qty).replace(0.0, np.nan)
    t['RMultiple'] = t['TradePL'] / dollar_risk

    # Holding time already computed; ensure numeric
    if 'HoldMins' not in t.columns:
        t['HoldMins'] = np.nan

    # Commission per round trip & Net P/L
    t['Commission'] = cfg.commission_per_round_trip
    t['NetPL'] = t['TradePL'] - t['Commission']

    return t


#METRICS

def compute_metrics(trades: pd.DataFrame, cfg: BacktestConfig) -> dict:
    df = trades.copy()
    pl = df['NetPL'].fillna(0.0)

    win_rate = float((pl > 0).mean() * 100.0)
    profit_factor = _profit_factor(pl)
    aw_al = _avg_win_loss_ratio(pl)
    total_net = float(pl.sum())

    equity = cfg.initial_capital + pl.cumsum()
    max_dd = _max_drawdown(equity)  # negative

    expectancy_R = _expectancy_R(df['RMultiple'])

    # Per-trade Sharpe (simple, not annualized)
    trade_returns = pl / cfg.initial_capital
    sharpe = float(trade_returns.mean() / trade_returns.std(ddof=1)) if trade_returns.std(ddof=1) > 0 else np.nan

    return {
        "strategy_name": cfg.strategy_name,
        "version": cfg.version,
        "timeframe": cfg.timeframe,
        "initial_capital": cfg.initial_capital,
        "num_trades": int(len(df)),
        "win_rate_pct": win_rate,
        "profit_factor": profit_factor,
        "avg_win_over_avg_loss": aw_al,
        "total_net_profit": total_net,
        "max_drawdown_pct": max_dd * 100.0,
        "expectancy_R": expectancy_R,
        "per_trade_sharpe": sharpe
    }


#RUNNER

def run_backtest(tos_csv_path: str,
                 cfg: BacktestConfig,
                 ohlcv_csv_path: Optional[str] = None) -> Tuple[pd.DataFrame, dict]:
    os.makedirs(cfg.outdir(), exist_ok=True)

    # Load & clean
    raw = load_tos_strategy_report(tos_csv_path)

    # Build trades
    trades = build_trades(raw)

    # Optional indicators
    if ohlcv_csv_path is not None and os.path.exists(ohlcv_csv_path):
        bars = pd.read_csv(ohlcv_csv_path)
        ind = compute_indicators(bars, cfg)
        trades = attach_indicators_to_trades(trades, ind)

    # Features
    trades = feature_engineering(trades, cfg)

    # Metrics
    metrics = compute_metrics(trades, cfg)

    # Save artifacts
    trades_out = os.path.join(cfg.outdir(), "trades_enriched.csv")
    metrics_out = os.path.join(cfg.outdir(), "metrics.json")
    cfg_out = os.path.join(cfg.outdir(), "config.json")

    trades.to_csv(trades_out, index=False)
    with open(metrics_out, "w") as f:
        json.dump(metrics, f, indent=2)
    with open(cfg_out, "w") as f:
        json.dump(asdict(cfg), f, indent=2)

    # Plot equity curve
    equity = cfg.initial_capital + trades['NetPL'].fillna(0.0).cumsum()
    plt.figure(figsize=(9, 4.5))
    plt.plot(equity.index, equity.values)
    plt.ylabel("Equity ($)")
    plt.xlabel("Trade #")
    plt.title(f"Equity Curve — {cfg.strategy_name} ({cfg.timeframe})")
    plot_path = os.path.join(cfg.outdir(), "equity_curve.png")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=160)
    plt.close()

    return trades, metrics


# MAIN 

if __name__ == "__main__":
    import argparse
    import glob
    from pathlib import Path

    parser = argparse.ArgumentParser(
        description="Run TOS backtest on one or more CSV files."
    )
    parser.add_argument("--csv", type=str, default="StrategyReports_MESXCME_81425.csv",
                        help="Path to a TOS Strategy Report CSV, or a glob like 'data/*.csv'.")
    parser.add_argument("--bars", type=str, default=None,
                        help="Optional OHLCV CSV for indicators: columns Date,Open,High,Low,Close,Volume.")
    parser.add_argument("--strategy", type=str, default="SuperSignal_v7",
                        help="Strategy name label to store in outputs.")
    parser.add_argument("--timeframe", type=str, default="15m",
                        help="Timeframe label for outputs (e.g., 15m, 5m, 1h).")
    parser.add_argument("--capital", type=float, default=10000.0,
                        help="Initial capital.")
    parser.add_argument("--commission", type=float, default=2.02,
                        help="Commission per round trip.")
    parser.add_argument("--ema_fast", type=int, default=9, help="EMA fast period.")
    parser.add_argument("--ema_slow", type=int, default=15, help="EMA slow period.")
    parser.add_argument("--atr_period", type=int, default=7, help="ATR period.")
    parser.add_argument("--vwap", action="store_true", help="Compute VWAP (requires bars).")
    parser.add_argument("--outroot", type=str, default=None,
                        help="Optional override for output root (defaults to CWD/Backtests).")
    args = parser.parse_args()

    # Resolve csv paths (single or many)
    csv_paths = sorted(glob.glob(args.csv)) if any(ch in args.csv for ch in "*?[]") else [args.csv]
    if not csv_paths:
        raise FileNotFoundError(f"No CSV files matched: {args.csv}")

    # Prepare optional bars path
    ohlcv_csv_path = args.bars if args.bars and os.path.exists(args.bars) else None
    if args.bars and not ohlcv_csv_path:
        print(f"[WARN] Bars file not found: {args.bars} (indicators/R-multiple may be NaN)")

    # Run each CSV with its own dated output folder
    for csv_path in csv_paths:
        # Build config for this run
        cfg = BacktestConfig(
            strategy_name=args.strategy,
            instruments=("/MES",),               
            timeframe=args.timeframe,
            session_hours=("08:30", "15:00"),    
            lookback_months=12,
            initial_capital=args.capital,
            commission_per_round_trip=args.commission,
            atr_period=args.atr_period,
            ema_fast=args.ema_fast,
            ema_slow=args.ema_slow,
            vwap=args.vwap,
            version="1.0.0"
        )

        #override output root
        if args.outroot:
            day = datetime.now().strftime("%Y-%m-%d")
            safe_strategy = cfg.strategy_name.replace(" ", "_")
            outdir = os.path.join(args.outroot, f"{day}_{safe_strategy}_{cfg.timeframe}")
            os.makedirs(outdir, exist_ok=True)
            # Monkey-patch cfg.outdir() to use custom root
            cfg.outdir = lambda od=outdir: od 

        print(f"\n[RUN] CSV: {csv_path}")
        trades_df, metrics = run_backtest(csv_path, cfg, ohlcv_csv_path)
        print(json.dumps(metrics, indent=2))
        print(f"Saved outputs to: {cfg.outdir()}")
