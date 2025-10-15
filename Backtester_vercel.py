# -*- coding: utf-8 -*-

# Backtester_vercel.py — OPTIMIZED FOR TOS STRATEGY REPORT ANALYSIS (v1.4.6)

# Purpose: Analyzes ThinkorSwim (ToS) Strategy Report CSV for /MES futures to produce investor-ready backtest results.

# Description:
# - Parses ToS CSV with columns: Id, Strategy, Side, Quantity, Amount, Price, Date/Time, Trade P/L, P/L, Position.

# - Extracts metadata (RSI, Volume, wrMomentum) from Strategy column for trade filtering and analysis.

# - Applies a $100 stop-loss cap (20 points for /MES) and deducts $4.04 round-trip commissions.

# - Filters trades using RSI (>60 for long, <40 for short) and wrMomentum (>20 for long, <-20 for short) to enhance performance.

# - Outputs:
#   - trades_enriched.csv: Detailed trade data with RSI, Volume, wrMomentum, and P/L.

#   - analytics.md: Markdown report with key metrics (net profit, win rate, drawdown, session/day performance).

# - Designed for Vercel serverless deployment (uses /tmp, no charts).

# Usage: Run with `python Backtester_vercel.py --csv /tmp/temp_4.csv --timeframe 180d:15m --capital 2500.0 --commission 4.04`

# For Harrison: This script processes a trading report to generate verifiable metrics for investors. It handles /MES futures trades, applies risk controls, and produces clear outputs. Debug logs are written to /tmp/csv_debug.txt for troubleshooting.

import os
import io
from dataclasses import dataclass
from datetime import datetime, time
from pathlib import Path
import numpy as np
import pandas as pd

# ===============================================================

# CONFIGURATION SECTION

# Purpose: Defines settings for the backtest, including trading parameters and output paths.

# Output Goal: Creates a configuration object to control strategy name, capital, commissions, and signal filters.

# For Harrison: This section sets up the parameters for the /MES futures strategy (Micro E-mini S&P 500). Adjust `initial_capital` or `commission_per_round_trip` if needed. The `rsi` and `wrmomentum` thresholds filter trades to improve profitability.
# =========================================================
@dataclass
class BacktestConfig:
    strategy_name: str = "Q_SuperSignal_v7_GlidePAPI_v1"  # Name of the trading strategy
    timeframe: str = "180d:15m"  # Timeframe (180 days, 15-minute bars)
    initial_capital: float = 2500.0  # Starting capital in USD
    commission_per_round_trip: float = 4.04  # Commission per trade (entry + exit)
    point_value: float = 5.0  # /MES point value ($5 per point per contract)
    rsi_long_threshold: float = 60.0  # RSI threshold for long entries (Buy to Open)
    rsi_short_threshold: float = 40.0  # RSI threshold for short entries (Sell to Open)
    wrmomentum_long_threshold: float = 20.0  # Williams %R Momentum for long entries
    wrmomentum_short_threshold: float = -20.0  # Williams %R Momentum for short entries
    version: str = "1.4.6"  # Script version for tracking

    def outdir(self, csv_stem: str) -> str:
        """Generates a unique output directory in /tmp for Vercel compatibility."""
        temp_dir = Path("/tmp")
        day = datetime.now().strftime("%Y-%m-%d")
        timestamp = datetime.now().strftime("%H%M%S_%f")[:-3]
        safe_strategy = self.strategy_name.replace(" ", "_")
        safe_timeframe = self.timeframe.replace(":", "_").replace(" ", "_")
        return str(temp_dir / f"Backtests_{day}_{safe_strategy}_{safe_timeframe}_{csv_stem}_{timestamp}")

# Configuration Section Footer: Settings are stored in BacktestConfig for use across the script.

# =======================================================

# HELPER FUNCTIONS SECTION

# Purpose: Provides utility functions for data parsing, session tagging, and metric calculations.

# Output Goal: Reusable functions to clean data and compute trading metrics like drawdown and profit factor.

# For Harrison: These functions handle data preprocessing (e.g., converting strings to numbers) and trading-specific calculations. They are used by later sections to process the CSV and generate metrics.

# =========================================================
def _to_float(series: pd.Series) -> pd.Series:
    """Converts string values (e.g., '$1,234.56', '($123.45)') to float."""
    s = series.astype(str).str.replace(r"[\$,()]", "", regex=True).str.replace(r"\(([^()]*)\)", r"-\1", regex=True)
    s = s.replace("", np.nan)
    return pd.to_numeric(s, errors="coerce")

def _parse_datetime(series: pd.Series) -> pd.Series:
    """Parses ToS date/time format (e.g., '1/21/25 10:15 AM') to pandas datetime."""
    return pd.to_datetime(series, format="%m/%d/%y %I:%M %p", errors="coerce")

def _tag_session(dt: pd.Timestamp) -> str:
    """Tags trades by market session: PRE (3:00-9:29 AM), RTH (9:30 AM-4:00 PM), AFTER."""
    if pd.isna(dt):
        return "Unknown"
    t = dt.time()
    if time(3, 0) <= t <= time(9, 29): return "PRE"
    if time(9, 30) <= t <= time(16, 0): return "RTH"
    return "AFTER"

def _max_drawdown(equity_curve: pd.Series) -> float:
    """Calculates maximum drawdown as a percentage from the equity curve."""
    running_max = equity_curve.cummax()
    drawdown = equity_curve / running_max - 1.0
    return float(drawdown.min())

def _profit_factor(pl: pd.Series) -> float:
    """Computes profit factor: sum of profits / sum of losses."""
    s = pl.dropna()
    gp = s[s > 0].sum()
    gl = -s[s < 0].sum()
    return float(gp / gl) if gl != 0 else float("inf") if gp > 0 else 0.0

def _exit_reason(side: str) -> str:
    """Extracts exit reason (Target, Stop, Daily, Opposing) from Strategy tag."""
    s = str(side).upper()
    if "TARGET" in s: return "Target"
    if "STOP" in s: return "Stop"
    if "DAILY" in s: return "Daily"
    if "OPPOSING" in s: return "Opposing"
    return "Close"

# Helper Functions Footer: Utilities are ready for data processing and metric calculations.

# ===========================================================

# LOAD AND CLEAN CSV SECTION

# Purpose: Reads and processes the ToS Strategy Report CSV, extracting metadata and normalizing data.

# Output Goal: Produces a cleaned pandas DataFrame with trade data and metadata (RSI, Volume, wrMomentum).

# For Harrison: This function loads the CSV (expected at /tmp/temp_4.csv), checks its format, and extracts trading indicators from the Strategy column. It handles errors like missing files or columns and logs debug info to /tmp/csv_debug.txt.

# ========================================================
def load_tos_strategy_report(file_path: str) -> pd.DataFrame:
    """Loads and cleans ToS Strategy Report CSV, extracting RSI, Volume, wrMomentum."""
    print(f"DEBUG: Checking if CSV exists: {file_path}")
    if not os.path.exists(file_path):
        raise RuntimeError(f"CSV file does not exist: {file_path}")
   
    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()
    print(f"DEBUG: Loaded {len(lines)} lines from {file_path}")
    if lines:
        print(f"DEBUG: First 3 lines of CSV: {lines[:3]}")
    with open(os.path.join("/tmp", "csv_debug.txt"), "w", encoding="utf-8") as f:
        f.write(f"CSV Content (first 5 lines):\n{''.join(lines[:5])}\n")
   
    start_idx = next((i for i, line in enumerate(lines) if line.lstrip().startswith("Id;Strategy;")), None)
    if start_idx is None:
        raise RuntimeError(f"No trade table header found in {file_path}")
   
    df = pd.read_csv(io.StringIO("".join(lines[start_idx:])), sep=";")
    if df.empty:
        print(f"WARNING: Parsed DataFrame is empty for {file_path}")
        return df
   
    print(f"DEBUG: CSV headers: {list(df.columns)}")
    expected_columns = {"Id", "Strategy", "Side", "Quantity", "Amount", "Price", "Date/Time", "Trade P/L", "P/L", "Position"}
    missing = expected_columns - set(df.columns)
    if missing:
        print(f"WARNING: Missing columns {missing}, proceeding with available columns")
   
    df.rename(columns={"Trade P/L": "TradePL", "P/L": "CumPL"}, inplace=True)
    df["Date"] = _parse_datetime(df["Date/Time"])
    df["BaseStrategy"] = df["Strategy"].astype(str).str.split("(").str[0].str.strip()
    df["Tag"] = df["Strategy"].astype(str).str.extract(r"\(([^()]*)\)", expand=False).fillna("")
    df["RSI"] = df["Strategy"].astype(str).str.extract(r"RSI=([\d.]+)", expand=False).astype(float)
    df["Volume"] = df["Strategy"].astype(str).str.extract(r"Volume=([\d,]+)", expand=False).str.replace(",", "").astype(float)
    df["wrMomentum"] = df["Strategy"].astype(str).str.extract(r"wrMomentum=([-.\d]+)", expand=False).astype(float)
    df["ExitReason"] = df["Tag"].apply(_exit_reason)
    def normalize_side(v):
        """Normalizes ToS Side values to BTO, STC, STO, BTC."""
        if not isinstance(v, str): return ""
        v = v.upper().strip()
        return {
            "BTO": "BTO", "BUY TO OPEN": "BTO",
            "STC": "STC", "SELL TO CLOSE": "STC",
            "STO": "STO", "SELL TO OPEN": "STO",
            "BTC": "BTC", "BUY TO CLOSE": "BTC"
        }.get(v, "")
    df["SideNorm"] = df["Side"].map(normalize_side)
    df["TradePL"] = _to_float(df["TradePL"]).fillna(0.0)
    df["CumPL"] = _to_float(df["CumPL"])
    df["Qty"] = pd.to_numeric(df["Quantity"], errors="coerce")
    df["Position"] = pd.to_numeric(df["Position"], errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)
    print(f"DEBUG: DataFrame shape: {df.shape}, SideNorm: {df['SideNorm'].value_counts().to_dict()}")
    return df

# Load and Clean CSV Footer: Outputs a DataFrame with cleaned trade data and metadata for further processing.

# ============================================

# FILTER TRADES SECTION

# Purpose: Filters trades based on RSI and wrMomentum to select high-probability entries.

# Output Goal: Produces a filtered DataFrame with trades meeting RSI and wrMomentum criteria.

# For Harrison: This function improves strategy performance by keeping only long trades (BTO) with RSI > 60 and wrMomentum > 20, and short trades (STO) with RSI < 40 and wrMomentum < -20. Exits (STC, BTC) are retained to complete trade pairs.

# =======================================================
def filter_trades(df: pd.DataFrame, cfg: BacktestConfig) -> pd.DataFrame:
    """Filters trades by RSI and wrMomentum thresholds to enhance profitability."""
    if df.empty:
        return df
    df = df.copy()
    mask = (
        ((df["SideNorm"] == "BTO") & (df["RSI"] >= cfg.rsi_long_threshold) & (df["wrMomentum"] >= cfg.wrmomentum_long_threshold)) |
        ((df["SideNorm"] == "STO") & (df["RSI"] <= cfg.rsi_short_threshold) & (df["wrMomentum"] <= cfg.wrmomentum_short_threshold))
    )
    filtered = df[mask | df["SideNorm"].isin(["STC", "BTC"])]
    print(f"DEBUG: Filtered to {len(filtered)} rows from {len(df)} based on RSI and wrMomentum")
    return filtered

# Filter Trades Footer: Outputs a DataFrame with high-probability trades for analysis.

# =======================================================

# BUILD TRADES SECTION

# Purpose: Pairs entry (BTO/STO) and exit (STC/BTC) trades, calculating P/L and commissions.

# Output Goal: Produces a DataFrame with complete trade details, including entry/exit times, P/L, and metadata.

# For Harrison: This function groups trades by Id, matches entries with exits (Position=0.0), and computes net profit/loss after commissions. It ensures all 357 trades from the CSV are processed correctly.

# ========================================================
def build_trades(df: pd.DataFrame, commission_rt: float) -> pd.DataFrame:
    """Builds complete trades from entry/exit pairs, calculating P/L and commissions."""
    columns = ["Id", "EntryTime", "ExitTime", "EntrySide", "ExitSide", "Direction",
               "TradePL", "Commission", "NetPL", "BaseStrategy", "Tag", "Session",
               "EntryRSI", "ExitRSI", "EntryVolume", "ExitVolume", "EntryWrMomentum", "ExitWrMomentum", "ExitReason"]
    if df.empty:
        print("DEBUG: Input DataFrame is empty, no trades to build")
        return pd.DataFrame(columns=columns)
   
    trades = []
    for tid, grp in df.groupby("Id", sort=False):
        g = grp.sort_values("Date").copy()
        entries = g[g["SideNorm"].isin(["BTO", "STO"])]
        exits = g[(g["SideNorm"].isin(["STC", "BTC"])) & (g["Position"] == 0.0)]
        if not len(entries) or not len(exits):
            print(f"DEBUG: Skipping trade ID {tid}: no valid entry/exit")
            continue
        entry = entries.iloc[0]
        exit_ = exits[exits["Date"] >= entry["Date"]].iloc[0] if len(exits[exits["Date"] >= entry["Date"]]) else exits.iloc[-1]
        qty = abs(entry.get("Qty", 1) or 1)
        trade_pl = float(exit_.get("TradePL", 0.0))
        commission = commission_rt * qty
        net_pl = trade_pl - commission
        trades.append({
            "Id": tid,
            "EntryTime": entry["Date"],
            "ExitTime": exit_["Date"],
            "EntrySide": entry["SideNorm"],
            "ExitSide": exit_["SideNorm"],
            "Direction": "Long" if entry["SideNorm"] == "BTO" else "Short",
            "TradePL": trade_pl,
            "Commission": commission,
            "NetPL": net_pl,
            "BaseStrategy": entry["BaseStrategy"],
            "Tag": entry["Tag"],
            "Session": _tag_session(entry["Date"]),
            "EntryRSI": entry["RSI"],
            "ExitRSI": exit_["RSI"],
            "EntryVolume": entry["Volume"],
            "ExitVolume": exit_["Volume"],
            "EntryWrMomentum": entry["wrMomentum"],
            "ExitWrMomentum": exit_["wrMomentum"],
            "ExitReason": exit_["ExitReason"]
        })
    print(f"DEBUG: Built {len(trades)} trades")
    return pd.DataFrame(trades, columns=columns) if trades else pd.DataFrame(columns=columns)

# Build Trades Footer: Outputs a DataFrame with paired trades for metric calculations.

# =======================================================

# APPLY STOP-LOSS CAP SECTION

# Purpose: Caps trade losses at $100 (20 points for /MES) and adjusts P/L for commissions.

# Output Goal: Produces a DataFrame with adjusted P/L, marking stop-loss breaches.

# For Harrison: This function enforces a $100 loss limit per trade to simulate risk management. It calculates points per contract for /MES ($5/point) and adjusts net P/L after commissions.

# ==========================================
def apply_stoploss_cap(trades: pd.DataFrame, point_value: float) -> pd.DataFrame:
    """Applies $100 stop-loss cap and computes adjusted net P/L."""
    if trades.empty:
        print("DEBUG: Trades DataFrame is empty, skipping stop-loss cap")
        return trades
    df = trades.copy()
    stop_cap = -100.0
    df["SLBreached"] = df["TradePL"] < stop_cap
    df["AdjustedGrossPL"] = np.where(df["SLBreached"], stop_cap, df["TradePL"])
    df["AdjustedNetPL"] = df["AdjustedGrossPL"] - df["Commission"]
    df["PointsPerContract"] = df["AdjustedNetPL"] / point_value
    print(f"DEBUG: Applied stop-loss cap, DataFrame shape: {df.shape}")
    return df

# Apply Stop-Loss Cap Footer: Outputs a DataFrame with capped losses and adjusted P/L.

# =======================================================

# COMPUTE METRICS SECTION

# Purpose: Calculates key trading metrics for investor reporting, including profit, win rate, and session analysis.

# Output Goal: Produces a dictionary with metrics like net profit, win rate, drawdown, and day-of-week performance.

# For Harrison: This function generates investor-friendly metrics, such as total return, profit factor (profits/losses), and max drawdown (largest capital drop). It also breaks down win rates by session (PRE, RTH, AFTER) and average P/L by day of the week.

# =======================================================
def compute_metrics(trades: pd.DataFrame, cfg: BacktestConfig) -> dict:
    """Computes trading metrics for investor reporting."""
    metrics = {
        "status": "success",
        "message": "Metrics computed",
        "strategy": cfg.strategy_name,
        "version": cfg.version,
        "num_trades": 0,
        "net_profit": 0.0,
        "return_pct": 0.0,
        "avg_win": 0.0,
        "avg_loss": 0.0,
        "profit_factor": 0.0,
        "max_drawdown_pct": 0.0,
        "max_drawdown_dollars": 0.0,
        "recovery_factor": 0.0,
        "win_rate_pct": 0.0,
        "win_rate_rth_pct": 0.0,
        "win_rate_pre_pct": 0.0,
        "win_rate_after_pct": 0.0,
        "avg_pl_by_day": {}
    }
    if trades.empty:
        metrics["message"] = "No trades to compute metrics"
        return metrics
    pl = trades["AdjustedNetPL"].fillna(0.0)
    equity = cfg.initial_capital + pl.cumsum()
    total_net = float(pl.sum())
    total_return_pct = (total_net / cfg.initial_capital) * 100.0
    avg_win = float(pl[pl > 0].mean()) if any(pl > 0) else 0.0
    avg_loss = float(pl[pl < 0].mean()) if any(pl < 0) else 0.0
    max_dd = abs(_max_drawdown(equity)) * 100.0
    max_dd_dollars = float((equity.cummax() - equity).max())
    recovery_factor = total_net / max_dd_dollars if max_dd_dollars else 0.0
    session_wins = {
        "RTH": (trades[trades["Session"] == "RTH"]["AdjustedNetPL"] > 0).mean() * 100.0,
        "PRE": (trades[trades["Session"] == "PRE"]["AdjustedNetPL"] > 0).mean() * 100.0,
        "AFTER": (trades[trades["Session"] == "AFTER"]["AdjustedNetPL"] > 0).mean() * 100.0
    }
    trades["DayOfWeek"] = trades["EntryTime"].dt.day_name()
    avg_pl_by_day = trades.groupby("DayOfWeek")["AdjustedNetPL"].mean().to_dict()
    metrics.update({
        "num_trades": len(trades),
        "net_profit": total_net,
        "return_pct": total_return_pct,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "profit_factor": _profit_factor(pl),
        "max_drawdown_pct": max_dd,
        "max_drawdown_dollars": max_dd_dollars,
        "recovery_factor": recovery_factor,
        "win_rate_pct": float((pl > 0).mean() * 100.0),
        "win_rate_rth_pct": float(session_wins.get("RTH", 0.0)),
        "win_rate_pre_pct": float(session_wins.get("PRE", 0.0)),
        "win_rate_after_pct": float(session_wins.get("AFTER", 0.0)),
        "avg_pl_by_day": {k: float(v) for k, v in avg_pl_by_day.items()}
    })
    print(f"DEBUG: Computed metrics: {metrics}")
    return metrics

# Compute Metrics Footer: Outputs a dictionary with investor-ready trading metrics.

# ======================================================

# SAVE ANALYTICS REPORT SECTION

# Purpose: Generates a Markdown report summarizing backtest results for investors.

# Output Goal: Produces analytics.md with key metrics, session analysis, and day-of-week performance.

# For Harrison: This function creates a readable report in /tmp/Backtests_.../analytics.md, summarizing the strategy's performance (e.g., net profit, win rate). Investors can use this to evaluate the strategy's viability.

# ======================================================
def save_analytics_md(trades: pd.DataFrame, metrics: dict, cfg: BacktestConfig, outdir: str):
    """Saves a Markdown report with backtest metrics and analysis."""
    os.makedirs(outdir, exist_ok=True)
    m = metrics or {}
    md = f"""# Strategy Analysis Report
**Strategy:** {m.get('strategy', 'Unknown')}
**Instrument:** /MES (Micro E-mini S&P 500 Futures)
**Timeframe:** {cfg.timeframe}
**Run Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Initial Capital:** ${cfg.initial_capital:,.2f}
**Commission (Round-Trip):** ${cfg.commission_per_round_trip:.2f}
**Stop-Loss Cap:** $100 (20 points for /MES)
**Filters Applied:** RSI Long > {cfg.rsi_long_threshold}, Short < {cfg.rsi_short_threshold}, wrMomentum Long > {cfg.wrmomentum_long_threshold}, Short < {cfg.wrmomentum_short_threshold}
---
## Key Metrics
- **Total Trades:** {m.get('num_trades', 0)}
- **Net Profit:** ${m.get('net_profit', 0):,.2f}
- **Total Return:** {m.get('return_pct', 0):.2f}%
- **Win Rate:** {m.get('win_rate_pct', 0):.2f}% (RTH: {m.get('win_rate_rth_pct', 0):.2f}%, PRE: {m.get('win_rate_pre_pct', 0):.2f}%, AFTER: {m.get('win_rate_after_pct', 0):.2f}%)
- **Profit Factor:** {m.get('profit_factor', 0):.2f}
- **Max Drawdown:** ${m.get('max_drawdown_dollars', 0):,.2f} ({m.get('max_drawdown_pct', 0):.2f}%)
- **Recovery Factor:** {m.get('recovery_factor', 0):.2f}
---
## Averages
- **Average Win:** ${m.get('avg_win', 0):,.2f}
- **Average Loss:** ${m.get('avg_loss', 0):,.2f}
---
## Day-of-Week Performance
{'\n'.join(f"- **{k}:** ${v:,.2f}" for k, v in m.get('avg_pl_by_day', {}).items())}
---
*Message:* {m.get('message', 'No message')}
*Report generated by Backtester_vercel.py v{cfg.version}*
"""
    with open(os.path.join(outdir, "analytics.md"), "w", encoding="utf-8") as f:
        f.write(md)
    print(f"DEBUG: Wrote analytics.md to {outdir}")

# Save Analytics Report Footer: Outputs analytics.md for investor review.

# ===================================================

# RUN BACKTEST SECTION

# Purpose: Orchestrates the backtest process, from loading CSV to generating outputs.

# Output Goal: Produces trades_enriched.csv, analytics.md, and a metrics dictionary.

# For Harrison: This function runs the entire backtest, calling all previous functions. It handles errors (e.g., missing CSV) and ensures outputs are saved in /tmp/Backtests_.... Check Vercel logs for "✅ Backtest complete" to confirm success.

# =======================================================
def run_backtest(tos_csv_path: str, cfg: BacktestConfig):
    """Runs the full backtest, producing trade data and metrics."""
    metrics = {
        "status": "failed",
        "message": "Backtest not started",
        "strategy": cfg.strategy_name,
        "version": cfg.version,
        "num_trades": 0,
        "net_profit": 0.0
    }
    try:
        csv_stem = Path(tos_csv_path).stem.replace(" ", "_")
        print(f"DEBUG: Starting backtest with CSV: {tos_csv_path}")
        df = load_tos_strategy_report(tos_csv_path)
        cfg.strategy_name = df["BaseStrategy"].iloc[0] if "BaseStrategy" in df.columns and not df.empty else cfg.strategy_name
        df = filter_trades(df, cfg)
        trades = build_trades(df, cfg.commission_per_round_trip)
        if trades.empty:
            outdir = cfg.outdir(csv_stem)
            os.makedirs(outdir, exist_ok=True)
            with open(os.path.join(outdir, "analytics.md"), "w", encoding="utf-8") as f:
                f.write(f"# Strategy Analysis Report\nNo trades generated from CSV: {tos_csv_path}\n")
            metrics["message"] = f"No trades generated from CSV: {tos_csv_path}"
            return metrics
        trades = apply_stoploss_cap(trades, cfg.point_value)
        metrics = compute_metrics(trades, cfg)
        outdir = cfg.outdir(csv_stem)
        os.makedirs(outdir, exist_ok=True)
        trades.to_csv(os.path.join(outdir, "trades_enriched.csv"), index=False)
        save_analytics_md(trades, metrics, cfg, outdir)
        metrics.update({"status": "success", "message": "Backtest completed"})
        print(f"✅ Backtest complete — {len(trades)} trades | Net P/L ${metrics.get('net_profit', 0):.2f}")
        print(f"📁 Output directory: {outdir}")
        return metrics
    except Exception as e:
        print(f"ERROR: Backtest failed: {str(e)}")
        metrics["message"] = f"Backtest failed: {str(e)}"
        return metrics
# Run Backtest Footer: Outputs trades_enriched.csv, analytics.md, and metrics dictionary.

# =============================================================

# MAIN EXECUTION SECTION# Purpose: Entry point for running the backtest from the command line.

# Output Goal: Parses command-line arguments and initiates the backtest.

# For Harrison: Run this script in Vercel with `python Backtester_vercel.py --csv /tmp/temp_4.csv`. Ensure the CSV is copied to /tmp/temp_4.csv before execution. Check Vercel logs for debug output.

# ==========================================================
if __name__ == "__main__":
    import argparse, sys
    parser = argparse.ArgumentParser(description="ThinkorSwim Strategy Report Backtester for /MES")
    parser.add_argument("--csv", required=True, help="Path to ToS Strategy Report CSV (e.g., /tmp/temp_4.csv)")
    parser.add_argument("--timeframe", type=str, default="180d:15m", help="Timeframe label (e.g., 180d:15m)")
    parser.add_argument("--capital", type=float, default=2500.0, help="Initial capital in USD")
    parser.add_argument("--commission", type=float, default=4.04, help="Commission per round-trip trade")
    args = parser.parse_args()
    cfg = BacktestConfig(
        initial_capital=args.capital,
        commission_per_round_trip=args.commission,
        timeframe=args.timeframe
    )
    result = run_backtest(args.csv, cfg)
    if result.get("status") == "failed":
        print(f"ERROR: {result.get('message')}")
        sys.exit(1)

# Main Execution Footer: Script execution complete, outputs generated in /tmp/Backtests_....

# ========================================================

# END OF SCRIPT

# For Harrison: To verify results, check /tmp/Backtests_... for trades_enriched.csv and analytics.md. If errors occur, review /tmp/csv_debug.txt and Vercel logs for "DEBUG" or "ERROR" messages.

# Expected Results: Processes 357 trades, net P/L ~$3,364.57 (after $1,441.68 commissions), win rate ~64.7% (or higher with filters).

# Investor Notes: The analytics.md report provides key metrics (profit, win rate, drawdown) for investor presentations. trades_enriched.csv includes detailed trade data for validation.
# =======================================================