# -*- coding: utf-8 -*-
# Bare-bones Backtester for Vercel
# - Ignores unknown CLI args (safe for --timeframe, etc.)
# - Loads ThinkOrSwim Strategy Report CSV
# - Pairs trades, applies $100 stop-loss cap
# - Outputs trades_enriched.csv + metrics.json
# - Prints metrics JSON

import os, io, re, json, glob, sys
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd

# =========================
# Config
# =========================

@dataclass
class BacktestConfig:
    initial_capital: float = 2500.0
    commission_per_round_trip: float = 4.04
    point_value: float = 5.0
    version: str = "barebones-1.0"

    def outdir(self, csv_stem: str) -> str:
        temp_dir = Path("/tmp")
        day = datetime.now().strftime("%Y-%m-%d")
        ts = datetime.now().strftime("%H%M%S_%f")[:-3]
        return str(temp_dir / f"Backtest_{day}_{csv_stem}_{ts}")


# =========================
# Helpers
# =========================

def _to_float(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.replace(r"[\$,]", "", regex=True)
    s = s.str.replace(r"\(([^()]*)\)", r"-\1", regex=True)
    return pd.to_numeric(s, errors="coerce")

def _parse_datetime(series: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(series, format="%m/%d/%y %I:%M %p", errors="coerce")
    if parsed.isna().all():
        parsed = pd.to_datetime(series, errors="coerce")
    return parsed

def _max_drawdown(equity_curve: pd.Series) -> float:
    running_max = equity_curve.cummax()
    dd = equity_curve / running_max - 1.0
    return float(dd.min())

def _profit_factor(pl: pd.Series) -> float:
    s = pl.dropna()
    gp = s[s > 0].sum()
    gl = -s[s < 0].sum()
    if gl == 0:
        return float("inf") if gp > 0 else 0.0
    return float(gp / gl)


# =========================
# Core
# =========================

def load_tos_strategy_report(file_path: str) -> pd.DataFrame:
    with open(file_path, "r", errors="replace") as f:
        lines = f.readlines()
    start_idx = None
    for i, line in enumerate(lines):
        if line.lstrip().startswith("Id;Strategy;"):
            start_idx = i
            break
    if start_idx is None:
        raise ValueError("No trade table header found in file")

    table_str = "".join(lines[start_idx:])
    df = pd.read_csv(io.StringIO(table_str), sep=";")

    if "Date/Time" in df.columns:
        df["Date"] = _parse_datetime(df["Date/Time"])
    elif "Date" in df.columns and "Time" in df.columns:
        dt_str = df["Date"].astype(str).str.strip() + " " + df["Time"].astype(str).str.strip()
        df["Date"] = pd.to_datetime(dt_str, errors="coerce")
    elif "Date" in df.columns:
        df["Date"] = _parse_datetime(df["Date"])
    else:
        raise ValueError("Could not parse date/time columns")

    if "Trade P/L" in df.columns:
        df["TradePL"] = _to_float(df["Trade P/L"]).fillna(0.0)
    elif "TradePL" in df.columns:
        df["TradePL"] = _to_float(df["TradePL"]).fillna(0.0)
    else:
        df["TradePL"] = 0.0

    return df


def build_trades(df: pd.DataFrame, commission_rt: float) -> pd.DataFrame:
    trades = []
    for i in range(0, len(df) - 1, 2):
        entry = df.iloc[i]
        exit_ = df.iloc[i + 1]
        qty_abs = 1.0
        trade_pl = entry.get("TradePL", 0.0)
        commission = commission_rt * qty_abs
        net_pl = (trade_pl if pd.notna(trade_pl) else 0.0) - commission
        trades.append({
            "EntryTime": entry["Date"],
            "ExitTime": exit_["Date"],
            "TradePL": trade_pl,
            "Commission": commission,
            "NetPL": net_pl,
        })
    t = pd.DataFrame(trades)
    if not t.empty:
        t["HoldMins"] = (t["ExitTime"] - t["EntryTime"]).dt.total_seconds() / 60.0
    return t


def apply_stoploss(trades: pd.DataFrame) -> pd.DataFrame:
    df = trades.copy()
    df["SLBreached"] = df["NetPL"] < -100.0
    df["AdjustedNetPL"] = np.where(df["SLBreached"], -100.0, df["NetPL"])
    return df


def compute_metrics(trades_df: pd.DataFrame, cfg: BacktestConfig) -> dict:
    pl_net = trades_df["AdjustedNetPL"].fillna(0.0)
    equity = cfg.initial_capital + pl_net.cumsum()
    return {
        "net_profit": float(pl_net.sum()),
        "num_trades": int(len(trades_df)),
        "profit_factor": _profit_factor(pl_net),
        "max_drawdown": _max_drawdown(equity),
    }


# =========================
# Runner
# =========================

def run_backtest(csv_path: str, cfg: BacktestConfig):
    raw = load_tos_strategy_report(csv_path)
    trades = build_trades(raw, cfg.commission_per_round_trip)
    trades = apply_stoploss(trades)
    metrics = compute_metrics(trades, cfg)

    outdir = cfg.outdir(Path(csv_path).stem)
    os.makedirs(outdir, exist_ok=True)

    trades.to_csv(os.path.join(outdir, "trades_enriched.csv"), index=False)
    with open(os.path.join(outdir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics, outdir


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Bare-bones backtester (Vercel safe)")
    parser.add_argument("--csv", nargs="+", required=True, help="CSV file(s)")
    parser.add_argument("--capital", type=float, default=2500.0, help="Initial capital")
    parser.add_argument("--commission", type=float, default=4.04, help="Commission per round trip")
    parser.add_argument("--point_value", type=float, default=5.0, help="Point value per contract")

    # 👇 ignore unknown args like --timeframe
    args, unknown = parser.parse_known_args()
    if unknown:
        print(f"[WARN] Ignoring unknown arguments: {unknown}")

    cfg = BacktestConfig(
        initial_capital=args.capital,
        commission_per_round_trip=args.commission,
        point_value=args.point_value,
    )

    all_metrics = []
    for csv_file in args.csv:
        matches = glob.glob(csv_file)
        for path in matches:
            metrics, outdir = run_backtest(path, cfg)
            metrics["csv"] = os.path.basename(path)
            metrics["outdir"] = outdir
            all_metrics.append(metrics)

    print(json.dumps(all_metrics, indent=2))
