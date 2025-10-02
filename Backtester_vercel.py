# -*- coding: utf-8 -*-
# ==========================================================
# Backtester_vercel.py v1.6.1 (Stable, Full Inline Functions)
# ==========================================================

import os, io, re, json, argparse, glob, sys
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Tuple
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# =========================
# Config
# =========================
@dataclass
class BacktestConfig:
    strategy_name: str = ""
    instruments: Tuple[str, ...] = ("/MES",)
    timeframe: str = "180d:15m"
    initial_capital: float = 2500.0
    commission_per_round_trip: float = 4.04
    point_value: float = 5.0
    version: str = "1.6.1"
    algo_params: dict = None
    def __post_init__(self):
        if self.algo_params is None:
            self.algo_params = {"ATRFactor_Fixed": 2.2, "StopLossCap": 100.0}
    def outdir(self, csv_stem: str, instrument: str, strategy_label: str) -> str:
        temp_dir = Path("/tmp")
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
    return pd.to_numeric(s, errors="coerce")

def _parse_datetime(series: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(series, format="%m/%d/%y %I:%M %p", errors="coerce")
    if parsed.isna().all():
        parsed = pd.to_datetime(series, errors="coerce")
    return parsed

def _max_drawdown(equity_curve: pd.Series) -> float:
    running_max = equity_curve.cummax()
    drawdown = equity_curve / running_max - 1.0
    return float(drawdown.min())

def _profit_factor(pl: pd.Series) -> float:
    s = pl.dropna()
    gp = s[s > 0].sum()
    gl = -s[s < 0].sum()
    if gl == 0: return float("inf") if gp > 0 else 0.0
    return float(gp / gl)

def _exit_reason(text: str) -> str:
    if not text or pd.isna(text): return "Close"
    s = str(text).upper().strip()
    if any(w in s for w in ["TARGET","TGT","TP","PROFIT"]): return "Target"
    if any(w in s for w in ["STOP","LOSS LIMIT"]): return "Stop"
    if any(w in s for w in ["TIME","TIMED"]): return "Time"
    return "Close"

def normalize_symbol(sym: str) -> str:
    s = str(sym).strip()
    if not s: return "/UNK"
    if s.startswith("/"): return s.upper()
    return f"/{s.upper()}"

# =========================
# Load Strategy Report
# =========================
def load_tos_strategy_report(file_path: str, cfg: BacktestConfig) -> pd.DataFrame:
    with open(file_path, "r", errors="replace") as f:
        lines = f.readlines()
    start_idx = None
    for i, line in enumerate(lines):
        if line.lstrip().startswith("Id;Strategy;"):
            start_idx = i
            break
    if start_idx is None:
        raise ValueError("No header found in file.")
    table_str = "".join(lines[start_idx:])
    df = pd.read_csv(io.StringIO(table_str), sep=";")

    if "Date/Time" in df.columns:
        df["Date"] = _parse_datetime(df["Date/Time"])
    elif "Date" in df.columns and "Time" in df.columns:
        dt_str = df["Date"].astype(str)+" "+df["Time"].astype(str)
        df["Date"] = pd.to_datetime(dt_str, errors="coerce")
    elif "Date" in df.columns:
        df["Date"] = _parse_datetime(df["Date"])
    else:
        raise ValueError("No Date column found.")

    if "Trade P/L" in df.columns:
        df["TradePL"] = _to_float(df["Trade P/L"]).fillna(0.0)
    else:
        df["TradePL"] = 0.0

    df["BaseStrategy"] = cfg.strategy_name
    if "Symbol" in df.columns:
        df["Symbol"] = df["Symbol"].astype(str).map(normalize_symbol)
    else:
        df["Symbol"] = "/UNK"

    return df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)

# =========================
# Build Trades (patched to always include NetPL)
# =========================
def build_trades(df: pd.DataFrame, commission_rt: float):
    trades=[]
    i=0
    while i < len(df)-1:
        entry, exit_ = df.iloc[i], df.iloc[i+1]
        side_entry=str(entry.get("Side","")).upper()
        side_exit=str(exit_.get("Side","")).upper()
        if any(x in side_entry for x in ["BTO","STO"]) and any(x in side_exit for x in ["STC","BTC"]):
            qty_abs = 1.0
            direction = "Long" if "BTO" in side_entry else "Short"
            trade_pl = pd.to_numeric(exit_.get("TradePL"), errors="coerce")
            commission = commission_rt * qty_abs
            net_pl = (trade_pl if pd.notna(trade_pl) else 0.0)-commission
            trades.append({
                "EntryTime": entry["Date"],
                "ExitTime": exit_["Date"],
                "TradePL": trade_pl,
                "Commission": commission,
                "NetPL": net_pl,
                "ExitReason": _exit_reason(exit_.get("Side")),
                "Direction": direction,
                "QtyAbs": qty_abs
            })
            i+=2
        else:
            i+=1

    # ✅ Ensure consistent schema (always has NetPL column)
    if trades:
        t=pd.DataFrame(trades)
        t["HoldMins"]=(t["ExitTime"]-t["EntryTime"]).dt.total_seconds()/60.0
    else:
        t=pd.DataFrame(columns=["EntryTime","ExitTime","TradePL","Commission","NetPL",
                                "ExitReason","Direction","QtyAbs","HoldMins"])
    return t,0

# =========================
# Stop-Loss Adjustments
# =========================
def apply_stoploss_corrections(trades: pd.DataFrame, point_value: float):
    df=trades.copy()
    df["RawLossExceeds100"]=df["NetPL"]<-100.0
    df["AdjustedNetPL"]=np.where(df["NetPL"]<-100.0,-100.0,df["NetPL"])
    df["PointsPerContract"]=np.where(df["QtyAbs"]>0, df["AdjustedNetPL"]/(point_value*df["QtyAbs"]), 0.0)
    return df

# =========================
# Compute Metrics (Detailed)
# =========================
def compute_metrics(trades_df: pd.DataFrame, cfg: BacktestConfig, scope_label: str, non_rth_trades=0):
    if trades_df.empty:
        return {"scope":scope_label,"strategy_name":cfg.strategy_name,"num_trades":0}
    df=trades_df.copy()
    pl=df["AdjustedNetPL"]
    equity=cfg.initial_capital+pl.cumsum()
    wins, losses = pl[pl>0], pl[pl<0]
    return {
        "scope":scope_label,
        "strategy_name":cfg.strategy_name,
        "num_trades":len(df),
        "net_profit":float(pl.sum()),
        "profit_factor":_profit_factor(pl),
        "win_rate_pct":float((pl>0).mean()*100.0),
        "avg_win":float(wins.mean()) if not wins.empty else 0.0,
        "avg_loss":float(losses.mean()) if not losses.empty else 0.0,
        "expectancy":float(pl.mean()),
        "max_drawdown_pct":abs(_max_drawdown(equity))*100.0,
        "sharpe_ratio":float(pl.mean()/pl.std()) if pl.std()!=0 else 0.0,
        "stoploss_hits":int((df["RawLossExceeds100"]).sum()),
        "avg_hold_mins":float(df["HoldMins"].mean()) if "HoldMins" in df else 0.0
    }

# =========================
# Save Visuals
# =========================
def save_visuals_and_tables(trades_df: pd.DataFrame, cfg: BacktestConfig, outdir: str, title_suffix="ALL"):
    os.makedirs(outdir, exist_ok=True)
    pl=trades_df["AdjustedNetPL"].fillna(0.0)
    equity=cfg.initial_capital+pl.cumsum()
    plt.figure(figsize=(9,4)); plt.plot(equity.index,equity.values)
    plt.title("Equity Curve"); plt.grid(True)
    plt.savefig(os.path.join(outdir,f"equity_curve_{title_suffix}.png")); plt.close()

# =========================
# Markdown Report
# =========================
def generate_analytics_md(trades_all, trades_rth, metrics, cfg, non_rth_trades, outdir):
    md = [
        f"# Strategy Report ({cfg.version})",
        f"**Trades:** {metrics['num_trades']}",
        f"**Net Profit:** ${metrics['net_profit']:.2f}",
        f"**Win Rate:** {metrics['win_rate_pct']:.2f}%",
        f"**Profit Factor:** {metrics['profit_factor']:.2f}",
        f"**Expectancy:** {metrics['expectancy']:.2f}",
        f"**Max Drawdown:** {metrics['max_drawdown_pct']:.2f}%",
        f"**Sharpe Ratio:** {metrics['sharpe_ratio']:.2f}",
        f"**Stop-Loss Hits:** {metrics['stoploss_hits']}",
        f"**Avg Hold (mins):** {metrics['avg_hold_mins']:.1f}",
    ]
    with open(os.path.join(outdir,"analytics.md"),"w") as f: f.write("\n".join(md))

# =========================
# Run Backtest
# =========================
def run_backtest_for_instrument(df_raw,instrument,cfg,csv_stem):
    cfg.strategy_name=Path(csv_stem).stem
    instr=normalize_symbol(instrument or "/UNK")
    pv=5.0 if instr.upper()=="/MES" else 2.0
    cfg.point_value=pv
    outdir=cfg.outdir(csv_stem,instr,cfg.strategy_name)

    trades_all,_=build_trades(df_raw,cfg.commission_per_round_trip)
    trades_all=apply_stoploss_corrections(trades_all,cfg.point_value)
    metrics_all=compute_metrics(trades_all,cfg,"ALL")

    os.makedirs(outdir,exist_ok=True)
    trades_all.to_csv(os.path.join(outdir,"trades_enriched.csv"),index=False)
    with open(os.path.join(outdir,"metrics.json"),"w") as f: json.dump(metrics_all,f,indent=2)
    save_visuals_and_tables(trades_all,cfg,outdir)
    generate_analytics_md(trades_all,trades_all,metrics_all,cfg,0,outdir)
    with open(os.path.join(outdir,"config.json"),"w") as f: json.dump(asdict(cfg),f,indent=2)
    return trades_all,trades_all,metrics_all,outdir

def run_backtest(tos_csv_path,cfg):
    csv_stem=Path(tos_csv_path).stem.replace(" ","_")
    raw=load_tos_strategy_report(tos_csv_path,cfg)
    symbols=raw["Symbol"].dropna().unique().tolist() or ["/MES"]
    results=[]
    for instr in symbols:
        trades_all,trades_rth,metrics,outdir=run_backtest_for_instrument(raw,instr,cfg,csv_stem)
        results.append({"instrument":instr,"metrics":metrics,"outdir":outdir})
    return results

# =========================
# Main
# =========================
if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--csv",nargs="+",required=True)
    parser.add_argument("--timeframe",type=str,default="180d:15m")
    parser.add_argument("--capital",type=float,default=2500.0)
    parser.add_argument("--commission",type=float,default=4.04)
    parser.add_argument("--point_value",type=float,default=5.0)
    args=parser.parse_args()

    resolved=[]
    for item in args.csv:
        matches=glob.glob(item)
        resolved.extend(matches if matches else [item])
    csv_paths=sorted({str(Path(p)) for p in resolved if Path(p).exists()})
    if not csv_paths:
        print("[ERROR] No CSVs found",file=sys.stderr); sys.exit(1)

    cfg=BacktestConfig(timeframe=args.timeframe,initial_capital=args.capital,
                       commission_per_round_trip=args.commission,point_value=args.point_value)

    all_metrics=[]
    for csv_path in csv_paths:
        print(f"[RUN] {csv_path}")
        results=run_backtest(csv_path,cfg)
        for r in results:
            m=r["metrics"]; m["csv"]=Path(csv_path).name
            all_metrics.append(m)

    consolidated=Path("/tmp")/f"metrics_consolidated_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(consolidated,"w") as f: json.dump(all_metrics,f,indent=2)
    print(f"[DONE] {len(csv_paths)} CSV(s). Consolidated metrics at {consolidated}")
    sys.exit(0)

# =========================
# End of Backtester_vercel.py v1.6.1
# =========================
