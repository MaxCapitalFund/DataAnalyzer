# -*- coding: utf-8 -*-
# Backtester_vercel.py
# Lean backtester for TOS Strategy Report (serverless-friendly)
# Outputs trades.csv, metrics.json, analytics.md
# - Stop-loss cap: fixed -$100 per trade per contract
# - Commissions tracked (reported only, not deducted from NetPL)
# - No charts

import os, io, re, json, warnings
from dataclasses import dataclass, asdict
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
    strategy_name: str = ""
    instruments: Tuple[str,...] = ("/MES",)
    timeframe: str = "180d:15m"
    session_hours_rth: Tuple[str,str] = ("09:30","16:00")
    initial_capital: float = 2500.0
    commission_per_round_trip: float = 4.04
    version: str = "1.4.1"

    def outdir(self, csv_stem: str, instrument: str, strategy_label: str) -> str:
        temp_dir = Path("/tmp")
        day = datetime.now().strftime("%Y-%m-%d")
        timestamp = datetime.now().strftime("%H%M%S_%f")[:-3]
        safe_strategy = (strategy_label or "Unknown").replace(" ","_")
        safe_instr = (instrument or "UNK").replace("/","")
        return str(temp_dir / f"Backtests_{day}_{safe_strategy}_{self.timeframe}_{safe_instr}_{csv_stem}_{timestamp}")

# =========================
# Helpers
# =========================

def _to_float(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.replace(r"[\$,]","",regex=True)
    s = s.str.replace(r"\(([^()]*)\)",r"-\1",regex=True)
    return pd.to_numeric(s,errors="coerce")

def _parse_datetime(series: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(series,format="%m/%d/%y %I:%M %p",errors="coerce")
    if parsed.isna().all():
        parsed = pd.to_datetime(series,errors="coerce")
    return parsed

PRE_START, PRE_END = time(3,0), time(9,29)
OPEN_START, OPEN_END = time(9,30), time(11,30)
LUNCH_START, LUNCH_END = time(11,30), time(14,0)
CLOSE_START, CLOSE_END = time(14,0), time(16,0)

def _tag_session(dt: pd.Timestamp) -> str:
    if pd.isna(dt): return "Unknown"
    t = dt.time()
    if PRE_START <= t <= PRE_END: return "PRE"
    if OPEN_START <= t <= OPEN_END: return "OPEN"
    if LUNCH_START <= t <= LUNCH_END: return "LUNCH"
    if CLOSE_START <= t <= CLOSE_END: return "CLOSING"
    return "OTHER"

def _exit_reason(text: str) -> str:
    s = str(text).upper()
    if any(w in s for w in ["TARGET","TGT","TP","PROFIT"]): return "Target"
    if any(w in s for w in ["STOP","SL","STOPPED"]): return "Stop"
    if any(w in s for w in ["TIME","TIMED","TIME EXIT"]): return "Time"
    if any(w in s for w in ["MANUAL","MKT CLOSE","FLATTEN"]): return "Manual"
    return "Close"

# =========================
# Load
# =========================

def load_tos_strategy_report(file_path: str) -> pd.DataFrame:
    with open(file_path,"r",errors="replace") as f:
        lines=f.readlines()
    start_idx=None
    for i,line in enumerate(lines):
        if line.lstrip().startswith("Id;Strategy;"): start_idx=i; break
    if start_idx is None: raise ValueError("No trade table header found.")
    df=pd.read_csv(io.StringIO("".join(lines[start_idx:])),sep=";")
    if "Date/Time" in df.columns: df["Date"]=_parse_datetime(df["Date/Time"])
    elif "Date" in df.columns and "Time" in df.columns:
        df["Date"]=pd.to_datetime(df["Date"].astype(str)+" "+df["Time"].astype(str),errors="coerce")
    else: df["Date"]=_parse_datetime(df["Date"])
    df["TradePL"]=_to_float(df.get("Trade P/L",df.get("TradePL",0.0))).fillna(0.0)
    df["Side"]=df.get("Side","")
    df["Price"]=df.get("Price",np.nan)
    df["Qty"]=pd.to_numeric(df.get("Quantity",df.get("Qty",np.nan)),errors="coerce")
    df=df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)
    return df

# =========================
# Trades
# =========================

def build_trades(df: pd.DataFrame, commission_rt: float) -> pd.DataFrame:
    trades=[]; i=0
    OPEN_RX=r"\b(BTO|STO|OPEN)\b"; CLOSE_RX=r"\b(STC|BTC|CLOSE)\b"
    while i < len(df)-1:
        e,x=df.iloc[i],df.iloc[i+1]
        if re.search(OPEN_RX,str(e["Side"]).upper()) and re.search(CLOSE_RX,str(x["Side"]).upper()):
            qty=abs(float(e.get("Qty") or 1))
            direction="Long" if "BTO" in str(e["Side"]).upper() else "Short"
            trades.append({
                "EntryTime":e["Date"],"ExitTime":x["Date"],
                "QtyAbs":qty,"GrossPL":x.get("TradePL"),
                "Commission":commission_rt*qty,"NetPL":x.get("TradePL"),
                "Direction":direction,"ExitReason":_exit_reason(x.get("Side"))
            }); i+=2
        else: i+=1
    t=pd.DataFrame(trades)
    if not t.empty: t["HoldMins"]=(t["ExitTime"]-t["EntryTime"]).dt.total_seconds()/60.0
    return t

# =========================
# Stoploss (fixed -$100)
# =========================

def apply_stoploss_corrections(trades: pd.DataFrame) -> pd.DataFrame:
    df=trades.copy()
    cap=-100.0*df["QtyAbs"].fillna(1)
    df["AdjustedNetPL"]=np.where(df["NetPL"]<cap,cap,df["NetPL"])
    return df

# =========================
# Metrics + Analytics
# =========================

def compute_metrics(trades: pd.DataFrame, cfg: BacktestConfig) -> dict:
    if trades.empty: return {}
    pl,gross=trades["AdjustedNetPL"],trades["GrossPL"]
    metrics={
        "strategy_name":cfg.strategy_name,"num_trades":len(trades),
        "net_profit":float(pl.sum()),"gross_profit":float(gross.sum()),
        "commissions_total":float(trades["Commission"].sum()),
        "win_rate_pct":(pl>0).mean()*100,
        "bto_count":(trades["Direction"]=="Long").sum(),
        "sto_count":(trades["Direction"]=="Short").sum(),
        "bto_net_pl":trades.loc[trades["Direction"]=="Long","AdjustedNetPL"].sum(),
        "sto_net_pl":trades.loc[trades["Direction"]=="Short","AdjustedNetPL"].sum()
    }
    return metrics

def generate_analytics_md(trades: pd.DataFrame, metrics: dict, cfg: BacktestConfig, outdir: str):
    if trades.empty: return
    md=["# Strategy Analysis Report",
        f"**Strategy:** {metrics.get('strategy_name','')}",
        f"**Run Date:** {datetime.now().strftime('%Y-%m-%d')}",
        f"**Trades:** {metrics.get('num_trades',0)}",
        "---",
        "## Key Performance",
        f"- Net Profit: ${metrics.get('net_profit',0):.2f}",
        f"- Gross Profit: ${metrics.get('gross_profit',0):.2f}",
        f"- Total Commissions: ${metrics.get('commissions_total',0):.2f}",
        f"- Win Rate: {metrics.get('win_rate_pct',0):.2f}%"
    ]
    md+=["---","## Entry Breakdown",
         f"- BTO Trades: {metrics.get('bto_count',0)} | Net ${metrics.get('bto_net_pl',0):.2f}",
         f"- STO Trades: {metrics.get('sto_count',0)} | Net ${metrics.get('sto_net_pl',0):.2f}"]
    # Hold time
    hold=trades["HoldMins"].dropna()
    if not hold.empty:
        md+=["---","## Holding Time",
             f"- Average Hold: {hold.mean():.1f} min",
             f"- Median Hold: {hold.median():.1f} min",
             f"- Longest Hold: {hold.max():.1f} min",
             f"- Shortest Hold: {hold.min():.1f} min"]
    # Day of Week
    trades["Day"]=pd.to_datetime(trades["ExitTime"]).dt.day_name()
    dow=trades.groupby("Day")["AdjustedNetPL"].agg(["count","sum","mean"])
    if not dow.empty:
        md+=["---","## Day of Week"]
        for d,r in dow.iterrows():
            md.append(f"- {d}: {int(r['count'])} trades | Net ${r['sum']:.2f} | Exp ${r['mean']:.2f}")
    # Session
    trades["Session"]=trades["EntryTime"].apply(_tag_session)
    sess=trades.groupby("Session")["AdjustedNetPL"].agg(["count","sum","mean"])
    if not sess.empty:
        md+=["---","## Sessions"]
        for s,r in sess.iterrows():
            md.append(f"- {s}: {int(r['count'])} trades | Net ${r['sum']:.2f} | Exp ${r['mean']:.2f}")
    # Exit Reasons
    ex=trades.groupby("ExitReason")["AdjustedNetPL"].agg(["count","mean"])
    if not ex.empty:
        md+=["---","## Exit Reasons"]
        for e,r in ex.iterrows():
            md.append(f"- {e}: {int(r['count'])} trades | Exp ${r['mean']:.2f}")
    with open(os.path.join(outdir,"analytics.md"),"w") as f: f.write("\n".join(md))

# =========================
# Runner
# =========================

def run_backtest(tos_csv_path: str, cfg: BacktestConfig):
    raw=load_tos_strategy_report(tos_csv_path)
    trades=build_trades(raw,cfg.commission_per_round_trip)
    trades=apply_stoploss_corrections(trades)
    metrics=compute_metrics(trades,cfg)
    outdir=cfg.outdir(Path(tos_csv_path).stem,"/MES",cfg.strategy_name)
    os.makedirs(outdir,exist_ok=True)
    trades.to_csv(os.path.join(outdir,"trades.csv"),index=False)
    with open(os.path.join(outdir,"metrics.json"),"w") as f: json.dump(metrics,f,indent=2)
    generate_analytics_md(trades,metrics,cfg,outdir)
    return metrics

if __name__=="__main__":
    import argparse,glob,sys
    p=argparse.ArgumentParser()
    p.add_argument("--csv",nargs="+",required=True)
    a=p.parse_args()
    cfg=BacktestConfig()
    for path in a.csv:
        for f in glob.glob(path):
            print(run_backtest(f,cfg))

# === End of Backtester_vercel.py ===
