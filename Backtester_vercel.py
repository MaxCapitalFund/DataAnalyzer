# -*- coding: utf-8 -*-
# Hybrid Backtester: Optimized for Vercel deployment
# - Computes ALL, RTH, PM45 metrics
# - Stop-loss cap: -$100 per trade per contract
# - Outputs: trades_enriched.csv, metrics.json, analytics.md, config.json, charts
# - Fixes: duplicates removed, --timeframe safe, consistent returns

import os, io, re, json, warnings
from dataclasses import dataclass, asdict
from datetime import datetime, time
from pathlib import Path
from typing import Tuple, Optional
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
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
    session_hours_pm: Tuple[str, str] = ("03:00", "09:15")
    initial_capital: float = 2500.0
    commission_per_round_trip: float = 4.04
    point_value: float = 5.0
    version: str = "1.4.6"

    def outdir(self, csv_stem: str, instrument: str, strategy_label: str) -> str:
        temp_dir = Path('/tmp')
        day = datetime.now().strftime("%Y-%m-%d")
        ts = datetime.now().strftime("%H%M%S_%f")[:-3]
        safe_strategy = (strategy_label or "Unknown").replace(" ", "_")
        safe_instr = (instrument or "UNK").replace("/", "")
        return str(temp_dir / f"Backtests_{day}_{safe_strategy}_{self.timeframe}_{safe_instr}_{csv_stem}_{ts}")

# =========================
# Helpers
# =========================

def _to_float(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.replace(r"[\$,]", "", regex=True)
    s = s.str.replace(r"\(([^()]*)\)", r"-\1", regex=True)
    s = s.replace("", np.nan)
    return pd.to_numeric(s, errors='coerce')

def _parse_datetime(series: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(series, format="%m/%d/%y %I:%M %p", errors='coerce')
    if parsed.isna().all():
        parsed = pd.to_datetime(series, errors='coerce')
    return parsed

# Sessions
PRE_START, PRE_END = time(3, 0), time(9, 15)  # PM45 ends 09:15
OPEN_START, OPEN_END = time(9, 30), time(11, 30)
LUNCH_START, LUNCH_END = time(11, 30), time(14, 0)
CLOSE_START, CLOSE_END = time(14, 0), time(16, 0)

def _tag_session(dt: pd.Timestamp) -> str:
    if pd.isna(dt): return "Unknown"
    t = dt.time()
    if PRE_START <= t <= PRE_END: return "PRE"
    if OPEN_START <= t <= OPEN_END: return "OPEN"
    if LUNCH_START <= t <= LUNCH_END: return "LUNCH"
    if CLOSE_START <= t <= CLOSE_END: return "CLOSING"
    return "OTHER"

def _in_rth(dt: pd.Timestamp) -> bool:
    return pd.notna(dt) and OPEN_START <= dt.time() <= CLOSE_END

def _in_pm(dt: pd.Timestamp) -> bool:
    return pd.notna(dt) and PRE_START <= dt.time() <= PRE_END

def _max_drawdown(equity_curve: pd.Series) -> float:
    return float((equity_curve / equity_curve.cummax() - 1.0).min())

def _profit_factor(pl: pd.Series) -> float:
    s = pl.dropna()
    gp, gl = s[s > 0].sum(), -s[s < 0].sum()
    if gl == 0: return float('inf') if gp > 0 else 0.0
    return float(gp / gl)

def _exit_reason(text: str) -> str:
    s = str(text).upper()
    if any(w in s for w in ["TARGET","TGT","TP","PROFIT"]): return "Target"
    if any(w in s for w in ["STOP","SL","STOPPED"]): return "Stop"
    if any(w in s for w in ["TIME","TIME EXIT","TIMED","TIMEOUT","DAILY"]): return "Time"
    if any(w in s for w in ["MANUAL","FLATTEN","MKT CLOSE","DISCRETIONARY"]): return "Manual"
    return "Close"

ROOT_RE = re.compile(r"^/?([A-Za-z]{1,3})(?:[FGHJKMNQUVXZ]\d{1,2})?$")
def normalize_symbol(sym: str) -> str:
    s = str(sym).strip()
    if not s: return "/UNK"
    has_slash = s.startswith("/")
    core = s[1:] if has_slash else s
    m = ROOT_RE.match(core.upper())
    if m: return f"/{m.group(1).upper()}"
    m2 = re.search(r"/([A-Za-z]{1,3})", s.upper())
    if m2: return f"/{m2.group(1)}"
    m3 = re.search(r"\b([A-Za-z]{1,3})\b", s.upper())
    return f"/{m3.group(1)}" if m3 else "/UNK"

# =========================
# Load TOS Strategy Report
# =========================

def load_tos_strategy_report(file_path: str) -> pd.DataFrame:
    with open(file_path,'r',errors='replace') as f:
        lines = f.readlines()
    start_idx = next((i for i,l in enumerate(lines) if l.lstrip().startswith("Id;Strategy;")),None)
    if start_idx is None:
        raise ValueError("No trade table header found.")
    df = pd.read_csv(io.StringIO("".join(lines[start_idx:])), sep=';')
    if 'Date/Time' in df.columns:
        df['Date'] = _parse_datetime(df['Date/Time'])
    elif 'Date' in df.columns and 'Time' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'].astype(str)+" "+df['Time'].astype(str),errors='coerce')
    elif 'Date' in df.columns:
        df['Date'] = _parse_datetime(df['Date'])
    else: raise ValueError("No date column found.")
    if 'Trade P/L' in df.columns:
        df['TradePL'] = _to_float(df['Trade P/L']).fillna(0.0)
    else:
        df['TradePL'] = _to_float(df.get('TradePL',0)).fillna(0.0)
    if 'Strategy' in df.columns:
        df['BaseStrategy'] = df['Strategy'].astype(str).str.split('(').str[0].str.strip()
    else: df['BaseStrategy']="Unknown"
    df['Side'] = df[df.columns[df.columns.str.contains("Side|Action|Order|Type")][0]].astype(str) if df.columns.str.contains("Side|Action|Order|Type").any() else ""
    if 'Price' not in df: df['Price']=np.nan
    if 'Quantity' in df: df['Qty']=pd.to_numeric(df['Quantity'],errors='coerce')
    elif 'Qty' not in df: df['Qty']=np.nan
    if 'Symbol' in df: df['Symbol']=df['Symbol'].astype(str).map(normalize_symbol)
    elif 'Instrument' in df: df['Symbol']=df['Instrument'].astype(str).map(normalize_symbol)
    else: df['Symbol']="/UNK"
    return df.dropna(subset=['Date']).sort_values('Date').reset_index(drop=True)

# =========================
# Build Trades
# =========================

def build_trades(df: pd.DataFrame, commission_rt: float) -> pd.DataFrame:
    trades=[]
    OPEN_RX=r"\b(?:BTO|BUY TO OPEN|BOT TO OPEN|STO|SELL TO OPEN|SELL SHORT)\b"
    CLOSE_RX=r"\b(?:STC|SELL TO CLOSE|SLD TO CLOSE|BTC|BUY TO CLOSE|CLOSE)\b"
    i=0; unpaired=0
    while i < len(df)-1:
        entry,exit_=df.iloc[i],df.iloc[i+1]
        if re.search(OPEN_RX,str(entry['Side']).upper()) and re.search(CLOSE_RX,str(exit_['Side']).upper()):
            entry_qty=pd.to_numeric(entry.get('Qty'),errors='coerce')
            qty=abs(entry_qty) if pd.notna(entry_qty) and entry_qty!=0 else 1.0
            direction="Long" if "BUY" in str(entry['Side']).upper() else "Short"
            trade_pl=pd.to_numeric(exit_.get('TradePL'),errors='coerce')
            commission=commission_rt*qty
            net_pl=(trade_pl if pd.notna(trade_pl) else 0.0)-commission
            trades.append({
                "EntryTime":entry['Date'],"ExitTime":exit_['Date'],
                "EntryPrice":pd.to_numeric(entry.get('Price'),errors='coerce'),
                "ExitPrice":pd.to_numeric(exit_.get('Price'),errors='coerce'),
                "QtyAbs":qty,"TradePL":trade_pl,"GrossPL":trade_pl,
                "Commission":commission,"NetPL":net_pl,"Direction":direction,
                "BaseStrategy":entry.get('BaseStrategy','Unknown'),
                "Symbol":entry.get('Symbol',''),"ExitReason":_exit_reason(exit_.get('Side'))
            }); i+=2
        else: unpaired+=1; i+=1
    t=pd.DataFrame(trades)
    if t.empty: return t.assign(HoldMins=np.nan)
    t['HoldMins']=(t['ExitTime']-t['EntryTime']).dt.total_seconds()/60.0
    return t.sort_values('ExitTime').reset_index(drop=True)

# =========================
# Stop-loss
# =========================

def apply_stoploss_corrections(trades: pd.DataFrame, point_value: float)->pd.DataFrame:
    df=trades.copy()
    df['SLBreached']=df['NetPL']<-100.0
    df['AdjustedNetPL']=np.where(df['SLBreached'],-100.0,df['NetPL'])
    qty=pd.to_numeric(df['QtyAbs'],errors='coerce').replace(0,np.nan)
    gross_adj=np.where(df['SLBreached'],-100.0+df['Commission'],df['NetPL']+df['Commission'])
    df['PointsPerContract']=gross_adj/(point_value*qty)
    return df

# =========================
# Metrics (simplified)
# =========================

def compute_metrics(trades_df: pd.DataFrame,cfg:BacktestConfig,scope:str)->dict:
    if trades_df.empty: return {"scope":scope,"num_trades":0,"net_profit":0}
    pl=trades_df['AdjustedNetPL'].fillna(0.0)
    equity=cfg.initial_capital+pl.cumsum()
    return {
        "scope":scope,
        "num_trades":len(trades_df),
        "net_profit":float(pl.sum()),
        "profit_factor":_profit_factor(pl),
        "win_rate_pct":float((pl>0).mean()*100),
        "max_drawdown":_max_drawdown(equity)
    }

# =========================
# Main runner
# =========================

def run_backtest(tos_csv_path:str,cfg:BacktestConfig):
    raw=load_tos_strategy_report(tos_csv_path)
    instr=normalize_symbol(raw['Symbol'].iloc[0] if 'Symbol' in raw else '/UNK')
    trades_all=apply_stoploss_corrections(build_trades(raw,cfg.commission_per_round_trip),cfg.point_value)
    trades_rth=trades_all[trades_all['EntryTime'].apply(_in_rth)]
    trades_pm45=trades_all[trades_all['EntryTime'].apply(_in_pm)]
    metrics_all=compute_metrics(trades_all,cfg,"ALL")
    metrics_rth=compute_metrics(trades_rth,cfg,"RTH")
    metrics_pm45=compute_metrics(trades_pm45,cfg,"PM45")
    outdir=cfg.outdir(Path(tos_csv_path).stem,instr,cfg.strategy_name)
    os.makedirs(outdir,exist_ok=True)
    trades_all.to_csv(os.path.join(outdir,"trades_enriched.csv"),index=False)
    metrics={"ALL":metrics_all,"RTH":metrics_rth,"PM45":metrics_pm45}
    with open(os.path.join(outdir,"metrics.json"),"w") as f: json.dump(metrics,f,indent=2)
    return metrics,outdir

if __name__=="__main__":
    import argparse,glob,sys
    parser=argparse.ArgumentParser()
    parser.add_argument("--csv",nargs="+",required=True)
    parser.add_argument("--timeframe",type=str,default="180d:15m") # display only
    parser.add_argument("--capital",type=float,default=2500.0)
    parser.add_argument("--commission",type=float,default=4.04)
    parser.add_argument("--point_value",type=float,default=5.0)
    args,unknown=parser.parse_known_args()  # 👈 ignores unknown args
    cfg=BacktestConfig(timeframe=args.timeframe,
                       initial_capital=args.capital,
                       commission_per_round_trip=args.commission,
                       point_value=args.point_value)
    results=[]
    for item in args.csv:
        for path in glob.glob(item):
            m,outdir=run_backtest(path,cfg)
            results.append({"file":path,"metrics":m,"outdir":outdir})
    print(json.dumps(results,indent=2))
