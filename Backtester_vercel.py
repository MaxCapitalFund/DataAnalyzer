# -*- coding: utf-8 -*-
# Clean Backtester for Vercel deployment
# Simplified but complete: trade pairing, $100 stop cap, full metrics

import os, io, re, json, glob, sys, warnings
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import pandas as pd

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
    version: str = "2.0.0"

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

def _exit_reason(text: str) -> str:
    s = str(text).upper()
    if any(w in s for w in ["TARGET","TGT","TP","PROFIT"]): return "Target"
    if any(w in s for w in ["STOP","SL","STOPPED"]): return "Stop"
    if any(w in s for w in ["TIME","TIME EXIT","TIMED","TIMEOUT","DAILY"]): return "Time"
    return "Close"

ROOT_RE = re.compile(r"^/?([A-Za-z]{1,3})")

def normalize_symbol(sym: str) -> str:
    s = str(sym).strip()
    if not s: return "/UNK"
    m = ROOT_RE.match(s.upper().lstrip("/"))
    return f"/{m.group(1)}" if m else "/UNK"

def _max_drawdown(equity_curve: pd.Series) -> float:
    if equity_curve.empty: return 0.0
    running_max = equity_curve.cummax()
    drawdown = equity_curve / running_max - 1.0
    return float(drawdown.min())

# =========================
# Load & Clean
# =========================
def load_tos_strategy_report(file_path: str) -> pd.DataFrame:
    with open(file_path, 'r', errors='replace') as f:
        lines = f.readlines()
    start_idx = None
    for i, line in enumerate(lines):
        if line.lstrip().startswith("Id;Strategy;"):
            start_idx = i; break
    if start_idx is None:
        raise ValueError("No trade table header found in file.")
    df = pd.read_csv(io.StringIO("".join(lines[start_idx:])), sep=';')

    if 'Date/Time' in df.columns:
        df['Date'] = _parse_datetime(df['Date/Time'])
    elif 'Date' in df.columns and 'Time' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'].astype(str)+" "+df['Time'].astype(str), errors='coerce')
    elif 'Date' in df.columns:
        df['Date'] = _parse_datetime(df['Date'])
    else:
        raise ValueError("No Date column found.")

    if 'Trade P/L' in df.columns:
        df['TradePL'] = _to_float(df['Trade P/L']).fillna(0.0)
    else:
        df['TradePL'] = 0.0

    df['CumPL'] = _to_float(df['P/L']) if 'P/L' in df.columns else np.nan
    if 'Strategy' in df.columns:
        df['BaseStrategy'] = df['Strategy'].astype(str).str.split('(').str[0].str.strip()
    else:
        df['BaseStrategy'] = "Unknown"

    side_col = None
    for cand in ['Side','Action','Order','Type']:
        if cand in df.columns: side_col = cand; break
    df['Side'] = df[side_col].astype(str) if side_col else ""
    if 'Price' not in df.columns: df['Price'] = np.nan
    if 'Qty' not in df.columns:
        qcol = 'Quantity' if 'Quantity' in df.columns else None
        df['Qty'] = pd.to_numeric(df[qcol], errors='coerce') if qcol else np.nan
    if 'Symbol' in df.columns:
        df['Symbol'] = df['Symbol'].astype(str).map(normalize_symbol)
    else:
        df['Symbol'] = "/MES"
    return df.dropna(subset=['Date']).sort_values('Date').reset_index(drop=True)

# =========================
# Build Trades
# =========================
def build_trades(df: pd.DataFrame, commission_rt: float) -> pd.DataFrame:
    trades = []
    def _safe(x): return pd.to_numeric(x, errors='coerce')

    OPEN_RX = r"\b(BTO|BUY TO OPEN|STO|SELL TO OPEN|SELL SHORT)\b"
    CLOSE_RX = r"\b(STC|SELL TO CLOSE|BTC|BUY TO CLOSE|CLOSE)\b"

    i=0
    while i < len(df)-1:
        entry, exit_ = df.iloc[i], df.iloc[i+1]
        if re.search(OPEN_RX,str(entry['Side']).upper()) and re.search(CLOSE_RX,str(exit_['Side']).upper()):
            entry_qty = _safe(entry.get('Qty')); qty_abs = abs(entry_qty) if pd.notna(entry_qty) and entry_qty!=0 else 1.0
            direction = 'Long' if 'BUY' in str(entry['Side']).upper() else 'Short'
            trade_pl = _safe(exit_.get('TradePL')); comm = commission_rt*qty_abs
            net_pl = (trade_pl if pd.notna(trade_pl) else 0.0) - comm
            trades.append({
                'EntryTime': entry['Date'],
                'ExitTime': exit_['Date'],
                'EntryPrice': _safe(entry.get('Price')),
                'ExitPrice': _safe(exit_.get('Price')),
                'QtyAbs': qty_abs,
                'TradePL': trade_pl,
                'Commission': comm,
                'NetPL': net_pl,
                'BaseStrategy': entry.get('BaseStrategy','Unknown'),
                'Symbol': entry.get('Symbol','/UNK'),
                'ExitReason': _exit_reason(exit_.get('Side')),
                'Direction': direction
            }); i+=2
        else: i+=1
    t=pd.DataFrame(trades)
    if t.empty: return t.assign(HoldMins=np.nan)
    t['HoldMins']=(t['ExitTime']-t['EntryTime']).dt.total_seconds()/60.0
    return t

# =========================
# Stop-loss
# =========================
def apply_stoploss_corrections(trades: pd.DataFrame, point_value: float) -> pd.DataFrame:
    df=trades.copy()
    df['SLBreached']=df['NetPL']<-100.0
    df['AdjustedNetPL']=np.where(df['SLBreached'],-100.0,df['NetPL'])
    qty_abs=pd.to_numeric(df['QtyAbs'],errors='coerce').replace(0,np.nan)
    gross_adj=np.where(df['SLBreached'],-100.0+df['Commission'],df['NetPL']+df['Commission'])
    df['PointsPerContract']=gross_adj/(point_value*qty_abs)
    return df

# =========================
# Metrics
# =========================
def compute_metrics(trades_df: pd.DataFrame, cfg: BacktestConfig) -> dict:
    if trades_df.empty: return {"num_trades":0}
    pl=trades_df['AdjustedNetPL'].fillna(0.0)
    equity=cfg.initial_capital+pl.cumsum()

    gross_profit=float(pl[pl>0].sum())
    gross_loss=float(pl[pl<0].sum())
    avg_win=float(pl[pl>0].mean()) if (pl>0).any() else 0.0
    avg_loss=float(pl[pl<0].mean()) if (pl<0).any() else 0.0
    expectancy=float(pl.mean()) if len(pl) else 0.0
    largest_win=float(pl.max()) if len(pl) else 0.0
    largest_loss=float(pl.min()) if len(pl) else 0.0
    net_profit=float(pl.sum())
    win_rate=float((pl>0).mean()*100)
    profit_factor=(gross_profit/abs(gross_loss)) if gross_loss!=0 else float('inf')
    max_dd=abs(_max_drawdown(equity))*100
    recov=net_profit/((equity.cummax()-equity).max() or 1)

    # Sharpe
    rets=pl/cfg.initial_capital
    if len(rets)>1 and rets.std(ddof=1)>0:
        sharpe=rets.mean()/rets.std(ddof=1)*np.sqrt(len(rets))
    else: sharpe=np.nan

    return {
        "num_trades":len(pl),
        "net_profit":net_profit,
        "gross_profit":gross_profit,
        "gross_loss":gross_loss,
        "total_return_pct":(net_profit/cfg.initial_capital*100),
        "win_rate":win_rate,
        "profit_factor":profit_factor,
        "avg_win":avg_win,
        "avg_loss":avg_loss,
        "largest_win":largest_win,
        "largest_loss":largest_loss,
        "expectancy":expectancy,
        "max_drawdown_pct":max_dd,
        "recovery_factor":recov,
        "sharpe_ratio":sharpe
    }

# =========================
# Runner
# =========================
def run_backtest(csv_path: str, cfg: BacktestConfig):
    csv_stem=Path(csv_path).stem
    raw=load_tos_strategy_report(csv_path)
    trades=build_trades(raw,cfg.commission_per_round_trip)
    trades=apply_stoploss_corrections(trades,cfg.point_value)
    metrics=compute_metrics(trades,cfg)
    outdir=cfg.outdir(csv_stem,"/MES",cfg.strategy_name)
    os.makedirs(outdir,exist_ok=True)
    trades.to_csv(os.path.join(outdir,"trades_enriched.csv"),index=False)
    with open(os.path.join(outdir,"metrics.json"),"w") as f: json.dump(metrics,f,indent=2)
    return metrics

# =========================
# CLI
# =========================
if __name__=="__main__":
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument("--csv",nargs="+",required=True)
    parser.add_argument("--capital",type=float,default=2500.0)
    parser.add_argument("--commission",type=float,default=4.04)
    parser.add_argument("--point_value",type=float,default=5.0)
    args,_=parser.parse_known_args()   # ignores unknown args like --timeframe

    cfg=BacktestConfig(initial_capital=args.capital,
                       commission_per_round_trip=args.commission,
                       point_value=args.point_value)

    allm=[]
    for path in args.csv:
        for f in glob.glob(path):
            if Path(f).exists():
                m=run_backtest(f,cfg)
                m["csv"]=Path(f).name
                allm.append(m)

    print(json.dumps(allm,indent=2))
