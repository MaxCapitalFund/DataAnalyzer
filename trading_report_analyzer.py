import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import StringIO

#Data formatting
def _to_float(s):
    s = s.astype(str).str.replace(r'[\$,]', '', regex=True)         
    s = s.str.replace(r'\(([^()]*)\)', r'-\1', regex=True)          
    s = s.replace('', np.nan)
    return pd.to_numeric(s, errors='coerce')

def analyze_trading_report(file_path):
    with open(file_path, 'r', errors='replace') as f:
        lines = f.readlines()

    start_idx = None
    for i, line in enumerate(lines):
        if line.startswith("Id;Strategy;"):
            start_idx = i
            break
    if start_idx is None:
        raise ValueError("No trade table header found in file.")

    table_str = "".join(lines[start_idx:])
    df = pd.read_csv(StringIO(table_str), sep=';')

    # Clean data
    df['TradePL'] = _to_float(df['Trade P/L']).fillna(0.0)
    df['CumPL']   = _to_float(df['P/L'])
    #Format to 11/27/24 3:45 AM
    df['Date']    = pd.to_datetime(df['Date/Time'], format="%m/%d/%y %I:%M %p", errors='coerce')
    df['BaseStrategy'] = df['Strategy'].str.split('(').str[0].str.strip()

    # Only count realized P/L on closing trades
    close_mask = df['Side'].str.contains('Close', case=False, na=False)
    df_close = df[close_mask].copy().sort_values('Date')

    total_pl     = df_close['TradePL'].sum()
    avg_pl       = df_close['TradePL'].mean()
    win_rate     = (df_close['TradePL'] > 0).mean() * 100
    largest_win  = df_close['TradePL'].max()
    largest_loss = df_close['TradePL'].min()


    strategy_perf = (df_close.groupby('BaseStrategy')['TradePL']
                     .agg(['count', 'sum', 'mean'])
                     .sort_values('sum', ascending=False))

    
    print("Summary")
    print(f"Total P/L: ${total_pl:,.2f}")
    print(f"Average P/L: ${avg_pl:,.2f}")
    print(f"Win rate: {win_rate:.2f}%")
    print(f"Biggest win: ${largest_win:,.2f}")
    print(f"Biggest loss: ${largest_loss:,.2f}\n")
    print("Strategy Performance")
    print(strategy_perf, "\n")

if __name__ == "__main__":
    analyze_trading_report("StrategyReports_MESXCME_81425.csv")
