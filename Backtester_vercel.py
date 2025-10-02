def build_trades(df: pd.DataFrame, commission_rt: float) -> pd.DataFrame:
    trades = []
    def _safe_num(x):
        return pd.to_numeric(x, errors='coerce')

    OPEN_RX  = r"(BTO|BUY TO OPEN|BOT|STO|SELL TO OPEN|SELL SHORT|OPEN)"
    CLOSE_RX = r"(STC|SELL TO CLOSE|SLD|BTC|BUY TO CLOSE|CLOSE)"

    i = 0
    while i < len(df) - 1:
        entry = df.iloc[i]
        exit_ = df.iloc[i + 1]
        side_entry = str(entry['Side']).upper()
        side_exit = str(exit_['Side']).upper()

        if re.search(OPEN_RX, side_entry) and re.search(CLOSE_RX, side_exit):
            entry_qty = _safe_num(entry.get('Qty'))
            qty_abs = abs(entry_qty) if pd.notna(entry_qty) and entry_qty != 0 else 1.0

            direction = 'Long' if 'BTO' in side_entry or 'BUY' in side_entry else 'Short'
            trade_pl = _safe_num(exit_.get('TradePL'))
            commission = commission_rt * qty_abs
            net_pl = (trade_pl if pd.notna(trade_pl) else 0.0) - commission

            trades.append({
                'EntryTime': entry['Date'],
                'ExitTime': exit_['Date'],
                'EntryPrice': _safe_num(entry.get('Price')),
                'ExitPrice': _safe_num(exit_.get('Price')),
                'EntryQty': entry_qty,
                'ExitQty': _safe_num(exit_.get('Qty')),
                'QtyAbs': qty_abs,
                'TradePL': trade_pl,
                'GrossPL': trade_pl,
                'Commission': commission,
                'NetPL': net_pl,
                'Direction': direction,
            })
            i += 2
        else:
            i += 1

    t = pd.DataFrame(trades)
    if t.empty:
        return pd.DataFrame(columns=['EntryTime','ExitTime','NetPL'])

    t = t.sort_values('ExitTime').reset_index(drop=True)
    t['HoldMins'] = (t['ExitTime'] - t['EntryTime']).dt.total_seconds() / 60.0
    return t
