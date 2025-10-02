# Re-parse with Strategy tags considered for entry/exit classification

df2 = df.copy()

# Extract BaseStrategy and Tag from Strategy column
df2["BaseStrategy"] = df2["Strategy"].astype(str).str.split("(").str[0].str.strip()
df2["Tag"] = df2["Strategy"].astype(str).str.extract(r"\(([^()]*)\)", expand=False).fillna("")

# Normalize Side to 4 codes again
def normalize_side(val: str) -> str:
    if not isinstance(val, str):
        return ""
    v = val.upper()
    if "BUY TO OPEN" in v or "BTO" in v:
        return "BTO"
    if "SELL TO CLOSE" in v or "STC" in v:
        return "STC"
    if "SELL TO OPEN" in v or "STO" in v:
        return "STO"
    if "BUY TO CLOSE" in v or "BTC" in v:
        return "BTC"
    return v.strip()

df2["SideNorm"] = df2["Side"].map(normalize_side)

# Clean date/time split
if "Date/Time" in df2.columns:
    dt = pd.to_datetime(df2["Date/Time"], errors="coerce")
    df2["Date"] = dt.dt.date
    df2["Time"] = dt.dt.time

# Build trades by sequencing through rows per BaseStrategy, pairing entries with exits
trade_counts = {}
total_trades = 0

for strat, grp in df2.groupby("BaseStrategy"):
    grp_sorted = grp.sort_values("Date")
    sides = grp_sorted["SideNorm"].tolist()
    tags = grp_sorted["Tag"].tolist()
    
    trades = 0
    open_side = None
    
    for s, t in zip(sides, tags):
        if s in ("BTO", "STO"):
            open_side = s
        elif s in ("STC", "BTC") and open_side:
            # Match long exit (BTO->STC) or short exit (STO->BTC)
            if (open_side == "BTO" and s == "STC") or (open_side == "STO" and s == "BTC"):
                trades += 1
                open_side = None
            else:
                # skip mismatches
                open_side = None
    
    trade_counts[strat] = trades
    total_trades += trades

trade_counts, total_trades
