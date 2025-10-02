# Clean and transform the user's CSV to the requested format

# Shorten strategy names (remove tags like (STO Entry), (BTC Stop), etc.)
df["StrategyClean"] = df["Strategy"].astype(str).str.split("(").str[0].str.strip()

# Normalize Side into 4 codes
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

df["Side"] = df["Side"].map(normalize_side)

# Split Date/Time column into Date and Time
if "Date/Time" in df.columns:
    dt = pd.to_datetime(df["Date/Time"], errors="coerce")
    df["Date"] = dt.dt.date
    df["Time"] = dt.dt.time

# Keep only relevant columns
clean_df = df[["Id", "StrategyClean", "Side", "Quantity", "Price", "Date", "Time", "Trade P/L"]].copy()
clean_df.rename(columns={"StrategyClean": "Strategy", "Trade P/L": "TradePL"}, inplace=True)

# Count round trips: pair BTO->STC and STO->BTC sequentially (ignoring Id since ToS assigns new Id each leg)
round_trips = 0
open_side = None

for _, row in clean_df.iterrows():
    side = row["Side"]
    if side in ("BTO", "STO"):
        open_side = side
    elif side in ("STC", "BTC") and open_side:
        if (open_side == "BTO" and side == "STC") or (open_side == "STO" and side == "BTC"):
            round_trips += 1
            open_side = None

(clean_df.head(12), round_trips)
