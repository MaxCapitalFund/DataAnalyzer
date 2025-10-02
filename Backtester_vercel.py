# trade_counter.py
import pandas as pd
import io
import re

def load_tos_strategy_report(file_path: str) -> pd.DataFrame:
    # Read file
    with open(file_path, "r", errors="replace") as f:
        lines = f.readlines()

    # Find header line
    start_idx = None
    for i, line in enumerate(lines):
        if line.lstrip().startswith("Id;Strategy;"):
            start_idx = i
            break
    if start_idx is None:
        raise ValueError("No trade table header found in file.")

    df = pd.read_csv(io.StringIO("".join(lines[start_idx:])), sep=";")

    # Clean strategy + tag
    df["StrategyClean"] = df["Strategy"].astype(str).str.split("(").str[0].str.strip()
    df["Tag"] = df["Strategy"].astype(str).str.extract(r"\(([^()]*)\)", expand=False).fillna("")

    # Normalize sides
    def normalize_side(val: str) -> str:
        if not isinstance(val, str): return ""
        v = val.upper()
        if "BUY TO OPEN" in v or "BTO" in v: return "BTO"
        if "SELL TO CLOSE" in v or "STC" in v: return "STC"
        if "SELL TO OPEN" in v or "STO" in v: return "STO"
        if "BUY TO CLOSE" in v or "BTC" in v: return "BTC"
        return v.strip()
    df["SideNorm"] = df["Side"].map(normalize_side)

    # Split Date + Time
    if "Date/Time" in df.columns:
        dt = pd.to_datetime(df["Date/Time"], errors="coerce")
        df["Date"] = dt.dt.date
        df["Time"] = dt.dt.time

    return df

def count_round_trips(df: pd.DataFrame) -> int:
    trades = 0
    open_side = None
    for side in df["SideNorm"]:
        if side in ("BTO", "STO"):
            open_side = side
        elif side in ("STC", "BTC") and open_side:
            if (open_side == "BTO" and side == "STC") or (open_side == "STO" and side == "BTC"):
                trades += 1
                open_side = None
            else:
                open_side = None
    return trades

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python trade_counter.py file.csv")
        sys.exit(1)

    file_path = sys.argv[1]
    df = load_tos_strategy_report(file_path)
    total_trades = count_round_trips(df)

    print(f"✅ Total round trips: {total_trades}")
    print(df.head(10))  # show first 10 rows cleaned
