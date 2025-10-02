def load_tos_strategy_report(file_path: str) -> pd.DataFrame:
    import re
    import pandas as pd
    import io

    # Read raw file
    with open(file_path, 'r', errors='replace') as f:
        lines = f.readlines()

    # Find header line
    start_idx = None
    for i, line in enumerate(lines):
        if line.lstrip().startswith("Id;Strategy;"):
            start_idx = i
            break
    if start_idx is None:
        raise ValueError("No trade table header found in file.")

    table_str = "".join(lines[start_idx:])
    df = pd.read_csv(io.StringIO(table_str), sep=";")

    # --- Cleaning ---

    # Strategy base + tag
    df["StrategyClean"] = df["Strategy"].astype(str).str.split("(").str[0].str.strip()
    df["Tag"] = df["Strategy"].astype(str).str.extract(r"\(([^()]*)\)", expand=False).fillna("")

    # Normalize Side to 4 codes
    def normalize_side(val: str) -> str:
        if not isinstance(val, str):
            return ""
        v = val.upper()
        if "BUY TO OPEN" in v or "BTO" in v: return "BTO"
        if "SELL TO CLOSE" in v or "STC" in v: return "STC"
        if "SELL TO OPEN" in v or "STO" in v: return "STO"
        if "BUY TO CLOSE" in v or "BTC" in v: return "BTC"
        return v.strip()

    df["SideNorm"] = df["Side"].map(normalize_side)

    # Clean numbers
    def _to_float(series: pd.Series) -> pd.Series:
        s = series.astype(str).str.replace(r"[\$,]", "", regex=True)
        s = s.str.replace(r"\(([^()]*)\)", r"-\1", regex=True)
        s = s.replace("", pd.NA)
        return pd.to_numeric(s, errors="coerce")

    for col in ["Quantity", "Amount", "Price", "Trade P/L", "P/L", "Position"]:
        if col in df.columns:
            df[col] = _to_float(df[col])

    # Split Date and Time
    if "Date/Time" in df.columns:
        dt = pd.to_datetime(df["Date/Time"], errors="coerce")
        df["Date"] = dt.dt.date
        df["Time"] = dt.dt.time

    # Final clean frame
    keep_cols = ["Id", "StrategyClean", "Tag", "SideNorm", "Quantity", "Price", "Date", "Time", "Trade P/L"]
    return df[[c for c in keep_cols if c in df.columns]].copy()
