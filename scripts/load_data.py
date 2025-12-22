import pandas as pd
import os

DATA_DIR = "data/okx"
OUT_DIR = "dataset"
os.makedirs(OUT_DIR, exist_ok=True)

def load_okx(symbol):
    """
    Loads OKX 1-minute OHLCV data from CSV in the format:

    datetime,open,high,low,close,vol
    """

    path = os.path.join(DATA_DIR, f"{symbol}_1m.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    # Read CSV with header
    df = pd.read_csv(path)

    # Ensure datetime is parsed correctly
    df["datetime"] = pd.to_datetime(df["datetime"])

    # Sort and reset index
    df = df.sort_values("datetime").reset_index(drop=True)

    # Standardize column names
    df = df.rename(columns={"vol": "volume"})

    # Keep consistent columns
    df = df[["datetime", "open", "high", "low", "close", "volume"]]

    return df


if __name__ == "__main__":
    symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "DOGEUSDT"]

    for sym in symbols:
        print(f"Loading {sym} ...")
        df = load_okx(sym)
        out_path = os.path.join(OUT_DIR, f"{sym}.parquet")
        df.to_parquet(out_path)
        print(f"Saved → {out_path} ({len(df)} rows)")

