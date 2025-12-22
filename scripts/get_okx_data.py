import os
import time
import requests
import pandas as pd

SAVE_DIR = "data/okx"
os.makedirs(SAVE_DIR, exist_ok=True)

ASSETS = ["BTC-USDT", "ETH-USDT", "SOL-USDT", "DOGE-USDT"]
BASE_URL = "https://www.okx.com/api/v5/market/history-candles"
BAR = "1m"
LIMIT = 300


def fetch_okx_pair(symbol):
    print(f"\nDownloading {symbol}...")

    all_rows = []
    after = None

    while True:
        params = {"instId": symbol, "bar": BAR, "limit": LIMIT}
        if after:
            params["after"] = after

        r = requests.get(BASE_URL, params=params)
        data = r.json()

        if "data" not in data or not isinstance(data["data"], list):
            print(f"  Bad response for {symbol}, stopping.")
            break

        rows = data["data"]

        if len(rows) == 0:
            print("  No more data.")
            break

        all_rows.extend(rows)

        after = rows[-1][0]  # next page anchor

        print(f"  Fetched {len(rows)} rows (oldest ts={after})")

        if len(rows) < LIMIT:
            break

        time.sleep(0.08)

    if not all_rows:
        print(f"  No valid data for {symbol}.")
        return None

    # KEEP ONLY FIRST 6 FIELDS: ts, open, high, low, close, vol
    clean_rows = [row[:6] for row in all_rows]

    df = pd.DataFrame(
        clean_rows,
        columns=["ts", "open", "high", "low", "close", "vol"]
    )

    df["datetime"] = pd.to_datetime(df["ts"], unit="ms")
    df = df.set_index("datetime")
    df = df.sort_index()
    df = df[["open", "high", "low", "close", "vol"]]

    outname = symbol.replace("-", "") + "_1m.csv"
    outpath = os.path.join(SAVE_DIR, outname)
    df.to_csv(outpath)

    print(f"Saved: {outpath} ({len(df)} rows)")
    return df


for asset in ASSETS:
    fetch_okx_pair(asset)

print("\n All OKX 1m OHLCV downloaded successfully!")



