import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

FACTORS_DIR = "factors"
OUT_DIR = "ml_data"
os.makedirs(OUT_DIR, exist_ok=True)

SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "DOGEUSDT"]

# CONFIG
TARGET_COL = "abs_ret_fwd_1m"   # Phase 2 target (volatility magnitude)
TIME_COL = "datetime"

# Columns that should NEVER be treated as features
EXCLUDE_COLS = {
    TIME_COL,
    "ret_fwd_1m",
    "abs_ret_fwd_1m",
    "sq_ret_fwd_1m",
}

TRAIN_FRAC = 0.7
VAL_FRAC   = 0.15
TEST_FRAC  = 0.15

# HELPERS

def time_split(df):
    n = len(df)
    i1 = int(n * TRAIN_FRAC)
    i2 = int(n * (TRAIN_FRAC + VAL_FRAC))
    return df.iloc[:i1], df.iloc[i1:i2], df.iloc[i2:]


# MAIN PIPELINE

def build_ml_dataset(symbol):
    print(f"\n=== Building ML dataset for {symbol} ===")

    path = os.path.join(FACTORS_DIR, f"{symbol}_features.parquet")
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    df = pd.read_parquet(path).sort_values(TIME_COL).reset_index(drop=True)

    # Safety checks
    if TARGET_COL not in df.columns:
        raise ValueError(f"{symbol}: target {TARGET_COL} not found")

    # Define features explicitly
    feature_cols = [
        c for c in df.columns
        if c not in EXCLUDE_COLS
        and np.issubdtype(df[c].dtype, np.number)
    ]

    print(f"Using {len(feature_cols)} features")

    # Split by time (NO SHUFFLING)
    train_df, val_df, test_df = time_split(df)

    # Scale features (fit on train only)
    scaler = StandardScaler()
    scaler.fit(train_df[feature_cols])

    train_df = train_df.copy()
    val_df   = val_df.copy()
    test_df  = test_df.copy()

    train_df.loc[:, feature_cols] = scaler.transform(train_df[feature_cols])
    val_df.loc[:, feature_cols]   = scaler.transform(val_df[feature_cols])
    test_df.loc[:, feature_cols]  = scaler.transform(test_df[feature_cols])

    # Keep only what ML needs
    keep_cols = [TIME_COL] + feature_cols + [TARGET_COL]

    train_df = train_df[keep_cols]
    val_df   = val_df[keep_cols]
    test_df  = test_df[keep_cols]

    # Save
    train_df.to_parquet(os.path.join(OUT_DIR, f"{symbol}_train.parquet"))
    val_df.to_parquet(os.path.join(OUT_DIR, f"{symbol}_val.parquet"))
    test_df.to_parquet(os.path.join(OUT_DIR, f"{symbol}_test.parquet"))

    print(
        f"[{symbol}] train={len(train_df)}, "
        f"val={len(val_df)}, test={len(test_df)}"
    )


# RUN
if __name__ == "__main__":
    print("=== Building ML datasets (Phase 2) ===")
    for sym in SYMBOLS:
        build_ml_dataset(sym)
    print("\n ML datasets ready in ml_data/")

