import os
import numpy as np
import pandas as pd

DATASET_DIR = "dataset"   # cleaned OHLCV parquet files
FACTORS_DIR = "factors"   # where we save feature matrices
os.makedirs(FACTORS_DIR, exist_ok=True)

SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "DOGEUSDT"]


def add_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add baseline features:
      - 1m return (ret_1m)
      - 1m forward return (ret_fwd_1m)
      - rolling volatility of returns (vol_5, vol_15, vol_30)
      - price momentum (mom_5, mom_15, mom_30)
    Assumes df has: datetime, open, high, low, close, volume
    """
    df = df.sort_values("datetime").reset_index(drop=True)

    # 1-minute realized return
    df["ret_1m"] = df["close"].pct_change()

    # 1-minute forward return (target)
    df["ret_fwd_1m"] = df["close"].shift(-1) / df["close"] - 1.0
    # # Phase 2 targets (state / volatility)
    # # Absolute forward return (volatility magnitude)
    df["abs_ret_fwd_1m"] = df["ret_fwd_1m"].abs()
    # Squared forward return (variance proxy)
    df["sq_ret_fwd_1m"] = df["ret_fwd_1m"] ** 2


    # Rolling volatility of returns
    for w in [5, 15, 30]:
        df[f"vol_{w}"] = df["ret_1m"].rolling(window=w).std()

    # Price momentum: k-minute return
    for w in [5, 15, 30]:
        df[f"mom_{w}"] = df["close"] / df["close"].shift(w) - 1.0

    return df


def add_microstructure_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add microstructure-inspired features:
      - abs & squared returns
      - sign of return
      - directional run-length (streaks)
      - signed volume
      - volume z-score (60m)
      - realized volatility over 60m
    Requires ret_1m to exist.
    """
    # Safety: drop rows with no ret_1m first
    # (but we do this at the very end globally)
    ret = df["ret_1m"]

    # 1) Return transformations
    df["ret_1m_abs"] = ret.abs()
    df["ret_1m_sq"] = ret ** 2
    df["ret_1m_sign"] = np.sign(ret).fillna(0.0)

    # 2) Directional run-length (streaks of same sign)
    # Zero returns are treated as "no direction" -> break streak
    direction = np.sign(ret).replace(0, np.nan)

    # When direction changes (or NaN), start a new group
    change = (direction != direction.shift()).astype(int)
    group_id = change.cumsum()

    # Run-length within each same-sign segment
    run_len = direction.groupby(group_id).cumcount() + 1
    run_len = run_len.where(direction.notna(), 0)  # zero where no direction

    df["run_len"] = run_len.fillna(0).astype(float)
    df["run_len_signed"] = df["run_len"] * df["ret_1m_sign"]

    # 3) Volume + direction
    df["signed_volume"] = df["volume"] * df["ret_1m_sign"]

    # 4) Volume z-score over a 60-minute window
    vol = df["volume"].astype(float)
    roll_mean = vol.rolling(window=60).mean()
    roll_std = vol.rolling(window=60).std()
    df["vol_zscore_60"] = (vol - roll_mean) / (roll_std + 1e-12)

    # 5) Realized volatility over 60 minutes
    roll_vol_60 = ret.rolling(window=60).std()
    df["vol_60"] = roll_vol_60
    df["log_vol_60"] = np.log(roll_vol_60 + 1e-12)

    return df


def build_features_for_symbol(symbol: str):
    in_path = os.path.join(DATASET_DIR, f"{symbol}.parquet")
    if not os.path.exists(in_path):
        raise FileNotFoundError(f"Dataset not found for {symbol}: {in_path}")

    df = pd.read_parquet(in_path)

    # Expect at least these columns
    required = {"datetime", "open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{symbol}: missing required columns: {missing}")

    # Add features
    df = add_basic_features(df)
    df = add_microstructure_features(df)

    # Final cleanup:
    #   - drop rows with any NaNs in numeric features (mainly from rolling)
    #   - keep only rows where ret_fwd_1m is available (no forward target at last row)
    df = df.dropna().reset_index(drop=True)

    out_path = os.path.join(FACTORS_DIR, f"{symbol}_features.parquet")
    df.to_parquet(out_path)
    print(f"[{symbol}] Saved features → {out_path} ({len(df)} rows)")


if __name__ == "__main__":
    print("=== Building factors with microstructure features ===")
    for sym in SYMBOLS:
        build_features_for_symbol(sym)
    print(" Done. Factors ready in 'factors/'")

