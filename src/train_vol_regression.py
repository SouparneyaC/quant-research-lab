import os
import numpy as np
import pandas as pd

from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error

# Config
DATA_DIR = "ml_data"
SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "DOGEUSDT"]

# Utility
def load_splits(symbol):
    train = pd.read_parquet(f"{DATA_DIR}/{symbol}_train.parquet")
    val   = pd.read_parquet(f"{DATA_DIR}/{symbol}_val.parquet")
    test  = pd.read_parquet(f"{DATA_DIR}/{symbol}_test.parquet")

    feature_cols = [
        c for c in train.columns
        if c not in ["datetime", "abs_ret_fwd_1m"]
    ]

    return train, val, test, feature_cols


def evaluate_regression(model, df, feature_cols):
    X = df[feature_cols]
    y = df["abs_ret_fwd_1m"]

    preds = model.predict(X)

    r2 = r2_score(y, preds)
    rmse = np.sqrt(mean_squared_error(y, preds))

    return r2, rmse

# Main loop
if __name__ == "__main__":

    print("\n=== Volatility (Absolute Return) Prediction Results ===\n")

    for symbol in SYMBOLS:
        print(f"--- {symbol} ---")

        train_df, val_df, test_df, feature_cols = load_splits(symbol)

        # 1. Ridge Regression
        ridge = Ridge(alpha=1.0)
        ridge.fit(train_df[feature_cols], train_df["abs_ret_fwd_1m"])

        r2_ridge, rmse_ridge = evaluate_regression(
            ridge, test_df, feature_cols
        )

        # 2. Random Forest
        rf = RandomForestRegressor(
            n_estimators=300,
            max_depth=8,
            n_jobs=-1,
            random_state=42
        )

        rf.fit(train_df[feature_cols], train_df["abs_ret_fwd_1m"])

        r2_rf, rmse_rf = evaluate_regression(
            rf, test_df, feature_cols
        )

        # Print (Table 2 values)
        print(
            f"Ridge: R² = {r2_ridge:.4f} | RMSE = {rmse_ridge:.6f}"
        )
        print(
            f"RF:    R² = {r2_rf:.4f} | RMSE = {rmse_rf:.6f}\n"
        )

    print("Volatility modeling complete.")