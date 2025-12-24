import os
import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score

import joblib
import matplotlib.pyplot as plt

# Paths
DATA_DIR = "ml_data"
MODEL_DIR = "models"
PRED_DIR = "predictions"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PRED_DIR, exist_ok=True)

SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "DOGEUSDT"]

# Load ML splits
def load_splits(symbol):
    train = pd.read_parquet(f"{DATA_DIR}/{symbol}_train.parquet")
    val   = pd.read_parquet(f"{DATA_DIR}/{symbol}_val.parquet")
    test  = pd.read_parquet(f"{DATA_DIR}/{symbol}_test.parquet")

    # Features = everything except datetime
    feature_cols = [c for c in train.columns if c not in ["datetime"]]

    return train, val, test, feature_cols


# Build directional target
def build_directional_target(df):
    """
    y_{t+1} = 1{ r_{t+1} > 0 }
    where r_{t+1} = close_{t+1} / close_t - 1
    """
    df = df.copy()
    df["ret_fwd_1m"] = df["close"].shift(-1) / df["close"] - 1.0
    df["target"] = (df["ret_fwd_1m"] > 0).astype(int)
    return df.dropna()


# Models
def train_logistic(train_df, feature_cols):
    model = LogisticRegression(max_iter=300, C=1.0)
    model.fit(train_df[feature_cols], train_df["target"])
    return model


def train_random_forest(train_df, feature_cols):
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        n_jobs=-1,
        random_state=42
    )
    model.fit(train_df[feature_cols], train_df["target"])
    return model


# Evaluation
def evaluate(model, df, feature_cols, label):
    X = df[feature_cols]
    y = df["target"]

    preds = model.predict(X)
    prob  = model.predict_proba(X)[:, 1]

    acc = accuracy_score(y, preds)
    auc = roc_auc_score(y, prob)
    f1  = f1_score(y, preds)

    print(f"[{label}] ACC={acc:.4f} | AUC={auc:.4f} | F1={f1:.4f}")
    return preds, prob, auc


# Run per asset
def run_for_symbol(symbol):
    print(f"\n=== Training ML models for {symbol} ===")

    train_df, val_df, test_df, feature_cols = load_splits(symbol)

    # Rebuild directional target
    train_df = build_directional_target(train_df)
    val_df   = build_directional_target(val_df)
    test_df  = build_directional_target(test_df)

    # Remove target columns from features
    feature_cols = [c for c in feature_cols if c not in ["ret_fwd_1m", "target"]]

    # Logistic Regression
    print("\nLogistic Regression:")
    log_model = train_logistic(train_df, feature_cols)

    evaluate(log_model, val_df, feature_cols, "VAL")
    test_preds, test_prob, test_auc = evaluate(log_model, test_df, feature_cols, "TEST")

    joblib.dump(log_model, f"{MODEL_DIR}/{symbol}_logreg.pkl")

    out = test_df[["datetime"]].copy()
    out["target"] = test_df["target"]
    out["pred"] = test_preds
    out["prob"] = test_prob
    out.to_csv(f"{PRED_DIR}/{symbol}_logreg_preds.csv", index=False)

    # Random Forest
    print("\nRandom Forest:")
    rf_model = train_random_forest(train_df, feature_cols)

    evaluate(rf_model, val_df, feature_cols, "VAL")
    test_preds, test_prob, test_auc = evaluate(rf_model, test_df, feature_cols, "TEST")

    joblib.dump(rf_model, f"{MODEL_DIR}/{symbol}_rf.pkl")

    # Feature importance
    importance = pd.Series(rf_model.feature_importances_, index=feature_cols)
    importance.sort_values(ascending=False).head(15).plot(
        kind="bar", figsize=(8, 3)
    )
    plt.title(f"{symbol} – Random Forest Feature Importance")
    plt.tight_layout()
    plt.savefig(f"{PRED_DIR}/{symbol}_rf_feature_importance.png")
    plt.close()

    print(f"Saved RF feature importance → {PRED_DIR}/{symbol}_rf_feature_importance.png")


# Main
if __name__ == "__main__":
    print("=== Training baseline directional models ===")
    for sym in SYMBOLS:
        run_for_symbol(sym)
    print("\n Directional modeling complete.")