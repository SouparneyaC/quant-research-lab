import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Paths & config
FACTORS_DIR = "factors"
PLOTS_DIR = os.path.join("plots", "diagnostics")
os.makedirs(PLOTS_DIR, exist_ok=True)

SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "DOGEUSDT"]

TARGET_CANDIDATES = ["ret_fwd_1m", "abs_ret_fwd_1m", "sq_ret_fwd_1m"]
RET_CANDIDATES = ["ret_1m"]


# Utilities
def find_column(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


# Diagnostics
def basic_info(df, symbol):
    print(f"\n=== {symbol}: basic info ===")
    print(f"Rows: {len(df)}")

    if "datetime" in df.columns:
        print(f"Date range: {df['datetime'].min()} → {df['datetime'].max()}")

    print("Columns:")
    for c in df.columns:
        print("  ", c)


def return_distribution(df, ret_col, symbol):
    if ret_col is None:
        return

    s = df[ret_col].dropna()
    s_clip = s.clip(s.quantile(0.01), s.quantile(0.99))

    plt.figure(figsize=(6, 4))
    s_clip.plot(kind="kde", linewidth=2)

    plt.title(f"{symbol} | Distribution of {ret_col} (1–99%)")
    plt.xlabel("Return")
    plt.ylabel("Density")
    plt.tight_layout()

    out = os.path.join(PLOTS_DIR, f"{symbol}_return_dist.png")
    plt.savefig(out, dpi=300)
    plt.close()

    print(f"[{symbol}] Return dist saved → {out}")
    print(
        f"[{symbol}] mean={s.mean():.6e}, std={s.std():.6e}, "
        f"skew={s.skew():.2f}, kurt={s.kurt():.2f}"
    )


def autocorr_plot(df, ret_col, symbol, max_lag=50):
    if ret_col is None:
        return

    s = df[ret_col].dropna()
    if len(s) < max_lag + 5:
        return

    lags = np.arange(1, max_lag + 1)
    acf_vals = [s.autocorr(lag) for lag in lags]

    plt.figure(figsize=(6, 4))
    plt.plot(lags, acf_vals, linewidth=2)
    plt.axhline(0.0, linestyle="--", color="black", linewidth=1)

    plt.title(f"{symbol} | ACF of {ret_col}")
    plt.xlabel("Lag (minutes)")
    plt.ylabel("Autocorrelation")
    plt.tight_layout()

    out = os.path.join(PLOTS_DIR, f"{symbol}_acf_{ret_col}.png")
    plt.savefig(out, dpi=300)
    plt.close()

    print(f"[{symbol}] ACF plot saved → {out}")
    print(f"[{symbol}] ACF(1) = {acf_vals[0]:.4f}")


def volatility_clustering(df, ret_col, symbol, window=60):
    if ret_col is None or "datetime" not in df.columns:
        return

    s = df.set_index("datetime")[ret_col].dropna()
    rv = s.rolling(window).std()

    rv_tail = rv.tail(5000)

    plt.figure(figsize=(7, 4))
    plt.plot(rv_tail.index, rv_tail, linewidth=1.5)

    plt.title(f"{symbol} | Rolling {window}m Volatility")
    plt.xlabel("Time")
    plt.ylabel("Rolling Std")
    plt.tight_layout()

    out = os.path.join(PLOTS_DIR, f"{symbol}_rolling_vol_{window}.png")
    plt.savefig(out, dpi=300)
    plt.close()

    print(f"[{symbol}] Rolling volatility plot saved → {out}")


def feature_target_correlations(df, target_col, symbol):
    if target_col is None:
        return

    num = df.select_dtypes(include=[np.number])
    if target_col not in num.columns:
        return

    corrs = (
        num.corr()[target_col]
        .drop(target_col)
        .sort_values(key=np.abs, ascending=False)
    )

    out = os.path.join(PLOTS_DIR, f"{symbol}_feature_target_corr.csv")
    corrs.to_csv(out, header=["corr"])

    print(f"[{symbol}] Feature–target correlations saved → {out}")
    print(f"[{symbol}] Top correlations:")
    print(corrs.head(8))


# Runner
def run_for_symbol(symbol):
    path = os.path.join(FACTORS_DIR, f"{symbol}_features.parquet")
    if not os.path.exists(path):
        print(f"Missing factors for {symbol}")
        return

    df = pd.read_parquet(path)

    ret_col = find_column(df, RET_CANDIDATES)
    target_col = find_column(df, TARGET_CANDIDATES)

    basic_info(df, symbol)
    return_distribution(df, ret_col, symbol)
    autocorr_plot(df, ret_col, symbol)
    volatility_clustering(df, ret_col, symbol)
    feature_target_correlations(df, target_col, symbol)


if __name__ == "__main__":
    print("=== Running paper-quality diagnostics ===")
    for sym in SYMBOLS:
        run_for_symbol(sym)
    print("\n Diagnostics complete. Figures ready for paper.")
