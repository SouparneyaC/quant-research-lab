import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

FACTORS_DIR = "factors"
PLOTS_DIR = os.path.join("plots", "vol_diagnostics")
os.makedirs(PLOTS_DIR, exist_ok=True)

SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "DOGEUSDT"]


def autocorr(series, max_lag=30):
    return [series.autocorr(lag) for lag in range(1, max_lag + 1)]


def plot_acf(series, title, fname, max_lag=30):
    acf_vals = autocorr(series, max_lag)
    plt.figure()
    plt.stem(range(1, max_lag + 1), acf_vals)
    plt.axhline(0, color="black", linewidth=0.5)
    plt.title(title)
    plt.xlabel("Lag (minutes)")
    plt.ylabel("Autocorrelation")
    plt.savefig(fname, bbox_inches="tight")
    plt.close()


def volatility_regime_table(df, symbol):
    """
    E[ abs_ret_fwd_1m | vol_60 quantile ]
    """
    tmp = df[["vol_60", "abs_ret_fwd_1m"]].dropna().copy()
    tmp["vol_bin"] = pd.qcut(tmp["vol_60"], q=5, labels=False)

    table = (
        tmp.groupby("vol_bin")["abs_ret_fwd_1m"]
        .agg(["mean", "median", "count"])
    )

    out_path = os.path.join(PLOTS_DIR, f"{symbol}_vol_regimes.csv")
    table.to_csv(out_path)

    print(f"\n[{symbol}] Volatility regime table:")
    print(table)
    print(f"[{symbol}] Saved → {out_path}")


def run_symbol(symbol):
    print(f"\n=== Volatility diagnostics for {symbol} ===")
    path = os.path.join(FACTORS_DIR, f"{symbol}_features.parquet")
    df = pd.read_parquet(path)

    # --- ACF of absolute returns ---
    abs_ret = df["ret_1m"].abs().dropna()
    plot_acf(
        abs_ret,
        f"{symbol} | ACF of |ret_1m|",
        os.path.join(PLOTS_DIR, f"{symbol}_acf_abs_ret.png")
    )

    # --- ACF of forward absolute returns ---
    abs_fwd = df["abs_ret_fwd_1m"].dropna()
    plot_acf(
        abs_fwd,
        f"{symbol} | ACF of |ret_fwd_1m|",
        os.path.join(PLOTS_DIR, f"{symbol}_acf_abs_ret_fwd.png")
    )

    print(f"[{symbol}] ACF plots saved")

    # --- Volatility regimes ---
    volatility_regime_table(df, symbol)


if __name__ == "__main__":
    print("=== Phase 2: Volatility Diagnostics ===")
    for sym in SYMBOLS:
        run_symbol(sym)
    print("\n Volatility diagnostics complete.")

