import os
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

from scipy.stats import norm, probplot  # for Gaussian comparison

# 0. Config
BASE_DIR = "/Users/aki/Desktop/Quant"

DATA_DIR = os.path.join(BASE_DIR, "data")
PLOTS_DIR = os.path.join(BASE_DIR, "plots")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

START_DATE = "2014-01-01"  # BTC has data from ~2014 onward

# 1. Helper to get adjusted close as 1D Series
def get_adj_close(symbol, start):
    data = yf.download(symbol, start=start)
    # yfinance gives MultiIndex: ('Close', 'SYMBOL')
    close_df = data["Close"]
    if isinstance(close_df, pd.DataFrame):
        prices = close_df[symbol]
    else:
        prices = close_df
    prices.name = symbol
    return prices

# 2. Download SPY and BTC-USD prices
spy_prices = get_adj_close("SPY", START_DATE)
btc_prices = get_adj_close("BTC-USD", START_DATE)

# Align on common dates
prices = pd.concat([spy_prices, btc_prices], axis=1, join="inner")
prices.columns = ["SPY", "BTC"]

# 3. Compute log returns
log_returns = np.log(prices).diff().dropna()
log_returns.columns = ["SPY", "BTC"]

# 4. Basic stats: mean, std, skew, kurtosis
stats_list = []
for symbol in ["SPY", "BTC"]:
    r = log_returns[symbol].dropna()
    stats_list.append({
        "asset": symbol,
        "n_obs": len(r),
        "mean": r.mean(),
        "std": r.std(),
        "skew": r.skew(),
        "kurtosis_excess": r.kurt()  # pandas: excess kurtosis (kurtosis - 3)
    })

stats_df = pd.DataFrame(stats_list).set_index("asset")

# 5. Extreme moves: |z| > 3,4,5 and compare to Gaussian expectation
def extreme_counts(r, thresholds=(3, 4, 5)):
    r = r.dropna()
    mu = r.mean()
    sigma = r.std()
    z = (r - mu) / sigma

    rows = []
    n = len(z)
    for t in thresholds:
        count = (np.abs(z) > t).sum()
        # two-sided Gaussian tail prob
        p_gauss = 2 * (1 - norm.cdf(t))
        expected = n * p_gauss
        rows.append({
            "threshold_sigma": t,
            "n_obs": n,
            "count_extreme": int(count),
            "expected_gaussian": expected,
            "prob_gaussian": p_gauss
        })
    return pd.DataFrame(rows)

tails_spy = extreme_counts(log_returns["SPY"])
tails_btc = extreme_counts(log_returns["BTC"])

tails_spy["asset"] = "SPY"
tails_btc["asset"] = "BTC"

tails_df = pd.concat([tails_spy, tails_btc], ignore_index=True)
tails_df = tails_df[["asset", "threshold_sigma", "n_obs",
                     "count_extreme", "expected_gaussian", "prob_gaussian"]]

# 6. Hill tail index (using absolute returns)
def hill_tail_index(r, k=100):
    x = np.sort(np.abs(r.dropna()))
    if len(x) <= k + 1:
        return np.nan
    x_k = x[-k - 1]  # threshold
    top = x[-k:]
    hill = np.mean(np.log(top) - np.log(x_k))
    if hill <= 0:
        return np.nan
    alpha = 1.0 / hill  # tail index alpha
    return alpha

hill_rows = []
for symbol in ["SPY", "BTC"]:
    r = log_returns[symbol]
    alpha = hill_tail_index(r, k=100)
    hill_rows.append({
        "asset": symbol,
        "hill_k": 100,
        "tail_index_alpha": alpha
    })

hill_df = pd.DataFrame(hill_rows).set_index("asset")

# 7. Plots: histogram + Gaussian overlay, QQ-plots
def plot_hist_with_gaussian(r, symbol):
    r = r.dropna()
    mu, sigma = r.mean(), r.std()

    plt.figure(figsize=(8, 5))
    # Histogram
    plt.hist(r, bins=100, density=True, alpha=0.6, label="Empirical")

    # Gaussian overlay
    x = np.linspace(mu - 5 * sigma, mu + 5 * sigma, 500)
    y = norm.pdf(x, mu, sigma)
    plt.plot(x, y, linewidth=2, label="Gaussian (same mean/std)")

    plt.title(f"Histogram of daily log returns: {symbol}")
    plt.xlabel("Log return")
    plt.ylabel("Density")
    plt.legend()
    out_path = os.path.join(PLOTS_DIR, f"day2_hist_{symbol}.png")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"Saved {out_path}")

def plot_qq(r, symbol):
    r = r.dropna()
    plt.figure(figsize=(6, 6))
    probplot(r, dist="norm", plot=plt)
    plt.title(f"QQ-plot vs Gaussian: {symbol}")
    out_path = os.path.join(PLOTS_DIR, f"day2_qq_{symbol}.png")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"Saved {out_path}")

for symbol in ["SPY", "BTC"]:
    r = log_returns[symbol]
    plot_hist_with_gaussian(r, symbol)
    plot_qq(r, symbol)

# 8. Save outputs
stats_path = os.path.join(DATA_DIR, "day2_stats_spy_btc.csv")
tails_path = os.path.join(DATA_DIR, "day2_tails_spy_btc.csv")
hill_path = os.path.join(DATA_DIR, "day2_hill_spy_btc.csv")

stats_df.to_csv(stats_path)
tails_df.to_csv(tails_path, index=False)
hill_df.to_csv(hill_path)

print(f"Saved stats to {stats_path}")
print(f"Saved tails to {tails_path}")
print(f"Saved Hill tail index to {hill_path}")
print("Done.")
