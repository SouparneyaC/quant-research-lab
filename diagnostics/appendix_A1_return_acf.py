import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf
from mpl_toolkits.mplot3d import Axes3D

# -----------------------------
# Configuration
# -----------------------------
DATA_DIR = "data/processed"
ASSETS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "DOGEUSDT"]
MAX_LAG = 50

# -----------------------------
# Load data and compute ACFs
# -----------------------------
acf_matrix = []

for asset in ASSETS:
    df = pd.read_parquet(f"{DATA_DIR}/{asset}.parquet")

    # compute 1-minute returns
    ret = df["close"].pct_change().dropna()

    # absolute returns (volatility proxy)
    abs_ret = np.abs(ret)

    # ACF of absolute returns
    acf_vals = acf(abs_ret, nlags=MAX_LAG, fft=True)[1:]  # skip lag 0
    acf_matrix.append(acf_vals)

acf_matrix = np.array(acf_matrix)

# -----------------------------
# Create surface grid
# -----------------------------
lags = np.arange(1, MAX_LAG + 1)
assets_idx = np.arange(len(ASSETS))

X, Y = np.meshgrid(lags, assets_idx)
Z = acf_matrix

# -----------------------------
# Plot (LinkedIn optimized)
# -----------------------------
plt.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 11,
})

fig = plt.figure(figsize=(9, 5))
ax = fig.add_subplot(111, projection="3d")

surf = ax.plot_surface(
    X, Y, Z,
    cmap="viridis",
    edgecolor="none",
    antialiased=True,
    alpha=0.95
)

# -----------------------------
# Formatting
# -----------------------------
ax.set_xlabel("Lag (minutes)", labelpad=8)
ax.set_ylabel("Asset", labelpad=8)
ax.set_zlabel("Volatility Persistence", labelpad=6)

ax.set_yticks(assets_idx)
ax.set_yticklabels(["BTC", "ETH", "SOL", "DOGE"])

ax.set_title(
    "Short-Horizon Volatility Persistence in Crypto Markets",
    pad=12,
    fontweight="bold"
)

ax.view_init(elev=28, azim=-135)

fig.colorbar(
    surf,
    shrink=0.6,
    aspect=12,
    pad=0.08,
    label="ACF of |1-min returns|"
)

plt.tight_layout()
plt.show()