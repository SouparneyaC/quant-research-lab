import numpy as np
import matplotlib.pyplot as plt

def plot_acf_paper(
    series,
    max_lag=30,
    title="",
    ylabel="Autocorrelation",
    save_path=None
):
    """
    Paper-style ACF plot with 95% confidence bands.
    """
    series = series.dropna()
    T = len(series)

    lags = np.arange(1, max_lag + 1)
    acf_vals = np.array([series.autocorr(lag) for lag in lags])

    # White-noise confidence bands
    conf = 2.0 / np.sqrt(T)

    plt.figure(figsize=(6.5, 4.5))
    plt.plot(lags, acf_vals, marker="o", linewidth=1.5)
    plt.axhline(0.0, color="black", linewidth=0.8)

    plt.fill_between(
        lags,
        -conf,
        conf,
        color="gray",
        alpha=0.25,
        label="95% confidence band"
    )

    plt.xlabel("Lag (minutes)")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(frameon=False)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
        plt.close()
    else:
        plt.show()

