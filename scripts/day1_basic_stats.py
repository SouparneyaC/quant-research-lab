import pandas as pd
import numpy as np
import yfinance as yf
import os

# ---------------------------------------------------------
# 1. Download price data
# ---------------------------------------------------------
data = yf.download("SPY", start="2010-01-01")
print("Columns in downloaded data:", data.columns)

# Pull the close as a 1D Series
close_df = data["Close"]

if isinstance(close_df, pd.DataFrame):
    prices = close_df["SPY"]
else:
    prices = close_df

print("\nType of prices:", type(prices))
print("First 5 prices:\n", prices.head())

# ---------------------------------------------------------
# 2. Compute returns (simple + log)
# ---------------------------------------------------------
simple_ret = prices.pct_change()
log_ret = np.log(prices).diff()

print("\nType of simple_ret:", type(simple_ret))
print("First 5 simple returns:\n", simple_ret.head())
print("\nType of log_ret:", type(log_ret))
print("First 5 log returns:\n", log_ret.head())

# ---------------------------------------------------------
# 3. Compute rolling volatility (20-day window)
# ---------------------------------------------------------
rolling_vol = log_ret.rolling(20).std() * np.sqrt(252)
print("\nType of rolling_vol:", type(rolling_vol))
print("First 5 rolling vol values:\n", rolling_vol.head())

# ---------------------------------------------------------
# 4. Combine into a clean table
# ---------------------------------------------------------
df = pd.DataFrame(
    {
        "price": prices,
        "simple_return": simple_ret,
        "log_return": log_ret,
        "rolling_volatility": rolling_vol,
    }
)

# ---------------------------------------------------------
# 5. Save output inside data/ (RELATIVE PATH - BEST PRACTICE)
# ---------------------------------------------------------

output_dir = "../data"               # <---- relative, not absolute
os.makedirs(output_dir, exist_ok=True)

save_path = os.path.join(output_dir, "day1_output.csv")
df.to_csv(save_path)

print(f"\nSaved to {save_path}")
print("\nDataFrame preview:")
print(df.head(40))
