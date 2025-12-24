import os
import sys
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

# --- THE FIX ---
# This adds the project root to the python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
if project_root not in sys.path:
    sys.path.append(project_root)

# Now these imports will work
from src.models.train_vol_regression import load_splits, evaluate_regression
from sklearn.linear_model import Ridge

# Import seaborn after your 'pip install'
try:
    import seaborn as sns
except ImportError:
    print("Seaborn not found. Run: pip install seaborn")
    sys.exit(1)

# Path fix to ensure it can see src/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from sklearn.linear_model import Ridge
from src.models.train_vol_regression import load_splits

# Ensure output directory exists
PLOT_DIR = "results/plots"
os.makedirs(PLOT_DIR, exist_ok=True)

def generate_performance_visuals(symbol):
    print(f"Generating results for {symbol}...")
    
    # 1. Load data
    train_df, _, test_df, feature_cols = load_splits(symbol)
    y_train = train_df["abs_ret_fwd_1m"]
    y_test = test_df["abs_ret_fwd_1m"]
    
    # 2. Train a quick Ridge model (matches your main research)
    model = Ridge(alpha=1.0)
    model.fit(train_df[feature_cols], y_train)
    preds = model.predict(test_df[feature_cols])

    # 3. Plot Predicted vs Actual
    plt.figure(figsize=(8, 6))
    sns.regplot(x=preds, y=y_test, 
                scatter_kws={'alpha':0.2, 's':10}, 
                line_kws={'color':'red', 'label':'Linear Fit'})
    
    # Add identity line (Perfect Prediction)
    max_val = max(y_test.max(), preds.max())
    plt.plot([0, max_val], [0, max_val], color='black', linestyle='--', label='Perfect Prediction')
    
    plt.title(f"{symbol} Volatility: Predicted vs Actual", fontsize=14)
    plt.xlabel("Predicted Volatility (Absolute Return)", fontsize=12)
    plt.ylabel("Actual Volatility (Absolute Return)", fontsize=12)
    plt.legend()
    plt.grid(alpha=0.3)
    
    save_path = f"{PLOT_DIR}/{symbol}_pred_vs_actual.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved → {save_path}")
    plt.close()

if __name__ == "__main__":
    for sym in ["BTCUSDT", "ETHUSDT", "SOLUSDT", "DOGEUSDT"]:
        try:
            generate_performance_visuals(sym)
        except Exception as e:
            print(f"Could not generate for {sym}: {e}")