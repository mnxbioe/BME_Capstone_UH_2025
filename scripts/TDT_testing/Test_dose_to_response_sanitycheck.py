# scripts/dose_to_response_fit.py
"""
Quantitative dose–response analysis.
Plots Stim RMS vs LFP response RMS with regression line and R² score.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.linear_model import LinearRegression

# ----------------------------------------------------------
# 1. Load all feature CSVs from data/interim/Exp_1
# ----------------------------------------------------------
root = Path(__file__).resolve().parents[1]
features_dir = root / "data" / "interim" / "Exp_1"
feature_files = list(features_dir.rglob("features.csv"))

df = pd.concat(pd.read_csv(f) for f in feature_files)
df = df[df["stim_rms"].notna() & df["lfp_response_rms"].notna()]
print(f"[info] Loaded {len(df):,} feature rows from {len(feature_files)} blocks")

# ----------------------------------------------------------
# 2. Fit linear regression StimRMS → LFPresponseRMS
# ----------------------------------------------------------
X = df["stim_rms"].values.reshape(-1, 1)
y = df["lfp_response_rms"].values
model = LinearRegression().fit(X, y)
r2 = model.score(X, y)

print(f"[fit] slope={model.coef_[0]:.4f}  intercept={model.intercept_:.2f}  R²={r2:.3f}")

# ----------------------------------------------------------
# 3. Plot with regression line
# ----------------------------------------------------------
plt.figure(figsize=(6,5))
plt.scatter(df["stim_rms"], df["lfp_response_rms"], s=10, alpha=0.3, label="trials")

# regression line
x_line = np.linspace(df["stim_rms"].min(), df["stim_rms"].max(), 100)
y_line = model.predict(x_line.reshape(-1,1))
plt.plot(x_line, y_line, color="red", lw=2, label=f"Fit (R²={r2:.2f})")

plt.xlabel("Stim RMS (a.u.)")
plt.ylabel("LFP response RMS (a.u.)")
plt.title("Dose–Response with Linear Fit")
plt.legend()
plt.tight_layout()

# ----------------------------------------------------------
# 4. Save under QA folder
# ----------------------------------------------------------
qa_dir = root / "data" / "processed" / "Exp_1" / "qa"
qa_dir.mkdir(parents=True, exist_ok=True)
out_path = qa_dir / "dose_response_fit.png"
plt.savefig(out_path, dpi=150)
plt.close()
print(f"[qa] Saved regression plot → {out_path}")
