import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os, math

os.makedirs("out", exist_ok=True)
df = pd.read_csv("out/ksol_xgb_eval.csv")

mae_log = np.mean(np.abs(df.y_true_log10 - df.y_pred_log10))
ss_res = np.sum((df.y_true_log10 - df.y_pred_log10) ** 2)
ss_tot = np.sum((df.y_true_log10 - np.mean(df.y_true_log10)) ** 2)
r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")

band = 0.3  # ±0.3 log10 ≈ 2×
lo = min(df.y_true_log10.min(), df.y_pred_log10.min()) - 0.2
hi = max(df.y_true_log10.max(), df.y_pred_log10.max()) + 0.2
xx = np.linspace(lo, hi, 200)

plt.figure(figsize=(7, 7))
plt.scatter(df.y_true_log10, df.y_pred_log10, alpha=0.65, s=26, edgecolors="none")
plt.plot(xx, xx, linestyle="--", linewidth=1.5)                 # identity
plt.fill_between(xx, xx - band, xx + band, alpha=0.15, label="±0.3 log10 (~2×)")
plt.xlim(lo, hi); plt.ylim(lo, hi)
plt.xlabel("True log10(KSol) [µM]")
plt.ylabel("Predicted log10(KSol) [µM]")
plt.title(f"Kinetic Solubility | XGBoost Parity\nMAE={mae_log:.3f} log10,  R²={r2:.3f}")
plt.legend(frameon=False, loc="upper left")
plt.grid(True, alpha=0.25)
plt.tight_layout()
plt.savefig("out/ksol_parity.png", dpi=300)
plt.show()
