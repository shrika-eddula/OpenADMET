import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

os.makedirs("out", exist_ok=True)

# Load evaluation CSV
df = pd.read_csv("out/hlm_xgb_eval.csv")

# Compute metrics
mae_log = np.mean(np.abs(df.y_true_log10 - df.y_pred_log10))
ss_res = np.sum((df.y_true_log10 - df.y_pred_log10) ** 2)
ss_tot = np.sum((df.y_true_log10 - np.mean(df.y_true_log10)) ** 2)
r2 = 1 - ss_res / ss_tot

print(f"HLM_Clint parity | MAE_log10={mae_log:.3f} | R²={r2:.3f}")

# Plot settings
lims = [min(df.y_true_log10.min(), df.y_pred_log10.min()) - 0.2,
        max(df.y_true_log10.max(), df.y_pred_log10.max()) + 0.2]
band = 0.3  # ±0.3 log10 ≈ 2× fold error

plt.figure(figsize=(6, 6))
plt.scatter(df.y_true_log10, df.y_pred_log10, alpha=0.6, s=25, edgecolors='none')
plt.plot(lims, lims, 'k--', lw=1.5, label="Ideal parity")
plt.fill_between(lims, [x - band for x in lims], [x + band for x in lims],
                 color="gray", alpha=0.15, label="±0.3 log10 (~2×)")
plt.xlim(lims)
plt.ylim(lims)
plt.xlabel("True log10(HLM Clint)")
plt.ylabel("Predicted log10(HLM Clint)")
plt.title(f"HLM Clint | XGBoost Parity\nMAE={mae_log:.3f}, R²={r2:.3f}")
plt.legend(frameon=False, loc="upper left")
plt.tight_layout()
plt.savefig("out/hlm_parity.png", dpi=300)
plt.show()
