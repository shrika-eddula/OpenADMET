# plot_parity_logd.py (no pie chart)
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

CSV_PATH = "out/logd_xgb_eval.csv"  # <-- change if needed
BAND = 0.5                          # ± band around y=x

df = pd.read_csv(CSV_PATH)
if not {"y_true","y_pred"}.issubset(df.columns):
    raise ValueError("CSV needs columns: y_true, y_pred")

y = df["y_true"].astype(float).to_numpy()
p = df["y_pred"].astype(float).to_numpy()
n = len(y)

# --- metrics ---
mae  = np.mean(np.abs(p - y))
mse  = np.mean((p - y) ** 2)
rmse = math.sqrt(mse)
ss_res = np.sum((y - p) ** 2)
ss_tot = np.sum((y - np.mean(y)) ** 2)
r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
r  = np.corrcoef(y, p)[0,1]

# --- scatter + bands ---
fig, ax = plt.subplots(figsize=(8, 7))
ax.scatter(y, p, alpha=0.7)

lo = min(y.min(), p.min()) - 0.1
hi = max(y.max(), p.max()) + 0.1
xx = np.linspace(lo, hi, 200)

ax.plot(xx, xx, color="red", lw=1.2)                    # identity
ax.plot(xx, xx + BAND, linestyle="--", color="red", alpha=0.7)
ax.plot(xx, xx - BAND, linestyle="--", color="red", alpha=0.7)

ax.set_xlim(lo, hi)
ax.set_ylim(lo, hi)
ax.set_xlabel("True LogD")
ax.set_ylabel("Predicted LogD")
ax.grid(True, alpha=0.3)

# --- metrics box (top-left) ---
txt = (f"$R^2$: {r2:.3f}\n"
       f"$r$: {r:.3f}\n"
       f"MAE: {mae:.2f}\n"
       f"MSE: {mse:.2f}\n"
       f"RMSE: {rmse:.2f}\n"
       f"Support: {n}")
ax.text(0.03, 0.97, txt, transform=ax.transAxes, va="top", ha="left",
        fontsize=11, fontweight="bold",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.9))

# --- band label + within-band stats ---
within = np.abs(p - y) <= BAND
pct_in = 100.0 * within.mean()
ax.text(0.70, 0.08, f"±{BAND} LogD ({pct_in:.0f}% within)", transform=ax.transAxes,
        fontsize=11, fontweight="bold")

plt.tight_layout()
plt.show()
