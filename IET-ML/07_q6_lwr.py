import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("data/air_quality_cleaned.csv")
# Predict PM2.5 from WindSpeed (clear inverse nonlinear-ish relationship) for interpretable 1D LWR,
# then also do a multi-feature version.
df_sorted = df.sort_values("WindSpeed").reset_index(drop=True)
X1 = df_sorted["WindSpeed"].values.reshape(-1,1)
y1 = df_sorted["PM2.5"].values

X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=0.25, random_state=42)

def lwr_predict(X_train, y_train, x_query, tau):
    """Locally Weighted Regression prediction at a single query point x_query.
    Uses Gaussian kernel weights and closed-form weighted least squares:
        theta = (X^T W X)^-1 X^T W y
    """
    m = X_train.shape[0]
    Xb = np.hstack([np.ones((m,1)), X_train])              # add bias term
    diffs = X_train[:,0] - x_query
    w = np.exp(-(diffs**2) / (2*tau**2))                    # Gaussian kernel weights
    W = np.diag(w)
    xq = np.array([1, x_query])
    try:
        theta = np.linalg.pinv(Xb.T @ W @ Xb) @ (Xb.T @ W @ y_train)
    except np.linalg.LinAlgError:
        theta = np.zeros(2)
    return xq @ theta

def lwr_batch(X_train, y_train, X_query, tau):
    return np.array([lwr_predict(X_train, y_train, xq[0], tau) for xq in X_query])

print("=== Locally Weighted Regression: effect of bandwidth tau ===")
taus = [0.3, 0.7, 1.5, 3.0, 6.0]
results = {}
for tau in taus:
    pred = lwr_batch(X1_train, y1_train, X1_test, tau)
    mae = mean_absolute_error(y1_test, pred)
    rmse = np.sqrt(mean_squared_error(y1_test, pred))
    r2 = r2_score(y1_test, pred)
    results[tau] = (mae, rmse, r2)
    print(f"  tau={tau:4.1f}  MAE={mae:6.3f}  RMSE={rmse:6.3f}  R2={r2:6.3f}")

best_tau = min(results, key=lambda t: results[t][1])
print(f"\nBest tau (lowest RMSE) = {best_tau}")

# Compare with standard Linear Regression (global model)
lr = LinearRegression().fit(X1_train, y1_train)
lr_pred = lr.predict(X1_test)
lr_mae = mean_absolute_error(y1_test, lr_pred)
lr_rmse = np.sqrt(mean_squared_error(y1_test, lr_pred))
lr_r2 = r2_score(y1_test, lr_pred)
print(f"\n=== Global Linear Regression (comparison) ===\nMAE={lr_mae:.3f}  RMSE={lr_rmse:.3f}  R2={lr_r2:.3f}")

# Plot LWR fit curve (best tau) vs Linear Regression vs actual data
xs_plot = np.linspace(X1.min(), X1.max(), 200).reshape(-1,1)
lwr_curve = lwr_batch(X1_train, y1_train, xs_plot, best_tau)
lr_curve = lr.predict(xs_plot)

plt.figure(figsize=(9,6))
plt.scatter(X1_train, y1_train, s=12, alpha=0.3, label="Training data")
plt.plot(xs_plot, lwr_curve, color="crimson", linewidth=2.5, label=f"LWR (tau={best_tau})")
plt.plot(xs_plot, lr_curve, color="black", linestyle="--", linewidth=2, label="Linear Regression")
plt.xlabel("Wind Speed (m/s)"); plt.ylabel("PM2.5 (µg/m³)")
plt.title("Locally Weighted Regression vs Linear Regression: PM2.5 vs Wind Speed")
plt.legend()
plt.tight_layout(); plt.savefig("figs/13_lwr_vs_linear.png", dpi=140); plt.close()

# Plot effect of tau (under/over-fitting)
plt.figure(figsize=(9,6))
plt.scatter(X1_train, y1_train, s=10, alpha=0.25, color="gray")
for tau, style in zip([0.3, 1.5, 6.0], ["-", "-", "-"]):
    curve = lwr_batch(X1_train, y1_train, xs_plot, tau)
    plt.plot(xs_plot, curve, linewidth=2, label=f"tau={tau}")
plt.title("Effect of Bandwidth (tau) on LWR Fit")
plt.xlabel("Wind Speed (m/s)"); plt.ylabel("PM2.5 (µg/m³)")
plt.legend()
plt.tight_layout(); plt.savefig("figs/14_lwr_tau_effect.png", dpi=140); plt.close()

import json
with open("data/q6_results.json","w") as f:
    json.dump({"tau_results": {str(k): v for k,v in results.items()}, "best_tau": best_tau,
               "linear_regression": {"mae": lr_mae, "rmse": lr_rmse, "r2": lr_r2}}, f, indent=2)
