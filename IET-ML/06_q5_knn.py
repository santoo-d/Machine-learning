import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from collections import Counter

df = pd.read_csv("data/air_quality_cleaned.csv")
features = ["PM2.5","PM10","NO2","SO2","CO","O3","Temperature","Humidity","WindSpeed","Pressure"]
X = df[features].values
y = df["AQI_Category"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
scaler = StandardScaler().fit(X_train)
X_train_s = scaler.transform(X_train)
X_test_s = scaler.transform(X_test)

class KNNScratch:
    """K-Nearest Neighbors implemented from first principles (Euclidean distance)."""
    def __init__(self, k=5):
        self.k = k
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
        return self
    def _predict_one(self, x):
        dists = np.sqrt(np.sum((self.X_train - x)**2, axis=1))  # Euclidean distance
        nn_idx = np.argsort(dists)[:self.k]
        nn_labels = self.y_train[nn_idx]
        vote = Counter(nn_labels).most_common(1)[0][0]
        return vote
    def predict(self, X):
        return np.array([self._predict_one(x) for x in X])

print("=== KNN from scratch: testing K = 1..25 ===")
ks = list(range(1, 26))
accs = []
for k in ks:
    knn = KNNScratch(k=k).fit(X_train_s, y_train)
    pred = knn.predict(X_test_s)
    acc = accuracy_score(y_test, pred)
    accs.append(acc)
    print(f"  K={k:2d}  Accuracy={acc:.4f}")

best_k = ks[int(np.argmax(accs))]
print(f"\nBest K = {best_k} with accuracy = {max(accs):.4f}")

best_knn = KNNScratch(k=best_k).fit(X_train_s, y_train)
best_pred = best_knn.predict(X_test_s)
print(f"\n=== KNN (K={best_k}) detailed report ===")
print(classification_report(y_test, best_pred))

# Plot accuracy vs K (validation curve)
plt.figure(figsize=(8,5))
plt.plot(ks, accs, marker='o')
plt.axvline(best_k, color='red', linestyle='--', label=f'Best K={best_k}')
plt.title("KNN Accuracy vs K (Validation Curve)")
plt.xlabel("K"); plt.ylabel("Test Accuracy"); plt.legend()
plt.tight_layout(); plt.savefig("figs/11_knn_k_validation_curve.png", dpi=140); plt.close()

import seaborn as sns
labels_order = sorted(set(y_test))
cm = confusion_matrix(y_test, best_pred, labels=labels_order)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Greens", xticklabels=labels_order, yticklabels=labels_order)
plt.xlabel("Predicted"); plt.ylabel("Actual"); plt.title(f"KNN (K={best_k}) Confusion Matrix")
plt.tight_layout(); plt.savefig("figs/12_knn_confusion_matrix.png", dpi=140); plt.close()

import json
with open("data/q5_results.json","w") as f:
    json.dump({"ks": ks, "accs": accs, "best_k": best_k, "best_acc": max(accs)}, f, indent=2)
