import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("data/air_quality_cleaned.csv")
features = ["PM2.5","PM10","NO2","SO2","CO","O3","Temperature","Humidity","WindSpeed","Pressure"]
X = df[features]
y = df["AQI_Category"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

# ---- Manual Information Gain illustration for the ROOT split ----
def entropy(labels):
    vals, counts = np.unique(labels, return_counts=True)
    p = counts / counts.sum()
    return -np.sum(p * np.log2(p))

root_entropy = entropy(y_train)
print(f"Entropy of full training set (root) = {root_entropy:.4f} bits")

def info_gain_numeric(feat_col, y, thresholds=5):
    vals = np.unique(np.quantile(feat_col, np.linspace(0.1,0.9,thresholds)))
    best_gain, best_t = -1, None
    for t in vals:
        left = y[feat_col <= t]; right = y[feat_col > t]
        if len(left)==0 or len(right)==0: continue
        w = len(left)/len(y)
        gain = entropy(y) - (w*entropy(left) + (1-w)*entropy(right))
        if gain > best_gain:
            best_gain, best_t = gain, t
    return best_gain, best_t

print("\nInformation Gain per attribute (best threshold split):")
ig_results = {}
for f in features:
    g, t = info_gain_numeric(X_train[f].values, y_train.values)
    ig_results[f] = (g, t)
    print(f"  {f:12s}: IG = {g:.4f}  (best threshold ~ {t:.2f})")

best_feat = max(ig_results, key=lambda k: ig_results[k][0])
print(f"\n=> Root attribute selected by Information Gain: {best_feat}")

# ---- Train sklearn Decision Tree (uses Gini by default; also fit entropy variant) ----
dt_gini = DecisionTreeClassifier(criterion="gini", max_depth=4, random_state=42)
dt_gini.fit(X_train, y_train)
pred_gini = dt_gini.predict(X_test)

dt_entropy = DecisionTreeClassifier(criterion="entropy", max_depth=4, random_state=42)
dt_entropy.fit(X_train, y_train)
pred_entropy = dt_entropy.predict(X_test)

print("\n=== Decision Tree (Gini) ===")
print("Accuracy:", accuracy_score(y_test, pred_gini))
print(classification_report(y_test, pred_gini))

print("=== Decision Tree (Entropy / Information Gain) ===")
print("Accuracy:", accuracy_score(y_test, pred_entropy))
print(classification_report(y_test, pred_entropy))

print("\nFeature importances (entropy tree):")
for f, imp in sorted(zip(features, dt_entropy.feature_importances_), key=lambda x:-x[1]):
    print(f"  {f:12s}: {imp:.4f}")

# Visualize tree
plt.figure(figsize=(20,10))
plot_tree(dt_entropy, feature_names=features, class_names=dt_entropy.classes_,
          filled=True, rounded=True, fontsize=8, max_depth=3)
plt.title("Decision Tree (Entropy Criterion, depth-limited view)")
plt.tight_layout(); plt.savefig("figs/07_decision_tree.png", dpi=140); plt.close()

# Confusion matrix
import seaborn as sns
cm = confusion_matrix(y_test, pred_entropy, labels=dt_entropy.classes_)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=dt_entropy.classes_, yticklabels=dt_entropy.classes_)
plt.xlabel("Predicted"); plt.ylabel("Actual"); plt.title("Decision Tree Confusion Matrix")
plt.tight_layout(); plt.savefig("figs/08_dt_confusion_matrix.png", dpi=140); plt.close()

import json
with open("data/q3_results.json","w") as f:
    json.dump({
        "root_entropy": root_entropy,
        "info_gain": {k: v[0] for k,v in ig_results.items()},
        "root_attribute": best_feat,
        "accuracy_gini": accuracy_score(y_test, pred_gini),
        "accuracy_entropy": accuracy_score(y_test, pred_entropy),
    }, f, indent=2)
