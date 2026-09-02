import numpy as np, pandas as pd, json
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

df = pd.read_csv("data/air_quality_cleaned.csv")
features = ["PM2.5","PM10","NO2","SO2","CO","O3","Temperature","Humidity","WindSpeed","Pressure"]
X = df[features].values; y = df["AQI_Category"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
scaler = StandardScaler().fit(X_train)
X_train_s, X_test_s = scaler.transform(X_train), scaler.transform(X_test)

models = {
    "Decision Tree": DecisionTreeClassifier(criterion="entropy", max_depth=4, random_state=42),
    "Naive Bayes": GaussianNB(),
    "KNN (K=15)": KNeighborsClassifier(n_neighbors=15),
    "MLP (sklearn, 16 hidden)": MLPClassifier(hidden_layer_sizes=(16,), max_iter=2000, random_state=42),
}

rows = []
for name, model in models.items():
    model.fit(X_train_s, y_train)
    pred = model.predict(X_test_s)
    acc = accuracy_score(y_test, pred)
    prec = precision_score(y_test, pred, average="macro", zero_division=0)
    rec = recall_score(y_test, pred, average="macro", zero_division=0)
    f1 = f1_score(y_test, pred, average="macro", zero_division=0)
    rows.append({"Model": name, "Accuracy": acc, "Precision(macro)": prec,
                 "Recall(macro)": rec, "F1(macro)": f1})
    print(f"{name:28s} Acc={acc:.4f}  Prec={prec:.4f}  Rec={rec:.4f}  F1={f1:.4f}")

# Include GA-optimized DT and best-K KNN from earlier steps + LWR regression metrics
with open("data/q5_results.json") as f: q5 = json.load(f)
with open("data/q6_results.json") as f: q6 = json.load(f)
with open("data/q7_results.json") as f: q7 = json.load(f)
with open("data/q8_results.json") as f: q8 = json.load(f)

results_df = pd.DataFrame(rows)
results_df.to_csv("data/model_comparison.csv", index=False)
print("\n=== Comparison Table ===")
print(results_df.round(4).to_string(index=False))

print(f"\nScratch KNN best K={q5['best_k']} test accuracy={q5['best_acc']:.4f}")
print(f"Scratch MLP (16 hidden) test accuracy={q7['test_acc']:.4f}")
print(f"GA-optimized DT (features={q8['selected_features']}) accuracy={q8['ga_accuracy']:.4f}")
print(f"LWR (regression, tau={q6['best_tau']}) RMSE={q6['tau_results'][str(q6['best_tau'])][1]:.3f}, "
      f"MAE={q6['tau_results'][str(q6['best_tau'])][0]:.3f}")

# Bar chart comparing accuracy across models
plt.figure(figsize=(10,6))
plt.bar(results_df["Model"], results_df["Accuracy"], color="teal")
plt.ylim(0,1); plt.xticks(rotation=20, ha="right")
plt.title("Test Accuracy Comparison Across Classification Models")
plt.ylabel("Accuracy")
for i,v in enumerate(results_df["Accuracy"]):
    plt.text(i, v+0.02, f"{v:.3f}", ha="center")
plt.tight_layout(); plt.savefig("figs/20_model_comparison_accuracy.png", dpi=140); plt.close()

# Grouped bar: precision/recall/F1
plt.figure(figsize=(10,6))
x = np.arange(len(results_df))
w = 0.25
plt.bar(x-w, results_df["Precision(macro)"], width=w, label="Precision")
plt.bar(x, results_df["Recall(macro)"], width=w, label="Recall")
plt.bar(x+w, results_df["F1(macro)"], width=w, label="F1-score")
plt.xticks(x, results_df["Model"], rotation=20, ha="right")
plt.ylim(0,1); plt.legend()
plt.title("Precision / Recall / F1 Comparison (macro-averaged)")
plt.tight_layout(); plt.savefig("figs/21_model_comparison_prf.png", dpi=140); plt.close()

# Scalability analysis: training time vs dataset size for each model
import time
sizes = [100, 300, 600, 900, len(X_train_s)]
scal_results = {name: [] for name in models}
for size in sizes:
    Xs, ys = X_train_s[:size], y_train[:size]
    for name, model in models.items():
        m2 = type(model)(**model.get_params())
        t0 = time.time()
        m2.fit(Xs, ys)
        scal_results[name].append(time.time()-t0)

plt.figure(figsize=(8,5))
for name, times in scal_results.items():
    plt.plot(sizes, times, marker='o', label=name)
plt.title("Scalability Analysis: Training Time vs Dataset Size")
plt.xlabel("Training set size"); plt.ylabel("Training time (s)")
plt.legend()
plt.tight_layout(); plt.savefig("figs/22_scalability_analysis.png", dpi=140); plt.close()

print("\nScalability (training time in seconds):")
for name, times in scal_results.items():
    print(f"  {name:28s}: {['%.4f'%t for t in times]}")
