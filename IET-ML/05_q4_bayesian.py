import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

df = pd.read_csv("data/air_quality_cleaned.csv")
features = ["PM2.5","PM10","NO2","SO2","CO","O3","Temperature","Humidity","WindSpeed","Pressure"]
X = df[features]; y = df["AQI_Category"]
classes = sorted(y.unique())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

# ---- Maximum Likelihood Estimation of Gaussian parameters per class for PM2.5 ----
print("=== MLE of Gaussian parameters (mean, std) per class, feature=PM2.5 ===")
mle_params = {}
for c in classes:
    vals = X_train.loc[y_train == c, "PM2.5"]
    mu_mle = vals.mean()          # MLE estimator of mean
    sigma_mle = vals.std(ddof=0)  # MLE estimator of std (biased, /N)
    mle_params[c] = (mu_mle, sigma_mle)
    print(f"  Class={c:10s}: mu_MLE={mu_mle:.2f}, sigma_MLE={sigma_mle:.2f}, n={len(vals)}")

# Prior probabilities P(class) via MLE (relative frequency)
priors = y_train.value_counts(normalize=True)
print("\nPrior probabilities P(Class) [MLE = relative frequency]:")
print(priors)

def gaussian_pdf(x, mu, sigma):
    return (1.0/(np.sqrt(2*np.pi)*sigma)) * np.exp(-0.5*((x-mu)/sigma)**2)

# Manual Bayes' theorem demo for a single test sample using ONLY PM2.5 feature
sample = X_test.iloc[0]
true_label = y_test.iloc[0]
print(f"\n--- Manual Bayes'-theorem demo for one sample (PM2.5={sample['PM2.5']:.2f}, true class={true_label}) ---")
posteriors = {}
evidence = 0
likelihoods = {}
for c in classes:
    mu, sigma = mle_params[c]
    like = gaussian_pdf(sample["PM2.5"], mu, sigma)
    likelihoods[c] = like
    evidence += like * priors[c]
for c in classes:
    post = (likelihoods[c] * priors[c]) / evidence
    posteriors[c] = post
    print(f"  P({c}|PM2.5) = P(PM2.5|{c})*P({c}) / P(PM2.5) = {likelihoods[c]:.5f}*{priors[c]:.3f}/{evidence:.5f} = {post:.4f}")
pred_manual = max(posteriors, key=posteriors.get)
print(f"  => Manual Bayes prediction (PM2.5-only): {pred_manual}  (true: {true_label})")

# ---- Full Gaussian Naive Bayes classifier (all features) ----
gnb = GaussianNB()
gnb.fit(X_train, y_train)
pred = gnb.predict(X_test)
acc = accuracy_score(y_test, pred)
print(f"\n=== Gaussian Naive Bayes (all 10 features) ===\nAccuracy: {acc:.4f}")
print(classification_report(y_test, pred))

cm = confusion_matrix(y_test, pred, labels=gnb.classes_)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Purples", xticklabels=gnb.classes_, yticklabels=gnb.classes_)
plt.xlabel("Predicted"); plt.ylabel("Actual"); plt.title("Naive Bayes Confusion Matrix")
plt.tight_layout(); plt.savefig("figs/09_nb_confusion_matrix.png", dpi=140); plt.close()

# Plot class-conditional Gaussians for PM2.5
plt.figure(figsize=(8,5))
xs = np.linspace(X["PM2.5"].min(), X["PM2.5"].max(), 300)
for c in classes:
    mu, sigma = mle_params[c]
    plt.plot(xs, gaussian_pdf(xs, mu, sigma), label=f"{c} (mu={mu:.1f}, sd={sigma:.1f})")
plt.title("Class-conditional Gaussian Likelihoods for PM2.5 (MLE fit)")
plt.xlabel("PM2.5"); plt.ylabel("Density"); plt.legend()
plt.tight_layout(); plt.savefig("figs/10_bayes_gaussian_pm25.png", dpi=140); plt.close()

import json
with open("data/q4_results.json","w") as f:
    json.dump({"accuracy_gnb": acc, "mle_params": {k: list(v) for k,v in mle_params.items()},
               "priors": priors.to_dict()}, f, indent=2)
