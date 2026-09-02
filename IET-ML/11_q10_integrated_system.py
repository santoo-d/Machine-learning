import numpy as np, pandas as pd, json
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier

df = pd.read_csv("data/air_quality_cleaned.csv")
features = ["PM2.5","PM10","NO2","SO2","CO","O3","Temperature","Humidity","WindSpeed","Pressure"]
X = df[features].values; y = df["AQI_Category"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
scaler = StandardScaler().fit(X_train)
best_model = MLPClassifier(hidden_layer_sizes=(16,), max_iter=2000, random_state=42)
best_model.fit(scaler.transform(X_train), y_train)

RECOMMENDATIONS = {
    "Good": "Air quality is satisfactory. Normal outdoor activities are safe for everyone.",
    "Moderate": "Acceptable air quality. Unusually sensitive individuals should consider reducing prolonged outdoor exertion.",
    "Poor": "Health effects possible for sensitive groups (children, elderly, respiratory/heart conditions). Limit prolonged outdoor exertion and consider masks (N95) outdoors.",
    "Hazardous": "Health warning of emergency conditions. Everyone should avoid outdoor physical activity, keep windows closed, use air purifiers, and follow local health advisories.",
}

def urban_air_quality_warning_system(sample: dict):
    """
    Integrated Urban Air-Quality Warning System.
    Input: dict of the 10 sensor readings.
    Output: predicted AQI category, confidence, and health recommendation.
    """
    x = np.array([[sample[f] for f in features]])
    x_s = scaler.transform(x)
    pred = best_model.predict(x_s)[0]
    proba = best_model.predict_proba(x_s)[0]
    confidence = float(np.max(proba))
    return {
        "predicted_category": pred,
        "confidence": round(confidence, 3),
        "recommendation": RECOMMENDATIONS[pred],
        "class_probabilities": {c: round(float(p),3) for c,p in zip(best_model.classes_, proba)}
    }

# Demonstrate the system on 4 representative scenarios
scenarios = {
    "Clean windy day": dict(zip(features, [45, 70, 25, 8, 0.5, 40, 28, 45, 6.5, 1015])),
    "Typical urban day": dict(zip(features, [105, 165, 48, 17, 1.2, 55, 27, 55, 2.5, 1012])),
    "Post-monsoon humid stagnant": dict(zip(features, [135, 210, 60, 22, 1.6, 35, 24, 78, 0.8, 1009])),
    "Winter smog emergency": dict(zip(features, [180, 260, 75, 28, 2.4, 30, 15, 65, 0.5, 1006])),
}

print("=== Integrated Urban Air-Quality Warning System: Demo ===\n")
results = {}
for name, s in scenarios.items():
    out = urban_air_quality_warning_system(s)
    results[name] = out
    print(f"Scenario: {name}")
    print(f"  Inputs: {s}")
    print(f"  --> Predicted category : {out['predicted_category']}  (confidence={out['confidence']})")
    print(f"  --> Recommendation     : {out['recommendation']}")
    print()

with open("data/q10_demo_results.json","w") as f:
    json.dump(results, f, indent=2)

# Visualization: dashboard-style summary of the 4 scenarios
fig, axes = plt.subplots(2,2, figsize=(12,8))
order = ["Good","Moderate","Poor","Hazardous"]
colors = {"Good":"#4CAF50","Moderate":"#FFC107","Poor":"#FF7043","Hazardous":"#B71C1C"}
for ax, (name, out) in zip(axes.flat, results.items()):
    probs = [out["class_probabilities"].get(c,0) for c in order]
    bars = ax.bar(order, probs, color=[colors[c] for c in order])
    ax.set_ylim(0,1)
    ax.set_title(f"{name}\nPredicted: {out['predicted_category']} ({out['confidence']*100:.0f}% conf.)", fontsize=10)
    ax.set_ylabel("Probability")
plt.suptitle("Integrated Warning System — Scenario Predictions", fontsize=13)
plt.tight_layout()
plt.savefig("figs/23_integrated_system_dashboard.png", dpi=140)
plt.close()
print("Saved dashboard visualization.")
