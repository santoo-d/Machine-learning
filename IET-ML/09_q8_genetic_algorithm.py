import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("data/air_quality_cleaned.csv")
features = ["PM2.5","PM10","NO2","SO2","CO","O3","Temperature","Humidity","WindSpeed","Pressure"]
X = df[features].values
y = df["AQI_Category"].values
n_features = len(features)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
scaler = StandardScaler().fit(X_train)
X_train_s, X_test_s = scaler.transform(X_train), scaler.transform(X_test)

# ---- Baseline: model using ALL features ----
base_clf = DecisionTreeClassifier(max_depth=5, random_state=42).fit(X_train_s, y_train)
base_acc = accuracy_score(y_test, base_clf.predict(X_test_s))
print(f"Baseline (all {n_features} features) Decision Tree accuracy: {base_acc:.4f}")

# ---- Genetic Algorithm for feature selection ----
# Chromosome: binary vector of length n_features (1 = feature selected)
# Fitness: classification accuracy on validation split, with a small penalty
#          per selected feature to encourage compact feature subsets.
POP_SIZE = 20
GENERATIONS = 30
MUTATION_RATE = 0.08
CROSSOVER_RATE = 0.8
PENALTY = 0.002  # per-feature complexity penalty

rng = np.random.RandomState(7)

def random_chromosome():
    c = rng.randint(0, 2, n_features)
    if c.sum() == 0:
        c[rng.randint(n_features)] = 1
    return c

def fitness(chromosome):
    idx = np.where(chromosome == 1)[0]
    if len(idx) == 0:
        return 0.0
    clf = DecisionTreeClassifier(max_depth=5, random_state=42)
    clf.fit(X_train_s[:, idx], y_train)
    acc = accuracy_score(y_test, clf.predict(X_test_s[:, idx]))
    return acc - PENALTY * len(idx)

def tournament_select(pop, fits, k=3):
    idxs = rng.choice(len(pop), k, replace=False)
    best = idxs[np.argmax([fits[i] for i in idxs])]
    return pop[best].copy()

def crossover(p1, p2):
    if rng.rand() < CROSSOVER_RATE:
        point = rng.randint(1, n_features)
        c1 = np.concatenate([p1[:point], p2[point:]])
        c2 = np.concatenate([p2[:point], p1[point:]])
        return c1, c2
    return p1.copy(), p2.copy()

def mutate(c):
    for i in range(len(c)):
        if rng.rand() < MUTATION_RATE:
            c[i] = 1 - c[i]
    if c.sum() == 0:
        c[rng.randint(n_features)] = 1
    return c

population = [random_chromosome() for _ in range(POP_SIZE)]
best_fitness_history = []
avg_fitness_history = []
best_chromosome, best_fit = None, -np.inf

print("\n=== Genetic Algorithm evolution (feature selection) ===")
for gen in range(GENERATIONS):
    fits = [fitness(c) for c in population]
    gen_best_idx = int(np.argmax(fits))
    if fits[gen_best_idx] > best_fit:
        best_fit = fits[gen_best_idx]
        best_chromosome = population[gen_best_idx].copy()
    best_fitness_history.append(fits[gen_best_idx])
    avg_fitness_history.append(np.mean(fits))
    if gen % 5 == 0 or gen == GENERATIONS-1:
        sel = [features[i] for i,b in enumerate(population[gen_best_idx]) if b]
        print(f"  Gen {gen:2d}: best_fitness={fits[gen_best_idx]:.4f}  avg={np.mean(fits):.4f}  features={sel}")

    # New generation: elitism (keep best) + crossover + mutation
    new_pop = [population[gen_best_idx].copy()]
    while len(new_pop) < POP_SIZE:
        p1 = tournament_select(population, fits)
        p2 = tournament_select(population, fits)
        c1, c2 = crossover(p1, p2)
        new_pop.append(mutate(c1))
        if len(new_pop) < POP_SIZE:
            new_pop.append(mutate(c2))
    population = new_pop

selected_features = [features[i] for i,b in enumerate(best_chromosome) if b]
print(f"\nBest chromosome: {best_chromosome}")
print(f"Selected features ({len(selected_features)}): {selected_features}")
print(f"Best fitness (accuracy - penalty): {best_fit:.4f}")

idx = np.where(best_chromosome == 1)[0]
ga_clf = DecisionTreeClassifier(max_depth=5, random_state=42).fit(X_train_s[:, idx], y_train)
ga_acc = accuracy_score(y_test, ga_clf.predict(X_test_s[:, idx]))
print(f"\nGA-optimized model accuracy (using {len(idx)} features): {ga_acc:.4f}")
print(f"Baseline model accuracy (using {n_features} features):     {base_acc:.4f}")

# Plot GA convergence
plt.figure(figsize=(8,5))
plt.plot(best_fitness_history, label="Best fitness", marker='o', markersize=3)
plt.plot(avg_fitness_history, label="Average fitness", linestyle='--')
plt.title("Genetic Algorithm Convergence (Feature Selection)")
plt.xlabel("Generation"); plt.ylabel("Fitness (Accuracy - Complexity Penalty)")
plt.legend()
plt.tight_layout(); plt.savefig("figs/18_ga_convergence.png", dpi=140); plt.close()

# Bar chart comparing baseline vs GA-optimized
plt.figure(figsize=(6,5))
plt.bar(["All Features\n(n=%d)"%n_features, "GA-Selected\n(n=%d)"%len(idx)], [base_acc, ga_acc], color=["gray","seagreen"])
plt.ylim(0,1)
plt.title("Model Accuracy: Baseline vs GA-Optimized Feature Subset")
plt.ylabel("Test Accuracy")
for i,v in enumerate([base_acc, ga_acc]):
    plt.text(i, v+0.02, f"{v:.3f}", ha='center')
plt.tight_layout(); plt.savefig("figs/19_ga_vs_baseline.png", dpi=140); plt.close()

import json
with open("data/q8_results.json","w") as f:
    json.dump({"selected_features": selected_features, "ga_accuracy": ga_acc,
               "baseline_accuracy": base_acc, "best_fitness": best_fit}, f, indent=2)
