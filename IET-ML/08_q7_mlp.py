import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

df = pd.read_csv("data/air_quality_cleaned.csv")
features = ["PM2.5","PM10","NO2","SO2","CO","O3","Temperature","Humidity","WindSpeed","Pressure"]
X = df[features].values
y_raw = df["AQI_Category"].values

classes = sorted(np.unique(y_raw))
class_to_idx = {c:i for i,c in enumerate(classes)}
y_idx = np.array([class_to_idx[c] for c in y_raw])
n_classes = len(classes)
Y_onehot = np.eye(n_classes)[y_idx]

X_train, X_test, y_train_oh, y_test_oh, y_train_idx, y_test_idx = train_test_split(
    X, Y_onehot, y_idx, test_size=0.25, random_state=42, stratify=y_idx)

scaler = StandardScaler().fit(X_train)
X_train_s = scaler.transform(X_train)
X_test_s = scaler.transform(X_test)

def sigmoid(z): return 1/(1+np.exp(-np.clip(z,-500,500)))
def sigmoid_deriv(a): return a*(1-a)
def softmax(z):
    z = z - np.max(z, axis=1, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=1, keepdims=True)

class MLP:
    """Multilayer Perceptron: input -> hidden(sigmoid) -> output(softmax),
    trained with backpropagation (gradient descent, cross-entropy loss)."""
    def __init__(self, n_in, n_hidden, n_out, lr=0.1, seed=42):
        rng = np.random.RandomState(seed)
        self.W1 = rng.randn(n_in, n_hidden) * np.sqrt(2.0/n_in)
        self.b1 = np.zeros((1, n_hidden))
        self.W2 = rng.randn(n_hidden, n_out) * np.sqrt(2.0/n_hidden)
        self.b2 = np.zeros((1, n_out))
        self.lr = lr
        self.loss_history = []

    def forward(self, X):
        self.z1 = X @ self.W1 + self.b1
        self.a1 = sigmoid(self.z1)
        self.z2 = self.a1 @ self.W2 + self.b2
        self.a2 = softmax(self.z2)
        return self.a2

    def backward(self, X, Y):
        m = X.shape[0]
        # Output layer error (softmax + cross-entropy simplifies to a2 - Y)
        dz2 = (self.a2 - Y) / m
        dW2 = self.a1.T @ dz2
        db2 = np.sum(dz2, axis=0, keepdims=True)
        da1 = dz2 @ self.W2.T
        dz1 = da1 * sigmoid_deriv(self.a1)
        dW1 = X.T @ dz1
        db1 = np.sum(dz1, axis=0, keepdims=True)
        # Gradient descent update
        self.W2 -= self.lr * dW2; self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1; self.b1 -= self.lr * db1

    def train(self, X, Y, epochs=1000, verbose_every=100):
        for e in range(epochs):
            out = self.forward(X)
            loss = -np.mean(np.sum(Y*np.log(out+1e-9), axis=1))
            self.loss_history.append(loss)
            self.backward(X, Y)
            if (e+1) % verbose_every == 0:
                print(f"  Epoch {e+1:4d}/{epochs}  Cross-Entropy Loss={loss:.4f}")

    def predict(self, X):
        return np.argmax(self.forward(X), axis=1)

print("=== Training MLP (1 hidden layer, sigmoid+softmax, backprop) ===")
print("Architecture: 10 inputs -> 16 hidden (sigmoid) -> 4 outputs (softmax)")
mlp = MLP(n_in=X_train_s.shape[1], n_hidden=16, n_out=n_classes, lr=0.5, seed=42)
mlp.train(X_train_s, y_train_oh, epochs=1500, verbose_every=150)

train_pred = mlp.predict(X_train_s)
test_pred = mlp.predict(X_test_s)
train_acc = accuracy_score(y_train_idx, train_pred)
test_acc = accuracy_score(y_test_idx, test_pred)
print(f"\nTrain Accuracy: {train_acc:.4f}")
print(f"Test  Accuracy: {test_acc:.4f}")
print("\nClassification report (test):")
print(classification_report(y_test_idx, test_pred, target_names=classes))

# Experiment with different hidden sizes / learning rates
print("\n=== Hyperparameter experiments ===")
exp_results = []
for hidden in [4, 8, 16, 32]:
    for lr in [0.05, 0.5]:
        m = MLP(n_in=X_train_s.shape[1], n_hidden=hidden, n_out=n_classes, lr=lr, seed=1)
        m.train(X_train_s, y_train_oh, epochs=800, verbose_every=10000)
        acc = accuracy_score(y_test_idx, m.predict(X_test_s))
        exp_results.append((hidden, lr, acc))
        print(f"  hidden={hidden:3d}  lr={lr:.2f}  test_acc={acc:.4f}")

# Plot training loss curve
plt.figure(figsize=(8,5))
plt.plot(mlp.loss_history)
plt.title("MLP Training Loss (Cross-Entropy) via Backpropagation")
plt.xlabel("Epoch"); plt.ylabel("Loss")
plt.tight_layout(); plt.savefig("figs/15_mlp_loss_curve.png", dpi=140); plt.close()

import seaborn as sns
cm = confusion_matrix(y_test_idx, test_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Oranges", xticklabels=classes, yticklabels=classes)
plt.xlabel("Predicted"); plt.ylabel("Actual"); plt.title("MLP Confusion Matrix")
plt.tight_layout(); plt.savefig("figs/16_mlp_confusion_matrix.png", dpi=140); plt.close()

# Plot hyperparameter comparison
plt.figure(figsize=(8,5))
hiddens = sorted(set(h for h,_,_ in exp_results))
for lr in sorted(set(l for _,l,_ in exp_results)):
    accs_lr = [a for h,l,a in exp_results if l==lr]
    plt.plot(hiddens, accs_lr, marker='o', label=f"lr={lr}")
plt.title("MLP Test Accuracy vs Hidden-Layer Size")
plt.xlabel("Hidden neurons"); plt.ylabel("Test Accuracy"); plt.legend()
plt.tight_layout(); plt.savefig("figs/17_mlp_hyperparam.png", dpi=140); plt.close()

import json
with open("data/q7_results.json","w") as f:
    json.dump({"train_acc": train_acc, "test_acc": test_acc,
               "experiments": exp_results}, f, indent=2)
