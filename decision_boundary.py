import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons

# --- ДАННЫЕ ---
def create_data(n=200):
    X, y = make_moons(n_samples=n, noise=0.2, random_state=42)
    return X, y.reshape(-1, 1)

# --- АКТИВАЦИИ ---
def relu(x): return np.maximum(0, x)
def relu_deriv(x): return (x > 0).astype(float)
def sigmoid(x): return 1 / (1 + np.exp(-x))
def sigmoid_deriv(x): s = sigmoid(x); return s * (1 - s)

# --- МОДЕЛИ ---
class TinyNN:
    def __init__(self, input_dim=2, hidden_dim=4, output_dim=1, lr=0.1):
        self.W1 = np.random.randn(input_dim, hidden_dim) * 0.1
        self.b1 = np.zeros((1, hidden_dim))
        self.W2 = np.random.randn(hidden_dim, output_dim) * 0.1
        self.b2 = np.zeros((1, output_dim))
        self.lr = lr
        self.losses, self.weights = [], []

    def forward(self, X):
        self.z1 = X @ self.W1 + self.b1
        self.a1 = relu(self.z1)
        self.z2 = self.a1 @ self.W2 + self.b2
        self.out = sigmoid(self.z2)
        return self.out

    def compute_loss(self, y, y_pred):
        eps = 1e-8
        return -np.mean(y * np.log(y_pred + eps) + (1 - y) * np.log(1 - y_pred + eps))

    def backward(self, X, y):
        m = X.shape[0]
        d_out = self.out - y
        dW2 = self.a1.T @ d_out / m
        db2 = np.sum(d_out, axis=0, keepdims=True) / m

        da1 = d_out @ self.W2.T
        dz1 = da1 * relu_deriv(self.z1)
        dW1 = X.T @ dz1 / m
        db1 = np.sum(dz1, axis=0, keepdims=True) / m

        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1

    def train(self, X, y, epochs=400):
        for epoch in range(epochs):
            y_pred = self.forward(X)
            loss = self.compute_loss(y, y_pred)
            self.backward(X, y)

            self.losses.append(loss)
            self.weights.append((self.W1.copy(), self.W2.copy()))

            if epoch % 100 == 0:
                print(f"[Base] Epoch {epoch}, Loss: {loss:.4f}")

# Глубокая сеть: два слоя
class DeepNN:
    def __init__(self, input_dim=2, hidden_dim=4, output_dim=1, lr=0.1):
        self.W1 = np.random.randn(input_dim, hidden_dim) * np.sqrt(2 / input_dim)
        self.b1 = np.zeros((1, hidden_dim))
        self.W2 = np.random.randn(hidden_dim, hidden_dim) * np.sqrt(2 / hidden_dim)
        self.b2 = np.zeros((1, hidden_dim))
        self.W3 = np.random.randn(hidden_dim, output_dim) * np.sqrt(2 / hidden_dim)
        self.b3 = np.zeros((1, output_dim))
        self.lr = lr
        self.losses, self.weights = [], []

    def forward(self, X):
        self.z1 = X @ self.W1 + self.b1
        self.a1 = relu(self.z1)
        self.z2 = self.a1 @ self.W2 + self.b2
        self.a2 = relu(self.z2)
        self.z3 = self.a2 @ self.W3 + self.b3
        self.out = sigmoid(self.z3)
        return self.out

    def compute_loss(self, y, y_pred):
        eps = 1e-8
        return -np.mean(y * np.log(y_pred + eps) + (1 - y) * np.log(1 - y_pred + eps))

    def backward(self, X, y):
        m = X.shape[0]
        d_out = self.out - y
        dW3 = self.a2.T @ d_out / m
        db3 = np.sum(d_out, axis=0, keepdims=True) / m

        da2 = d_out @ self.W3.T
        dz2 = da2 * relu_deriv(self.z2)
        dW2 = self.a1.T @ dz2 / m
        db2 = np.sum(dz2, axis=0, keepdims=True) / m

        da1 = dz2 @ self.W2.T
        dz1 = da1 * relu_deriv(self.z1)
        dW1 = X.T @ dz1 / m
        db1 = np.sum(dz1, axis=0, keepdims=True) / m

        self.W3 -= self.lr * dW3
        self.b3 -= self.lr * db3
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1

    def train(self, X, y, epochs=400):
        for epoch in range(epochs):
            y_pred = self.forward(X)
            loss = self.compute_loss(y, y_pred)
            self.backward(X, y)

            self.losses.append(loss)
            self.weights.append((self.W1.copy(), self.W3.copy()))

            if epoch % 100 == 0:
                print(f"[Deep] Epoch {epoch}, Loss: {loss:.4f}")

# --- ВИЗУАЛИЗАЦИИ ---
def plot_decision_boundary(model, X, y, title=""):
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = 0.01
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    grid = np.c_[xx.ravel(), yy.ravel()]
    probs = model.forward(grid).reshape(xx.shape)

    plt.contourf(xx, yy, probs, levels=[0, 0.5, 1], alpha=0.3, cmap='coolwarm')
    plt.scatter(X[:, 0], X[:, 1], c=y.ravel(), cmap='coolwarm', edgecolors='k')
    plt.title(title)
    plt.show()

def plot_losses(models):
    plt.figure(figsize=(10, 4))
    for name, model in models.items():
        plt.plot(model.losses, label=name)
    plt.title("Loss over epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_weight_evolutions(models, layer="W1", i=0, j=0):
    plt.figure(figsize=(10, 4))
    for name, model in models.items():
        values = [w[0 if layer == "W1" else 1][i, j] for w in model.weights]
        plt.plot(values, label=name)
    plt.title(f"Weight {layer}[{i},{j}] evolution")
    plt.xlabel("Epoch")
    plt.ylabel("Weight value")
    plt.legend()
    plt.grid(True)
    plt.show()

# --- ЗАПУСК ---
if __name__ == "__main__":
    X, y = create_data()

    models = {
        "Base (4 neurons)": TinyNN(hidden_dim=4, lr=0.1),
        "Wide (8 neurons)": TinyNN(hidden_dim=8, lr=0.1),
        "Deep (2 layers)": DeepNN(hidden_dim=4, lr=0.1)
    }

    for name, model in models.items():
        print(f"Training {name}...")
        model.train(X, y, epochs=400)

    for name, model in models.items():
        plot_decision_boundary(model, X, y, title=name)

    plot_losses(models)
    plot_weight_evolutions(models, layer="W1", i=0, j=0)
