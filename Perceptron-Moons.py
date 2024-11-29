
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split


X, y = make_moons(n_samples=1000, noise=0.2, random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)


def heaviside(x):
    return torch.where(x >= 0, torch.tensor(1.0), torch.tensor(0.0))


def perceptron_infer(x, params):
    w, b = params
    linear_output = x @ w + b
    return heaviside(linear_output)


def perceptron_loss(label, true_label):
    return (label != true_label).sum().item()


def optimizer_step(x, y, params, y_pred_precomputed=None):
    w, b = params
    if y_pred_precomputed is None:
        y_pred_precomputed = perceptron_infer(x, params)
    for i in range(x.shape[0]):
        if y_pred_precomputed[i] != y[i]:  
            w += (y[i] - y_pred_precomputed[i]) * x[i]
            b += (y[i] - y_pred_precomputed[i])
    return w, b


def initial_weight(x):
    w = torch.randn(x.shape[1], dtype=torch.float32)  
    b = torch.randn(1, dtype=torch.float32)  
    return w, b


def train_perceptron(x, y, epochs=10):
    params = initial_weight(x)
    for epoch in range(epochs):
        y_pred = perceptron_infer(x, params)
        loss = perceptron_loss(y_pred, y)
        print(f"Loss at {loss} in epoch {epoch}")
        params = optimizer_step(x, y, params, y_pred_precomputed=y_pred)
    return params
    

params = train_perceptron(X_train, y_train, epochs=10)


def plot_decision_boundary(params, X, y):
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = torch.meshgrid(torch.arange(x_min, x_max, 0.01),
        torch.arange(y_min, y_max, 0.01))
    
    
    grid = torch.stack([xx.ravel(), yy.ravel()], dim=1)
    Z = perceptron_infer(grid, params).reshape(xx.shape)

    
    plt.scatter(X[:, 0], X[:, 1], c=y, s=40, edgecolor="k", cmap="coolwarm")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title("Perzeptron Entscheidungsgrenze")
    plt.show()


plot_decision_boundary(params, X_test, y_test)


  






