import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score

# True function
def true_fun(X):
    return np.cos(1.5 * np.pi * X)

np.random.seed(0)
X = np.sort(np.random.rand(30))
y = true_fun(X) + np.random.randn(30) * 0.1
degrees = [1, 4, 15]

# Plot models
plt.figure(figsize=(14, 5))
X_test = np.linspace(0, 1, 100)[:, None]

for i, degree in enumerate(degrees, 1):
    model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
    model.fit(X[:, None], y)
    mse = -cross_val_score(model, X[:, None], y, scoring="neg_mean_squared_error", cv=10)

    plt.subplot(1, 3, i)
    plt.plot(X_test, model.predict(X_test), label="Model")
    plt.plot(X_test, true_fun(X_test), label="True function")
    plt.scatter(X, y, edgecolor="b", s=20, label="Samples")
    plt.title(f"Degree {degree}\nMSE = {mse.mean():.2e} (+/- {mse.std():.2e})")
    plt.legend(loc="best")

plt.tight_layout()
plt.show()
