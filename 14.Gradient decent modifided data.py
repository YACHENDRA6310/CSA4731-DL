import numpy as np
import matplotlib.pyplot as plt

def mean_squared_error(y_true, y_predicted):
    return np.sum((y_true - y_predicted)**2) / len(y_true)

def gradient_descent(x, y, iterations=1000, learning_rate=0.0001, stopping_threshold=1e-6):
    weight, bias = 0.1, 0.01
    n = len(x)
    costs, weights, prev_cost = [], [], None

    for i in range(iterations):
        y_pred = weight * x + bias
        cost = mean_squared_error(y, y_pred)
        if prev_cost and abs(prev_cost - cost) <= stopping_threshold: break
        prev_cost = cost
        costs.append(cost)
        weights.append(weight)

        weight -= learning_rate * -(2/n) * sum(x * (y - y_pred))
        bias -= learning_rate * -(2/n) * sum(y - y_pred)

    plt.plot(weights, costs)
    plt.scatter(weights, costs, color='red')
    plt.title("Cost vs Weights")
    plt.show()

    return weight, bias

def main():
    X = np.array([52.5, 63.4, 81.5, 47.5, 89.8, 55.1, 52.2, 39.3, 48.1, 52.5, 45.4, 54.3, 44.1, 58.2, 56.7, 48.9, 44.6, 60.3, 45.6, 38.8])
    Y = np.array([41.7, 78.8, 82.6, 91.5, 77.2, 78.2, 79.6, 59.2, 75.3, 71.3, 55.2, 82.5, 62.0, 75.4, 81.4, 60.7, 82.9, 97.4, 48.8, 56.9])

    weight, bias = gradient_descent(X, Y, iterations=2000)
    print(f"Estimated Weight: {weight}\nEstimated Bias: {bias}")

    Y_pred = weight * X + bias
    plt.scatter(X, Y, color='pink')
    plt.plot([min(X), max(X)], [min(Y_pred), max(Y_pred)], color='blue', linestyle='dashed')
    plt.show()

if __name__ == "__main__":
    main()
