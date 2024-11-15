import numpy as np
import matplotlib.pyplot as plt

def mean_squared_error(y_true, y_predicted):
    return np.sum((y_true - y_predicted) ** 2) / len(y_true)

def gradient_descent(x, y, iterations=2000, learning_rate=0.0001, stopping_threshold=1e-6):
    weight, bias = 0.1, 0.01
    n = float(len(x))
    costs, weights = [], []

    for i in range(iterations):
        y_predicted = (weight * x) + bias
        current_cost = mean_squared_error(y, y_predicted)
        if i > 0 and abs(costs[-1] - current_cost) <= stopping_threshold:
            break
        costs.append(current_cost)
        weights.append(weight)
        
        weight -= learning_rate * (-2/n) * sum(x * (y - y_predicted))
        bias -= learning_rate * (-2/n) * sum(y - y_predicted)
        
    plt.plot(weights, costs)
    plt.xlabel("Weight"), plt.ylabel("Cost")
    plt.show()
    return weight, bias

def main():
    X = np.array([32.5, 53.4, 61.5, 47.5, 59.8, 55.1, 52.2, 39.3, 48.1, 52.5, 45.4, 54.3, 44.1, 58.1, 56.7, 48.9, 44.6, 60.2, 45.6, 38.8])
    Y = np.array([31.7, 68.7, 62.5, 71.5, 87.2, 78.2, 79.6, 59.1, 75.3, 71.3, 55.1, 82.4, 62.0, 75.3, 81.4, 60.7, 82.8, 97.3, 48.8, 56.8])
    w, b = gradient_descent(X, Y)
    Y_pred = w * X + b

    plt.scatter(X, Y, color='red')
    plt.plot(X, Y_pred, color='blue', linestyle='dashed')
    plt.xlabel("X"), plt.ylabel("Y")
    plt.show()

if __name__ == "__main__":
    main()
