import random
import costs
import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def sigmoid_prime(x):
    return np.exp(-x) / np.power(1.0 + np.exp(-x), 2)


one_hots = [np.zeros((10, 1)) for _ in range(0, 10)]
for i in range(0, 10):
    one_hots[i][i][0] = 1.0


class Net:
    def __init__(self, sizes, cost_func):
        self.sizes = sizes
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]
        self.biases = [np.random.randn(x, 1) for x in sizes[1:]]
        self.cost_func = cost_func

    def feedforward(self, x):
        for w, b in zip(self.weights, self.biases):
            x = sigmoid(np.matmul(w, x) + b)
        return x

    def stochastic_gradient_descent(self, epoch, training_data, mini_batch_size, eta, test_data):
        for i in range(epoch):
            print(f"Commencing epoch {i}...")
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k + mini_batch_size]
                for k in range(0, len(training_data), mini_batch_size)
            ]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            total_cost = 0
            for x, y in test_data:
                a = self.feedforward(x)
                total_cost += self.cost_func.cost(a, one_hots[y], a.shape[0])
            total_cost /= len(test_data)
            print(f"Epoch {i} complete. Test cost: {total_cost}.")

    def update_mini_batch(self, mini_batch, eta):
        nabla_w = [np.zeros(x.shape) for x in self.weights]
        nabla_b = [np.zeros(x.shape) for x in self.biases]
        for x, y in mini_batch:
            del_nabla_w, del_nabla_b = self.backprop(x, y)
            nabla_w = [w + dw for w, dw in zip(nabla_w, del_nabla_w)]
            nabla_b = [b + db for b, db in zip(nabla_b, del_nabla_b)]
        self.weights = [w - (eta / len(mini_batch)) * dw for w, dw in zip(self.weights, nabla_w)]
        self.biases = [b - (eta / len(mini_batch)) * db for b, db in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        nabla_w = [np.zeros(x.shape) for x in self.weights]
        nabla_b = [np.zeros(x.shape) for x in self.biases]
        activation = x
        activations = [x]
        zs = []
        for w, b in zip(self.weights, self.biases):
            z = np.matmul(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        delta = self.cost_func.delta(zs[-1], activation, y)
        nabla_b[-1] = delta
        nabla_w[-1] = np.matmul(delta, activations[-2].transpose())
        for i in range(2, len(self.sizes)):
            delta = np.matmul(self.weights[-i + 1].transpose(), delta) * sigmoid_prime(zs[-i])
            nabla_b[-i] = delta
            nabla_w[-i] = np.matmul(delta, activations[-i - 1].transpose())
        return nabla_w, nabla_b
