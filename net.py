import random
import numpy as np
import matplotlib.pyplot as plt
import markov


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
        self.weights = [np.random.randn(y, x) / np.sqrt(x) for x, y in zip(sizes[:-1], sizes[1:])]
        self.biases = [np.random.randn(x, 1) for x in sizes[1:]]
        self.cost_func = cost_func

    def feedforward(self, x):
        for w, b in zip(self.weights, self.biases):
            x = sigmoid(np.matmul(w, x) + b)
        return x

    def stochastic_gradient_descent(self, epoch, training_data, mini_batch_size, eta, lmbda, test_data,
                                    monitor_evaluation_cost=False,
                                    render_evaluation=False,
                                    monitor_evaluation_rmse=False,
                                    render_rpd=False):
        for i in range(epoch):
            print(f"Commencing epoch {i}...")
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k + mini_batch_size]
                for k in range(0, len(training_data), mini_batch_size)
            ]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta, lmbda, len(training_data))

            print(f"Epoch {i} complete. ", end="")
            if monitor_evaluation_cost or render_evaluation or \
                    monitor_evaluation_rmse or \
                    render_rpd:
                total_cost = 0
                rmse = 0
                predicted = []
                expected = []
                rpd = []
                # test_data = sorted(test_data, key=lambda x: x[1])
                for x, y in test_data:
                    a = self.feedforward(x)
                    if len(predicted) < 100:
                        predicted.append(a[0][0])
                        expected.append(y[0][0])
                    rmse += np.power(np.sum(a - y), 2)
                    total_cost += self.cost_func.cost(a, y, len(test_data))
                    rpd.append(((a - y) / (np.abs(a) + np.abs(y)))[0][0])
                rmse = np.sqrt(rmse / len(test_data))
                if monitor_evaluation_cost:
                    print(f"Evaluation data cost: {total_cost}", end=", ")
                if monitor_evaluation_rmse:
                    print(f"Evaluation RMSE: {rmse}", end=", ")
                if render_evaluation:
                    fig = plt.figure()
                    ax = plt.axes()
                    ax.plot(predicted, color="b")
                    ax.plot(expected, color="r")
                    plt.show()
                if render_rpd:
                    fig = plt.figure()
                    ax = plt.axes()
                    ax.plot(rpd, color="b")
                    plt.show()
            print("done.")
        # Apply markov model?
        rpd = []
        for x, y in test_data[:6900]:
            a = self.feedforward(x)
            rpd.append(((a - y) / (np.abs(a) + np.abs(y)))[0][0])
        markov_model = markov.Markov(2)
        markov_model.train(rpd)

        # The 6900th data
        a = self.feedforward(test_data[6900][0])
        z = markov.one_hot_state(((a - test_data[6900][1]) / (np.abs(a) + np.abs(test_data[6900][1])))[0][0])
        mods = np.array([-0.75, -0.25, 0.25, 0.75])
        results = [[], [], []]
        rmse_mod = 0
        rmse_a = 0
        for i in range(6900, len(test_data), 2):
            preds = markov_model.predict(z)
            # ...predict the next one...
            for j in range(1, 3):
                if i + j >= len(test_data):
                    continue
                x = test_data[i + j][0]
                a = self.feedforward(x)[0][0]
                y = test_data[i + j][1][0][0]
                results[0].append(y)
                results[1].append(a)
                mod = (1 - np.dot(preds[j - 1], mods)) * a
                results[2].append(mod[0])
                rmse_a += np.power(np.sum(a - y), 2)
                rmse_mod += np.power(np.sum(mod - y), 2)
            z = preds[-1]
        print(f"RMSEmod: {np.sqrt(rmse / 100)}, RMSEa: {np.sqrt(rmse_a / 100)}")
        fig = plt.figure()
        ax = plt.axes()
        ax.plot(results[0], color="b")
        ax.plot(results[1], color="g")
        ax.plot(results[2], color="r")
        plt.show()


        print("Done")


    def update_mini_batch(self, mini_batch, eta, lmbda, n):
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
