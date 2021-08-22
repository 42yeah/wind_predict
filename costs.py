import numpy as np


class QuadraticCost:
    @staticmethod
    def sigmoid_prime(x):
        return np.exp(-x) / np.power(1.0 + np.exp(-x), 2)

    @staticmethod
    def delta(z, a, y):
        return (a - y) * QuadraticCost.sigmoid_prime(z)

    @staticmethod
    def cost(a, y, n):
        return np.sqrt(np.sum(np.power(a - y, 2))) / (2 * n)


class CrossEntropyCost:
    @staticmethod
    def delta(z, a, y):
        return a - y

    @staticmethod
    def cost(a, y, n):
        return -np.sum(y * np.log(a) + (1.0 - y) * np.log(1.0 - a)) / n
