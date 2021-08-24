import numpy as np


def statify_single(num):
    if num < -0.5:
        return 0
    if num < 0.0:
        return 1
    if num < 0.5:
        return 2
    return 3


statify = np.vectorize(statify_single)
fixes = [-0.75, -0.25, 0.25, 0.75]


def one_hot_state(num):
    state = statify_single(num)
    z = np.zeros((1, 4))
    z[0][state] = 1
    return z


class Markov:
    def __init__(self, k):
        self.k = k
        self.stochastic_matrices = [np.zeros((4, 4)) for i in range(k)]

    def train(self, err):
        sequence = statify(err)
        state_counts = [
            np.sum(x == 0 for x in sequence),
            np.sum(x == 1 for x in sequence),
            np.sum(x == 2 for x in sequence),
            np.sum(x == 3 for x in sequence),
        ]
        # print(f"Seq {sequence}; StateC {state_counts}")
        for index in range(1, len(sequence)):
            j = sequence[index]
            for t in range(-1, -self.k - 1, -1):
                if index + t < 0:
                    continue
                i = sequence[index + t]
                # print(f"indx {index}; t {t}; i {i}; j {j}")
                self.stochastic_matrices[-t - 1][i][j] += 1
        for mat in self.stochastic_matrices:
            for i in range(0, len(state_counts)):
                if state_counts[i] != 0:
                    mat[i, :] /= state_counts[i]
                else:
                    mat[i, :] = [0, 0, 0, 0]

    def predict(self, z):
        ret = []
        for i in range(0, self.k):
            z = np.matmul(z, self.stochastic_matrices[i])
            ret.append(z)
        return ret
