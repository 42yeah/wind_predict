import net
import costs
import huge_reader
import numpy as np
import random


if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    training_data, test_data = huge_reader.load_data()
    network = net.Net([5, 100, 1], costs.QuadraticCost)
    network.stochastic_gradient_descent(400, training_data, 3, 0.5, 1.0, test_data,
                                        monitor_evaluation_cost=True,
                                        render_evaluation=True,
                                        monitor_evaluation_rmse=True)
