import net
import costs
import huge_reader


if __name__ == "__main__":
    training_data, test_data = huge_reader.load_data()
    network = net.Net([5, 100, 1], costs.QuadraticCost)
    network.stochastic_gradient_descent(30, training_data, 10, 1.0, test_data,
                                        monitor_evaluation_cost=True)
