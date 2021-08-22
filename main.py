import net
import costs
import mnist_loader


if __name__ == "__main__":
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
    network = net.Net([784, 30, 10], costs.QuadraticCost)
    network.stochastic_gradient_descent(30, list(training_data), 10, 1.0, list(test_data),
                                        monitor_evaluation_accuracy=True)
