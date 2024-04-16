import numpy as np


# layer class
class Layer:
    def __init__(self, n_neurons, n_input):
        self.weights = np.random.uniform(
            -0.5, 0.5, size=(n_neurons, n_input)
        )  # random weights
        self.biases = np.random.uniform(-0.5, 0.5, size=(n_neurons, 1))  # random biases

    # forward
    def forward(self, inputs):
        self.inputs = inputs  # save inputs for backpropagation
        self.outputs = (
            np.dot(self.weights, self.inputs) + self.biases
        )  # calculate outputs for next layer
        return self.outputs

    # backward prop
    def backward(self, output_gradient, learning_rate):
        prev_layer_gradient = np.dot(
            self.weights.T, output_gradient
        )  # calculate gradient for previous layer
        weights_gradient = np.dot(
            output_gradient, self.inputs.T
        )  # calculate gradient for weights
        self.weights -= learning_rate * weights_gradient  # update weights
        self.biases -= learning_rate * output_gradient  # update biases
        return prev_layer_gradient


# activation classes
# linear
class Purelin:
    def __init__(self):
        pass

    def forward(self, x):
        return x

    def backward(self, output_gradient, learning_rate):
        return output_gradient


# sigmoid
class Logsig:
    def __init__(self):
        self.output = 0

    def forward(self, x):
        self.output = 1 / (1 + np.exp(-x))
        return self.output

    def backward(self, output_gradient, learning_rate):
        # return output_gradient * self.output * (1 - self.output)
        return np.multiply(output_gradient, self.output * (1 - self.output))


# tanh
class Tansig:
    def __init__(self):
        self.output = 0

    def forward(self, x):
        self.output = 2 / (1 + np.exp(-2 * x)) - 1
        return self.output

    def backward(self, output_gradient, learning_rate):
        return output_gradient * (1 - self.output * self.output)


# softmax
class Softmax:
    def __init__(self):
        self.output = 0  # initialize output

    def forward(self, x):
        self.output = np.exp(x) / np.sum(np.exp(x), axis=0)
        return self.output

    def backward(self, output_gradient, learning_rate):
        # may need reshaping (1, -1_
        deriv = self.output * np.identity(self.output.size) - np.dot(
            self.output.T, self.output
        )
        return np.dot(output_gradient, deriv)
        # another way
        # n = np.size(self.output)
        # return np.dot((np.identity(n) - self.output.T) * self.output, output_gradient)


# loss


def loss(y_true, y_pred):
    loss = np.mean(np.power(y_true - y_pred, 2))
    return loss


def loss_prime(y_true, y_pred):
    return 2 * (y_pred - y_true) / np.size(y_true)