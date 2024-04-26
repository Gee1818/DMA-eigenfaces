import numpy as np
import pandas as pd

# disable scientific notation
np.set_printoptions(suppress=True, precision=2)


class n_network:
    def __init__(self, params, layers, activations, loss_function):
        self.layers = layers  # list of layer clasess
        self.learning_rate = params["learning_rate"]
        self.max_epochs = params["epochs"]
        self.target_error = params["target_error"]
        self.weights = []
        self.biases = []
        self.activations = activations  # list of activations per layer
        self.loss = loss_function  # CrossEntropyLoss class in functions_nn.py
        self.accuracy = 0
        self.trained = False

    def train(self, X, Y):
        epoch = 0
        error = 10

        while error > self.target_error and epoch < self.max_epochs:
            error = 0
            for x, y in zip(X, Y):

                # Forward pass
                activation = self.single_forward(x)

                # Calculate loss and gradient
                error += self.loss.calc(y, activation)
                gradient = self.loss.prime(y, activation)

                # Backward pass
                layers_and_activs = [
                    val for pair in zip(self.layers, self.activations) for val in pair
                ]  # https://stackoverflow.com/a/7946825
                for layer in reversed(layers_and_activs):
                    gradient = layer.backward(gradient, self.learning_rate)

            # Update loop params
            epoch += 1
            error /= len(X)

            # Save parameters
            self.save_params()

            # print epoch error
            if epoch % 20 == 0:
                print(f"Epoch: {epoch} - Error: {error}")

            # adaptive learning rate
            if epoch % 4000 == 0:
                self.learning_rate /= 10

    def single_forward(self, x):  # Forward pass of a single observation
        # input
        activation = x
        # hidden + output layers
        for layer, activ_fun in zip(self.layers, self.activations):
            output = layer.forward(activation)  # z_j
            activation = activ_fun.forward(output)  # a_j
        return activation

    def predict(self, X):
        if self.trained:
            print("The model is not trained")
            return
        activations = [self.single_forward(x) for x in X]
        predictions = [np.argmax(activ, axis=0)[0] for activ in activations]
        return predictions

    def evaluate(self, X, Y):
        predictions = self.predict(X)
        correct_preds = [pred == np.argmax(y) for pred, y in zip(predictions, Y)]
        self.accuracy = np.mean(correct_preds)
        print(f"Accuracy: {self.accuracy}")
        return self.accuracy

    def save_params(self):
        self.weights += [layer.weights for layer in self.layers]
        self.biases += [layer.biases for layer in self.layers]
        
    def load_params(self):
    	for i, layer in enumerate(self.layers):
    		layer.weights = pd.read_csv(f"5.nn_params/weights_{i}.csv", header=None).to_numpy()
    		layer.biases = pd.read_csv(f"5.nn_params/biases_{i}.csv", header=None).to_numpy()

    def __str__(self) -> str:
        result = f"Neural Network with {len(self.layers)} layers:\n"
        for i, layer in enumerate(self.layers):
            result += f"Layer {i+1}:\n"
            result += f"Weights: {layer.weights}\n"
            result += f"Biases: {layer.biases}\n"
        return result

    def export_params(self):
        for i, layer in enumerate(self.layers):
            np.savetxt(f"5.nn_params/weights_{i}.csv", layer.weights, delimiter=",")
            np.savetxt(f"5.nn_params/biases_{i}.csv", layer.biases, delimiter=",")
        print("Parameters exported")
