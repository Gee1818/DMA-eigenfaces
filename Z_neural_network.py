# library import

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from functions_nn import *
from Z_clase_red import *

# set seed for reproducibility
np.random.seed(341)


##################### Training data #####################

X, Y = get_data("components_train.csv")


# one hot encoding


def one_hot_encode(array):
    unique = np.unique(array)
    df = pd.DataFrame(0, columns=unique, index=range(len(array)))
    for index, label in enumerate(array):
        df.loc[index, label] = 1
    return df.to_numpy(), unique


def one_hot_encode_test(array, unique_train):
    df = pd.DataFrame(0, columns=unique_train, index=range(len(array)))
    for index, label in enumerate(array):
        df.loc[index, label] = 1
    return df.to_numpy()


Y, unique_values = one_hot_encode(Y)

# # reshape
X = np.reshape(X, (X.shape[0], X.shape[1], 1))
Y = np.reshape(Y, (Y.shape[0], Y.shape[1], 1))

##################### Network Architecture #####################

x = X.shape[1]  # input features
y = Y.shape[1]  # output features
n1 = 12  # neurons in hidden layer 1
n2 = 12  # neurons in hidden layer 2
# n3 = 14  # neurons in hidden layer 3

layers = [Layer(n1, x), Layer(n2, n1), Layer(y, n2)]
activations = [Tansig(), Tansig(), Softmax()]

# Network parameters
params = {
    "learning_rate": 1,
    "epochs": 100,
    "target_error": 1e-4,
}

# Loss function
loss = CrossEntropyLoss()

# Instantiating the network
nn = n_network(params, layers, activations, loss)


##################### Training network #####################

nn.train(X, Y)

evaluations = nn.evaluate(X, Y)

print(f"Training accuracy: {evaluations}")


##################### Testing data #####################

df_train = pd.read_csv("components_train.csv")
X_test, Y_test = get_data_test("components_test.csv", df_train)

Y_test = one_hot_encode_test(Y_test, unique_values)

X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
Y_test = np.reshape(Y_test, (Y_test.shape[0], Y_test.shape[1], 1))

evaluations_t = nn.evaluate(X_test, Y_test)

print(f"Test accuracy: {evaluations_t}")



##################### Save the model #####################

save = input("Do you want to save the model? (y/n): ")
if save == "y":
    nn.export_params()
    nn_params = {
        "x": x,
        "y": y,
        "unique_values": unique_values.tolist(),
        "n1": n1,
        "n2": n2,
        "layers": [layer.__class__.__name__ for layer in layers],
        "activations": [activation.__class__.__name__ for activation in activations],
        "params": params,
        "loss": loss.__class__.__name__,
    }

    with open("5.nn_params/nn_params.json", "w") as file:
        json.dump(nn_params, file)

    print("Model saved")
