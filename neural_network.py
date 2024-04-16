# library import
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cv2.dnn import Layer

from functions_nn import *
from functions_nn import Logsig, Tansig, loss, loss_prime

# input data

# x = pd.read_csv("/home/ge/MCD/Data Mining Avanzado/dma-ros/datasets/cero.txt", sep="\t")
# convert to numpy
# X = np.array(x[["x1", "x2"]].values.tolist())
# Y = np.array(x["y"].values.tolist())

# # reshape
# X = np.reshape(X, (X.shape[0], X.shape[1], 1))
# Y = np.reshape(Y, (Y.shape[0], 1, 1))

# fake data for testing
X = np.random.randn(30, 60)  # 30 observations with 60 features
X = np.reshape(X, (X.shape[0], X.shape[1], 1))
Y = np.array(
    [
        [0, 0, 1, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
        [0, 0, 0, 1],
        [0, 0, 1, 0],
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [1, 0, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 0, 1, 0],
        [1, 0, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
        [0, 0, 0, 1],
        [1, 0, 0, 0],
        [1, 0, 0, 0],
        [1, 0, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
    ]
)
Y = np.reshape(Y, (Y.shape[0], Y.shape[1], 1))

# network architecture
x = 60  # input features
y = 4  # output features
n1 = 8  # neurons in hidden layer 1
n2 = 4  # neurons in hidden layer 2

network = [Layer(n1, x), Tansig(), Layer(n2, n1), Tansig(), Layer(y, n2), Logsig()]
# training parameters
learning_rate = 0.1
max_epochs = 10000
target_error = 1e-5
error = 10
epoch = 0

# training loop

while error > target_error and epoch < max_epochs:
    error = 0
    for x, y in zip(X, Y):
        # forward
        output = x
        for layer in network:
            # print(f"layer {layer}")
            # print(f"output {output}")
            output = layer.forward(output)

        # loss
        gradient = loss_prime(y, output)

        # backward
        for layer in reversed(network):
            gradient = layer.backward(gradient, learning_rate)

        # error
        error += loss(y, output)

    # update loop params
    epoch += 1
    error /= len(X)

    # print epoch error
    if epoch % 20 == 0:
        print(f"Epoch {epoch} Error {error}")

# final prediction
for x, y in zip(X, Y):
    output = x
    for layer in network:
        output = layer.forward(output)
    print(f"X: {x.flatten()} Y: {y.flatten()} Pred: {output.flatten()}")
