# library import
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from functions_nn import *
from functions_nn import Logsig, Tansig, loss, loss_prime

# from cv2.dnn import Layer


# input data

x = pd.read_csv("components_train.csv")
# convert to numpy
X = x.drop("Name", axis=1).to_numpy()
Y = np.array(x["Name"].values.tolist())

# one hot encoding


def one_hot_encode(array):
    unique = np.unique(array)
    df = pd.DataFrame(0, columns=unique, index=range(len(array)))
    for index, label in enumerate(array):
        df.loc[index, label] = 1
    return df.to_numpy(), unique


Y, unique_values = one_hot_encode(Y)

# # reshape
X = np.reshape(X, (X.shape[0], X.shape[1], 1))
Y = np.reshape(Y, (Y.shape[0], Y.shape[1], 1))


# network architecture
x = 60  # input features
y = Y.shape[1]  # output features
n1 = 18  # neurons in hidden layer 1
n2 = 18  # neurons in hidden layer 2

network = [Layer(n1, x), Tansig(), Layer(n2, n1), Tansig(), Layer(y, n2), Logsig()]
# training parameters
learning_rate = 0.1
max_epochs = 2000
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

# disable scientific notation
np.set_printoptions(suppress=True, precision=2)
# final prediction
for x, y in zip(X, Y):
    output = x
    for layer in network:
        output = layer.forward(output)
    print(f"Y: {y.flatten()} Pred: {output.flatten()}")


image_sum = np.sum(Y, axis=0)
dataset = np.column_stack(
    (unique_values, image_sum, np.zeros(np.shape(Y[0]), dtype=int))
)
df = pd.DataFrame(data=dataset, columns=["Names", "img_qty", "True_Prediction"])

# export results to csv with Y and output

for x, y in zip(X, Y):
    output = x
    for layer in network:
        output = layer.forward(output)
    prediction = np.argmax(output)
    if prediction == np.argmax(y):
        df.iloc[prediction, 2] = int(df.iloc[prediction, 2]) + 1

print(df)
