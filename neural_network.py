# library import
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from functions_nn import *
from functions_nn import Logsig, Tansig, cross_loss, cross_loss_prime

# from cv2.dnn import Layer

# set seed for reproducibility
np.random.seed(341)


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

def one_hot_encode_test(array, unique_train):
    df = pd.DataFrame(0, columns=unique_train, index=range(len(array)))
    for index, label in enumerate(array):
        df.loc[index, label] = 1
    return df.to_numpy()


Y, unique_values = one_hot_encode(Y)

# # reshape
X = np.reshape(X, (X.shape[0], X.shape[1], 1))
Y = np.reshape(Y, (Y.shape[0], Y.shape[1], 1))


# network architecture
x = 60  # input features
y = Y.shape[1]  # output features
n1 = 14  # neurons in hidden layer 1
n2 = 14  # neurons in hidden layer 2
n3 = 14  # neurons in hidden layer 3

network = [
    Layer(n1, x),
    Tansig(),
    Layer(n2, n1),
    Tansig(),
    Layer(n3, n2),
    Tansig(),
    Layer(y, n3),
    Softmax(),
]
# training parameters
learning_rate = 1
max_epochs = 10000
target_error = 1e-2
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

        # loss comment the one you want to use
        # gradient = MSE_loss_prime(y, output)
        gradient = cross_loss_prime(y, output)

        # backward
        for layer in reversed(network):
            gradient = layer.backward(gradient, learning_rate)

        # error comment the one you want to use
        # error += MSE_loss(y, output)
        error += cross_loss(y, output)

    # update loop params
    epoch += 1
    error /= len(X)

    # print epoch error
    if epoch % 20 == 0:
        print(f"Epoch {epoch} Error {error}")
    if epoch % 4000 == 0:
        learning_rate = learning_rate / 10

# disable scientific notation
np.set_printoptions(suppress=True, precision=2)
# final prediction
for x, y in zip(X, Y):
    output = x
    for layer in network:
        output = layer.forward(output)
    # print(f"Y: {y.flatten()} Pred: {output.flatten()}")


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
    # TODO: define a minimun threshold for prediction
    prediction = np.argmax(output)
    # TODO: Add a row to df for indeterminate predictions
    if prediction == np.argmax(y):
        df.iloc[prediction, 2] = int(df.iloc[prediction, 2]) + 1

# TODO: Add a row for total image count and the total of correct predictions
print(df)
# convert to numeric , was giving error
df["True_Prediction"] = pd.to_numeric(df["True_Prediction"], errors="coerce")
df["img_qty"] = pd.to_numeric(df["img_qty"], errors="coerce")

# calculate accuracy
accuracy = df["True_Prediction"].sum() / df["img_qty"].sum()
print("Accuracy: {:.2%}".format(accuracy))


# ==============================================================


# Compare with test dataset
x_test = pd.read_csv("components_test.csv")

X_test = x_test.drop("Name", axis=1).to_numpy()
Y_test = np.array(x_test["Name"].values.tolist())

Y_test = one_hot_encode_test(Y_test, unique_values)

X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
Y_test = np.reshape(Y_test, (Y_test.shape[0], Y_test.shape[1], 1))

# Forward pass through the net
num, den = 0, 0
for x, y in zip(X_test, Y_test):
        output = x
        for layer in network:
            output = layer.forward(output)

        parsed_output = np.zeros(output.shape[0], dtype = int)
        guessed_response = np.argmax(output)
        parsed_output[guessed_response] = 1

        parsed_y = np.reshape(y, y.shape[0])

        #print("Output:", parsed_output)
        #print("Answer:", parsed_y)
        #print()

        if np.array_equal(parsed_y, parsed_output):
            num += 1
        den += 1

# Check accuracy on test set
print("Test accuracy: {frac} ({num} correct out of {den}).".format(num = num, den = den, frac = num/den))