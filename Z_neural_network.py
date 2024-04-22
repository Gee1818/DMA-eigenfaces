# library import

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from functions_nn import *
from Z_clase_red import *

# set seed for reproducibility
np.random.seed(341)


##################### Training data #####################

x = pd.read_csv("components_train.csv")
# convert to numpy
X = x.drop("Name", axis=1)
X = standardize(X)
X = X.to_numpy()
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

##################### Network Architecture #####################

x = 60  # input features
y = Y.shape[1]  # output features
n1 = 12  # neurons in hidden layer 1
n2 = 12  # neurons in hidden layer 2
#n3 = 14  # neurons in hidden layer 3

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

# Training network

nn.train(X, Y)
evaluations = nn.evaluate(X, Y)

print(nn)



##################### Testing data #####################