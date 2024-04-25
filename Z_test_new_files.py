import json
import sys

sys.path.append("helper_scripts/")

import numpy as np
import pandas as pd
from face_extraction import *

from functions import *
from functions_nn import *
from Z_clase_red import *

test_raw_directory = "4.test_your_files/"
test_processed_directory = "4.test_your_files/preprocessed_test_files/"


face_detection_test(test_raw_directory, test_processed_directory)
images, names = read_images(test_processed_directory)


# Load train eigenvectors and mean face
eigenvectors = pd.read_csv(
    "reduced_eigenvectors.csv", header=None, dtype=float
).to_numpy()
mean_face = pd.read_csv("mean_face.csv", header=None).to_numpy().flatten()

# substract mean face
images_centered = substract_mean_face(images, mean_face)

# Project faces to eigenspace

projected_faces = create_eigenface_space(eigenvectors, images_centered)

# standardize projected faces
projected_faces = standardize(projected_faces)

# reshape
projected_faces = np.reshape(
    projected_faces, (projected_faces.shape[0], projected_faces.shape[1], 1)
)

##################### Network Architecture #####################

with open("5.nn_params/nn_params.json", "r") as json_file:
    nn_architecture = json.load(json_file)


x = nn_architecture["x"]  # input features
y = nn_architecture["y"]  # input features
n1 = nn_architecture["n1"]  # neurons in hidden layer 1
n2 = nn_architecture["n2"]  # neurons in hidden layer 2
# n3 = 14  # neurons in hidden layer 3

layers = [Layer(n1, x), Layer(n2, n1), Layer(y, n2)]
activation_map = {
    "Tansig": Tansig(),
    "Logsig": Logsig(),
    "Softmax": Softmax(),
    "Purelin": Purelin(),
}
activations = []
for activation in nn_architecture["activations"]:
    activations.append(activation_map[activation])


# Network parameters
params = nn_architecture["params"]

# Loss function
loss_map = {"CrossEntropyLoss": CrossEntropyLoss()}
loss = loss_map[nn_architecture["loss"]]

# Instantiating the network
nn = n_network(params, layers, activations, loss)


# Load trained weights
for i, layer in enumerate(nn.layers):
    print(f"i: {i}")
    layer.weights = pd.read_csv(f"5.nn_params/weights_{i}.csv", header=None).to_numpy()
    layer.biases = pd.read_csv(f"5.nn_params/biases_{i}.csv", header=None).to_numpy()


# Predict
possible_predictions = nn_architecture["unique_values"]
predictions = nn.predict(projected_faces)

# Get names
names = [possible_predictions[pred] for pred in predictions]

# Need to read to code to see if and where the filenames are stored
# for i in range(len(names)):
#     print(f"The file {filenames[i]} is {names[i]}")
