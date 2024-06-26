import json

import numpy as np
import pandas as pd

from helper_scripts.functions_eigen import *
from helper_scripts.functions_nn import *
from helper_scripts.clase_red import *
from helper_scripts.face_detection import *



test_raw_directory = "4.test_your_files/"
test_processed_directory = "4.test_your_files/preprocessed_test_files/"


face_detection_test(test_raw_directory, test_processed_directory)
images, names = read_images(test_processed_directory)


# Load train eigenvectors and mean face
eigenvectors = pd.read_csv(
    "reduced_eigenvectors.csv",
    header=None )

# Remove imaginary part (as strings) from eigenvectors.
eigenvectors = eigenvectors.map(lambda x: re.sub(r'\(|\)|\+0j', '', x) if isinstance(x, str) else x)

# Convert to numpy array
eigenvectors = eigenvectors.astype(float).to_numpy()

mean_face = pd.read_csv("mean_face.csv", header=None).to_numpy().flatten()

# substract mean face
images_centered = substract_mean_face(images, mean_face)

# Project faces to eigenspace

projected_faces = create_eigenface_space(eigenvectors, images_centered)

# standardize projected faces
df_train = pd.read_csv("components_train.csv")
projected_faces = standardize_test(projected_faces, df_train)

# reshape
projected_faces = np.reshape(
    projected_faces, (projected_faces.shape[0], projected_faces.shape[1], 1)
)

##################### Network Architecture #####################

with open("3.nn_params/nn_params.json", "r") as json_file:
    nn_architecture = json.load(json_file)


x = nn_architecture["x"]    # number of input features
y = nn_architecture["y"]    # number of outputs
n1 = nn_architecture["n1"]  # neurons in hidden layer 1
n2 = nn_architecture["n2"]  # neurons in hidden layer 2

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
nn.load_params()

# Predict
possible_predictions = nn_architecture["unique_values"]
predictions = nn.predict(projected_faces)

# Get names
predicted_names = [possible_predictions[pred] for pred in predictions]
print(predictions)

# Need to read to code to see if and where the filenames are stored
for i in range(len(names)):
    print(f"The file {names[i]} is {predicted_names [i]}")
