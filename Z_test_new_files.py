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

x = 60  # input features
y = 19
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


# Load trained weights
for i in range(len(layers)):
    nn.weights.append(
        pd.read_csv(f"5.nn_params/weights_{i}.csv", header=None).to_numpy()
    )
    nn.biases.append(pd.read_csv(f"5.nn_params/biases_{i}.csv", header=None).to_numpy())

# Predict
nn.predict(projected_faces)
