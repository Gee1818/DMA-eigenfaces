import sys
sys.path.append('helper_scripts/')

from functions import *
from functions_nn import *
from face_extraction import *

import pandas as pd
import numpy as np


test_raw_directory = "4.test_your_files/"
test_processed_directory = "4.test_your_files/preprocessed_test_files/"


face_detection_test(test_raw_directory, test_processed_directory)
images, names = read_images(test_processed_directory)

print("Loaded images..")
print(names)

print(images.shape)

# Load train eigenvectors and mean face
eigenvectors = pd.read_csv("reduced_eigenvectors.csv", header=None).to_numpy()
mean_face = pd.read_csv("mean_face.csv", header=None).to_numpy().flatten()

print("mean_face shape: ", mean_face.shape)
print("mean_face type: ", type(mean_face))
print("\n")

print("Images matrix:")
print(images)
print(type(images))
print("\n")

print("eigenvectors shape: ", eigenvectors.shape)
print("eigenvectors type: ", type(eigenvectors))
print("\n")

# Calculate reduced space
eig_space = create_eigenface_space(eigenvectors, images)
print(type(eig_space))
print("\n")


df_eig_space = pd.DataFrame(eig_space)
#print(df_eig_space.head(4))
print("\n")

print(df_eig_space[0].iloc[0])
# Standardize the reduced space
# Add name in column 0
'''
df_eig_space = standardize(df_eig_space)

print(type(df_eig_space))

df_eig_space.insert(0, "Name", names)
print(df_eig_space.head(5))

'''




