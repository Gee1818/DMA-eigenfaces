import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.lib.type_check import real

from helper_scripts.functions_eigen import *

# Define path for input images
path_og = "2.cropped_photos/"

number_of_images_train = 10 # per person
n_components = 20


def main():

    # Read images
    images, names = read_images(
        path_og
    )  # Images matrix -->(num_images, 900), names -->list

    # Select training set
    train, test, train_idx, test_idx, train_names, test_names = select_training_set(
        images, names, number_of_images_train
    )  # train --> matrix(num_images*names, 900)

    train, train_names = augment_training_set(train, train_names)

    # Calculate mean face
    mean_face = calculate_mean_face(train)  # array(900,)

    # Save mean face
    mean_face_df = pd.DataFrame(mean_face)
    mean_face_df.to_csv("mean_face.csv", index=False, header=False)

    # Subtract mean face from images
    train_standarized = substract_mean_face(
        train, mean_face
    )  # matrix(num_images*names, 900)

    # Calculate covariance matrix
    covariance_matrix = calculate_covariance(train_standarized)  # matrix(900, 900)

    # Get Eeigenvalues and eigenvectors
    eigenvalues, eigenvectors = calculate_eigenvalues(
        covariance_matrix
    )  # array(900,), matrix(900, 900)

    # print(explained_variance[:n_components])
    reduced_eigenvectors = eigenvectors[:, :n_components]  # matrix(900, n_components)

    # Save reduced eigenvectors to a csv
    red_eig = pd.DataFrame(reduced_eigenvectors.real)
    red_eig.to_csv("reduced_eigenvectors.csv", index=False, header=False)

    # Project images into the reduced eigenface space
    reduced_eigenface_space = create_eigenface_space(
        reduced_eigenvectors, train_standarized
    )  # matrix(num_images*names, n_components)

    # Create two DataFrames (train + test) with the first n_components PCs of each image
    for tag, split, names in [
        ("train", train, train_names),
        ("test", test, test_names),
    ]:
        centered_imgs = substract_mean_face(split, mean_face)
        eigenspace = create_eigenface_space(reduced_eigenvectors, centered_imgs)
        pca_df = pd.DataFrame(eigenspace)
        pca_df.columns = ["PC" + str(i) for i in range(1, n_components + 1)]
        pca_df.insert(0, "Name", names, True)
        pca_df.sort_values(by="Name", inplace=True)
        pca_df.to_csv(f"components_{tag}.csv", index=False)
        if tag == "train":
            train_df = pca_df.copy()


if __name__ == "__main__":
    main()