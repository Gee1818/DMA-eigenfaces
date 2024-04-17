import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.lib.type_check import real

from functions import *

# Define path for input images
# path = "/home/frank/maestria_mcd/nuestras_caras/Nuestras Caras-20240321T193416Z-001/test_photos/"
path = "/home/ge/MCD/Data Mining Avanzado/DMA-eigenfaces-output/"


def main():

    # Read images
    images, names = read_images(
        path
    )  # Images matrix -->(num_images, 900), names -->list

    # Select training set
    train, test, train_idx, test_idx, train_names, test_names = select_training_set(
        images, names, 8
    )  # train --> matrix(num_images*names, 900)

    # Calculate mean face
    mean_face = calculate_mean_face(train)  # array(900,)

    # Plot mean face
    # plt.imshow(mean_face.reshape(30, 30), cmap="gray")
    # plt.show()

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

    # Plot eigenfaces
    # plot_images(np.real(eigenvectors.T), range(1, 21), 4, 5, 0, 20)

    # Calculate explained variance
    # explained_variance = calculate_explained_variance(eigenvalues)  # array(900,)
    n_components = 60
    ##print(explained_variance[:n_components])
    reduced_eigenvectors = eigenvectors[:, :n_components]  # matrix(900, n_components)

    # Project images into the reduced eigenface space
    reduced_eigenface_space = create_eigenface_space(
        reduced_eigenvectors, train_standarized
    )  # matrix(num_images*names, n_components)

    # Add a function to find the closest face to every face in the test set

    print("\n")
    print("Calculando las caras mas cercanas..")
    print(f"Matrix dimensions: {reduced_eigenface_space.shape}")
    print(f"Comparing {reduced_eigenface_space.shape[0]} vectors with each other..")

    # Distances between vectors
    distances = np.array(
        reduced_eigenface_space.shape[0]
        * [reduced_eigenface_space.shape[0] * [float("nan")]]
    )

    for i in range(reduced_eigenface_space.shape[0]):
        for j in range(reduced_eigenface_space.shape[0]):

            if i // 8 == j // 8:  # if vectors belong to the same person...
                continue  # ...skip

            # Calculate distance between ith and jth vector
            dist = np.linalg.norm(
                reduced_eigenface_space[i] - reduced_eigenface_space[j]
            )

            # Add to matrix
            distances[i][j] = dist
            distances[j][i] = dist

    index_max_dist = np.unravel_index(np.nanargmax(distances), distances.shape)

    # Closest faces
    index_min_dist = np.unravel_index(np.nanargmin(distances), distances.shape)
    comb_min_dis = [train_names[index_min_dist[0]], train_names[index_min_dist[1]]]
    print("Min distance:", distances[index_min_dist])
    print("Closest faces are:", comb_min_dis)

    # Most distant faces
    index_max_dist = np.unravel_index(np.nanargmax(distances), distances.shape)
    comb_max_dis = [train_names[index_max_dist[0]], train_names[index_max_dist[1]]]
    print("Max distance:", distances[index_max_dist])
    print("Most distant faces are:", comb_max_dis)

    # Print the faces
    # Get the faces matrix
    distant_faces = np.column_stack(
        (images[train_idx[index_max_dist[0]]], images[train_idx[index_max_dist[1]]])
    )
    # Get the names:
    names = [train_names[index_max_dist[0]], train_names[index_max_dist[1]]]
    # Plot:
    plot_images(np.real(distant_faces.T), [names[0], names[1]], 1, 2, 0, 2)

    # Create two DataFrames (train + test) with the first 60 PCs of each image
    for tag, split, names in [
        ("train", train, train_names),
        ("test", test, test_names),
    ]:
        centered_imgs = substract_mean_face(split, mean_face)
        eigenspace = create_eigenface_space(reduced_eigenvectors, centered_imgs)
        pca_df = pd.DataFrame(eigenspace)
        pca_df.columns = ["PC" + str(i) for i in range(1, n_components + 1)]
        pca_df.insert(0, "Name", names, True)
        pca_df.to_csv(f"test2-components_{tag}.csv", index=False)

    # Testing PCA accuracy.
    # Result should be a vector with boolean values which results
    # from comparing the predicted name with the real name

    """
    evaluate_test(
        test, test_names, reduced_eigenvectors, train_names, reduced_eigenface_space
    )

    """

    # Project new face into eigenvectors space
    # i = 5
    # new_face, new_name = test[i], test_names[i]
    # new_face_projected = calculate_eigenface(
    #     new_face, mean_face, reduced_eigenvectors
    # )  # array(n_components,)

    # # Find the closest face
    # closest_face_index = find_closest_face(reduced_eigenface_space, new_face_projected)
    # closest_face_name = train_names[closest_face_index]
    # closest_face = train[closest_face_index]

    # # Plot closest face
    # # Create a figure and axes for subplots
    # fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # # Plot closest face
    # axes[0].imshow(closest_face.reshape(30, 30), cmap="gray")
    # axes[0].set_title(closest_face_name)

    # # Plot new face
    # axes[1].imshow(new_face.reshape(30, 30), cmap="gray")
    # axes[1].set_title(new_name)

    # # Display the subplots
    # plt.show()

    # Plot comparison images

    # Plot images
    # plot_images(train, train_names, 2, 4, 24, 32)
    # plot_images(images, names, 5, 5, 30, 55)
    from sklearn.decomposition import PCA

    pca = PCA(n_components=60)

    train = pca.fit_transform(train)

    df = pd.DataFrame(train)

    df.to_csv("train_pca.csv", index=False)


if __name__ == "__main__":
    main()

