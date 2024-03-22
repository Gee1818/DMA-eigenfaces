import os

import cv2
import matplotlib.pyplot as plt
import numpy as np


# Read training set
def read_images(path):
    images = []
    names = []
    for image in os.listdir(path):
        img = cv2.imread(path + image)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (30, 30))
        images.append(img.flatten())
        names.append(image.split("-")[0])
    return np.array(images), names


# Select training set
def select_training_set(images, names, num_images):
    unique_names = list(set(names))
    dict_names = {}
    for name in unique_names:
        dict_names[name] = [0, 0]
    train_images = []
    test_images = []
    for i in range(len(images)):
        if dict_names[names[i]][0] < num_images:
            train_images.append(images[i])
            dict_names[names[i]][0] += 1
        else:
            test_images.append(images[i])
            dict_names[names[i]][1] += 1
    for name, counts in dict_names.items():
        print(f"{name}: Train images = {counts[0]}, Test images = {counts[1]}")
    return np.array(train_images), np.array(test_images)


# Compute mean face and substract from faces
def calculate_mean_face(images):
    return np.mean(images, axis=0)


# Substract the mean face from the images
def substract_mean_face(images, mean_face):
    return images - mean_face


# Calculate covariance
def calculate_covariance(images):
    return np.cov(images.T)


# Calculate eigenvalues and eigenvectors from covariance matrix
def calculate_eigenfaces(covariance_matrix):
    eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)
    sort_eigenvalues = eigenvalues[np.argsort(eigenvalues)[::-1]]
    sort_eigenvectors = eigenvectors[:, np.argsort(eigenvalues)[::-1]]
    return sort_eigenvalues, sort_eigenvectors


# Calculate explained variance
def calculate_explained_variance(eigenvalues):
    total = np.sum(eigenvalues)
    explained_variance = [(i / total) * 100 for i in eigenvalues]
    return np.cumsum(explained_variance)


# Create eigenface projection space
def create_eigenface_space(eigenvectors, mean_face, images):
    print(eigenvectors.shape)
    print(mean_face.shape)
    print(images.shape)
    return np.dot(images - mean_face, eigenvectors)


# Calculate eigenface for a new face
def calculate_eigenface(new_face, mean_face, reduced_eigenvectors):
    subtracted_new_face = new_face - mean_face
    print(subtracted_new_face.shape)
    return np.dot(reduced_eigenvectors, subtracted_new_face.T)


# Calculate Euclidean distance between eigenface and new face
def calculate_euclidean_distance(eigenface, new_face):
    return np.linalg.norm(eigenface - new_face)


# Find the closest face
def find_closest_face(eigenfaces, new_face):
    distances = []
    for eigenface in eigenfaces:
        distances.append(calculate_euclidean_distance(eigenface, new_face))
    return np.argmin(distances)


# Display the result


# Plot images
def plot_images(images, names, width, height, start_idx, end_idx):
    num_images = len(images[start_idx:end_idx])
    if num_images == 1:
        fig, ax = plt.subplots(1, 1, figsize=(30, 30))
        ax.imshow(images[start_idx].reshape(30, 30), cmap="gray")
        ax.set_title(names[start_idx])
        ax.axis("off")
    else:
        fig, axes = plt.subplots(width, height, figsize=(30, 30))
        for i, ax in enumerate(axes.flat):
            if i < num_images and start_idx + i <= end_idx:
                ax.imshow(images[start_idx + i].reshape(30, 30), cmap="gray")
                ax.set_title(names[start_idx + i])

    plt.show()


# Define path for input images
path = "/home/ge/MCD/Data Mining Avanzado/DMA-eigenfaces-output/"

# Read images
images, names = read_images(path)  # Images matrix -->(num_images, 900), names -->list

# Select training set
train, test = select_training_set(
    images, names, 6
)  # train --> matrix(num_images*names, 900)

# Calculate mean face
mean_face = calculate_mean_face(train)  # array(900,)

# Plot mean face
# plt.imshow(mean_face.reshape(30, 30), cmap="gray")
# plt.show()

# Subtract mean face from images
subtracted_images = substract_mean_face(
    train, mean_face
)  # matrix(num_images*names, 900)

# Calculate covariance matrix
covariance_matrix = calculate_covariance(subtracted_images)  # matrix(900, 900)

# Get Eeigenvalues and eigenvectors
eigenvalues, eigenvectors = calculate_eigenfaces(
    covariance_matrix
)  # array(900,), matrix(900, 900)

# Calculate explained variance
explained_variance = calculate_explained_variance(eigenvalues)  # array(900,)
n_components = 70
print(explained_variance[:n_components])
reduced_eigenvectors = eigenvectors[:, :n_components]  # matrix(900, n_components)

# Project images into the reduced eigenface space
reduced_eigenface_space = create_eigenface_space(
    reduced_eigenvectors, mean_face, subtracted_images
)  # matrix(num_images*names, n_components)


# Calculate eigenface for a new face
new_face = test[0]  # array(900,)
eigenface = calculate_eigenface(new_face, mean_face, reduced_eigenvectors)
# Plot new face
plt.imshow(eigenface.reshape(30, 30), cmap="gray")
plt.show()


# Find the closest face
closest_face = find_closest_face(reduced_eigenface_space, eigenface)
print(f"Closest face: {names[train[closest_face]]}")


# Plot images
# plot_images(images, names, 1, 1, 0, 1)
# plot_images(images, names, 5, 5, 30, 55)
