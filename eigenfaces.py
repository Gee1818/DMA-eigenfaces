import os

import cv2
import matplotlib.pyplot as plt
import numpy as np


# Read training set
def read_images(path):
    images = []
    names = []
    for image in os.listdir(path):
        img = cv2.imread(path + image, cv2.IMREAD_GRAYSCALE)
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
            train_images.append(i)
            dict_names[names[i]][0] += 1
        else:
            test_images.append(i)
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


# Calculate eigenvalues and eigenvectors


# Calculate eigenfaces


# Create reduced eigenface space


# Calculate eigenface for a new face


# Calculate Euclidean distance between eigenface and new face


# Find the closest face


# Display the result


# Plot images
def plot_images(images, names, width, height, start_idx, end_idx):
    num_images = len(images[start_idx:end_idx])
    print(num_images)
    print(images[start_idx:end_idx].shape)
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
images, names = read_images(path)

# Select training set
train, test = select_training_set(images, names, 8)

print(train.shape)
print(test.shape)
# Calculate mean face
mean_face = calculate_mean_face(images)

# Plot mean face
# plt.imshow(mean_face.reshape(30, 30), cmap="gray")
# plt.show()

# Subtract mean face from images
subtracted_images = substract_mean_face(images, mean_face)

# Calculate covariance matrix
covariance_matrix = calculate_covariance(subtracted_images)

# Plot images
# plot_images(images, names, 1, 1, 0, 1)
# plot_images(images, names, 3, 3, 30, 38)
