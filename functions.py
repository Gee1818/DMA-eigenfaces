import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import random
from numpy.lib.type_check import real


# Read training set
def read_images(path):
    images = []
    names = []
    
    for image in os.listdir(path):
        img_dir = os.path.join(path, './'+image)
        img = cv2.imread(img_dir)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (30, 30))
        images.append(img.flatten())
        names.append(image.split("-")[0])
    return np.array(images), names


# Select training set
def select_training_set(images, names, num_images):
    unique_names = sorted(list(set(names)))
    dict_names = {}
    train_images, train_images_idx, train_names = [], [], []
    test_images, test_images_idx, test_names = [], [], []

    random.seed(505)

    # Iterate through the people
    for unique_name in unique_names:
        # Get the indices of every photo belonging to this person
        name_indices = [i for i, name in enumerate(names) if name == unique_name]
        # Randomly select n (num_images) of these indices
        train_indices = random.sample(name_indices, num_images)
        train_images_idx += train_indices
        # The non-selected indices go to test
        test_indices = list(set(name_indices).difference(set(train_indices)))
        test_images_idx += test_indices
        # Count train and test images for each person
        dict_names[unique_name] = [len(train_indices), len(test_indices)]

    for idx in train_images_idx:
        train_images.append(images[idx])
        train_names.append(names[idx])

    for idx in test_images_idx:
        test_images.append(images[idx])
        test_names.append(names[idx])    
    
    print("=============================")
    print("Name       | n_train | n_test")
    print("-----------------------------")
    for name, counts in sorted(dict_names.items()):
        print("{:<10} | {:>7} | {:>6}".format(name, counts[0], counts[1]))
    print("=============================")
    return np.array(train_images), np.array(test_images), train_images_idx, test_images_idx, train_names, test_names


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
def calculate_eigenvalues(covariance_matrix):
    eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    return eigenvalues, eigenvectors


# Calculate explained variance
def calculate_explained_variance(eigenvalues):
    total = np.sum(eigenvalues)
    explained_variance = [(i / total) * 100 for i in eigenvalues]
    return np.cumsum(explained_variance)


# Create eigenface projection space
def create_eigenface_space(eigenvectors, images):
    return np.dot(images, eigenvectors).real


# Project new face
def calculate_eigenface(new_face, mean_face, eigenvectors):
    new_standarized_face = substract_mean_face(new_face, mean_face)
    projected_new_face = np.dot(new_standarized_face, eigenvectors).real
    return projected_new_face


# Find the closest face
def find_closest_face(eigenvectors, new_face_projected):
    distances = []
    for i in range(len(eigenvectors)):
        distance = np.linalg.norm(eigenvectors[i] - new_face_projected)
        distances.append(distance)
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


# Function to evaluate all test set
def evaluate_test(
    test_images, test_names, reduced_eigenvectors, train_names, reduced_eigenface_space
):
    results_test_name = []
    results_predicted_name = []
    results = []
    for i in range(len(test_images)):
        new_face = test_images[i]
        # Append new name to results
        results_test_name.append(test_names[i])
        # Project new face into eigenvectors space
        new_face_projected = calculate_eigenface(
            new_face, mean_face, reduced_eigenvectors
        )
        # Find the closest face from the training set
        closest_face_index = find_closest_face(
            reduced_eigenface_space, new_face_projected
        )
        # Append the predicted name to results
        results_predicted_name.append(train_names[closest_face_index])
    # Compare the predicted name with the real name
    for i in range(len(results_test_name)):
        if results_test_name[i] == results_predicted_name[i]:
            results.append(True)
        else:
            results.append(False)
        # print(f"Real name: {results_test_name[i]}, Predicted name: {results_predicted_name[i]}, Result: {results[i]}")
    # Count the number of correct predictions
    correct_predictions = sum(results)
    print(f"Correct predictions: {correct_predictions} from {len(results)}")

    # Show the incorrect predictions
    incorrect_predictions = [i for i, x in enumerate(results) if not x]
    for i in incorrect_predictions:
        print(
            f"Real name: {results_test_name[i]}, Predicted name: {results_predicted_name[i]}"
        )
