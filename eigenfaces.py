import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.lib.type_check import real

from functions import *

# Define path for input images
path_og = "1.cropped_photos/"
path_mirrored = "2.mirrored_images/" 
number_of_images_train = 10
     
def main():
    
    # Read images
    images, names = read_images(path_og)  # Images matrix -->(num_images, 900), names -->list
    
    images_mirrored, names_mirrored = read_images(path_mirrored)  # Images matrix -->(num_images, 900), names -->list
    
    images = np.concatenate((images, images_mirrored), axis=0)
    names = names + names_mirrored
    
    print(images.shape)
    
    # Select training set
    train, test, train_idx, test_idx, train_names, test_names = select_training_set(images, names, number_of_images_train)  # train --> matrix(num_images*names, 900)

    # Sort train set
    train_idx = [idx for _, idx in sorted(zip(train_names, train_idx))]
    train = [img for _, img in sorted(zip(train_names, train), key=lambda x: x[0])]
    train_names = sorted(train_names)

    # Sort test set
    test_idx = [idx for _, idx in sorted(zip(test_names, test_idx))]
    test = [img for _, img in sorted(zip(test_names, test), key=lambda x: x[0])]
    test_names = sorted(test_names)
    
    # Calculate mean face
    mean_face = calculate_mean_face(train)  # array(900,)
    
    
    # Save mean face
    mean_face_df = pd.DataFrame(mean_face)
    mean_face_df.to_csv("mean_face.csv", index=False, header= False)
    
    # Plot mean face
    #plt.imshow(mean_face.reshape(30, 30), cmap="gray")
    #plt.show()

    # Subtract mean face from images
    train_standarized = substract_mean_face(
        train, mean_face
    )  # matrix(num_images*names, 900)

    # Calculate covariance matrix
    covariance_matrix = calculate_covariance(train_standarized)  # matrix(900, 900)

    # Get Eeigenvalues and eigenvectors
    eigenvalues, eigenvectors = calculate_eigenvalues(covariance_matrix)  # array(900,), matrix(900, 900)

    # Plot eigenfaces
    #plot_images(np.real(eigenvectors.T), range(1, 21), 4, 5, 0, 20)

    # Calculate explained variance
    
    #explained_variance = calculate_explained_variance(eigenvalues)  # array(900,)
    n_components = 60
    
    #print(explained_variance[:n_components])
    reduced_eigenvectors = eigenvectors[:, :n_components]  # matrix(900, n_components)
    
    # Save reduced eigenvectors to a csv
    red_eig = pd.DataFrame(reduced_eigenvectors)
    red_eig.to_csv("reduced_eigenvectors.csv", index=False, header= False)
    
    # Project images into the reduced eigenface space
    reduced_eigenface_space = create_eigenface_space(
        reduced_eigenvectors, train_standarized
    )  # matrix(num_images*names, n_components)


    
    # Create two DataFrames (train + test) with the first 60 PCs of each image
    for (tag, split, names) in [("train", train, train_names), ("test", test, test_names)]:
        centered_imgs = substract_mean_face(split, mean_face)
        eigenspace = create_eigenface_space(reduced_eigenvectors, centered_imgs)
        pca_df = pd.DataFrame(eigenspace)
        pca_df.columns = ["PC"+str(i) for i in range(1, n_components+1)]
        pca_df.insert(0, "Name", names, True)
        pca_df.sort_values(by="Name", inplace=True)
        pca_df.to_csv(f"components_{tag}.csv", index=False)
        if tag == "train": 
            train_df = pca_df.copy()
    
    
    
    # Add a function to find the closest face to every face in the test set

   

    print("\n")
    print("Calculando las caras mas cercanas..")
    print(f"Comparing {train_df.shape[0]} vectors with each other..")

    
    # Distances between vectors
    distances = np.array(train_df.shape[0] * [train_df.shape[0] * [float('nan')]])
    
    for i in range(train_df.shape[0]):
        for j in range(train_df.shape[0]):
            
            if i // number_of_images_train  == j // number_of_images_train : # if vectors belong to the same person...
                continue # ...skip
            
            # Calculate distance between ith and jth vector
            dist = np.linalg.norm(train_df.iloc[i, 1:] - train_df.iloc[j, 1:])
            
            # Add to matrix
            distances[i][j] = dist
            distances[j][i] = dist
            
    #index_max_dist = np.unravel_index(np.nanargmax(distances), distances.shape)
                    
    
    # Closest faces
    index_min_dist = np.unravel_index(np.nanargmin(distances), distances.shape)
    comb_min_dis = [train_names[index_min_dist[0]], train_names[index_min_dist[1]]]
    print("Min distance:", distances[index_min_dist])
    print("Closest faces are:", comb_min_dis)
    
    # Print the closest faces
    # Get the faces matrix
    distant_faces = np.column_stack((images[train_idx[index_min_dist[0]]], images[train_idx[index_min_dist[1]]]))
    # Get the names:
    names = [train_names[index_min_dist[0]], train_names[index_min_dist[1]]]
    # Plot:
    plot_images(np.real(distant_faces.T), [names[0] , names[1]], 1, 2, 0, 2)
    
    
    
    # Most distant faces
    index_max_dist = np.unravel_index(np.nanargmax(distances), distances.shape)
    comb_max_dis = [train_names[index_max_dist[0]], train_names[index_max_dist[1]]]
    print("Max distance:", distances[index_max_dist])
    print("Most distant faces are:", comb_max_dis)
    
    # Print the most distant faces
    # Get the faces matrix
    distant_faces = np.column_stack((images[train_idx[index_max_dist[0]]], images[train_idx[index_max_dist[1]]]))
    # Get the names:
    names = [train_names[index_max_dist[0]], train_names[index_max_dist[1]]]
    # Plot:
    plot_images(np.real(distant_faces.T), [names[0] , names[1]], 1, 2, 0, 2)
    
    

if __name__ == "__main__":
    main()
    
    