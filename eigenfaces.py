import matplotlib.pyplot as plt
import numpy as np
from numpy.lib.type_check import real

from functions import *

# Define path for input images
path = "/home/frank/maestria_mcd/nuestras_caras/Nuestras Caras-20240321T193416Z-001/1.test_photos/"
mirror_path = "/home/frank/maestria_mcd/nuestras_caras/Nuestras Caras-20240321T193416Z-001/2.mirrored_images/"
   
     
def main():
    
    # Read images
    images, names = read_images(path)  # Images matrix -->(num_images, 900), names -->list

    
    # Flip images horizontally - DESCOMENTAR LINEA DE ABAJO PARA GENERAR IMAGENES ESPEJO --> DEBEMOS MOVERLO A HELPER_SCRIPTS.
    #mirror_images(path, mirror_path)
    mir_img, mir_names = read_images(mirror_path)

    print(f"Original images: {images.shape} \n Original names: {len(names)}")
    print(f"Mirrored images: {mir_img.shape} \n Mirrored names: {len(mir_names)}")

    # Le pongo train para ver si funcionan los indices.
    train = np.vstack((images, mir_img))
    train_names = names + mir_names

    print(f"Total images: {images.shape} \n Total names: {len(names)}")


    
    
    '''
    # Select training set
    train, test, train_names, test_names = select_training_set(
        images, names, 8
    )  # train --> matrix(num_images*names, 900)
    '''
    
    # Calculate mean face
    mean_face = calculate_mean_face(train)  # array(900,)

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
    eigenvalues, eigenvectors = calculate_eigenvalues(
        covariance_matrix
    )  # array(900,), matrix(900, 900)

    # Plot eigenfaces
    #plot_images(np.real(eigenvectors.T), range(1, 21), 4, 5, 0, 20)

    # Calculate explained variance
    #explained_variance = calculate_explained_variance(eigenvalues)  # array(900,)
    n_components = 60
    ##print(explained_variance[:n_components])
    reduced_eigenvectors = eigenvectors[:, :n_components]  # matrix(900, n_components)

    # Project images into the reduced eigenface space
    reduced_eigenface_space = create_eigenface_space(
        reduced_eigenvectors, train_standarized
    )  # matrix(num_images*names, n_components)


    # Find the closest face to every face in the test set

    print("\n")
    print("Calculando las caras mas cercanas..")
    print(f"Matrix dimensions: {reduced_eigenface_space.shape}") 
    print(f"Comparing {reduced_eigenface_space.shape[0]} vectors with each other..")

    
    # Create am empty dictionary to store the distances and indexes involved for each name combination.
    distance = {}

    for i in range(reduced_eigenface_space.shape[0]):
        name = train_names[i]
        for j in range(reduced_eigenface_space.shape[0]):
            if name != train_names[j]:
                distance_iter = np.linalg.norm(
                    reduced_eigenface_space[i] - reduced_eigenface_space[j]
                )
                name_comb = f"{name} - {train_names[j]}"
                inv_name_comb = f"{train_names[j]} - {name}"
                if not inv_name_comb in distance:
                    distance[name_comb] = [distance_iter ,[i,j]] # Save the distance + save the 2 indexes
                
    
    # Closest faces
    # Find the minimun score
    min_dist = min(distance.values())[0]

    # Get all the keys for the min distance
    comb_min_dis = [key for key, value in distance.items() if value[0] == min_dist] # len = 1


    # Print results!
    print("Closest faces are:", comb_min_dis)
    '''
    #print faces
    # Get the faces matrix
    closest_faces = np.column_stack(([images[distance[comb_min_dis[0]][1][0]], # First face, comb_min_dis requires [0] cause its a list of lenght 1 
                                      images[distance[comb_min_dis[0]][1][1]]]))  # Second face
    
    # Get the names:
    names = comb_min_dis[0].split(" - ")
    plot_images(np.real(closest_faces.T), [names[0] , names[1]], 1, 2, 0, 2)
    '''
    
    # More distant faces
    max_dist = max(distance.values())[0]

    # Get all the keys for the max distance
    comb_max_dis = [key for key, value in distance.items() if value[0] == max_dist]

    # Print results
    print("More distant faces are:", comb_max_dis)
    
    '''
    # Print the faces
    # Get the faces matrix
    distant_faces = np.column_stack(([images[distance[comb_max_dis[0]][1][0]], # First face, comb_min_dis requires [0] cause its a list of lenght 1 
                                      images[distance[comb_max_dis[0]][1][1]]]))  # Second face
    # Get the names:
    names = comb_max_dis[0].split(" - ")
    plot_images(np.real(distant_faces.T), [names[0] , names[1]], 1, 2, 0, 2)
    '''
    
    
    
    
    
    
    
    
    '''
    # Create two DataFrames (train + test) with the first 60 PCs of each image
    for (tag, split, names) in [("train", train, train_names), ("test", test, test_names)]:
        centered_imgs = substract_mean_face(split, mean_face)
        eigenspace = create_eigenface_space(reduced_eigenvectors, centered_imgs)
        pca_df = pd.DataFrame(eigenspace)
        pca_df.columns = ["PC"+str(i) for i in range(1, n_components+1)]
        pca_df.insert(0, "Name", names, True)
        pca_df.to_csv(f"components_{tag}.csv", index=False)
    

    '''
    
    # Testing PCA accuracy. 
    # Result should be a vector with boolean values which results
    # from comparing the predicted name with the real name

    
    '''
    evaluate_test(
        test, test_names, reduced_eigenvectors, train_names, reduced_eigenface_space
    )

    '''


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


if __name__ == "__main__":
    main()
    
    