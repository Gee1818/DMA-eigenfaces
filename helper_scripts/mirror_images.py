import os
from PIL import Image



def mirror_images(path_in, path_out):
    # Iterate over all files in the source directory
    for filename in os.listdir(path_in):
        # Check if the file is an image
        if filename.endswith(".png") or filename.endswith(".jpg") or filename.endswith(".jpeg"):
            # Open the image
            image_path = os.path.join(path_in, filename)
            image = Image.open(image_path)
            
            # Mirror the image horizontally
            mirrored_image = image.transpose(Image.FLIP_LEFT_RIGHT)
            
            # Generate the save path for the mirrored image
            save_path = os.path.join(path_out, filename)
            
            # Save the mirrored image
            os.makedirs(save_path, exist_ok=True) #comentar si genera algun error
            mirrored_image.save(save_path)
            

path_photos = "../1.test_photos/"
path_output = "../2.mirrored_images/"

mirror_images(path_photos, path_output)
