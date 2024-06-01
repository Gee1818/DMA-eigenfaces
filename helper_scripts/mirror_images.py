import os
from PIL import Image
from time import time


def mirror_images(path_in, path_out):
    # Iterate over all files in the source directory
    for filename in sorted(os.listdir(path_in)):
        # Check if the file is an image
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            # Open the image
            image_path = os.path.join(path_in, filename)
            image = Image.open(image_path)
            
            # Mirror the image horizontally
            mirrored_image = image.transpose(Image.FLIP_LEFT_RIGHT)
            
            # Generate the save path for the mirrored image
            save_path = os.path.join(path_out, filename)
            
            # Save the mirrored image
            os.makedirs(path_out, exist_ok=True) #comentar si genera algun error
            mirrored_image.save(save_path)
            


if __name__ == "__main__":
	start = time()

	path_photos = "../1.cropped_photos/"
	path_output = "../2.mirrored_images/"
	
	print("Mirroring images...")
	mirror_images(path_photos, path_output)
	success_msg = "Done! Mirrored all images in {secs:.2f} seconds."
	print(success_msg.format(secs = time() - start))
