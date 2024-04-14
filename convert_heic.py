import cv2
import os
import pillow_heif
import numpy as np
from time import time

# Keep track of time
start = time()

# Get list of HEIF and HEIC files in directory
directory = "/home/frank/maestria_mcd/nuestras_caras/Nuestras Caras-20240321T193416Z-001/Nuestras Caras/"

names = [dir for dir in os.listdir(directory)]

# Counter: number of images with HEIC extension
img_count = 0

print("Processing...")

# For each person...
for person in names:
    currentdir = directory + person + "/"
    # ...and for each picture of said person...
    for file in os.listdir(currentdir):
        # ...if picture has HEIC extension...
        if file.lower().endswith(".heic"):
            # ...convert to JPEG and save it as such
            heif_file = pillow_heif.open_heif(currentdir + file, convert_hdr_to_8bit=False, bgr_mode=True)
            image = np.array(heif_file)
            image_directory = currentdir + file[:-5] + ".jpg"
            cv2.imwrite(image_directory, image)

# Let user know the program has finished
success_msg = "Done! Converted {img_count} images in {secs:.2f} seconds."
print(success_msg.format(img_count = img_count, secs = time() - start))

# Convert each file to JPEG
# for filename in files:
#     image = Image.open(os.path.join(directory, filename))
#     image.convert("RGB").save(
#         os.path.join(directory, os.path.splitext(filename)[0] + ".jpg")
#     )
