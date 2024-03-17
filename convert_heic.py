import cv2
import os
import pillow_heif
import numpy as np


# Get list of HEIF and HEIC files in directory
directory = "/home/ge/MCD/Data Mining Avanzado/DMA-eigenfaces-photos/"

names = []
for dir in os.listdir(directory):
    names.append(dir)

for person in names:
    currentdir = directory + person + "/"
    for file in os.listdir(currentdir):
        if file.lower().endswith(".heic"):
            print(currentdir + file)
            heif_file = pillow_heif.open_heif(currentdir + file, convert_hdr_to_8bit=False, bgr_mode=True)
            image = np.array(heif_file)
            cv2.imwrite(currentdir + file + ".jpg", image)

# Convert each file to JPEG
# for filename in files:
#     image = Image.open(os.path.join(directory, filename))
#     image.convert("RGB").save(
#         os.path.join(directory, os.path.splitext(filename)[0] + ".jpg")
#     )
