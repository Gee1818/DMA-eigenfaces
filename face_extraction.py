import os

import cv2
import pyheif
from PIL import Image

# Set path to photos

path = "/home/ge/MCD/Data Mining Avanzado/DMA-eigenfaces-photos/"
output_path = "/home/ge/MCD/Data Mining Avanzado/DMA-eigenfaces-output/"
# Commented for thios to work in REPL
script_path = os.path.dirname(__file__) + "/"
print(script_path)
# script_path = "/home/ge/MCD/Data Mining Avanzado/DMA-eigenfaces/"

# Read names
names = []
for dir in os.listdir(path):
    names.append(dir)
names.count

# Load the cascade
casc_path = script_path + "haarcascade_frontalface_alt.xml"
face_cascade = cv2.CascadeClassifier(casc_path)


def crop_img(raw_img, face_classifier, name, number, output_path):
    img = cv2.imread(raw_img)
    # Convert to grey
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Detect the face
    face = face_classifier.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=6)
    for x, y, w, h in face:
        # cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
        face = gray_img[y : y + h, x : x + w]
        face = cv2.resize(face, (30, 30))
        # cv2.imshow("face", face)
        number = str(number)
        cv2.imwrite(output_path + name + "-" + number + ".jpg", face)
    # cv2.imshow("img", img)
    cv2.waitKey(0)


# TODO: Convert images in .HEIC format


for person in names:
    current_dir = path + person + "/"
    print("Current person is: " + person)
    i = 0
    for file in os.listdir(current_dir):
        if file[-4:] != "HEIC":
            i += 1
            current_img = current_dir + file
            print("Current file is: " + file)
            crop_img(current_img, face_cascade, person, i, output_path)
