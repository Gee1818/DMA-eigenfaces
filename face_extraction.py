import os

import cv2

# Set path to photos

path = "/home/ge/MCD/Data Mining Avanzado/DMA-eigenfaces-photos/"
output_path = "/home/ge/MCD/Data Mining Avanzado/DMA-eigenfaces-output/"
# Commented for thios to work in REPL
# script_path = os.path.dirname(__file__)
script_path = "/home/ge/MCD/Data Mining Avanzado/DMA-eigenfaces/"

# Read names
names = []
for dir in os.listdir(path):
    names.append(dir)
names.count

# Load the cascade
casc_path = script_path + "haarcascade_frontalface_alt2.xml"
face_cascade = cv2.CascadeClassifier(casc_path)


def crop_img(raw_img, face_classifier):
    img = cv2.imread(raw_img)
    # Convert to grey
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Detect the face
    face = face_classifier.detectMultiScale(gray_img, 1.1, 4)
    for x, y, w, h in face:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
        face = img[y : y + h, x : x + w]
        # cv2.imshow("face", face)
        cv2.imwrite("face.jpg", face)
    # cv2.imshow("img", img)
    cv2.waitKey(0)


test = "/home/ge/MCD/Data Mining Avanzado/DMA-eigenfaces-photos/Lautaro/20240309_112220.jpg"
print(test)
crop_img(test, face_cascade)


# TODO: Set a loop for all the folders and all the files. Write as number-person.jpg
