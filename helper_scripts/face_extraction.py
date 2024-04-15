import os
import cv2
from PIL import Image
from time import time

# Keep track of time
start = time()

# Set path to photos
path = "/home/frank/maestria_mcd/nuestras_caras/Nuestras Caras-20240321T193416Z-001/Nuestras Caras/"
output_path = "/home/frank/maestria_mcd/nuestras_caras/Nuestras Caras-20240321T193416Z-001/test_photos/"

# Commented for thios to work in REPL
script_path = os.path.dirname(__file__) + "/"
#print(script_path)
# script_path = "/home/ge/MCD/Data Mining Avanzado/DMA-eigenfaces/"

# Read names
names = [dir for dir in os.listdir(path)]

# Load the cascade
frontal_cascade_path = os.path.join(script_path, 'haarcascade_frontalface_alt.xml')
profile_cascade_path = os.path.join(script_path, 'haarcascade_profileface.xml')
frontal_face_detector = cv2.CascadeClassifier(frontal_cascade_path)
profile_face_detector = cv2.CascadeClassifier(profile_cascade_path)

def get_face(image):
    # Read image
    img = cv2.imread(image)
    # Convert to grayscale
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Detect faces (frontal)
    areas = frontal_face_detector.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=6)
    
    # If no frontal face was detected, look for profile face
    if len(areas) == 0:
        areas = profile_face_detector.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=2)
        # If still no face was detected, move on
        if len(areas) == 0:
            return None

    # Else, take the first detected "face" in the image
    [x, y, w, h] = areas[0]
    
    # Crop and resize
    face = gray_img[y:y+h, x:x+w]
    face = cv2.resize(face, (30, 30))
    
    return face


img_count = 0
fail_count = 0

for person in names:
    current_dir = path + person + "/"
    print("Processing {person}'s pictures...".format(person = person))
    person_count = 0
    for file in os.listdir(current_dir):
        if file[-4:] == "HEIC":
            continue
        img_count += 1
        person_count += 1
        current_img = current_dir + file
        filename = person + "-" + str(person_count)
        
        face = get_face(current_img)
        if face is None:
            fail_count += 1
        else:
            cv2.imwrite(output_path + filename + ".jpg", face)


success_msg = "Done! Processed {img_count} images in {secs:.0f} seconds."
warning_msg = "Warning: no faces were detected in {fail_count} of the input images."
print(success_msg.format(img_count = img_count, secs = time() - start))
if fail_count > 0:
    print(warning_msg.format(fail_count = fail_count))