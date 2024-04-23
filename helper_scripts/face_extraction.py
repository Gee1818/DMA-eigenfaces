import os
import cv2
from time import time

# Load the cascade
script_path = os.path.dirname(__file__) + "/"
frontal_cascade_path = os.path.join(script_path, 'haarcascade_frontalface_alt.xml')
profile_cascade_path = os.path.join(script_path, 'haarcascade_profileface.xml')
frontal_face_detector = cv2.CascadeClassifier(frontal_cascade_path)
profile_face_detector = cv2.CascadeClassifier(profile_cascade_path)

   
# Function to extract face from image
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
    
# Function to extract faces from a directory with folders inside
def face_extraction(input_path, output_path):
    img_count = 0
    fail_count = 0

    names = [dir for dir in os.listdir(input_path)]

    for person in names:
        current_dir = os.path.join(input_path, person)
        print("Processing {person}'s pictures...".format(person=person))
        person_count = 0
        for file in os.listdir(current_dir):
            if file[-4:] == "HEIC":
                continue
            img_count += 1
            person_count += 1
            current_img = os.path.join(current_dir, file)
            filename = person + "-" + str(person_count)
            
            face = get_face(current_img)
            if face is None:
                fail_count += 1
            else:
                cv2.imwrite(os.path.join(output_path, filename + ".jpg"), face)

    success_msg = "Done! Processed {img_count} images in {secs:.0f} seconds."
    warning_msg = "Warning: no faces were detected in {fail_count} of the input images."
    print(success_msg.format(img_count=img_count))

def face_detection_test(input_path, output_path):
    
    img_count = 0
    
    files = [file for file in os.listdir(input_path) if os.path.isfile(os.path.join(input_path, file))]
    
    
    for file in files:
        
        print(f"Processing {file}...")
        
        if file[-4:] == "HEIC":
                continue
        
        # Get face from file
        current_dir = os.path.join(input_path, file)
        print(current_dir)
        
        face = get_face(current_dir)
            
        if face is None:
            print("No face detected in {file}".format(file=file))
        else:
            print("saving face...\n")
            cv2.imwrite(os.path.join(output_path, file), face)
            #print("face saved in {output_path}".format(output_path=output_path))
        img_count += 1
    



def main():
    face_extraction("../3.nuestras_caras_raw/" , "../1.cropped_photos/")

if __name__ == "__main__":
    main()
    