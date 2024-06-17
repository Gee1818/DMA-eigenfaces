import os
import cv2

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



# Find and extract faces from test images (/4._test_your_files)
def face_detection_test(input_path, output_path):
    img_count = 0
    files = [file for file in os.listdir(input_path) if os.path.isfile(os.path.join(input_path, file))]
    
    for file in sorted(files):       
        print(f"Processing {file}...")
        
        if file[-4:] == "HEIC":
            continue
        
        # Get face from file
        current_dir = os.path.join(input_path, file)
        print(current_dir)
        
        face = get_face(current_dir)
            
        if face is None:
            print("No face detected in {file}\n".format(file=file))
        else:
            print("saving face...\n")
            cv2.imwrite(os.path.join(output_path, file), face)
        
        img_count += 1