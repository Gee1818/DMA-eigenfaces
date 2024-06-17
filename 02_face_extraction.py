import os
import cv2
from time import time
from helper_scripts.face_detection import get_face

# Keep track of time
start = time()

# Function to extract faces from a directory with folders inside
def face_extraction(input_path, output_path):
    img_count = 0
    fail_count = 0

    names = sorted(os.listdir(input_path))

    for person in names:
        current_dir = os.path.join(input_path, person)
        print("Processing {person}'s pictures...".format(person=person))
        person_count = 0
        for file in sorted(os.listdir(current_dir)):
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
                os.makedirs(output_path, exist_ok=True)
                cv2.imwrite(os.path.join(output_path, filename + ".jpg"), face)

    success_msg = "Done! Processed {img_count} images in {secs:.0f} seconds."
    warning_msg = "Warning: no faces were detected in {fail_count} of the input images."
    print(success_msg.format(img_count = img_count, secs = time() - start))
    if fail_count > 0:
    	print(warning_msg.format(fail_count = fail_count))
    


if __name__ == "__main__":
    face_extraction("./1.nuestras_caras_raw/" , "./2.cropped_photos/")