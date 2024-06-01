import os
import cv2
from PIL import Image
from time import time


def change_brightness(path_in, path_out):
    # Iterate over all files in the source directory
    for filename in sorted(os.listdir(path_in)):
        # Check if the file is an image
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            # Open the image
            image_path = os.path.join(path_in, filename)
            image = cv2.imread(image_path)
            
            # Increase / reduce brightness
            brighter_image = increase_brightness(image, value = 30)
            darker_image   = decrease_brightness(image, value = 30)
            
            brighter_image2 = increase_brightness(image, value = 15)
            darker_image2   = decrease_brightness(image, value = 15)
            # Generate the save path for the mirrored image
            save_path = os.path.join(path_out, filename)
            
            # Save the mirrored image
            os.makedirs(path_out, exist_ok=True) #comentar si genera algun error
            cv2.imwrite(os.path.join(path_out, filename.split(".")[0] + "b" + ".jpg"), brighter_image)
            cv2.imwrite(os.path.join(path_out, filename.split(".")[0] + "d" + ".jpg"), darker_image)
            cv2.imwrite(os.path.join(path_out, filename.split(".")[0] + "b2" + ".jpg"), brighter_image2)
            cv2.imwrite(os.path.join(path_out, filename.split(".")[0] + "d2" + ".jpg"), darker_image2)
            
def increase_brightness(img, value=30):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    lim = 255 - value
    v[v > lim] = 255
    v[v <= lim] += value

    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img
            

def decrease_brightness(img, value=30):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    lim = value
    v[v < lim] = 0
    v[v >= lim] -= value

    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img



if __name__ == "__main__":
	start = time()

	path_photos = "../1.cropped_photos/"
	path_output = "../6.brightness/"
	
	print("Changing brightness...")
	change_brightness(path_photos, path_output)
	success_msg = "Done! Changed brightness in {secs:.2f} seconds."
	print(success_msg.format(secs = time() - start))
