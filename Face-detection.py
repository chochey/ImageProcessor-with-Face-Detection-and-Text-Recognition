import zipfile
import cv2 as cv
import numpy as np
from PIL import Image
import pytesseract
import logging

# Load the pre-trained face cascade classifier
face_cascade = cv.CascadeClassifier(
    "./haarcascade-files/haarcascade_frontalface_default.xml"
)

# the zip name and keyword for image search
wordToSearch = "Mark"
filename = "images.zip"


# Function to find faces and clip them from an image
def clip_faces(image_data):
    try:
        img = cv.imdecode(np.frombuffer(image_data, np.uint8), cv.IMREAD_COLOR)
        if img is None:
            raise Exception("Failed to decode image data")

        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 6)

        return img, faces
    except Exception as e:
        logging.error(str(e))
        return None, []


# Open the ZIP file in read mode
with zipfile.ZipFile("./zip-Files/" + filename, "r") as zip_ref:
    image_files = [name for name in zip_ref.namelist() if name.lower().endswith(".png")]

    for image_file in image_files:
        with zip_ref.open(image_file) as img_file:
            image_data = img_file.read()

        img, faces = clip_faces(image_data)

        if img is None:
            continue

        text = pytesseract.image_to_string(img, config="--psm 1 --oem 3")

        if wordToSearch in text:
            if len(faces) > 0:
                img_with_faces = img.copy()

                # Create a horizontal list of faces
                face_list = []
                for i, (x, y, w, h) in enumerate(faces):
                    face_roi = img[y : y + h, x : x + w]
                    resized_face = cv.resize(face_roi, (100, 100))
                    face_list.append(resized_face)

                combined_faces = np.hstack(face_list)

                thumbnail_filename = f"combined_faces_{image_file}"
                cv.imwrite(thumbnail_filename, combined_faces)

                print(f"Faces found in '{image_file}': {len(faces)}")
            else:
                print(
                    f"'{wordToSearch}' was found in the file '{image_file}' but no faces were detected."
                )
