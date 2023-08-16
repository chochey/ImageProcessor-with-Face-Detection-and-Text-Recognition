import zipfile
import cv2 as cv
import numpy as np
from PIL import Image
import pytesseract
import logging
import os

# Define the keyword to search for
wordToSearch = "Mark"

# Load the pre-trained face cascade classifier
face_cascade = cv.CascadeClassifier(
    "./haarcascade-files/haarcascade_frontalface_default.xml"
)


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


# Set the directory where zip files and images are located
input_dir = "./input_data/"
output_dir = "./output_data/"

# Create the output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# List all the zip files in the directory
zip_files = [file for file in os.listdir(input_dir) if file.lower().endswith(".zip")]

# Process images within zip files
for zip_file in zip_files:
    with zipfile.ZipFile(os.path.join(input_dir, zip_file), "r") as zip_ref:
        for zip_info in zip_ref.infolist():
            if zip_info.is_dir():
                continue

            print(f"Processing image file: {zip_info.filename}")

            image_data = zip_ref.read(zip_info.filename)

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

                    thumbnail_filename = f"combined_faces_{zip_info.filename}"
                    output_path = os.path.join(output_dir, thumbnail_filename)
                    cv.imwrite(output_path, combined_faces)

                    print(f"Faces found in '{zip_info.filename}': {len(faces)}")
                else:
                    print(
                        f"'{wordToSearch}' was found in the file '{zip_info.filename}' but no faces were detected."
                    )

# Process images outside zip files
for image_file in os.listdir(input_dir):
    if image_file.lower().endswith((".png", ".jpg", ".jpeg")):
        print(f"Processing image file: {image_file}")

        image_path = os.path.join(input_dir, image_file)
        with open(image_path, "rb") as img_file:
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
                output_path = os.path.join(output_dir, thumbnail_filename)
                cv.imwrite(output_path, combined_faces)

                print(f"Faces found in '{image_file}': {len(faces)}")
            else:
                print(
                    f"'{wordToSearch}' was found in the file '{image_file}' but no faces were detected."
                )

print("Program Completed")
