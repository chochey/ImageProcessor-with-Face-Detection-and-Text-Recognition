import zipfile
import math
from PIL import Image
import pytesseract
import cv2 as cv
import numpy as np

# loading the face detection classifier
face_cascade = cv.CascadeClassifier("readonly/haarcascade_frontalface_default.xml")

# Load images, perform OCR, and detect faces
xyz = {}
zip_file_location = "./newspaper.zip"
with zipfile.ZipFile(zip_file_location, "r") as f:
    for starter in f.infolist():
        with f.open(starter.filename) as file:
            img = Image.open(file).convert("RGB")
            xyz[starter.filename] = {"person 1.jpg": img}


for i in xyz.keys():
    text = pytesseract.image_to_string(xyz[i]["person 1.jpg"])
    xyz[i]["text"] = text
    cvc = np.array(xyz[i]["person 1.jpg"])
    vg = cv.cvtColor(cvc, cv.COLOR_BGR2GRAY)
    bxing = face_cascade.detectMultiScale(vg, 1.3, 5)
    xyz[i]["faces"] = []
    for x, y, w, h in bxing:
        face = xyz[i]["person 1.jpg"].crop((x, y, x + w, y + h))
        xyz[i]["faces"].append(face)
    for face in xyz[i]["faces"]:
        face.thumbnail((100, 100), Image.ANTIALIAS)


def search_and_display(keyword):
    for i in xyz:
        if keyword.lower() in xyz[i]["text"].lower() and len(xyz[i]["faces"]) != 0:
            print("Result found in file {}".format(i))
            h = math.ceil(len(xyz[i]["faces"]) / 5)
            cs = Image.new("RGB", (500, 100 * h))
            accum = 0
            accum_2 = 0
            for img in xyz[i]["faces"]:
                cs.paste(img, (accum, accum_2))
                if accum + 100 == cs.width:
                    accum = 0
                    accum_2 += 100
                else:
                    accum += 100
            cs.show()  # Use .show() to display images


# Search for a specific keyword
search_and_display("BUSINESS")
