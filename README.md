# ImageProcessor with Face Detection and Text Recognition

A versatile Python script for processing images, detecting features, and performing advanced image analysis tasks.

- This program will unzip and search through images for a keyword and create thumbnails of all the face/s located within the images. It can run more than one image at a time and will create a .png of all the face/s located in each file it found the keyword.

## Features

- Detects faces, objects, or other features in images using various methods.
- Performs text recognition on images using Tesseract or other OCR tools.
- Generates composite images with detected features.
- Applies filters and transformations to images for analysis.

## Getting Started

These instructions will help you set up and run the script on your local machine.

### Prerequisites

- Python 3.x
- Required Python packages: numpy, opencv-python, pytesseract, Pillow

You can install the required packages using the following command:

```bash
pip install opencv-python numpy Pillow pytesseract

```

## Running the Code

- Put your .zip file into a folder called "zip-Files".
- Open Face-detection.py and update the following variables:
- wordToSearch with the word you are searching for in the images.
- filename with the name of the zipfile containing images.
- Run Face-detection.py script.
