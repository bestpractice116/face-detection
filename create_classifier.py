import numpy as np
from PIL import Image
import os
import cv2
import re


# Method to train custom classifier to recognize face
import re

def train_classifer(name):
    path = os.path.join(os.getcwd() + "/data/" + name + "/")
    faces = []
    ids = []

    def preprocess_image(image_path, target_size=(100, 100)):
        img = Image.open(image_path).convert('L')  # Convert to grayscale
        img = img.resize(target_size, Image.ANTIALIAS)
        img_array = np.array(img, 'uint8')
        return img_array

    for root, dirs, files in os.walk(path):
        pictures = files

    for pic in pictures:
        imgpath = os.path.join(path, pic)
        img_array = preprocess_image(imgpath)  # Preprocess the image
        match = re.match(r'^(\d+)_', pic)  # Use regular expression to extract number
        if match:
            number = match.group(1)  # Extract the captured number
            id = int(number)  # Convert the extracted number to an integer
            faces.append(img_array)
            ids.append(id)

    ids = np.array(ids)

    batch_size = 32
    num_batches = len(faces) // batch_size

    # Train the classifier in batches
    clf = cv2.face.LBPHFaceRecognizer_create()
    for i in range(num_batches):
        start = i * batch_size
        end = (i + 1) * batch_size
        clf.update(faces[start:end], ids[start:end])

    # Train the remaining data
    remaining_faces = faces[num_batches * batch_size:]
    remaining_ids = ids[num_batches * batch_size:]
    clf.update(remaining_faces, remaining_ids)

    clf.write("./data/classifiers/" + name + "_classifier.xml")
