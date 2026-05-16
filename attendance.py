import cv2
import numpy as np
from PIL import Image
import os

path = 'dataset'

faces = []
ids = []

for image in os.listdir(path):

    img_path = os.path.join(path, image)

    face_img = Image.open(img_path).convert('L')

    imageNp = np.array(face_img, 'uint8')

    id = int(image.split(".")[1])

    faces.append(imageNp)
    ids.append(id)

ids = np.array(ids)

clf = cv2.face.LBPHFaceRecognizer_create()

clf.train(faces, ids)

if not os.path.exists("trainer"):
    os.makedirs("trainer")

clf.write("trainer/classifier.xml")

print("Training completed")