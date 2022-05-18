from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import mysql.connector
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import imutils
import cv2, os, urllib.request
import numpy as np
from django.conf import settings
import random
import string


mydb = mysql.connector.connect(
    host="localhost",
    user="root",
    password="",
    database="facemask"
)

mycursor = mydb.cursor()

model = load_model(os.path.join(settings.BASE_DIR, 'LIVE/model2-010.modelmodel-010.h5'))


class output(object):
    def __init__(self):
        self.webcam = cv2.VideoCapture(0)

    def __del__(self):
        self.webcam.release()

    def get_frame(self):
        labels_dict = {0: 'NO MASK', 1: 'MASK'}
        color_dict = {0: (0, 0, 255), 1: (0, 255, 0)}
        print(i)
        size = 4

        # We load the xml file
        classifier = cv2.CascadeClassifier(os.path.join(settings.BASE_DIR, 'LIVE/haarcascade_frontalface_default.xml'))

        success, im = self.webcam.read()

        im = cv2.flip(im, 1, 1)  # Flip to act as a mirror

        # Resize the image to speed up detection
        mini = cv2.resize(im, (im.shape[1] // size, im.shape[0] // size))

        # detect MultiScale / faces
        faces = classifier.detectMultiScale(mini)

        # Draw rectangles around each face
        for f in faces:
            (x, y, w, h) = [v * size for v in f]  # Scale the shapesize backup
            # Save just the rectangle faces in SubRecFaces
            face_img = im[y:y + h, x:x + w]
            resized = cv2.resize(face_img, (150, 150))
            normalized = resized / 255.0
            reshaped = np.reshape(normalized, (1, 150, 150, 3))
            reshaped = np.vstack([reshaped])
            result = model.predict(reshaped)

            label = np.argmax(result, axis=1)[0]
            if label == 0:
                while True:
                    letters = string.ascii_lowercase
                    filename = ''.join(random.choice(letters) for i in range(10)) + '.jpg'
                    if not os.path.isfile(os.path.join('C:\\Users\\thisa\\Documents\\Detected\\', filename)):
                        break;
                cv2.imwrite('C:\\Users\\thisa\\Documents\\Detected\\' + filename, face_img)
                sql = "INSERT INTO detection (img_path) VALUES (%s)"
                val = "C:\\Users\\thisa\\Documents\\Detected\\" + filename,
                print(val)
                mycursor.execute(sql, val)
                mydb.commit()

            cv2.rectangle(im, (x, y), (x + w, y + h), color_dict[label], 2)
            cv2.rectangle(im, (x, y - 40), (x + w, y), color_dict[label], -1)
            cv2.putText(im, labels_dict[label], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # Show the image
        ret, op, = cv2.imencode('.jpg', im)
        return op.tobytes()
