import os
import cv2
import numpy as np


class FaceRecognizer:
    def __init__(self, train_path):
        self.train_path = train_path
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self.face_list = []
        self.class_list = []
        self.person_names = []

    def load_dataset(self):
        self.face_list = []
        self.class_list = []
        self.person_names = os.listdir(self.train_path)

        for idx, name in enumerate(self.person_names):
            full_path = os.path.join(self.train_path, name)

            for img_name in os.listdir(full_path):
                img_path = os.path.join(full_path, img_name)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

                if img is None:
                    continue

                detected_faces = self.face_cascade.detectMultiScale(
                    img, scaleFactor=1.2, minNeighbors=5
                )

                if len(detected_faces) < 1:
                    continue

                for x, y, w, h in detected_faces:
                    face_img = img[y:y + h, x:x + w]
                    self.face_list.append(face_img)
                    self.class_list.append(idx)

    def get_faces(self):
        return self.face_list, self.class_list

    def get_person_names(self):
        return self.person_names
