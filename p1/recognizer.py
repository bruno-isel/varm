import os
import cv2
import numpy as np


class FaceRecognizer:
    UNKNOWN_THRESHOLD = 100

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
        self.person_names = ['Angelina Jolie', 'Bruno Rodrigues', 'Meda', 'Pedro M Jorge']

        for idx, name in enumerate(self.person_names):
            full_path = os.path.join(self.train_path, name)
            faces_found = 0

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

                x, y, w, h = max(detected_faces, key=lambda f: f[2] * f[3])
                face_img = img[y:y + h, x:x + w]
                self.face_list.append(face_img)
                self.class_list.append(idx)
                faces_found += 1

            print(f"  {name}: {faces_found} faces carregadas")

    def get_faces(self):
        return self.face_list, self.class_list

    def get_person_names(self):
        return self.person_names

    def train(self):
        if not self.face_list:
            raise RuntimeError("Sem dados. Chama load_dataset() primeiro.")

        model = cv2.face.LBPHFaceRecognizer_create()
        model.train(self.face_list, np.array(self.class_list))
        self.model = model
        print(f"Treino concluído com {len(self.face_list)} faces de {len(self.person_names)} pessoas.")

    def predict(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img

        detected_faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)
        if len(detected_faces) < 1:
            return None, None

        x, y, w, h = detected_faces[0]
        face_img = gray[y:y + h, x:x + w]

        label, confidence = self.model.predict(face_img)
        name = self.person_names[label] if confidence < self.UNKNOWN_THRESHOLD else "Desconhecido"
        return name, confidence

    def evaluate(self, test_path):
        correct = 0
        total = 0

        for person in os.listdir(test_path):
            person_path = os.path.join(test_path, person)
            if not os.path.isdir(person_path):
                continue
            for img_name in os.listdir(person_path):
                img = cv2.imread(os.path.join(person_path, img_name))
                if img is None:
                    continue
                name, confidence = self.predict(img)
                total += 1
                if name == person:
                    correct += 1
                conf_str = f"{confidence:.1f}" if confidence is not None else "N/A"
                print(f"Real: {person} | Previsto: {name} | Confiança: {conf_str}")

        accuracy = 100 * correct / total if total > 0 else 0
        print(f"\nAccuracy: {correct}/{total} ({accuracy:.1f}%)")
        return accuracy

    def live(self):
        cap = cv2.VideoCapture(1, cv2.CAP_AVFOUNDATION)
        if not cap.isOpened():
            cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
        if not cap.isOpened():
            print("Erro: não foi possível abrir a webcam.")
            return
        print("A iniciar webcam... pressiona 'q' para sair.")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)

            for x, y, w, h in faces:
                face_img = gray[y:y + h, x:x + w]
                label, confidence = self.model.predict(face_img)
                name = self.person_names[label] if confidence < self.UNKNOWN_THRESHOLD else "Desconhecido"

                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, f"{name} ({confidence:.0f})", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            cv2.imshow("Face Recognizer", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
