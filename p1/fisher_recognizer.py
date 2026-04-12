import os
import cv2
import numpy as np


class FisherFaceRecognizer:
    UNKNOWN_THRESHOLD = 2000

    def __init__(self, train_path):
        self.train_path = train_path
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self.face_list = []
        self.class_list = []
        self.person_names = []
        self.target_size = None
        self.glasses = cv2.imread('glasses.png', cv2.IMREAD_UNCHANGED)
        self.hat = cv2.imread('hat.png', cv2.IMREAD_UNCHANGED)
        self.unknown_mask = cv2.imread('???.png', cv2.IMREAD_UNCHANGED)

    def overlay(self, frame, overlay_img, x, y, w, h):
        if overlay_img is None or w <= 0 or h <= 0:
            return
        resized = cv2.resize(overlay_img, (w, h))

        fh, fw = frame.shape[:2]
        x1, y1 = max(x, 0), max(y, 0)
        x2, y2 = min(x + w, fw), min(y + h, fh)
        if x1 >= x2 or y1 >= y2:
            return

        ox1, oy1 = x1 - x, y1 - y
        ox2, oy2 = ox1 + (x2 - x1), oy1 + (y2 - y1)
        roi = resized[oy1:oy2, ox1:ox2]

        if roi.shape[2] == 4:
            alpha = roi[:, :, 3] / 255.0
            for c in range(3):
                frame[y1:y2, x1:x2, c] = (
                    alpha * roi[:, :, c] + (1 - alpha) * frame[y1:y2, x1:x2, c]
                )
        else:
            frame[y1:y2, x1:x2] = roi

    def load_dataset(self):
        self.face_list = []
        self.class_list = []
        self.person_names = sorted(
            d for d in os.listdir(self.train_path)
            if os.path.isdir(os.path.join(self.train_path, d))
        )

        for idx, name in enumerate(self.person_names):
            full_path = os.path.join(self.train_path, name)
            faces_found = 0

            for img_name in os.listdir(full_path):
                img_path = os.path.join(full_path, img_name)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

                if img is None:
                    continue

                if self.target_size is None:
                    self.target_size = (img.shape[1], img.shape[0])

                img = cv2.resize(img, self.target_size)

                self.face_list.append(img)
                self.class_list.append(idx)
                faces_found += 1

            print(f"  {name}: {faces_found} faces carregadas")

    def train(self):
        if not self.face_list:
            raise RuntimeError("Sem dados. Chama load_dataset() primeiro.")

        labels = np.array(self.class_list)
        num_classes = len(set(self.class_list))

        if num_classes < 2:
            raise RuntimeError(f"Precisa de pelo menos 2 classes, tem {num_classes}.")

        model = cv2.face.FisherFaceRecognizer_create(num_components=num_classes - 1)
        model.train(self.face_list, labels)
        self.model = model
        print(f"Treino concluído com {len(self.face_list)} faces de {num_classes} pessoas.")

    def predict(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        gray = cv2.resize(gray, self.target_size)

        label, confidence = self.model.predict(gray)
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
                img = cv2.imread(os.path.join(person_path, img_name), cv2.IMREAD_GRAYSCALE)
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
            faces = self.face_cascade.detectMultiScale(
                gray, scaleFactor=1.2, minNeighbors=8, minSize=(80, 80)
            )

            for x, y, w, h in faces:
                face_img = gray[y:y + h, x:x + w]
                face_img = cv2.resize(face_img, self.target_size)

                label, confidence = self.model.predict(face_img)
                name = self.person_names[label] if confidence < self.UNKNOWN_THRESHOLD else "Desconhecido"

                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, f"{name} ({confidence:.0f})", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                if name == 'Bruno Rodrigues':
                    gw, gh = w, int(h * 0.35)
                    gy = y + int(h * 0.25)
                    self.overlay(frame, self.glasses, x, gy, gw, gh)
                elif name == 'Pedro M Jorge':
                    hw = int(w * 1.4)
                    hh = int(h * 0.9)
                    hx = x - (hw - w) // 2
                    hy = y - int(h * 0.85)
                    self.overlay(frame, self.hat, hx, hy, hw, hh)
                elif name == 'Desconhecido':
                    uw = int(w * 0.9)
                    uh = int(h * 0.9)
                    ux = x + (w - uw) // 2
                    uy = y - int(h * 0.95)
                    self.overlay(frame, self.unknown_mask, ux, uy, uw, uh)

            cv2.imshow("Fisher Face Recognizer", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
