import os
import cv2
import numpy as np


class FisherFaceRecognizer:
    UNKNOWN_THRESHOLD = 3000

    def __init__(self, train_path):
        self.train_path = train_path
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self.face_list = []
        self.class_list = []
        self.person_names = ['Bruno Rodrigues', 'Meda', 'Pedro M Jorge']
        self.target_size = None

    def load_dataset(self):
        self.face_list = []
        self.class_list = []

        for idx, name in enumerate(self.person_names):
            full_path = os.path.join(self.train_path, name)
            faces_found = 0

            for img_name in os.listdir(full_path):
                img_path = os.path.join(full_path, img_name)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

                if img is None:
                    continue

                # Guardar o tamanho da primeira imagem como referência
                if self.target_size is None:
                    self.target_size = (img.shape[1], img.shape[0])

                # FisherFaces exige todas as imagens do mesmo tamanho
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

        # FisherFaces precisa de num_components <= num_classes - 1
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
            faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)

            for x, y, w, h in faces:
                face_img = gray[y:y + h, x:x + w]
                face_img = cv2.resize(face_img, self.target_size)

                label, confidence = self.model.predict(face_img)
                name = self.person_names[label] if confidence < self.UNKNOWN_THRESHOLD else "Desconhecido"

                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, f"{name} ({confidence:.0f})", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            cv2.imshow("Fisher Face Recognizer", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='FisherFaces Recognizer')
    parser.add_argument('--dataset', type=str, default='dataset_normalized', help='Pasta do dataset normalizado')
    parser.add_argument('--image', type=str, help='Caminho para uma imagem a reconhecer')
    parser.add_argument('--live', action='store_true', help='Modo webcam em tempo real')
    args = parser.parse_args()

    recognizer = FisherFaceRecognizer(f'{args.dataset}/train')
    recognizer.load_dataset()
    recognizer.train()

    if args.live:
        recognizer.live()
    elif args.image:
        img = cv2.imread(args.image)
        if img is None:
            print(f"Erro: não foi possível abrir '{args.image}'")
        else:
            name, confidence = recognizer.predict(img)
            if name is None:
                print("Nenhuma face detetada na imagem.")
            else:
                print(f"Pessoa reconhecida: {name} (confiança: {confidence:.1f})")
    else:
        recognizer.evaluate(f'{args.dataset}/test')
