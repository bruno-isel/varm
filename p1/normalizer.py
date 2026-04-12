import cv2
import numpy as np


class FaceNormalizer:
    # Posições alvo dos olhos segundo MPEG-7
    LEFT_EYE_POS = (16, 24)   # (coluna, linha)
    RIGHT_EYE_POS = (31, 24)  # (coluna, linha)
    OUTPUT_SIZE = (46, 56)     # (largura, altura)

    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self.eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_eye.xml'
        )
        self.eye_cascade_glasses = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_eye_tree_eyeglasses.xml'
        )

    def detect_face(self, gray):
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        if len(faces) == 0:
            return None

        # Retorna as coordenadas da face maior
        return max(faces, key=lambda f: f[2] * f[3])

    def detect_eyes(self, gray, face_rect):
        fx, fy, fw, fh = face_rect
        # Procurar olhos nos 65% superiores da face
        upper_region = gray[fy:fy + int(fh * 0.65), fx:fx + fw]
        # Equalizar histograma para melhorar contraste
        upper_region = cv2.equalizeHist(upper_region)

        # Tentar com o cascade normal primeiro, depois com óculos
        # Relaxar progressivamente: minNeighbors, scaleFactor, minSize
        for cascade in [self.eye_cascade, self.eye_cascade_glasses]:
            for scale, min_neighbors, min_size in [
                (1.05, 5, 15), (1.05, 3, 15), (1.05, 2, 15),
                (1.03, 2, 10), (1.03, 1, 10)
            ]:
                eyes = cascade.detectMultiScale(
                    upper_region, scaleFactor=scale, minNeighbors=min_neighbors, minSize=(min_size, min_size)
                )
                if len(eyes) >= 2:
                    break
            if len(eyes) >= 2:
                break

        if len(eyes) >= 2:
            # Ordenar por tamanho e pegar os 2 maiores
            eyes = sorted(eyes, key=lambda e: e[2] * e[3], reverse=True)[:2]

            # Centro de cada olho em coordenadas ABSOLUTAS da imagem original
            centers = [(fx + ex + ew // 2, fy + ey + eh // 2) for ex, ey, ew, eh in eyes]

            # Ordenar: olho esquerdo (menor x) e olho direito (maior x)
            centers.sort(key=lambda c: c[0])
            return centers[0], centers[1]

        # Fallback: estimar posição dos olhos a partir da geometria da face
        left_eye = (fx + int(fw * 0.3), fy + int(fh * 0.35))
        right_eye = (fx + int(fw * 0.7), fy + int(fh * 0.35))
        return left_eye, right_eye

    def align_face(self, face_img, left_eye, right_eye):
        # Diferença entre os olhos
        dx = right_eye[0] - left_eye[0]
        dy = right_eye[1] - left_eye[1]

        # Ângulo de inclinação (em graus)
        angle = np.degrees(np.arctan2(dy, dx))

        # Ponto central entre os olhos — usamos como eixo de rotação
        center = (float(left_eye[0] + right_eye[0]) / 2,
                  float(left_eye[1] + right_eye[1]) / 2)

        # Matriz de rotação (escala 1.0 = sem zoom)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

        # Rodar a imagem inteira
        aligned = cv2.warpAffine(face_img, rotation_matrix, (face_img.shape[1], face_img.shape[0]))

        return aligned

    def normalize(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img

        # Passo 1 — Detetar a face
        face_rect = self.detect_face(gray)
        if face_rect is None:
            return None

        # Passo 2 — Detetar os olhos (coordenadas absolutas)
        left_eye, right_eye = self.detect_eyes(gray, face_rect)
        if left_eye is None or right_eye is None:
            return None

        # Passo 3 — Rodar a imagem inteira para alinhar os olhos
        aligned = self.align_face(gray, left_eye, right_eye)

        # Passo 4 — Escalar e recortar
        # Distância atual vs alvo
        current_dist = right_eye[0] - left_eye[0]
        target_dist = self.RIGHT_EYE_POS[0] - self.LEFT_EYE_POS[0]  # 31 - 16 = 15
        scale = target_dist / current_dist

        # Redimensionar a imagem inteira
        new_w = int(aligned.shape[1] * scale)
        new_h = int(aligned.shape[0] * scale)
        scaled = cv2.resize(aligned, (new_w, new_h))

        # Posição do olho esquerdo após escalar
        left_x = int(left_eye[0] * scale)
        left_y = int(left_eye[1] * scale)

        # Recortar para que o olho esquerdo fique em (16, 24)
        crop_x = left_x - self.LEFT_EYE_POS[0]
        crop_y = left_y - self.LEFT_EYE_POS[1]

        # Adicionar padding se o crop sair dos limites
        out_w, out_h = self.OUTPUT_SIZE
        pad_left = max(0, -crop_x)
        pad_top = max(0, -crop_y)
        pad_right = max(0, (crop_x + out_w) - new_w)
        pad_bottom = max(0, (crop_y + out_h) - new_h)

        if pad_left > 0 or pad_top > 0 or pad_right > 0 or pad_bottom > 0:
            scaled = cv2.copyMakeBorder(scaled, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_REPLICATE)
            crop_x += pad_left
            crop_y += pad_top

        return scaled[crop_y:crop_y + out_h, crop_x:crop_x + out_w]


if __name__ == '__main__':
    import argparse
    import os

    parser = argparse.ArgumentParser(description='Normalizar dataset segundo MPEG-7')
    parser.add_argument('--input', type=str, default='dataset', help='Pasta do dataset original')
    parser.add_argument('--output', type=str, default='dataset_normalized', help='Pasta de saída')
    args = parser.parse_args()

    normalizer = FaceNormalizer()
    success = 0
    fail = 0

    for split in ['train', 'test']:
        split_in = os.path.join(args.input, split)
        if not os.path.isdir(split_in):
            continue

        for person in os.listdir(split_in):
            person_in = os.path.join(split_in, person)
            if not os.path.isdir(person_in):
                continue

            person_out = os.path.join(args.output, split, person)
            os.makedirs(person_out, exist_ok=True)

            for img_name in os.listdir(person_in):
                img = cv2.imread(os.path.join(person_in, img_name))
                if img is None:
                    continue

                result = normalizer.normalize(img)
                if result is not None:
                    cv2.imwrite(os.path.join(person_out, img_name), result)
                    success += 1
                else:
                    fail += 1
                    print(f"  Falhou: {split}/{person}/{img_name}")

    print(f"\nNormalizadas: {success} | Falharam: {fail}")
    print(f"Output: {args.output}/")
