import argparse
import cv2
from recognizer import FaceRecognizer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, help='Caminho para uma imagem a reconhecer')
    parser.add_argument('--live', action='store_true', help='Modo webcam em tempo real')
    args = parser.parse_args()

    recognizer = FaceRecognizer('dataset/train')
    recognizer.load_dataset()
    recognizer.train()

    if args.live:
        recognizer.live()
    elif args.image:
        img = cv2.imread(args.image)
        if img is None:
            print(f"Erro: não foi possível abrir '{args.image}'")
            return
        name, confidence = recognizer.predict(img)
        if name is None:
            print("Nenhuma face detetada na imagem.")
        else:
            print(f"Pessoa reconhecida: {name} (confiança: {confidence:.1f})")
    else:
        recognizer.evaluate('dataset/test')

if __name__ == '__main__':
    main()
