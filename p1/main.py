import argparse
import os
import cv2
from fisher_recognizer import FisherFaceRecognizer
from normalizer import FaceNormalizer


def sync_normalized(raw_root, norm_root):
    if not os.path.isdir(raw_root):
        return
    normalizer = None
    added = 0
    failed = 0
    for split in ('train', 'test'):
        split_in = os.path.join(raw_root, split)
        if not os.path.isdir(split_in):
            continue
        for person in os.listdir(split_in):
            person_in = os.path.join(split_in, person)
            if not os.path.isdir(person_in):
                continue
            person_out = os.path.join(norm_root, split, person)
            for img_name in os.listdir(person_in):
                out_path = os.path.join(person_out, img_name)
                if os.path.exists(out_path):
                    continue
                img = cv2.imread(os.path.join(person_in, img_name))
                if img is None:
                    continue
                if normalizer is None:
                    normalizer = FaceNormalizer()
                    print("Novas fotos detetadas — a normalizar...")
                os.makedirs(person_out, exist_ok=True)
                result = normalizer.normalize(img)
                if result is not None:
                    cv2.imwrite(out_path, result)
                    added += 1
                else:
                    failed += 1
                    print(f"  Falhou: {split}/{person}/{img_name}")
    if normalizer is not None:
        print(f"Normalização: +{added} novas | {failed} falharam")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='dataset_normalized', help='Pasta do dataset (ex: dataset_normalized)')
    parser.add_argument('--image', type=str, help='Caminho para uma imagem a reconhecer')
    parser.add_argument('--live', action='store_true', help='Modo webcam em tempo real')
    args = parser.parse_args()

    train_path = f'{args.dataset}/train'
    test_path = f'{args.dataset}/test'

    sync_normalized('dataset', args.dataset)

    recognizer = FisherFaceRecognizer(train_path)
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
        recognizer.evaluate(test_path)


if __name__ == '__main__':
    main()
