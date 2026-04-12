# VARM — 1º Projeto: Face Detection, Recognition e Augmented Reality

Aplicação de reconhecimento facial com augmented reality em tempo real, desenvolvida para a cadeira de Computer Vision and Mixed Reality (VARM), ISEL.

## Como correr

### Dependências

```bash
pip install opencv-contrib-python numpy
```

(O pacote é `opencv-contrib-python` e não `opencv-python` porque precisamos do módulo `cv2.face`, onde vive o `FisherFaceRecognizer`.)

### Modos de execução

A partir da pasta [p1/](p1/):

```bash
# 1. Webcam em tempo real com overlays AR (modo principal)
python main.py --live

# 2. Avaliar o modelo sobre o test set (default se não for dado nenhum modo)
python main.py

# 3. Reconhecer uma única imagem
python main.py --image caminho/para/foto.jpg
```

No modo `--live`, carrega em `q` para sair. Os overlays AR são automáticos: óculos em cima do Bruno, chapéu em cima do Pedro, e um ponto de interrogação em cima de caras desconhecidas.

### Adicionar fotos novas

1. Copiar as fotos para `dataset/train/<Nome>/` (ou `dataset/test/<Nome>/`).
2. Correr `python main.py --live`.
3. O `main.py` deteta automaticamente as fotos novas, passa-as pelo `FaceNormalizer` (pipeline MPEG-7) e guarda-as em `dataset_normalized/`. Fotos já normalizadas são ignoradas (sincronização incremental).
4. O modelo FisherFaces é re-treinado em cada arranque com o conteúdo actual de `dataset_normalized/`.

Para adicionar uma pessoa nova basta criar a pasta correspondente; os `person_names` são descobertos dinamicamente a partir das subpastas de `train/`.

