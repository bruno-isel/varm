# VARM — P2: Marker Based Augmented Reality

Aplicação de AR baseada em marcadores ArUco com OpenCV/Python.

## Dependências

```bash
pip install opencv-contrib-python==4.9.0.80 numpy==1.26.4
```

> Usar `opencv-contrib-python` (não `opencv-python`) para ter acesso ao módulo `cv2.aruco`.

---

## Como correr

### 1. Calibrar a câmara

```bash
python calibrate.py
```

Controlos: `SPACE` captura frame · `c` calibra e guarda · `q` sai

Gera `camera_calibration.npz` com os parâmetros intrínsecos da câmara.

Para verificar o resultado (mostra feed original vs sem distorção):

```bash
python calibrate.py --check
```

### 2. Testar deteção ArUco (opcional)

```bash
python aruco_detector.py
```

Mostra o feed da câmara com eixos de pose desenhados sobre cada marcador detetado.

### 3. Aplicação AR principal

```bash
python main.py
```

Sobrepõe objetos 3D wireframe sobre os marcadores detetados:

| ID | Objeto |
|----|--------|
| 0  | Cubo (ciano) |
| 1  | Pirâmide (verde) |
| outro | Eixos XYZ |

---

## ⚠️ Passos manuais pendentes (fazer antes de correr pela primeira vez)

O código está completo mas é necessário preparar o material físico antes de executar.

### A. Imprimir o tabuleiro de xadrez para calibração

- Usar um tabuleiro com **9×6 cantos internos** (10×7 quadrados)
- Imprimir em A4 e colar numa superfície rígida plana (cartão, pasta)
- Medir o tamanho real de um quadrado em metros e atualizar `SQUARE_SIZE` em `calibrate.py`
  - Default: `0.025` (2.5 cm) — ajustar conforme a impressão
- Tabuleiro de exemplo: https://docs.opencv.org/master/pattern.png

### B. Gerar e imprimir os marcadores ArUco

- Aceder a https://chev.me/arucogen/
- Selecionar dicionário **6x6 (250)** e gerar pelo menos os IDs **0** e **1**
- Tamanho sugerido: 5–8 cm de lado
- Após imprimir, medir o lado real e atualizar `MARKER_SIZE` em `aruco_detector.py` e `main.py`
  - Default: `0.05` (5 cm) — ajustar conforme a impressão

### C. Tirar as fotos de calibração

- Com o tabuleiro impresso e a webcam ligada, correr `python calibrate.py`
- Capturar **mínimo 10 frames** com o tabuleiro em posições, ângulos e distâncias variados
- O erro de reprojeção (RMS) ideal é abaixo de **1.0 px**

---

## Estrutura do projeto

```
p2/
├── calibrate.py            # calibração da câmara com chessboard
├── aruco_detector.py       # deteção ArUco + estimação de pose
├── main.py                 # aplicação AR principal
├── camera_calibration.npz  # gerado após calibrar (não commitar)
└── README.md
```
