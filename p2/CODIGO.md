# VARM P2 — Explicação Detalhada do Código

**Aluno:** Bruno Rodrigues (52323)  
**Stack:** Python 3.10 + opencv-contrib-python 4.9.0.80 + numpy 1.26.4

---

## Índice

1. [Conceitos fundamentais](#conceitos-fundamentais)
2. [calibrate.py](#calibratepy)
3. [aruco_detector.py](#aruco_detectorpy)
4. [main.py](#mainpy)
5. [generate_markers.py](#generate_markerspy)
6. [Pipeline completo](#pipeline-completo)

---

## Conceitos fundamentais

### Sistemas de coordenadas

O projeto envolve três sistemas de coordenadas distintos:

```
Mundo 3D (marcador)  →  Câmara 3D  →  Imagem 2D (píxeis)
       [rvec, tvec]          [mtx, dist]
```

- **Espaço do marcador** — origem no centro do marcador, Z aponta para cima (para a câmara), unidades em cm
- **Espaço da câmara** — origem no centro óptico da câmara
- **Espaço da imagem** — píxeis 2D no plano da imagem

### Matriz intrínseca K

A câmara é modelada por uma matriz 3×3:

```
K = [ fx   0   cx ]
    [  0  fy   cy ]
    [  0   0    1 ]
```

- `fx`, `fy` — distâncias focais em píxeis (quantos píxeis correspondem a 1 cm a 1 cm de distância)
- `cx`, `cy` — ponto principal (centro óptico projetado na imagem, idealmente o centro do frame)

### Vetor de Rodrigues vs Matriz de Rotação

A rotação pode ser representada de duas formas:
- **Vetor de Rodrigues** `rvec` (3×1) — direção = eixo de rotação, magnitude = ângulo em radianos
- **Matriz de rotação** `R` (3×3) — forma explícita, usada para cálculos geométricos

`cv2.Rodrigues()` converte entre as duas representações.

---

## calibrate.py

### Objetivo

Calcular os parâmetros intrínsecos da câmara (matriz K e coeficientes de distorção) a partir de imagens de um tabuleiro de xadrez com geometria conhecida.

### Constantes

```python
CHESSBOARD = (9, 6)    # cantos internos (cols, rows) — não os quadrados totais
SQUARE_SIZE = 2.5      # cm — tamanho real medido do quadrado impresso
```

### `objp` — pontos 3D do tabuleiro

```python
objp = np.zeros((CHESSBOARD[0] * CHESSBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHESSBOARD[0], 0:CHESSBOARD[1]].T.reshape(-1, 2)
objp *= SQUARE_SIZE
```

Cria uma grelha de 54 pontos (9×6) no plano Z=0 com coordenadas reais em cm:

```
(0,0,0)  (2.5,0,0)  (5.0,0,0) ...
(0,2.5,0) (2.5,2.5,0) ...
```

Estes são os pontos "verdadeiros" do mundo físico — o algoritmo sabe exatamente onde cada canto deve estar.

### `capture_mode()` — captura de frames

```python
found, corners = cv2.findChessboardCorners(gray, CHESSBOARD, None)
```

Deteta os 54 cantos internos do tabuleiro na imagem em cinzento. Devolve `found=True` e as posições 2D aproximadas.

```python
corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), CRITERIA)
```

Refina as posições dos cantos com precisão subpixel usando um critério de convergência. A janela `(11,11)` define a área de busca em redor de cada canto.

A cada `SPACE` pressionado, guarda o par:
- `obj_points` ← posições 3D reais (sempre o mesmo `objp`)
- `img_points` ← posições 2D detetadas neste frame específico

### `_run_calibration()` — calibração

```python
rms, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
    obj_points, img_points, img_size, None, None
)
```

Com N pares (3D→2D), resolve o sistema de equações para encontrar K e dist que minimizam o erro de reprojeção. Devolve:

- **`mtx`** — matriz intrínseca K
- **`dist`** — 5 coeficientes de distorção `[k1, k2, p1, p2, k3]` (radial e tangencial)
- **`rms`** — erro médio de reprojeção em píxeis (quão bem os parâmetros reproduzem os pontos observados)

```python
new_mtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
```

Calcula uma nova matriz K ajustada para a imagem sem distorção, preservando todos os píxeis (`alpha=1`).

### `check_mode()` — verificação visual

```python
undistorted = cv2.undistort(frame, mtx, dist, None, new_mtx)
```

Aplica a correção de distorção a cada frame. Ao comparar original vs corrigido é possível ver as linhas retas que antes eram curvas.

---

## aruco_detector.py

### Objetivo

Detetar marcadores ArUco em tempo real e visualizar a pose estimada com eixos 3D. Funciona como ferramenta de diagnóstico standalone.

### Criação do detetor

```python
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
params = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, params)
```

- `DICT_6X6_250` — dicionário com 250 marcadores possíveis, cada um codificado numa grelha 6×6 de bits (64 bits por marcador, com redundância para correção de erros)
- `DetectorParameters` — configura thresholds adaptativos, limites de tamanho, taxa de correção de erros, etc.

### Ajuste dos parâmetros de deteção

```python
params.adaptiveThreshWinSizeMin = 3
params.adaptiveThreshWinSizeMax = 53
params.minMarkerPerimeterRate = 0.02
params.errorCorrectionRate = 0.6
```

Aumentar `adaptiveThreshWinSizeMax` melhora a deteção com iluminação não uniforme. `errorCorrectionRate=0.6` permite até 60% de bits incorretos antes de rejeitar o marcador.

### Deteção e pose

```python
corners, ids, rejected = detector.detectMarkers(gray)
```

- `corners` — lista de arrays `(1, 4, 2)` com os 4 cantos de cada marcador em píxeis
- `ids` — IDs dos marcadores detetados
- `rejected` — candidatos rejeitados (útil para debug)

```python
rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
    corners, MARKER_SIZE, mtx, dist
)
```

Para cada marcador, resolve internamente o PnP (Perspective-n-Point) com os 4 cantos como correspondência 3D→2D. Devolve `rvec` e `tvec` para cada marcador.

```python
cv2.drawFrameAxes(show, mtx, dist, rvec, tvec, MARKER_SIZE * 0.5)
```

Desenha os eixos X (vermelho), Y (verde), Z (azul) sobre o marcador com comprimento `MARKER_SIZE * 0.5`.

### Modo debug (`d`)

```python
thresh = cv2.adaptiveThreshold(gray, 255,
    cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 21, 7)
```

Mostra a imagem binarizada que o algoritmo usa internamente para encontrar os quadrados do marcador. Útil para perceber porque é que um marcador não é detetado.

---

## main.py

### Objetivo

Aplicação AR principal: deteta marcadores e sobrepõe objetos 3D wireframe alinhados com cada marcador, usando `solvePnPRansac` + `Rodrigues` para estimar a pose.

### `MARKER_OBJ_PTS` — cantos do marcador em 3D

```python
_h = MARKER_SIZE / 2.0
MARKER_OBJ_PTS = np.array([
    [-_h,  _h, 0],   # topo-esquerdo
    [ _h,  _h, 0],   # topo-direito
    [ _h, -_h, 0],   # baixo-direito
    [-_h, -_h, 0],   # baixo-esquerdo
], dtype=np.float32)
```

Define os 4 cantos do marcador no seu referencial local (Z=0, unidades cm). A ordem corresponde à ordem que `detectMarkers` devolve os cantos (sentido horário a partir do topo-esquerdo).

### Definição dos objetos 3D

Os objetos existem no **espaço do marcador**: origem no centro, Z positivo aponta para cima (para fora do marcador).

**Cubo** (`_cube_edges`):
```python
# 4 vértices base no plano Z=0 (plano do marcador)
# 4 vértices topo no plano Z=h (acima do marcador)
v = [(-s/2,-s/2,0), (s/2,-s/2,0), (s/2,s/2,0), (-s/2,s/2,0),
     (-s/2,-s/2,h), (s/2,-s/2,h), (s/2,s/2,h), (-s/2,s/2,h)]
```

**Pirâmide** (`_pyramid_edges`):
```python
# 4 vértices base + 1 ápice no centro a altura h
v = [(-s/2,-s/2,0), (s/2,-s/2,0), (s/2,s/2,0), (-s/2,s/2,0),
     (0, 0, h)]
```

### Estimação de pose — `solvePnPRansac` + `Rodrigues`

```python
ok, rvec, tvec, _ = cv2.solvePnPRansac(
    MARKER_OBJ_PTS, corner[0], mtx, dist
)
```

Resolve o problema PnP (Perspective-n-Point): dados 4 pontos 3D conhecidos (`MARKER_OBJ_PTS`) e as suas projeções 2D na imagem (`corner[0]`), encontra a rotação e translação que explicam essa correspondência.

O **RANSAC** (Random Sample Consensus) torna o algoritmo robusto: testa várias combinações aleatórias dos pontos e escolhe o modelo que maximiza os inliers, rejeitando cantos mal detetados.

Devolve:
- `rvec` (3×1) — vetor de rotação de Rodrigues
- `tvec` (3×1) — translação em cm `[x, y, z]` no referencial da câmara

```python
R, _ = cv2.Rodrigues(rvec)
angle = np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1.0, 1.0)))
```

Converte o vetor de rotação para a matriz de rotação 3×3. O ângulo total de rotação é extraído via `trace(R)`: quando `R = I` (sem rotação), `trace = 3` e `angle = 0°`; quando inclinado, o ângulo aumenta.

### `draw_object()` — projeção 3D→2D

```python
projected, _ = cv2.projectPoints(vertices, rvec, tvec, mtx, dist)
```

Aplica a transformação completa a cada vértice 3D:

```
P_marcador  →  [R|t]  →  P_câmara  →  K  →  P_imagem  →  distorção  →  pixel
```

1. `[R|t]` (de `rvec`/`tvec`) — transforma do espaço do marcador para o espaço da câmara
2. `K` (mtx) — projeta do 3D para o plano 2D da imagem
3. `dist` — aplica distorção radial/tangencial da lente

```python
for a, b in edges:
    cv2.line(frame, tuple(projected[a]), tuple(projected[b]), color, 2)
```

Liga os vértices projetados com linhas para formar o wireframe.

### Por que é que o objeto cola ao marcador

`rvec` e `tvec` são recalculados a cada frame com base nos cantos detetados nesse frame. Quando a câmara ou o marcador se move, as posições dos cantos em píxeis mudam, o `solvePnPRansac` recalcula a pose, e `projectPoints` reprojecta os vértices nas novas posições — o objeto segue automaticamente.

### `OBJECTS` dict — extensibilidade

```python
OBJECTS = {0: _cube_edges, 1: _pyramid_edges}
```

Para adicionar um novo objeto para o ID 2:
```python
def _meu_objeto(s=4.0):
    vertices = np.float32([...])
    edges = [(a, b), ...]
    return vertices, edges

OBJECTS[2] = _meu_objeto
```

---

## generate_markers.py

### Objetivo

Gerar imagens dos marcadores ArUco programaticamente, sem depender de ferramentas externas.

```python
img = cv2.aruco.generateImageMarker(aruco_dict, marker_id, MARKER_SIZE_PX)
```

`generateImageMarker` (equivalente a `drawMarker` em versões anteriores do OpenCV) desenha o marcador com ID `marker_id` do dicionário `aruco_dict` numa imagem quadrada de `MARKER_SIZE_PX` píxeis.

O marcador gerado tem borda preta por defeito. Ao imprimir, garantir pelo menos **1–2 cm de margem branca** em todos os lados — sem ela o algoritmo de deteção não encontra a transição preto→branco da borda exterior.

---

## Pipeline completo

```
1. generate_markers.py
   └─ generateImageMarker() → marker_0.png, marker_1.png → imprimir

2. calibrate.py
   ├─ findChessboardCorners() + cornerSubPix() → pontos 2D precisos
   ├─ calibrateCamera()                        → mtx, dist, rms
   ├─ getOptimalNewCameraMatrix()              → new_mtx
   └─ undistort()                              → camera_calibration.npz

3. main.py (cada frame)
   ├─ ArucoDetector.detectMarkers()           → corners, ids
   ├─ drawDetectedMarkers()                   → borda verde na imagem
   ├─ [por cada marcador]
   │   ├─ solvePnPRansac(MARKER_OBJ_PTS, corners_2D, mtx, dist) → rvec, tvec
   │   ├─ Rodrigues(rvec)                    → R, angle
   │   ├─ draw_object()
   │   │   ├─ projectPoints(vertices_3D, rvec, tvec, mtx, dist) → pixels_2D
   │   │   └─ cv2.line() por cada aresta     → wireframe na imagem
   │   └─ drawFrameAxes()                    → eixos XYZ
   └─ imshow()                               → janela AR
```

---

## Diferença entre `estimatePoseSingleMarkers` e `solvePnPRansac`

| | `estimatePoseSingleMarkers` | `solvePnPRansac` |
|---|---|---|
| Uso | Uma chamada para todos os marcadores | Uma chamada por marcador |
| Robustez | Usa todos os 4 cantos sempre | RANSAC ignora outliers |
| Controlo | Interno / opaco | Explícito — vês os pontos 3D |
| `Rodrigues` | Implícito | Explícito com `cv2.Rodrigues()` |
| Usado em | `aruco_detector.py` | `main.py` |
