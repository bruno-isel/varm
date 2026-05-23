# VARM — P2: Marker Based Augmented Reality

## Objetivo

Aplicação de AR baseada em marcadores ArUco com OpenCV/Python. Integração opcional com Unity3D.

**Aluno:** Bruno Rodrigues (52323)  
**Cadeira:** Computer Vision and Mixed Reality (VARM), ISEL MEIM  
**Stack:** Python 3.10 + opencv-contrib-python 4.9.0.80 + numpy 1.26.4

---

## Requisitos Obrigatórios

### a) Calibração da Câmara (`calibrate.py`)
- Usar padrão de tabuleiro de xadrez (chessboard)
- Calcular parâmetros intrínsecos (matriz K) e coeficientes de distorção
- Guardar em ficheiro `.npz` para uso posterior
- Funções OpenCV obrigatórias:
  - `findChessboardCorners()`, `drawChessboardCorners()`, `cornerSubPix()`
  - `calibrateCamera()`, `getOptimalNewCameraMatrix()`, `undistort()`
- Output: `camera_calibration.npz` com chaves `mtx` e `dist`

### b) Deteção ArUco + Estimação de Pose (`aruco_detector.py`)
- Detetar marcadores ArUco em tempo real (webcam)
- Estimar pose de câmara por marcador (View Transformation)
- Funções OpenCV ArUco obrigatórias:
  - `getPredefinedDictionary()`, `DetectorParameters()`, `ArucoDetector()`
  - `detectMarkers()`, `drawDetectedMarkers()`, `estimatePoseSingleMarkers()`
  - `drawFrameAxes()` para visualizar eixos de cada marcador
- Usar dicionário `DICT_6X6_250` (ou similar)
- Gerar marcadores em https://chev.me/arucogen/

### c) Registo de Objetos Virtuais em OpenCV (`main.py`)
- Sobrepor objetos virtuais alinhados com cada marcador detetado
- **Objeto diferente por ID de marcador** (e.g., cubo, seta, texto, imagem)
- Projetar geometria 3D sobre a imagem da câmara com:
  - `solvePnPRansac()`, `projectPoints()`, `Rodrigues()`
- Ler parâmetros de calibração do `.npz` no arranque

---

## Requisito Opcional — Integração Unity3D

### Arquitetura
```
Python/OpenCV (servidor)  <--UDP-->  Unity3D (cliente)
  - porta 5000: frames JPEG (fragmentados, 4-byte header)
  - porta 5001: pose JSON  {"id":X,"rvec":[[rx,ry,rz]],"tvec":[[tx,ty,tz]]}
```

### Python lado servidor
```python
import socket, json
UNITY_IP   = '127.0.0.1'
PORT_FRAME = 5000
PORT_POSE  = 5001
MAX_UDP    = 60000  # limite seguro UDP

sock_frame = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock_pose  = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

def sendFrame(sock, frame, ip, port):
    _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 60])
    data = buffer.tobytes()
    num_chunks = (len(data) + MAX_UDP - 1) // MAX_UDP
    for i in range(num_chunks):
        chunk = data[i*MAX_UDP:(i+1)*MAX_UDP]
        header = i.to_bytes(2,'big') + num_chunks.to_bytes(2,'big')
        sock.sendto(header + chunk, (ip, port))

# No loop principal:
sendFrame(sock_frame, frame, UNITY_IP, PORT_FRAME)
for rvec, tvec, ids_i in zip(rotation_vectors, translation_vectors, ids):
    pose_data = {'id': int(ids_i), 'rvec': rvec.tolist(), 'tvec': tvec.tolist()}
    msg = json.dumps(pose_data, separators=(',',':')).encode()
    sock_pose.sendto(msg, (UNITY_IP, PORT_POSE))
```

### Unity3D
- **Template:** 3D Built-in Render Pipeline (NÃO URP)
- **Unity versão:** 6000.0.x LTS
- **Resolução Game View:** 640×480 Fixed

**Hierarquia da cena:**
```
SampleScene
├── Main Camera         ← script PoseReceiver.cs
├── Directional Light
├── EventSystem
├── Canvas (Screen Space - Camera)
│   └── RawImage        ← script VideoReceiver.cs
└── ARPivot_XX          ← pivot vazio por marcador
    └── ARObject_XX     ← objeto 3D filho do pivot (Position Y=0.5)
```

**VideoReceiver.cs** — recebe frames UDP porta 5000, reassembla chunks, exibe em RawImage  
**PoseReceiver.cs** — recebe pose UDP porta 5001, converte OpenCV→Unity, posiciona AR object

#### Conversão de coordenadas OpenCV → Unity
```
Posição:  (tx, -ty, tz)   # negar Y
Rotação:  Rodrigues → matriz R → aplicar sinal Y → Quaternion
```

#### Matriz de projeção Unity (substituir fieldOfView)
```csharp
projMatrix[0,0] = 2f * fx / w;
projMatrix[0,2] = 1f - 2f * cx / w;
projMatrix[1,1] = 2f * fy / h;
projMatrix[1,2] = -1f + 2f * cy / h;
projMatrix[2,2] = -(far + near) / (far - near);
projMatrix[2,3] = -2f * far * near / (far - near);
projMatrix[3,2] = -1f;
cam.projectionMatrix = projMatrix;
```

---

## Estrutura de Ficheiros Esperada

```
p2/
├── CLAUDE.md
├── calibrate.py            # captura chessboard + calibrateCamera → .npz
├── aruco_detector.py       # deteção ArUco + estimação pose standalone
├── main.py                 # aplicação AR principal (webcam + overlays)
├── unity_server.py         # extensão com envio UDP (opcional)
├── camera_calibration.npz  # output da calibração (não commitar)
├── markers/                # imagens dos marcadores geradas
└── UnityProject/           # projeto Unity (opcional)
    └── Assets/Scripts/
        ├── VideoReceiver.cs
        └── PoseReceiver.cs
```

---

## Funções OpenCV de Referência

| Objetivo | Função |
|---|---|
| Detetar cantos do chessboard | `findChessboardCorners()` |
| Renderizar cantos | `drawChessboardCorners()` |
| Subpixel accuracy | `cornerSubPix()` |
| Calibrar câmara | `calibrateCamera()` |
| Matriz ótima | `getOptimalNewCameraMatrix()` |
| Corrigir distorção | `undistort()` |
| Pose de objeto 3D-2D | `solvePnPRansac()` |
| Projetar pontos 3D | `projectPoints()` |
| Rodrigues rotation | `Rodrigues()` |
| Desenhar eixos | `drawFrameAxes()` |
| Dicionário ArUco | `getPredefinedDictionary()` |
| Parâmetros detetor | `DetectorParameters()` |
| Construtor detetor | `ArucoDetector()` |
| Detetar marcadores | `detectMarkers()` |
| Desenhar marcadores | `drawDetectedMarkers()` |
| Estimar pose marcador | `estimatePoseSingleMarkers()` |

---

## Dependências

```bash
pip install opencv-contrib-python==4.9.0.80 numpy==1.26.4
```

**IMPORTANTE:** usar `opencv-contrib-python` (não `opencv-python`) para ter acesso ao módulo `cv2.aruco`.

---

## Ordem de Implementação

1. **`calibrate.py`** — calibrar câmara, guardar `camera_calibration.npz`
2. **`aruco_detector.py`** — detetar marcadores e visualizar eixos de pose
3. **`main.py`** — adicionar objetos virtuais diferentes por ID de marcador
4. *(opcional)* **`unity_server.py`** + **VideoReceiver.cs** + **PoseReceiver.cs**

---

## Notas Importantes

- Correr calibração com pelo menos 10–15 imagens do chessboard em posições/ângulos variados
- O tamanho físico do marcador impresso (em metros) é necessário para `estimatePoseSingleMarkers`
- No modo Unity: arrancar Unity **primeiro**, depois Python (pacotes UDP iniciais perdem-se caso contrário)
- Shader do video background Unity: `Unlit/Texture` (Built-in pipeline apenas; não funciona em URP)
- ARObjects NÃO devem ser filhos da Main Camera
- Usar `separators=(',',':')` em `json.dumps` para simplificar parsing em C#
- Criar material do video background em código (não no Inspector) para evitar perda de referência em runtime

## Links de Referência

- [1] https://docs.opencv.org/master/dc/dbb/tutorial_py_calibration.html
- [4] https://docs.opencv.org/4.9.0/d9/d6d/tutorial_table_of_content_aruco.html
- [5] https://chev.me/arucogen/ (gerador de marcadores ArUco)
- [6] https://mecaruco2.readthedocs.io/en/latest/notebooks_rst/Aruco/aruco_basics.html
