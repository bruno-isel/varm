# Face Recognizer - Notas de Desenvolvimento

## Algoritmo Utilizado

**LBPH (Local Binary Pattern Histogram)** para reconhecimento facial, com **Haar Cascade Classifier** para deteção de faces.

Pipeline: `frame` → CascadeClassifier (deteta face) → LBPH (identifica pessoa)

## Problemas Encontrados e Soluções

### 1. Threshold para "Desconhecido"

**Problema:** O LBPH devolve sempre o nome mais próximo, mesmo para pessoas que não estão no dataset. Não havia forma de dizer "Desconhecido".

**Solução:** Adicionámos um `UNKNOWN_THRESHOLD`. A confiança no LBPH é uma distância — quanto menor, melhor o match:
- `< 50` → match muito confiante
- `50–80` → razoável
- `> 100` → provável desconhecido

Começámos com threshold de `80`, mas muitas faces conhecidas estavam a ser marcadas como "Desconhecido" (confiânças de 82, 83, 91). Subimos para `100` e a accuracy melhorou de **52% → 65%**.

### 2. Faces no fundo classificadas com o label errado

**Problema:** No `load_dataset()`, o código guardava **todas** as faces detetadas numa imagem com o label dessa pessoa. Se uma foto do Bruno tivesse a Meda no fundo, essa face era guardada como "Bruno", poluindo o treino.

**Antes:**
```python
for x, y, w, h in detected_faces:
    face_img = img[y:y + h, x:x + w]
    self.face_list.append(face_img)
    self.class_list.append(idx)
```

**Depois:**
```python
x, y, w, h = max(detected_faces, key=lambda f: f[2] * f[3])
face_img = img[y:y + h, x:x + w]
self.face_list.append(face_img)
self.class_list.append(idx)
```

**Solução:** Guardar apenas a **face maior** de cada imagem, que é quase sempre a pessoa principal da foto.

### 3. Dataset desequilibrado

**Problema:** A Angelina Jolie tinha 87 fotos de treino, enquanto a Meda tinha apenas 9 e o Pedro 5. Isto causava confusões frequentes — a Meda era constantemente classificada como Bruno.

**Solução:** Adicionar mais fotos por pessoa, especialmente das pessoas com poucas amostras. Após adicionar fotos da Meda, a classificação dela passou de 0/3 para 3/3 correto.

### 4. Normalização MPEG-7

**Problema:** As imagens do dataset tinham tamanhos e orientações diferentes, o que reduz a precisão do reconhecimento.

**Solução:** Implementámos normalização segundo a recomendação MPEG-7:
- Imagem monocromática (256 níveis de cinza)
- Dimensão: 56 linhas × 46 colunas
- Olhos alinhados horizontalmente na linha 24, colunas 16 e 31

Ao aplicar esta normalização ao dataset celebrity, a accuracy subiu de **62.33% → 67.85%** e a precisão macro de 0.61 para 0.74.

### 5. `cv2.face` não disponível

**Problema:** `AttributeError: module 'cv2' has no attribute 'face'`

**Solução:** Instalar `opencv-contrib-python` em vez de `opencv-python`:
```bash
pip install opencv-contrib-python
```

### 6. Webcam no Mac

**Problema:** `OpenCV: not authorized to capture video (status 0)`

**Solução:**
- Ir a **Definições do Sistema → Privacidade e Segurança → Câmara**
- Ativar a permissão para o terminal/VS Code
- Reiniciar a aplicação após dar permissão
- Usar `cv2.VideoCapture(1, cv2.CAP_AVFOUNDATION)` para selecionar a câmara correta (índice 0 pode abrir a câmara do iPhone via Continuity Camera)
