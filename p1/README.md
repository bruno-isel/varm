# Face Recognizer - Notas de Desenvolvimento

## Algoritmos Utilizados

- **Haar Cascade Classifier** — deteção de faces e olhos
- **LBPH (Local Binary Pattern Histogram)** — reconhecimento facial (dataset original)
- **FisherFaces (PCA + LDA)** — reconhecimento facial (dataset normalizado)

Pipeline de normalização: `foto original` → Face Detection → Eye Detection → Rotação → Escala + Recorte → `imagem 56×46 MPEG-7`

Pipeline de reconhecimento: `frame` → CascadeClassifier (deteta face) → FisherFaces/LBPH (identifica pessoa)

## Resultados

| Configuração | Accuracy |
|-------------|----------|
| LBPH + dataset original (sem normalização) | 52% |
| LBPH + dataset original (threshold ajustado) | 65% |
| LBPH + dataset normalizado MPEG-7 | 0% (threshold desajustado) |
| FisherFaces + dataset normalizado MPEG-7 (sem padding/equalização) | 50% |
| FisherFaces + dataset normalizado MPEG-7 (com padding + equalização) | **80%** |

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

### 5. Normalização com dataset próprio — cadeia de problemas

Ao aplicar o normalizer MPEG-7 ao nosso dataset (Bruno, Meda, Pedro), encontrámos uma sequência de problemas:

**5a. Nenhuma imagem normalizada (1/146 sucesso)**

O normalizer detetava a face, recortava-a (ex: 100×100), e depois tentava escalar para que os olhos ficassem a 15px de distância. Isto reduzia a face para ~34×34 — demasiado pequeno para recortar 46×56.

**Solução:** Refazer o `normalize()` para trabalhar na **imagem original inteira** em vez da face recortada. Assim há pixels suficientes para escalar e recortar. Os olhos são detetados na face, mas as coordenadas são convertidas para absolutas na imagem original.

**5b. Imagens carregadas mas `predict` devolvia `None` (0% accuracy)**

O `recognizer.py` tentava detetar faces (com `CascadeClassifier`) nas imagens normalizadas de 56×46. Estas imagens já são faces recortadas — demasiado pequenas para o cascade funcionar.

**Solução:** Adicionar flag `normalized=True` ao recognizer. Quando ativo, o `load_dataset()` e o `predict()` usam as imagens diretamente sem deteção de face.

**5c. Tudo classificado como "Desconhecido" (0% accuracy)**

Com o dataset normalizado, as confiânças do LBPH eram muito altas (~150) porque:
- Poucas imagens de treino (19 faces no total)
- Imagens pequenas (56×46) geram histogramas menos distintos
- O threshold estava a 100, bloqueando todas as previsões

**Solução:** Subir o `UNKNOWN_THRESHOLD` para 200. Os valores de confiança dependem do dataset — com imagens normalizadas e poucas amostras, os valores são naturalmente mais altos.

### 6. Crop negativo na normalização (imagens perdidas)

**Problema:** Ao normalizar as fotos do Pedro Jorge, o recorte saía fora dos limites da imagem (coordenadas negativas). Isto acontecia porque o olho esquerdo ficava muito perto da borda — após escalar, a posição alvo (coluna 16) ficava fora da imagem.

**Solução:** Adicionar **padding** (bordas replicadas) à imagem escalada antes de recortar com `cv2.copyMakeBorder()`. Assim garante-se espaço suficiente para o recorte 46×56 mesmo quando os olhos estão perto das bordas.

### 7. Poucas imagens normalizadas (1/146 → 27/33)

**Problema:** A deteção de olhos falhava na maioria das imagens, especialmente com iluminação fraca ou pouco contraste.

**Solução:** Adicionar **equalização de histograma** (`cv2.equalizeHist()`) na região dos olhos antes da deteção. Esta técnica redistribui o brilho da imagem para usar toda a gama de 0–255, melhorando o contraste e tornando os padrões de claro/escuro (que o Haar Cascade usa) mais visíveis. A taxa de sucesso passou de 1 para 27 imagens normalizadas.

### 8. FisherFaces vs LBPH

**Problema:** O LBPH com dataset normalizado tinha confiânças muito altas (~150) e a accuracy era baixa.

**Solução:** Trocar para **FisherFaces** (PCA + LDA). O FisherFaces maximiza a separação entre classes e funciona melhor com imagens do mesmo tamanho (requisito do MPEG-7). A accuracy subiu de 50% para **80%** após combinar FisherFaces com padding e equalização de histograma.

O FisherFaces usa uma escala de confiança diferente do LBPH (valores na casa dos milhares em vez de 0–200), por isso o threshold para "Desconhecido" foi ajustado para 3000.

### 9. `cv2.face` não disponível (opencv-contrib)

**Problema:** `AttributeError: module 'cv2' has no attribute 'face'`

**Solução:** Instalar `opencv-contrib-python` em vez de `opencv-python`:
```bash
pip install opencv-contrib-python
```

### 10. Webcam no Mac

**Problema:** `OpenCV: not authorized to capture video (status 0)`

**Solução:**
- Ir a **Definições do Sistema → Privacidade e Segurança → Câmara**
- Ativar a permissão para o terminal/VS Code
- Reiniciar a aplicação após dar permissão
- Usar `cv2.VideoCapture(1, cv2.CAP_AVFOUNDATION)` para selecionar a câmara correta (índice 0 pode abrir a câmara do iPhone via Continuity Camera)
