# Face Pipeline

Pipeline de reconhecimento facial em Python com:

- detecção de rostos com MTCNN ou YuNet
- alinhamento facial em 5 pontos
- geração de embeddings com ArcFace local
- indexação e busca vetorial no Qdrant
- CLIs para imagem, vídeo, webcam, RealSense e carga da base

## Visão geral

O fluxo principal do repositório é este:

```text
imagem / vídeo / webcam
-> MTCNN ou YuNet detecta rostos e keypoints
-> recorte e alinhamento facial em 112x112
-> ArcFace local gera o embedding
-> Qdrant indexa ou consulta o vetor
-> resultado final e artefatos de depuração
```

Pontos importantes:

- O ArcFace usado em runtime vem de [`utils/local_arcface_backbones.py`](utils/local_arcface_backbones.py).
- O peso padrão vem de `models/ms1mv3_arcface_r100_fp16/backbone.pth`; se esse caminho não existir, o código usa `models/backbone.pth`.
- O diretório `insightface-master/` está no repositório como referência, mas não é a dependência ativa do ArcFace na pipeline principal.
- O MTCNN vem da dependência `facenet-pytorch`, carregada em [`utils/arcface_inference.py`](utils/arcface_inference.py).
- O YuNet usa `models/face_detection_yunet_2023mar.onnx` e OpenCV DNN, com backend `auto`, `cpu` ou `cuda`.

## Requisitos

- Python 3.10 ou superior
- `uv`
- Docker, se você quiser subir o Qdrant localmente
- Uma GPU NVIDIA com CUDA é opcional, mas ajuda bastante na inferência. O projeto roda tanto em GPUs dedicadas x86_64 (ex.: RTX) quanto em placas Jetson/Tegra ARM64 (ex.: AGX Orin). Os passos que dependem da GPU abaixo são parametrizados por versão de CUDA e pela *compute capability* da sua placa.

## Instalação

No diretório `face-recognition`:

```bash
python -m venv .venv
```

No Linux:

```bash
source .venv/bin/activate
uv pip install -r requirements.txt
```

Para usar GPU, instale o PyTorch compatível com a sua plataforma. A fonte do wheel depende da arquitetura:

**GPU dedicada em x86_64 (ex.: RTX):** use o índice oficial do PyTorch, escolhendo o sufixo `cuXXX` compatível com o seu driver. Exemplo para CUDA 12.8:

```bash
uv pip install --force-reinstall \
  torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cu128
```

Troque `cu128` por `cu126`, `cu121` etc. conforme o seu driver (`nvidia-smi` mostra a versão de CUDA suportada).

**Jetson / Tegra (aarch64, ex.: AGX Orin):** os wheels de `download.pytorch.org/whl/cuXXX` são x86_64 e **não funcionam** aqui — eles instalam um PyTorch sem CUDA. Use os wheels do JetPack publicados pela NVIDIA para a sua versão de JetPack/CUDA (índice `https://pypi.jetson-ai-lab.io` ou o índice oficial da NVIDIA para o seu JetPack). Instale a tríade `torch`/`torchvision`/`torchaudio` da mesma versão de JetPack.

Em qualquer um dos casos, valide que a GPU foi reconhecida:

```bash
.venv/bin/python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.version.cuda)"
```

Se `torch.cuda.is_available()` for `False` ou `torch.version.cuda` for `None`, você instalou um build de CPU — revise a fonte do wheel para a sua plataforma.

O wheel do PyTorch pode atualizar o NumPy para `2.x`. Reinstale o NumPy compatível antes de compilar o OpenCV:

```bash
uv pip install "numpy<2"
```

Depois compile o OpenCV CUDA com o script da próxima seção. O `requirements.txt` usa `numpy<2` porque o binding Python do OpenCV compilado localmente foi validado com NumPy 1.26.x.

Observação: `facenet-pytorch` declara compatibilidade com `torch<2.3` e `torchvision<0.18`, então `uv pip check` pode apontar conflito quando você instala um PyTorch mais novo (necessário para CUDA recente). Na prática, o import de `facenet_pytorch.MTCNN` foi validado com versões mais novas do Torch; mantenha o PyTorch da sua plataforma e ignore esse aviso de resolução de versão, desde que a validação de `torch.cuda.is_available()` acima passe.

## OpenCV com suporte CUDA

O pacote `opencv-python`/`opencv-python-headless` do `pip` normalmente não vem compilado com CUDA. Para usar YuNet com `cv2.dnn.DNN_BACKEND_CUDA`, compile o OpenCV localmente dentro da `.venv`.

O `requirements.txt` usa `./vendor/opencv-python-cuda-stub` no lugar do wheel PyPI. Esse stub registra a dependência `opencv-python` para pacotes como `ultralytics`, mas não instala outro `cv2` por cima do build CUDA.

O script `scripts/build_opencv_cuda.sh` é parametrizável por variáveis de ambiente, então funciona em qualquer host NVIDIA desde que você informe o toolkit e a arquitetura da sua GPU:

- `CUDA_ROOT` — diretório do CUDA Toolkit (ex.: `/usr/local/cuda-12.6`)
- `CUDA_ARCH_BIN` — a *compute capability* da sua GPU, **sem ela o binário não roda os kernels CUDA na placa**
- `VENV_DIR` / `PYTHON_BIN` — onde instalar o binding Python
- `CUDNN_INCLUDE_DIR` / `CUDNN_LIB_DIR` / `CUDNN_LIBRARY` — onde estão os headers e a biblioteca do cuDNN

Sobre o cuDNN, há duas situações diferentes conforme a plataforma:

- **x86_64:** o cuDNN normalmente vem do pacote `nvidia-cudnn-cu12` (instalado junto com o PyTorch `cuXXX`), dentro da própria `.venv`. Por isso o script **assume por padrão** o caminho do site-packages (`.venv/.../site-packages/nvidia/cudnn/`) e você não precisa informar as variáveis `CUDNN_*`.
- **Jetson / Tegra:** o cuDNN vem do JetPack e fica em caminhos de sistema, **não** no site-packages. O default do script não serve aqui, então você **precisa** sobrescrever `CUDNN_INCLUDE_DIR`, `CUDNN_LIB_DIR` e `CUDNN_LIBRARY` apontando para o cuDNN do JetPack. Confirme os caminhos exatos da sua imagem com `dpkg -L libcudnn8` (ou `libcudnn9`) e `find / -name 'libcudnn.so*' 2>/dev/null`.

O script detecta automaticamente a major version do cuDNN disponível (8.x ou 9.x), então não há suposição fixa de versão.

Descubra a *compute capability* da sua GPU em <https://developer.nvidia.com/cuda-gpus>. Valores comuns:

| GPU | `CUDA_ARCH_BIN` |
| --- | --- |
| RTX 40xx (Ada) | `8.9` |
| RTX 30xx (Ampere) | `8.6` |
| Jetson AGX/Orin (Ampere) | `8.7` |
| Jetson Xavier (Volta) | `7.2` |

Verifique a GPU e o compilador CUDA (ajuste o caminho do toolkit ao seu sistema):

```bash
nvidia-smi
$CUDA_ROOT/bin/nvcc --version
```

Depois rode o script informando o toolkit e a arquitetura da sua placa. Exemplo para uma RTX 40xx com CUDA 12.6:

```bash
CUDA_ROOT=/usr/local/cuda-12.6 CUDA_ARCH_BIN=8.9 scripts/build_opencv_cuda.sh
```

Exemplo para uma Jetson AGX Orin (o cuDNN vem do JetPack, então é preciso apontar `CUDNN_*` para os caminhos de sistema; ajuste conforme o `find` acima):

```bash
CUDA_ROOT=/usr/local/cuda-12.6 \
CUDA_ARCH_BIN=8.7 \
CUDNN_INCLUDE_DIR=/usr/include \
CUDNN_LIB_DIR=/usr/lib/aarch64-linux-gnu \
CUDNN_LIBRARY=/usr/lib/aarch64-linux-gnu/libcudnn.so \
scripts/build_opencv_cuda.sh
```

O script usa `.venv/bin/python3` por padrão. Se você usa outro ambiente, defina `VENV_DIR` ou `PYTHON_BIN`:

```bash
VENV_DIR=/caminho/para/.venv scripts/build_opencv_cuda.sh
PYTHON_BIN=/caminho/para/python scripts/build_opencv_cuda.sh
```

Valide a instalação:

```bash
.venv/bin/python - <<'PY'
import cv2

print("cv2:", cv2.__version__)
print("cuda devices:", cv2.cuda.getCudaEnabledDeviceCount())
print("DNN_BACKEND_CUDA:", cv2.dnn.DNN_BACKEND_CUDA)
print("DNN_TARGET_CUDA:", cv2.dnn.DNN_TARGET_CUDA)
print("FaceDetectorYN:", hasattr(cv2, "FaceDetectorYN"))

for line in cv2.getBuildInformation().splitlines():
    if "GUI:" in line or "NVIDIA CUDA" in line or "cuDNN" in line or "NVIDIA GPU arch" in line:
        print(line)
PY
```

Saída esperada:

```text
GUI: GTK3
cuda devices: 1
NVIDIA CUDA: YES
cuDNN: YES
```

Para usar YuNet na GPU, passe `backend_id` e `target_id` ao criar o detector:

```python
detector = cv2.FaceDetectorYN.create(
    "models/face_detection_yunet_2023mar.onnx",
    "",
    (w, h),
    0.6,
    0.3,
    5000,
    cv2.dnn.DNN_BACKEND_CUDA,
    cv2.dnn.DNN_TARGET_CUDA,
)
```

Nos CLIs deste projeto, use:

```bash
python3 cli/webcam_face_recognition.py --camera realsense --detector yunet --yunet-backend cuda --device cuda
```

No `cli/webcam_face_recognition.py`, a telemetria da GPU fica ativa por padrão no overlay e inclui uso, memória, temperatura, consumo instantâneo em watts, energia acumulada da sessão, clocks, P-state, memória Torch e driver. Use `--hide-gpu-stats` para ocultar ou `--gpu-stats-interval 0.5` para mudar o intervalo de amostragem.

O FPS mostrado no overlay usa média móvel, com janela padrão de 30 frames. Ajuste com `--fps-window 60` se quiser ainda mais estabilidade. No modo webcam, `torch.compile` fica desligado por padrão para evitar travamento na primeira face; se quiser testar compile/autotune, use `--arcface-compile`.

O reconhecimento aceita uma identidade quando pelo menos um candidato passa os critérios de score e margem. O default é `--min-identity-votes 1`; aumente esse valor apenas se você quiser exigir múltiplas imagens da mesma pessoa entre os vizinhos retornados.

Se você atualizar/recriar a `.venv`, repita a build ou restaure uma cópia desse OpenCV compilado. Depois da build, `uv pip install -r requirements.txt` deve manter o stub local e não sobrescrever o `cv2` CUDA.
Se reinstalar as dependências depois da build, repita também a instalação do PyTorch da sua plataforma e o `numpy<2` da seção de instalação para evitar que o resolvedor troque o Torch CUDA ou suba o NumPy para `2.x`.

## Modelo ArcFace

O modelo principal esperado por este projeto é:

| Modelo | Finalidade | Caminho local esperado | Arquivo usado na inferência | Link |
| --- | --- | --- | --- | --- |
| ArcFace R100 MS1MV3 FP16 | Embedder padrão da pipeline | `models/ms1mv3_arcface_r100_fp16/` | `backbone.pth` | https://huggingface.co/marcelohaps/arcface-torch/tree/main |

Observações:

- Para a inferência, o arquivo realmente necessário é `backbone.pth`.
- Se você passar o diretório do modelo em vez do arquivo, o código resolve automaticamente para `backbone.pth`.
- Os demais arquivos dentro dessa pasta são artefatos de treino e não são necessários para o fluxo principal de reconhecimento.

## Subindo o Qdrant com Docker

Para desenvolvimento local, a forma mais simples é rodar o Qdrant em um container Docker.

### 1. Baixar a imagem

```bash
docker pull qdrant/qdrant
```

### 2. Criar um volume persistente

```bash
docker volume create qdrant_storage
```

O volume nomeado funciona igual em Linux, macOS e Windows, e costuma ser mais portável do que montar uma pasta local diretamente.

### 3. Subir a instância

No Linux/macOS (bash), use `\` para continuar a linha:

```bash
docker run -d \
  --name qdrant \
  -p 6333:6333 \
  -p 6334:6334 \
  -v qdrant_storage:/qdrant/storage \
  qdrant/qdrant
```

No Windows com PowerShell, troque a barra invertida por crase (`` ` ``):

```powershell
docker run -d `
  --name qdrant `
  -p 6333:6333 `
  -p 6334:6334 `
  -v qdrant_storage:/qdrant/storage `
  qdrant/qdrant
```

### 4. Verificar se está no ar

```bash
docker ps
docker logs qdrant
```

Depois disso, o Qdrant ficará acessível em:

- API HTTP: `http://localhost:6333`
- Dashboard: `http://localhost:6333/dashboard`
- gRPC: `localhost:6334`

### 5. Parar e subir novamente

Parar:

```bash
docker stop qdrant
```

Subir de novo:

```bash
docker start qdrant
```

### 6. Remover a instância

```bash
docker rm -f qdrant
```

Se quiser apagar também os dados persistidos:

```bash
docker volume rm qdrant_storage
```

## Estrutura mínima esperada

```text
face-recognition/
|- cli/
|- utils/
|- models/
|  `- ms1mv3_arcface_r100_fp16/
|     `- backbone.pth
|- data/
|  |- people/
|  `- sample_data/
`- README.md
```

## Uso rápido

Todos os comandos abaixo partem do diretório `face-recognition`.

Antes de usar as CLIs, deixe o Qdrant rodando e confirme que o modelo está em `models/ms1mv3_arcface_r100_fp16/backbone.pth` ou `models/backbone.pth`.

### 1. Indexar uma única imagem no Qdrant

```bash
python cli/upload_face.py data/people/marcelo/meas.jpeg --name marcelo --entity-id marcelo --collection Rostos
```

Esse comando:

- detecta o rosto principal
- alinha a face
- gera o embedding
- cria a coleção no Qdrant, se necessário
- indexa o vetor com payload
- salva artefatos em `output/uploads/`

### 2. Indexar todas as imagens de uma pessoa

```bash
python cli/upload_face.py --entity-dir data/people/marcelo --collection Rostos
```

### 3. Indexar várias pessoas de uma vez

Estrutura esperada:

```text
data/people/
|- marcelo/
|  |- 1.jpg
|  `- 2.jpg
|- gus/
`- walter/
```

Comando:

```bash
python cli/upload_face.py --people-dir data/people --collection Rostos
```

### 4. Reconhecer uma pessoa em uma imagem

```bash
python cli/recognize_face.py data/sample_data/tobey.jpg --collection Rostos --save-pipeline-dir runs/tobey_pipeline
```

Saída esperada:

- JSON no terminal com o melhor match
- lista de candidatos agrupados por identidade
- imagens de depuração em `runs/tobey_pipeline/`

### 5. Reconhecimento em vídeo

```bash
python cli/video_face_recognition.py data/sample_data/mitch_cam.mp4 --collection Rostos --no-display --output-video runs/mitch_out.mp4
```

Opções úteis:

- `--save-pipeline-dir runs/debug_video` exporta imagens das etapas
- `--save-pipeline-frame 30` escolhe qual frame exportar
- `--detect-interval 3` reduz o custo de detecção entre frames
- `--disable-temporal-cache` desliga o reuso de resultados entre frames

### 6. Reconhecimento em webcam ou câmera IP

Webcam local:

```bash
python cli/webcam_face_recognition.py --camera 0 --collection Rostos
```

Câmera IP:

```bash
python cli/webcam_face_recognition.py --camera http://192.168.1.10:4747/video --collection Rostos
```

### 7. Benchmark da pipeline

Benchmark em imagem:

```bash
python benchmark_inference.py data/sample_data/tobey.jpg
```

Benchmark em vídeo:

```bash
python benchmark_inference.py --video-path data/sample_data/mitch_cam.mp4 --video-max-frames 120
```

## Como a pipeline funciona

### Detecção

- A detecção pode usar `MTCNN` da biblioteca `facenet-pytorch` ou `YuNet` via OpenCV DNN.
- A função `detect_faces_mtcnn()` em [`utils/arcface_inference.py`](utils/arcface_inference.py) retorna `bbox`, `score`, área e os 5 keypoints faciais.
- A função `detect_faces_yunet()` em [`utils/yunet_inference.py`](utils/yunet_inference.py) usa `models/face_detection_yunet_2023mar.onnx` e pode rodar em `--yunet-backend auto`, `cpu` ou `cuda`.
- Em vídeo, a classe `FaceRecognizer` pode redimensionar o frame para acelerar a etapa de detecção.

### Alinhamento

- Quando a detecção já retorna os 5 keypoints, a face é alinhada diretamente.
- Quando o alinhamento falha, o código tenta recuperar a face a partir do recorte disponível.
- Se `--require-alignment` não estiver ativo, existe fallback para redimensionamento simples do crop.

### Embedding

- O embedding é gerado por `ArcFaceEmbedder`.
- A arquitetura local é carregada de [`utils/local_arcface_backbones.py`](utils/local_arcface_backbones.py).
- O peso padrão é `models/ms1mv3_arcface_r100_fp16/backbone.pth`; se esse arquivo não existir, o código usa `models/backbone.pth`.

### Busca e decisão de identidade

- [`utils/qdrant.py`](utils/qdrant.py) encapsula criação de coleção, `upsert` e consultas vetoriais.
- [`utils/identity_matching.py`](utils/identity_matching.py) agrupa os vizinhos por identidade e escolhe o melhor match com base em score, margem e quantidade mínima de votos.

### Vídeo em tempo real

- `FaceRecognizer`, em [`utils/face_pipeline.py`](utils/face_pipeline.py), processa múltiplas faces em batch por frame.
- O cache temporal permite reutilizar resultados quando as caixas permanecem estáveis entre frames.
- O intervalo de detecção pode ser controlado com `--detect-interval`.

## Principais módulos

| Caminho | Função |
| --- | --- |
| `cli/upload_face.py` | Indexa uma imagem, um diretório de uma pessoa ou um diretório inteiro de pessoas no Qdrant. Também salva artefatos da face processada. |
| `cli/recognize_face.py` | Recebe uma imagem, gera embedding, consulta o Qdrant e retorna o melhor match e os candidatos agrupados por identidade. |
| `cli/video_face_recognition.py` | Processa um arquivo de vídeo frame a frame, desenha reconhecimentos, salva vídeo de saída e exporta artefatos da pipeline. |
| `cli/webcam_face_recognition.py` | Faz reconhecimento em tempo real com webcam USB, câmera IP ou RealSense. |
| `utils/face_pipeline.py` | Núcleo da aplicação. Contém `compute_primary_face_embedding()` e a classe `FaceRecognizer`, com batching, cache temporal e integração com Qdrant. |
| `utils/arcface_inference.py` | Carrega o ArcFace local, faz pré-processamento, gera embeddings, detecta rostos com MTCNN e executa alinhamento facial. |
| `utils/yunet_inference.py` | Detecta rostos com YuNet via OpenCV DNN, incluindo backend CUDA quando disponível. |
| `utils/local_arcface_backbones.py` | Implementa o backbone local do ArcFace (`r18`, `r34`, `r50`, `r100`, `r200`) sem depender de código externo no fluxo principal. |
| `utils/qdrant.py` | Wrapper para criar coleções, indexar pontos e consultar vetores no Qdrant. |
| `utils/identity_matching.py` | Resolve a identidade final a partir dos resultados brutos do Qdrant. |
| `utils/drawing.py` | Desenha `bbox`, nome e score nas imagens e frames processados. |
| `utils/pipeline_artifacts.py` | Gera painéis PNG de cada etapa da pipeline para depuração e apresentação. |
| `benchmark_inference.py` | Benchmark principal da pipeline atual em imagem ou vídeo. |

## Dados incluídos

| Caminho | Conteúdo |
| --- | --- |
| `data/sample_data/` | Imagens e vídeos de exemplo para testes rápidos e demonstrações |
| `data/people/` | Galeria de pessoas para indexação no Qdrant |
| `data/cfp-dataset/` | Estrutura de dataset usada em avaliações auxiliares |
| `notebooks/` | Testes e explorações manuais |

## Arquivos para começar

Se você quiser entender o projeto rapidamente, a melhor ordem é:

1. [`utils/face_pipeline.py`](utils/face_pipeline.py)
2. [`utils/arcface_inference.py`](utils/arcface_inference.py)
3. [`utils/local_arcface_backbones.py`](utils/local_arcface_backbones.py)
4. [`cli/upload_face.py`](cli/upload_face.py)
5. [`cli/recognize_face.py`](cli/recognize_face.py)
6. [`cli/video_face_recognition.py`](cli/video_face_recognition.py)
