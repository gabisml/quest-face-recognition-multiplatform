# Pipeline de benchmark — reconhecimento facial em edge

Suite de benchmark inspirada em MLPerf Inference (single-stream) e no
protocolo FRUITS/WebFace260M. Mede a pipeline de reconhecimento facial
em 5 etapas (`read`, `detection`, `alignment`, `embedding`, `search`)
mais o total, em múltiplos hardwares, gerando os números que preenchem
as 5 tabelas LaTeX da dissertação.

## Estrutura

```
bench/
├── __init__.py
├── config.py            # registro de hardwares, constantes (K, N, warmup, ...)
├── hardware.py          # system_info.json e detecções (driver, governor, etc.)
├── probes.py            # CPU/RAM (psutil), NVML, tegrastats (Jetson), vcgencmd (RPi)
├── sync.py              # cuda_sync — falha alto se CUDA-sync falhar
├── artifacts.py         # pré-computa bboxes / faces alinhadas / embeddings
├── microbench.py        # Benchmark 1 (por etapa, K×N)
├── e2e_video.py         # Benchmark 2 (vídeo, FPS sustentado, P95)
├── system_monitor.py    # sampling de CPU/GPU/temp/potência a 500 ms
├── aggregate.py         # parquets → CSVs agregados (mediana/P95/máx por etapa)
├── generate_tables.py   # CSVs → 5 fragmentos .tex
├── README.md
├── requirements.txt
└── benchmark_subset.json  # manifesto K-imagens (vem do tar, não versionado)
```

Saídas vão em `results/<hardware_id>/<timestamp>/`:
- `system_info.json` — versões, drivers, modo de potência, governor, steady-state UTC.
- `stage_raw.parquet` — uma linha por medição (K×N×6).
- `stage_aggregated.csv` — mediana / P95 / máx por etapa.
- `system_metrics.parquet` — amostragem temporal CPU/GPU/...
- `system_aggregated.csv` — médias/picos por hardware.
- `e2e_video.csv` — FPS, P95 e fração-no-orçamento por vídeo.
- `torch_memory.json` — `max_memory_allocated` / `max_memory_reserved` (GB).

Cada hardware grava sua própria pasta `results/<hardware_id>/<timestamp>/`
auto-contida. A etapa final (após coletar todos os hardwares) é
`bench.aggregate` + `bench.generate_tables`, que produz os 5 fragmentos
LaTeX em `results/_aggregated/` (ver "Consolidação e geração das tabelas
LaTeX" no fim do README).

## Layout esperado no destino

Antes de medir, o repositório no hardware-alvo precisa ter este conteúdo
(comandos para obter cada item estão em "Em cada hardware → 0"). Todos
os paths são relativos à raiz do repo (`/home/<user>/face-recognition/`),
exceto o cache do HuggingFace, que é global.

```
face-recognition/
├── bench/
│   ├── benchmark_subset.json        # manifest K=500 (do tar)
│   ├── artifacts.npz                # bboxes + aligned + embeddings (do tar)
│   └── artifacts/                   # 500 .npy BGR crus (do tar)
│       └── <image_id>.npy
├── models/
│   ├── yunet/
│   │   └── face_detection_yunet_2023mar.onnx   # detector (já no git)
│   └── ms1mv3_arcface_r100_fp16/
│       └── backbone.pth                         # ArcFace R100 (download manual)
├── data/
│   ├── people/Walter/                # identidade enrolada no Qdrant (Benchmark 2)
│   │   ├── walter1.jpg
│   │   └── walter2.jpeg
│   └── videos/clip.mp4               # só para o Benchmark 2 (e2e_video)
└── .venv/                            # ambiente Python (uv pip install …)

~/.cache/huggingface/hub/
└── datasets--marcelohaps--lfw/       # LFW raw — `bench.dataset_lfw download`
    ├── blobs/<sha>                   # JPEGs lidos pelo stage `read`
    └── snapshots/<rev>/…
```

| Item | Tamanho | Origem |
|---|---|---|
| `bench/benchmark_subset.json` + `artifacts.npz` + `artifacts/` | ~150 MB | `bench-artifacts.tar.gz` em [marcelohaps/face-bench-artifacts](https://huggingface.co/datasets/marcelohaps/face-bench-artifacts) |
| `models/yunet/face_detection_yunet_2023mar.onnx` | ~340 KB | versionado no git (`git clone` já traz) |
| `models/ms1mv3_arcface_r100_fp16/backbone.pth` | ~250 MB | [marcelohaps/arcface-torch](https://huggingface.co/marcelohaps/arcface-torch) (não versionado no git por tamanho) |
| `data/people/Walter/walter{1,2}.{jpg,jpeg}` | ~400 KB | identidade enrolada no Qdrant (Benchmark 2) — em [marcelohaps/face-bench-artifacts](https://huggingface.co/datasets/marcelohaps/face-bench-artifacts) |
| `data/videos/clip.mp4` | ~7 MB | vídeo de teste para Benchmark 2 — em [marcelohaps/face-bench-artifacts](https://huggingface.co/datasets/marcelohaps/face-bench-artifacts) |
| LFW raw no cache do HF | ~180 MB | `python -m bench.dataset_lfw download` |
| OpenCV-CUDA compilado | — | ver instruções em [/README.md](../README.md) ("OpenCV com suporte CUDA") |
| Qdrant ouvindo em 6333 | — | container `qdrant/qdrant` (local ou centralizado) |

> Para o Benchmark 1 sozinho, `data/` é opcional (LFW basta). O Benchmark 2
> exige `data/videos/clip.mp4` **e** o Walter enrolado no Qdrant — Walter é
> a identidade-alvo presente em `clip.mp4`.

## Em cada hardware

`<hardware_id>` é um destes:
`rtx3090 | rtx4060 | dgx_spark | jetson_agx_orin | jetson_orin_nano | rpi5 | rpi4 | coral_dev`.

### 0 — Pré-requisitos (dados, modelos, Qdrant)

Rode tudo desta seção a partir da **raiz do repositório**
(`/.../face-recognition/`). Os subpassos baixam os arquivos que ficam
fora do git e populam o Qdrant. Faça 1× por hardware (exceto enrollment,
que é 1× por servidor de Qdrant).

#### 0.1 — Autenticar no HuggingFace

```bash
hf auth login    # token gratuito; basta 1× por hardware
```

#### 0.2 — Artefatos do Benchmark 1 (microbench)

Os 3 artefatos consumidos pelo `microbench` (`benchmark_subset.json`,
`artifacts.npz`, `artifacts/<image_id>.npy` ×500) estão empacotados em
`bench-artifacts.tar.gz` (~80 MB) em
[marcelohaps/face-bench-artifacts](https://huggingface.co/datasets/marcelohaps/face-bench-artifacts):

```bash
hf download marcelohaps/face-bench-artifacts \
    bench-artifacts.tar.gz \
    --repo-type dataset \
    --local-dir bench/

tar xzf bench/bench-artifacts.tar.gz
```

#### 0.3 — Dados do Benchmark 2 (vídeo + identidade)

Necessário só pra `bench.e2e_video`. Mesmo dataset:

```bash
hf download marcelohaps/face-bench-artifacts \
    data/videos/clip.mp4 \
    data/people/Walter/walter1.jpg \
    data/people/Walter/walter2.jpeg \
    --repo-type dataset --local-dir .
```

#### 0.4 — LFW raw (stage `read` do microbench)

O `bench-artifacts.tar.gz` traz só os dados pós-decodificação. O stage
`read` faz `cv2.imread` nas JPEGs originais do LFW — os paths no
`benchmark_subset.json` são guardados **relativos ao cache do HF**
(`datasets--marcelohaps--lfw/blobs/<sha>`); como o cache é
content-addressed, esse trecho é idêntico em toda máquina e
`bench.artifacts.resolve_image_path` o rebaseia no cache local na leitura:

```bash
python -m bench.dataset_lfw download
```

#### 0.5 — Backbone do ArcFace R100

Necessário pro stage `embedding` (microbench) e pra enrollment. Não
versionado no git por tamanho (~250 MB):

```bash
hf download marcelohaps/arcface-torch \
    backbone.pth \
    --local-dir models/ms1mv3_arcface_r100_fp16/
```

#### 0.6 — Validar o destino

```bash
python -c "
import numpy as np, json, os
import cv2
from bench.artifacts import resolve_image_path
m = json.load(open('bench/benchmark_subset.json'))
d = np.load('bench/artifacts.npz', allow_pickle=False)
n_npy = len([f for f in os.listdir('bench/artifacts') if f.endswith('.npy')])
assert len(m['images']) == 500
assert d['aligned_bgr'].shape == (500, 112, 112, 3)
assert d['embeddings'].shape == (500, 512)
assert n_npy == 500
missing = [e['path'] for e in m['images'] if not resolve_image_path(e['path']).exists()]
assert not missing, f'LFW raw ausente em {len(missing)} paths (ex: {missing[0]}) — rode bench.dataset_lfw download'
first = resolve_image_path(m['images'][0]['path'])
sample = cv2.imread(str(first), cv2.IMREAD_COLOR)
assert sample is not None, f'cv2.imread falhou em {first}'
assert os.path.exists('models/ms1mv3_arcface_r100_fp16/backbone.pth'), 'backbone.pth ausente — rode 0.5'
assert os.path.exists('data/videos/clip.mp4'), 'clip.mp4 ausente — rode 0.3 (apenas Benchmark 2)' if False else True
print('OK — artefatos íntegros, LFW raw e backbone acessíveis no destino')
"
```

#### 0.7 — Subir o Qdrant e enrolar as identidades

O stage `search` (microbench) e o `e2e_video` consultam a coleção
`Rostos` no Qdrant. Como o servidor pode ser centralizado, **enrolar
1× por servidor**, não por hardware.

Instalar/subir o Qdrant localmente: ver [../README.md](../README.md).
Pré-requisitos pro enrollment: deps instaladas (passo 1), Qdrant
rodando e o `backbone.pth` no lugar (passo 0.5).

```bash
# Benchmark 1: 100 identidades do LFW (stage `search` do microbench).
python -m bench.dataset_lfw enroll \
    --n 100 \
    --collection Rostos \
    --qdrant-url http://localhost:6333 \
    --device cpu

# Benchmark 2: o Walter (única identidade-alvo presente em clip.mp4).
python scripts/enroll_and_recognize_people.py --only Walter
```

Após os dois, a coleção `Rostos` tem 102 pontos (100 LFW + 2 do Walter).


### 1 — Instalar dependências

```bash
pip install -r requirements.txt
pip install -r bench/requirements.txt

# E uma das opções abaixo, conforme o HW:
pip install pynvml                 # NVIDIA desktops
sudo -H pip install jetson-stats   # Jetsons
# RPi: vcgencmd já vem no Raspberry Pi OS
# Coral: nada extra (psutil já está em bench/requirements.txt)
```

### 2 — Congelar estado térmico/clock (antes de medir)

```bash
sudo cpupower frequency-set -g performance     # desktops x86 (NUC 4060, DGX Spark)
sudo nvpmodel -m 0 && sudo jetson_clocks       # Jetsons (MAX-N + lock)
sudo cpufreq-set -g performance                # RPi 4/5 e Coral Dev Board

# Confirmar que pegou:
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor    # → performance
```

### 3 — Benchmark 1 (microbenchmark, ~30 min)

```bash
python -m bench.microbench \
    --hardware-id <hardware_id> \
    --artifacts bench/artifacts.npz \
    --qdrant-url http://localhost:6333 \
    --device cuda                              # ou cpu em RPi/Coral
```

Em hardwares sem GPU dedicada acrescente `--embedder-precision fp32`.
Em hardwares offline, acrescente `--skip-search`.

> O estágio `search` exige a coleção `Rostos` já enrolada (passo 0.7).
> Sem isso, rode com `--skip-search`.

### 4 — Benchmark 2 (end-to-end em vídeo, ~6 min)

Roda o pipeline completo (`FaceRecognizer`) frame-a-frame no `clip.mp4`,
medindo FPS e latência por frame após 60 s de warm-up + 300 s de janela.
Como o vídeo é mais curto que a janela (191 s), o script faz loop
automaticamente até preencher os 300 s.

Pré-requisitos:
- `data/videos/clip.mp4` baixado (passo 0.3)
- Walter enrolado em `Rostos` (passo 0.7)
- YuNet em `models/yunet/face_detection_yunet_2023mar.onnx`

```bash
python -m bench.e2e_video \
    --hardware-id <hardware_id> \
    --videos data/videos/clip.mp4 \
    --qdrant-url http://localhost:6333 \
    --device cuda --warmup-s 60 --duration-s 300
```

Em hardwares sem GPU dedicada use `--device cpu` e
`--embedder-precision fp32`. Métricas registradas por vídeo
(`results/<hw_id>/<timestamp>/e2e_video.csv`):

| Coluna | Significado |
|---|---|
| `fps_mean` | FPS sustentado na janela steady-state |
| `latency_{median,p95,max}_ms` | latência por frame |
| `frac_in_{33,66,100}ms` | fração de frames dentro dos orçamentos FRUITS |
| `n_frames` | nº de frames medidos (warm-up excluído) |

### 5 — Conferir saída

Cada execução cria um diretório novo com timestamp UTC. Microbench e
e2e_video geram diretórios separados (são chamadas distintas).

```bash
ls results/<hardware_id>/<timestamp>/

# Após `bench.microbench`:
#   stage_raw.parquet    stage_aggregated.csv
#   system_info.json     system_metrics.parquet
#   torch_memory.json

# Após `bench.e2e_video`:
#   e2e_video.csv        e2e_per_frame.parquet
#   system_info.json     system_metrics.parquet
#   torch_memory.json
```

## Consolidação e geração das tabelas LaTeX (1× no final)

Após coletar `results/<hardware_id>/.../` em todos os hardwares-alvo, rode
**uma vez** na máquina que vai compilar a dissertação. `bench.aggregate`
percorre todos os diretórios em `results/`, agrega métricas por
hardware (e por rodada, se houver múltiplas) e grava CSVs unificados em
`results/_aggregated/`. `bench.generate_tables` converte esses CSVs em
fragmentos `.tex` prontos pra colar.

```bash
python -m bench.aggregate          # → results/_aggregated/{stage,system,e2e_video}_aggregated.csv
python -m bench.generate_tables    # → results/_aggregated/table{1..5}_*.tex
```

Saída:
- `table1_cpu_ram.tex` — Hardware × CPU × CPU% × RAM
- `table2_gpu_vram_temp_power.tex` — Hardware × GPU% × VRAM × Temp × Watts
- `table3_latency_median.tex` — Hardware × 6 etapas (mediana)
- `table4_latency_p95.tex` — Hardware × 6 etapas (P95)
- `table5_latency_max.tex` — Hardware × 6 etapas (máx)

Cole o conteúdo de cada `.tex` dentro do seu `\begin{table} ... \end{table}`.
Células com sensor indisponível ficam `--`.


