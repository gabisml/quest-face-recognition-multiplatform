# Coral Dev Board (Edge TPU)

Scripts que rodam **na placa**, mais os resultados que ficaram medidos. Recuperados de
`/home/mendel/data/` na Coral Dev Board, onde foram escritos soltos, fora do repo.

O pipeline PyTorch do repo não roda aqui: a placa não tem torch para aarch64/glibc 2.28.
Tudo passa por TFLite int8 compilado para a Edge TPU, com os `.tflite` e os tensores de
referência gerados no WSL e copiados para a placa.

## Arquivos

| Arquivo | O que faz | Origem na placa |
|---|---|---|
| `EDGETPU_OPS.md` | Mapeamento das operações aceitas e recusadas pela Edge TPU, dos 16 logs do compilador | escrito aqui |
| `map_tflite_graph.py` | Mapeia o grafo de um `.tflite` e aponta o que quebraria na compilação, sem depender do compilador | escrito aqui |
| `setup_venv39.sh` | Recria o venv Python 3.9 com acesso à Edge TPU | `coral_deploy/tools/setup_venv39.sh` |
| `edgetpu_shim.py` | Substitui `pycoral.utils.edgetpu` (o wheel oficial não carrega nesta glibc) | `coral_deploy/tools/edgetpu_shim.py` |
| `bench_yunet_cpu.py` | YuNet no CPU via `cv2.FaceDetectorYN`: `detect()` puro por resolução, escalonamento por thread, custo do resize | `_bench_yunet/bench_yunet.py` |
| `yunet_tflite_tpu.py` | YuNet 120×160 em TFLite no CPU ou na TPU, com decode dos 1076 priors + NMS em Python | `yunet-tpu/yunet_tflite_tpu.py` |
| `arcface_board_bench.py` | ArcFace r100 int8: latência TPU vs CPU + cross-check do embedding contra o mesmo int8 rodado no WSL | `arcface-tpu/06_board_bench.py` |
| `mobilefacenet_board_bench.py` | MobileFaceNet w600k int8 na TPU: latência + acurácia das 490 faces | `compact-fr/mbf_board.py` |
| `bench_tflite_latency.py` | Latência de qualquer `.tflite`, TPU ou CPU (entrada aleatória; só latência é válida) | `compact-fr/bench_tpu.py` |
| `bench_embedding_search.py` | Custo da busca por força bruta em numpy, por tamanho de galeria | escrito aqui |
| `resize_video.py` | Reescala o vídeo antes do benchmark, para o resize não entrar na medição | escrito aqui |
| `e2e_video_tpu.py` | e2e em vídeo: decode → detect → align → embed → busca, com timing por etapa | escrito aqui |
| `arcface_tpu_eval.log` | Saída da avaliação do ArcFace int8 na TPU | `arcface-tpu/tpu_eval.log` |
| `arcface_cpu_bench.log` | Saída do ArcFace int8 no CPU da placa | `arcface-tpu/cpu_bench.log` |
| `mobilefacenet_board.log` | Saída do MobileFaceNet w600k int8, TPU e CPU | medido na placa |
| `compact_fr_tpu_latency.log` | Latência dos 12 modelos compactos, TPU e CPU | medido na placa |
| `yunet_cpu.log` | Saída do `bench_yunet_cpu.py` | medido na placa |
| `embedding_search.log` | numpy vs faiss na busca de embedding | medido na placa |
| `e2e_video.log` | e2e em vídeo, as três configurações de TPU/CPU | medido na placa |
| `e2e_sustained.log` | e2e na janela do protocolo: warm-up 60 s + 300 s medidos | medido na placa |
| `edgeface_graph_map.log` | Grafo dos 3 EdgeFace int8, op por op, com o veredito de cada | gerado no WSL |
| `yunet_graph_map.log` | Logs de compilação das 3 variantes do YuNet, grafo e análise de memória | gerado no WSL |
| `face.jpg` | Imagem de teste (512×355, 4 faces) usada pelos benches de YuNet | `yunet-tpu/face.jpg` |

`bench_yunet_cpu.py` teve o `sys.path` e o caminho da imagem ajustados para funcionar de
dentro do repo; `setup_venv39.sh` passou a receber o diretório do venv como argumento.
O resto está como estava na placa.

## Resultados medidos

**ArcFace r100 int8, 490 faces (245 pares) do subset de verificação:**

| | latência | throughput |
|---|---|---|
| Edge TPU | 153 ms/face | 6,5 faces/s |
| CPU, 4 threads | 1090 ms | 0,92 faces/s |
| CPU, 1 thread | 3403 ms | 0,29 faces/s |

A TPU é ~7× o CPU com 4 threads. Os 153 ms/face vêm do laço de avaliação completo
(75,1 s para 490 faces), incluindo `set_tensor`/`get_tensor` — não é o `invoke()` isolado.

**Acurácia — a quantização int8 não custou nada nesse subset:**

| | acurácia | cos(pos) | cos(neg) |
|---|---|---|---|
| fp32 (referência) | 97,55 % | 0,6257 | 0,0092 |
| int8 na Edge TPU | 97,55 % | 0,6103 | 0,0109 |

Drift do embedding vs fp32: média 0,9786, p5 0,9712, mínimo 0,9582.

**MobileFaceNet w600k int8, mesmas 490 faces** (`mobilefacenet_board.log`):

| | latência | throughput | acurácia |
|---|---|---|---|
| Edge TPU | 5,11 ms (P95 5,75) | 196 faces/s | 96,53 % |
| CPU, 4 threads | 191,5 ms | 5,2 faces/s | — |
| CPU, 1 thread | 282,8 ms | 3,5 faces/s | — |

Aqui a TPU é **37×** o CPU com 4 threads, e a int8 também não custou acurácia (fp32
96,53 % → TPU int8 96,53 %; drift vs fp32 média 0,9686, p5 0,9523). As 490 faces saem em
2,5 s, contra 75,1 s do ArcFace r100.

O trade-off entre os dois: o ArcFace r100 dá 1,02 ponto de acurácia a mais (97,55 % vs
96,53 %) por 30× a latência (153 ms vs 5,11 ms). Para vídeo na placa só o MobileFaceNet
fecha a conta.

### Os 12 modelos compactos da tabela do EdgeFace

Rodada dos `_edgetpu.tflite` produzidos em `PDC/compact-fr-tpu/` (ver
`compact_fr_tpu_latency.log`). "ops" é o que o `edgetpu_compiler` 16.0 conseguiu mapear.

| Modelo | params (M)¹ | ops na TPU | TPU | CPU 4t | veredito |
|---|---|---|---|---|---|
| **MobileFaceNet** | 1,0 | **70/70** | **1,84 ms** | 31,4 ms | 17× — o mais rápido |
| **MobileFaceNet w600k** | 3,4² | **170/170** | **5,02 ms** | 192,0 ms | 38× |
| ShuffleFaceNet 0.5× | 0,53 | 17/388 | 35,5 ms | — | roda em CPU |
| MixFaceNet-XS | 0,53 | 9/448 | 80,4 ms | — | roda em CPU |
| ShuffleMixFaceNet-XS | 0,53 | 9/514 | 80,6 ms | — | roda em CPU |
| MixFaceNet-S | 1,04 | 9/443 | 127,6 ms | **81,0 ms** | delegate 1,6× pior |
| ShuffleMixFaceNet-S | 1,04 | 9/513 | 128,9 ms | **89,3 ms** | delegate 1,4× pior |
| ShuffleFaceNet 1.5× | 2,66 | 16/772 | 165,2 ms | — | roda em CPU |
| ShuffleFaceNet 2× | 5,71 | 16/976 | 285,7 ms | — | roda em CPU |
| VarGFaceNet | 6,46 | 39/1229 | 336,8 ms | — | roda em CPU |
| PocketNetM-128 | 1,69 | 29/5774 | 1367 ms | — | inviável |
| PocketNetM-256 | 1,75 | 29/5774 | 1370 ms | — | inviável |

¹ medido no ONNX exportado (`compact-fr-tpu/artifacts/stage_a*.csv`), não o número do paper.
Na família MixFaceNet os dois divergem por um fator ~3 — o `params_M` medido de S (1,04)
coincide com o do XS do paper (1,04), o que sugere que os configs do repo `mixfacenets`
estão deslocados em relação à nomenclatura do artigo. Conferir antes de citar.
² MobileFaceNet w600k é o `buffalo_s` do insightface, não está na tabela do EdgeFace.

**Só dois modelos rodam na Edge TPU.** Os outros dez carregam e executam, mas com 9 a 39
ops mapeadas de centenas ou milhares — o grafo fica no CPU e o delegate só acrescenta o
custo de transferência do subgrafo. Em MixFaceNet-S e ShuffleMixFaceNet-S isso é
mensurável: **com o delegate é mais lento que sem** (127,6 vs 81,0 ms; 128,9 vs 89,3 ms).
Para esses modelos, usar a TPU é pior que ignorá-la.

As três causas de rejeição, pelos logs do compilador:

1. **`More than one subgraph is not supported`** — MixFaceNet e ShuffleMixFaceNet. O
   `MUL`+`LOGISTIC`+`STRIDED_SLICE` do bloco de atenção parte o grafo em vários subgrafos,
   e o compilador aceita só um.
2. **`CONV_2D` versão 6** — PocketNet, ShuffleFaceNet e VarGFaceNet quantizam para uma
   versão de op que o `tflite-runtime` 2.7.0 desta imagem não registra (`Invalid model`).
   Os `_dgc_` são a tentativa de contornar; compilam, mas o desdobramento explode o
   modelo (PocketNetM-128 vai de 2,2 MB para 33,4 MB) e continuam quase todos no CPU.
3. **Conv agrupada** — VarGFaceNet tem 49 convoluções agrupadas, que viram `CONV_2D` v6.

**Atenção: os 12 modelos do estágio A têm pesos aleatórios.** `stage_a_screen.py` instancia
as arquiteturas sem carregar checkpoint nenhum (`reg["MobileFaceNet"] = lambda:
mfnet.MobileFaceNet()`), e não há nenhum `state_dict`/`torch.load` nos três scripts de
estágio A. A latência acima é válida; **acurácia desses modelos não é mensurável** sem
antes conseguir pesos treinados. O único com pesos reais é o `mbf_w600k` (insightface
`buffalo_s/w600k_mbf.onnx`), que é justamente o único com número de acurácia.

### YuNet no CPU — o detector é o gargalo

`yunet_cpu.log`, imagem 1280×720:

| entrada | `detect()` | resize | ponta a ponta | faces |
|---|---|---|---|---|
| long_side 640 | 69,0 ms | 11,7 ms | **82,9 ms** (12,1 FPS) | 4 |
| long_side 320 | 16,0 ms | 22,9 ms | **41,6 ms** (24,0 FPS) | 4 |

Escalonamento por thread em 640: 200,0 ms com 1 thread, 113,6 com 2, 68,8 com 4 — ganho
sublinear, os 4 cores Cortex-A53 saturam.

Isso reordena as prioridades: **o detector custa 17× o embedder**. Com o w600k a 5,02 ms na
TPU, um frame completo em 640 é ~88 ms, dos quais 94 % são YuNet no CPU. Trocar o embedder
por um de 1,84 ms economizaria 3,2 ms — 3,6 % do frame. Quem paga a conta é a detecção, e
o caminho é ou baixar o `long_side` (320 corta 41 ms) ou tirar o YuNet do CPU.

Note que em `long_side` baixo o resize passa a dominar: em 320 o resize (22,9 ms) custa mais
que o próprio `detect()` (16,0 ms), porque a imagem de entrada é grande. Reduzir a captura
na fonte vale mais que reduzir a entrada do detector.

### Recuperação de embedding: numpy, não faiss

Galeria real deste projeto: **~50 rostos**. Nesse tamanho a busca é 102 KB de dados e
custa ~0,1 ms — 0,1 % de um frame de 88 ms. Qualquer índice é overhead puro.

Medido na placa (`embedding_search.log`), 512 dims, top-5, cosseno = produto interno
porque os embeddings já saem normalizados:

| N | numpy 1 face | faiss 1 face | numpy 4 faces | faiss 4 faces |
|---|---|---|---|---|
| 100 | 0,122 ms | 0,100 ms | 0,234 ms | 0,101 ms |
| 1 000 | **0,399** | 0,982 | 1,563 | **1,116** |
| 10 000 | **2,850** | 9,248 | 14,577 | **9,890** |
| 100 000 | **25,568** | 91,239 | 157,092 | **100,906** |

`faiss-cpu` 1.13.0 instala sem problema nesta imagem (wheel aarch64, 10,8 MB), mas o
`IndexFlatIP` é **3,6× mais lento que o numpy** para uma consulta. A razão é conhecida: o
faiss só usa BLAS acima de `distance_compute_blas_threshold` (20 consultas); abaixo disso
percorre a galeria com kernel próprio, e perde do GEMV do OpenBLAS. Ele só ganha em lote,
onde o numpy degrada: 4 consultas custam 5,1× uma no numpy contra 1,07× no faiss.

Nada disso importa a 50 rostos. **Use numpy** — `gallery.npy` (N,512) float32 normalizada
mais um JSON de rótulos, carregado uma vez, `sims = gallery @ emb` e `argmax` com limiar.
Sem servidor, sem índice, sem dependência nova.

O que vale trocar não é o numpy pelo faiss, é o **Qdrant pelo numpy**: subir um servidor num
board de 985 MB com ~580 MB livres para guardar 102 KB de vetores não se paga. O Qdrant faz
sentido no NUC e no Spark, não aqui.

Duas armadilhas medidas, para não perder tempo com elas:

- **Galeria em fp16 não acelera, destrói.** 1455 ms contra 22 ms do fp32 em N=100 000 —
  66× mais lento. O numpy não tem caminho de BLAS para fp16 em aarch64 e faz upcast
  elemento a elemento. Reduzir bytes da galeria não ajuda por esse caminho.
- **A galeria grande não cabe.** N=100 000 em fp32 são 204,8 MB dos ~580 MB livres, e a
  tentativa de manter cópias fp32+fp16+int8 ao mesmo tempo estourou a memória.

### YuNet na TPU: mapeia 66/66 e não serve

O `yunet_120x160_int8_out_uint8.tflite` mapeia **todas as 66 ops** na Edge TPU (o log está
em `PDC/yunet-edgetpu/`). A variante com saída float32 não: 3 `DEQUANTIZE` ficam de fora.

E mesmo mapeando inteiro, não compensa:

| | inferência | `detect()` completo |
|---|---|---|
| Edge TPU | 13,08 ms | 24,19 ms |
| CPU, 4 threads | 14,93 ms | **23,44 ms** |

**1,14× na inferência, e ponta a ponta o CPU ganha.** O modelo é pequeno demais (179 KB,
66 ops, entrada 120×160) para o custo fixo de transferir para o chip se pagar. Detectar 4
faces com os mesmos scores nas duas versões — a TPU acerta, só não acelera.

Note onde vai o tempo: dos 24 ms do `detect()`, só 13 são inferência. Os outros ~11 ms são
o resize mais o decode dos 1076 priors com NMS, em Python. **O decode custa quase tanto
quanto a rede.**

Os dois `_edgetpu.tflite` de `~/data/yunet-tpu/` têm o mesmo tamanho (11.338.560 bytes) mas
sha256 diferentes — são modelos distintos, não cópia um do outro, ao contrário do que este
README dizia antes.

### Benchmark e2e em vídeo

`e2e_video_tpu.py` mede a cadeia inteira: decode → detect → alinhamento 5 pontos → embed →
busca numpy. Ele nasceu porque o `bench/e2e_video.py` não rodava aqui — desde **26/08/2026
roda**, e a placa entra na mesma suíte dos outros alvos (ver *A suíte principal na placa*,
abaixo). Os imports de torch em `bench.system_monitor`, `bench.sync` e `bench.hardware` são
todos preguiçosos, e `pandas`/`fastparquet`/`imageio` entraram no `setup_venv39.sh`.

O `e2e_video_tpu.py` continua servindo ao recorte que a suíte não faz: comparar TPU e CPU
etapa a etapa em 640×480, sem groundtruth.

Vídeo em **640×480**, reescalado *antes* pelo `resize_video.py` para o resize não entrar na
conta. 600 frames medidos, 0,75 faces/frame, galeria de 50 rostos (`e2e_video.log`):

| etapa | CPU + TPU | TPU + TPU | tudo CPU |
|---|---|---|---|
| decode | 5,02 ms | 5,33 | 4,79 |
| **detect** | **24,72** | 36,35 | 24,41 |
| align | 1,23 | 1,24 | 1,19 |
| **embed** | **4,51** | 11,46 | **145,89** |
| search | 0,20 | 0,38 | 0,42 |
| soma | **35,67** | 54,75 | 176,70 |
| **FPS** | **28,03** | 18,26 | 5,66 |

**A Edge TPU vale 5× no e2e** — 28,03 contra 5,66 FPS — e o ganho inteiro está no embedder
(4,51 contra 145,89 ms por frame). Nada mais muda.

**Não coloque os dois modelos na TPU.** Com detector e embedder disputando o chip, o embed
vai de 4,51 para 11,46 ms e o detect de 24,72 para 36,35: 18,26 FPS contra 28,03. Uma Edge
TPU não guarda os parâmetros de dois modelos ao mesmo tempo, então cada alternância entre
eles recarrega. O detector fica no CPU, que é onde ele já era mais rápido de qualquer forma.

Com o embedder na TPU o gargalo virou o **detect**: 24,72 dos 35,67 ms, 69 % do frame. E
dentro dele ~8,5 ms são o decode dos priors em Python. Reescrever esse decode é o próximo
ganho real, maior que qualquer troca de modelo.

Baixar de 720p para 640×480 economizou 11,7 ms só de decode de vídeo (16,73 → 5,02), o que
levou a configuração boa de 20,34 para 28,03 FPS.

### Janela sustentada, no protocolo da suíte

A configuração boa (YuNet no CPU + MobileFaceNet w600k na Edge TPU) rodada com warm-up de
60 s e **300 s medidos**, igual ao `VIDEO_WARMUP_S` e `MIN_VIDEO_DURATION_S` do
[bench/config.py](../config.py) — 8.319 frames, 0,58 faces/frame (`e2e_sustained.log`):

| etapa | mean | P95 |
|---|---|---|
| decode | 5,83 ms | 7,88 |
| **detect** | **25,43** | 26,31 |
| align | 0,90 | 1,69 |
| embed | 3,43 | 6,20 |
| search | 0,12 | 0,19 |
| soma | **35,72** | 41,31 |

- **fps** 28,00 pela soma das etapas, **27,73** de relógio de parede
- **latência** p50 38,57 ms · p95 41,31 · max 93,54
- **`frac_in_33ms` 0,416** · `frac_in_66ms` 0,992 · `frac_in_100ms` 1,000
- **térmico** 58 → 60 °C, pico 66 °C
- **1º decil 29,06 fps → último decil 27,44** (razão **0,944**)

Dois pontos que só a janela longa revela:

**A placa não entrega 30 fps.** O `fps_mean` de 28 esconde que só **42 % dos frames** ficam
dentro de 33 ms. Para 15 fps (66 ms) ela entrega 99,2 %. Esse é o número honesto de
capacidade, e é a métrica que o resto da suíte usa.

**A razão sustentado/burst dá 0,944, abaixo do limiar de 0,95** que o
[bench/robustness.py](../robustness.py) usa para acender `slowdown_flag`. Mas a latência p50
quase não mexeu (38,04 → 38,86 ms, +2 %) e a temperatura subiu só 2 °C, com pico de 66 °C —
não parece throttling térmico. Parte vem dos ~9 rewinds do vídeo (`--loop` faz
`cap.set(POS_FRAMES, 0)`, que é seek), que também explicam o max de 93,54 ms. Para cravar
throttling seria preciso um vídeo de 300 s sem loop.

O resultado bate com a janela curta de 21 s (28,03 fps), o que dá confiança na medição — mas
os grupos B (energia) e C (recursos) da Coral continuam vazios: a placa não tem sensor de
potência, e este script não amostra CPU/RAM como o `bench/system_monitor.py`.

### A suíte principal na placa

Desde 26/08/2026 a Coral roda o mesmo `scripts/run_bench_e2e.sh` dos outros alvos, com o
mesmo vídeo do ChokePoint em 1280×720, o mesmo groundtruth e o mesmo `--yunet-input-long-side 0`
— então os números são comparáveis com NUC, Pi 5 e Jetson sem ressalva de protocolo.

Primeira rodada válida (`yunet` + `mobilefacenet_w600k` int8 na Edge TPU, 2292 quadros,
clocks em `performance` a 1,5 GHz, 55 °C):

| etapa | mediana | P95 |
|---|---|---|
| **detecção** (YuNet, CPU) | **160,69 ms** | 178,21 |
| alinhamento | 1,48 | 1,98 |
| **embedding** (Edge TPU) | **6,13** | 7,67 |
| busca (numpy, 50 rostos) | 0,88 | 1,12 |
| latência do quadro | **171,07** | 189,08 |

- **5,03 FPS**, `frac_in_33ms` = `frac_in_66ms` = `frac_in_100ms` = **0**
- detecção P=0,7919 R=0,9993 **F1=0,8836** — idêntico ao da Pi 5 e do Jetson Orin Nano no
  mesmo vídeo, o que confirma que o caminho do detector é o mesmo em todos os alvos
- reconhecimento **rank-1 e2e 0,7489** (1035/1382), macro-F1 0,8477

O detector é **94 % do quadro**. Os 160,7 ms contra os 82,9 ms medidos em 640 são o preço de
rodar o YuNet na resolução nativa do vídeo, que é o que o protocolo da suíte pede; a
configuração de 28 FPS mais acima usa 640×480 e não é a mesma medida.

Os 6,13 ms de embedding são a prova de que o chip está em uso: o mesmo modelo no CPU da placa
dá ~240 ms (ver a armadilha abaixo).

**Duas armadilhas do runtime desta imagem, as duas silenciosas:**

1. **Os `.tflite` int8 não compilados devolvem embedding nulo.** O `tflite-runtime` 2.7.0
   carrega `mobilefacenet_w600k_int8.tflite` e `arcface_r100_int8.tflite` sem erro e devolve o
   *zero point* nas 512 posições — depois do L2, vetor zero. A latência continua com cara de
   normal (240 ms/face no mbf com 4 threads), e nada a jusante pega, porque a contagem de faces
   vem do detector. Os mesmos arquivos rodam certo no LiteRT do desktop, então é o runtime da
   placa. O `TFLiteEmbedder` faz uma inferência de sonda na construção e aborta a rodada em vez
   de gravar precision/recall de vetor zerado. Consequência prática: **`int8_cpu` não é uma
   linha de base de acurácia nesta placa**; para latência pura, `bench_tflite_latency.py`.
2. **EdgeFace não tem caminho aqui.** Além de o `edgetpu_compiler` não mapear o grafo, o
   `edgeface_xs_gelutanh_int8.tflite` nem carrega: `Didn't find op for builtin opcode
   'CONV_2D' version '6'` — a mesma causa nº 2 da lista de rejeições acima. O arquivo está em
   `models/tflite/` para quem for requantizar; hoje ele não roda em precisão nenhuma.

Um terceiro defeito, esse já corrigido: até 26/08/2026 o `TFLiteEmbedder` desligava o delegate
sozinho quando o `device` começava com `cpu`, e no perfil da Coral o device é o do **detector**.
As rodadas rotuladas `int8_tpu` mediam o extrator no CPU, com número plausível e sem aviso —
e a verificação de contagem de faces do `run_bench_e2e.sh` não pega isso, porque a contagem não
muda. Hoje `use_tpu` é tri-estado e um pedido explícito de TPU levanta se o delegate não
carregar, em vez de cair no CPU.

### Ressalvas

- **Recall da detecção não medido.** O YuNet TFLite trabalha em 120×120 fixo; num vídeo
  640×480 ele achou 0,75 faces/frame. Faces pequenas ou distantes provavelmente escapam, e
  não comparei com o YuNet 2023mar do OpenCV em 640 para quantificar.
- **A janela é de 21 s** (600 frames), não captura throttling. O resto do benchmark usa
  warm-up de 60 s mais 300 s de medição.
- **1280×720 → 640×480 distorce**, a proporção vai de 1,778 para 1,333. O `resize_video.py`
  avisa. Se a distorção afeta a detecção, não foi medido.

## Ambiente da placa

- Coral Dev Board, Mendel 5 "Eagle" (Debian 10, glibc 2.28), kernel 4.14.98-imx, aarch64
- Edge TPU PCIe em `/dev/apex_0`, `libedgetpu1-std` 16.0
- `python3` do sistema é 3.7.3. O que se usa é **Python 3.9.25 via uv** (`~/.local/bin/python3.9`)
- `tflite-runtime` **2.7.0**, numpy 1.26.4, opencv-python-headless 4.11.0.86, pycoral 2.0.0 (só os adapters)

Três armadilhas que custaram tempo e estão resolvidas no `setup_venv39.sh` e no `edgetpu_shim.py`:

1. Os wheels `pycoral`/`tflite_runtime` cp38 e cp39 do repo google-coral são compilados
   contra GLIBC_2.29 (Debian 11) e não carregam nesta imagem. Só o cp37 casa com a
   glibc 2.28, e ele serve apenas ao Python 3.7 do sistema.
2. `tflite-runtime` **2.13.0 segfaulta** com o libedgetpu 16.0 desta imagem. Use 2.7.0 e
   não "atualize".
3. `pycoral.utils.edgetpu` depende de `_pywrap_coral.so`, que é justamente a peça que não
   carrega. O shim reimplementa `make_interpreter`/`list_edge_tpus` carregando o delegate
   direto do `libedgetpu.so.1` do sistema. Os adapters (`common`, `detect`) são Python puro
   e funcionam normais.

## Onde o repo e os artefatos ficam na placa

A `/home` interna tem 2 GB (≈1,4 GB livres) — não cabe. Tudo vive no cartão de 59 GB
montado em `/home/mendel/data`, com `~/PDC-face` como símlink para lá.

Os `.tflite` e `.npy` **não estão versionados** (são grandes e regeneráveis a partir dos
scripts de quantização no WSL).

A suíte principal (`run_bench_e2e.sh`) procura os extratores em **`models/tflite/`**, com o
nome do modelo como aparece na matriz e o par `_edgetpu` ao lado. Esses cinco arquivos foram
copiados para lá em 26/08/2026, do que já existia na placa mais o EdgeFace vindo do WSL:

```
models/tflite/  arcface_r100_int8.tflite (66 MB)              sha256 e5cd4355...
                arcface_r100_int8_edgetpu.tflite (67 MB)      sha256 f779dbdb...
                mobilefacenet_w600k_int8.tflite (3,9 MB)      sha256 17632822...
                mobilefacenet_w600k_int8_edgetpu.tflite (4,3) sha256 39f643e4...
                edgeface_xs_gelutanh_int8.tflite (2,3 MB)     sha256 29cf0825...  (não carrega)
```

O `_edgetpu` tem que ser a saída do compilador sobre **aquele mesmo** int8: comparar TPU
contra CPU com um int8 requantizado noutra rodada mistura acurácia com latência. O
`mobilefacenet_w600k_int8.tflite` aqui é o `mbf_w600k_int8.tflite` de `compact-fr-tpu`, não o
`models/tflite/mobilefacenet_w600k_int8.tflite` do WSL, que é outra quantização (sha
5b2a5a45...) e não tem par compilado.

Os do MobileFaceNet também ficam dentro do repo, em **`bench/coral/models/`** — a regra
`models/` do `.gitignore` da raiz já os exclui automaticamente, então dá para deixá-los ao
lado dos scripts sem sujar o `git status`:

```
bench/coral/models/     mbf_w600k_int8_edgetpu.tflite (4,3 MB)  sha256 39f643e4...
                        mbf_w600k_int8.tflite (3,9 MB)
                        mbf_eval_uint8.npy (18 MB), mbf_emb_fp32.npy,
                        mbf_pairs.npy, baseline_perm.npy
```

O resto continua solto na `~/data`, porque são 133 MB só de ArcFace e não vale duplicar:

```
~/data/arcface-tpu/     arcface_r100_int8.tflite (66 MB), arcface_r100_int8_edgetpu.tflite (67 MB),
                        edgeface_xxs_int8.tflite (1,6 MB), edgeface_xxs_sim_float32.tflite (5,2 MB),
                        eval_subset_uint8.npy (1,2 MB), eval_full_uint8.npy (18 MB),
                        emb_fp32.npy, emb_ref_int8.npy, baseline_pairs.npy, baseline_perm.npy,
                        xxs_eval_uint8.npy, xxs_emb_ref_int8.npy
~/data/compact-fr/      cópia dos artefatos do MobileFaceNet (origem do que está em
                        bench/coral/models/) + mobilefacenet_int8.tflite (1,2 MB) e
                        mobilefacenet_int8_edgetpu.tflite (1,4 MB)
~/data/yunet-tpu/       yunet_120x160_int8_out_uint8.tflite (179 KB),
                        yunet_120x160_int8_out_uint8_edgetpu.tflite (11 MB),
                        model_full_integer_quant_edgetpu.tflite (11 MB)
```

No WSL, os artefatos do MobileFaceNet são gerados em `PDC/compact-fr-tpu/artifacts/`.

O YuNet int8 de 179 KB vira 11 MB depois do `edgetpu_compiler`. Os dois
`_edgetpu.tflite` de `yunet-tpu/` têm o mesmo tamanho (11.338.560 bytes) mas sha256
diferentes: são modelos distintos, e o de 179 KB mapeia 66/66 ops (ver abaixo).

O modelo do YuNet TFLite vem do
[PINTO_model_zoo/144_YuNet](https://github.com/PINTO0309/PINTO_model_zoo); o decode dos
priors reproduz o de
[Kazuhito00/YuNet-ONNX-TFLite-Sample](https://github.com/Kazuhito00/YuNet-ONNX-TFLite-Sample),
porque o `cv2.FaceDetectorYN` faz isso em C++ e não aceita TFLite.

## Como rodar

```bash
# na placa, uma vez. O .venv do repo e o que o run_bench_e2e.sh procura; o repo ja
# mora no cartao, entao nao ha risco de encher a /home interna de 2 GB.
bash bench/coral/setup_venv39.sh "$PWD/.venv"
source .venv/bin/activate

# a suite principal, igual aos outros alvos (matriz padrao: arcface_r100 e
# mobilefacenet_w600k, so int8_tpu)
bash scripts/run_bench_e2e.sh
MODELOS=mobilefacenet_w600k bash scripts/run_bench_e2e.sh   # so um modelo

# YuNet: CPU (dentro do repo, usa utils/yunet_inference.py) e TPU
# o .onnx tem que estar em models/yunet/, que é onde DEFAULT_YUNET_MODEL aponta
python bench/coral/bench_yunet_cpu.py bench/coral/face.jpg
python bench/coral/yunet_tflite_tpu.py ~/data/yunet-tpu/yunet_120x160_int8_out_uint8_edgetpu.tflite \
    bench/coral/face.jpg --edgetpu

# MobileFaceNet: artefatos já estão no repo
cd bench/coral/models && python ../mobilefacenet_board_bench.py

# latência isolada de qualquer modelo compilado
python ../bench_tflite_latency.py mbf_w600k_int8_edgetpu.tflite

# ArcFace: rodar de dentro do diretório dos artefatos, que ficou fora do repo
cd ~/data/arcface-tpu && python ~/PDC-face/face-recognition/bench/coral/arcface_board_bench.py
```

O clone na placa está em `~/PDC-face/face-recognition` (símlink para o cartão) e na branch
`coral`.
