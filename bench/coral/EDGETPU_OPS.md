# Operações na Edge TPU: o que passa e o que não passa

Levantamento feito sobre **16 logs do `edgetpu_compiler` 16.0.384591198**, de modelos de
reconhecimento facial exportados PyTorch → ONNX → `onnx2tf` → TFLite int8, entrada 112×112
ou 120×160. Não é a documentação da Google: é o que este compilador, nesta versão, aceitou
ou recusou nestes modelos. Onde a causa não foi rastreada, está dito.

Fontes: `PDC/compact-fr-tpu/artifacts/*_edgetpu.log` (12), `PDC/yunet-edgetpu/*.log` (2),
`PDC/arcface-edgetpu/artifacts/arcface_r100_int8_edgetpu.log`,
`PDC/edgeface-edgetpu/pipeline_*.log`.

## As mensagens do compilador, e quais são veredito

O compilador emite uma linha por operador com uma destas situações. A distinção importa:
**só quatro delas dizem algo sobre a operação**; as outras dizem algo sobre o modelo.

| Mensagem | Significa | Veredito sobre a op? |
|---|---|---|
| `Mapped to Edge TPU` | vai para o chip | sim |
| `Operation not supported` | a op não existe no runtime da TPU | sim, definitivo |
| `Operation version not supported` | a op existe, a versão do builtin não | sim, para essa versão |
| `Operation is working on an unsupported data type` | tipo errado (float onde o chip exige int) | sim, para esse tipo |
| `Tensor has unsupported rank (up to 3 innermost dimensions mapped)` | tensor com mais de 3 dimensões internas | sim, para esse shape |
| `Operation is otherwise supported, but not mapped due to some unspecified limitation` | limite interno não explicado (tamanho, layout) | parcial |
| `More than one subgraph is not supported` | **colateral**, não veredito — ver adiante | **não** |

## Ops confirmadas mapeadas

Cada uma apareceu como `Mapped to Edge TPU` em pelo menos um modelo real. O número é em
quantos dos 16 modelos ela mapeou.

| Op | modelos | Op | modelos |
|---|---|---|---|
| `QUANTIZE` | 16 | `RESHAPE` | 6 |
| `PAD` | 16 | `SOFTMAX` | 3 |
| `CONV_2D` | 16 | `SLICE` | 3 |
| `DEPTHWISE_CONV_2D` | 14 | `RESIZE_NEAREST_NEIGHBOR` | 3 |
| `PRELU` | 12 | `RELU` | 3 |
| `ADD` | 10 | `MAX_POOL_2D` | 3 |
| `CONCATENATION` | 8 | `SPLIT` | 2 |
| `TRANSPOSE` | 7 | `MUL` | 2 |
| | | `FULLY_CONNECTED` | 2 |
| | | `MEAN` | 1 |
| | | `LOGISTIC` | 1 |

Dois modelos mapeiam **100 %** das ops, e são os dois que valem na prática:

- **ArcFace r100**, 311 ops: `CONV_2D` 103, `ADD` 99, `PRELU` 50, `MUL` 50, `PAD` 4,
  `QUANTIZE` 2, `FULLY_CONNECTED` 1, `RESHAPE` 1, `TRANSPOSE` 1
- **YuNet 120×160 (saída uint8)**, 66 ops: `CONV_2D` 22, `DEPTHWISE_CONV_2D` 21,
  `RESHAPE` 8, `MAX_POOL_2D` 5, `QUANTIZE` 4, `SLICE` 3, `PAD` 1, `CONCATENATION` 1,
  `SOFTMAX` 1

`PRELU` mapear em 12 dos 16 é o achado mais útil da lista: não há motivo para trocar PReLU
por ReLU só por medo do compilador. É a ativação padrão da família ArcFace e passa.

## Ops rejeitadas de fato

Filtrando o colateral, estas são as recusas reais:

| Op | Mensagem | Onde (contagem) | Observação |
|---|---|---|---|
| `PADV2` | `Operation not supported` | pocketnetm_128_dgc (6), pocketnetm_256_dgc (6) | a única recusa categórica. `PAD` mapeia em 16 modelos; `PADV2` (pad com valor constante configurável) não existe no chip |
| `ABS` | `Operation version not supported` | vargfacenet_dgc (18), pocketnetm_128_dgc (27), pocketnetm_256_dgc (27) | origem no grafo não rastreada |
| `DEQUANTIZE` | `Operation is working on an unsupported data type` | yunet saída float32 (3) | some ao trocar a saída para uint8: a mesma rede vira 66/66 |
| `RESHAPE` | `Tensor has unsupported rank` | shufflefacenet 0.5×/1.5×/2× (32 cada), shufflemixfacenet-s/xs (32), mixfacenet-s/xs (2) | mais de 3 dimensões internas |
| `TRANSPOSE` | `Tensor has unsupported rank` | shufflefacenet ×3 (16 cada), shufflemixfacenet ×2 (16), mixfacenet ×2 (1) | idem — é o `channel_shuffle` |
| `TRANSPOSE` | `unspecified limitation` | pocketnetm_128/256_dgc (40), shufflefacenet_2x (20), 1.5× (14), vargfacenet (3) | |
| `QUANTIZE` | `unspecified limitation` | 10 modelos, **sempre exatamente 1** | um único requantize sobra fora do chip em todo modelo que fragmenta |

O `channel_shuffle` do ShuffleNet é o caso mais claro: ele é implementado como
reshape → transpose → reshape sobre 5 dimensões, e é justamente o `Tensor has unsupported
rank` que fragmenta esses cinco modelos.

## A armadilha do `More than one subgraph is not supported`

**O compilador produz um único subgrafo.** A primeira op genuinamente não suportada corta o
grafo ali, e **tudo que vem depois recebe essa mensagem, suportado ou não**.

Exemplo do `mixfacenet-s_int8_edgetpu.log`:

```
CONV_2D    2   Mapped to Edge TPU
CONV_2D   79   More than one subgraph is not supported
```

`CONV_2D` é a op mais bem suportada que existe no chip — mapeia em todos os 16 modelos. As
79 rejeitadas não dizem nada sobre `CONV_2D`; dizem que caíram depois do corte.

Consequência prática: **contar essas linhas como "ops sem suporte" infla o diagnóstico
grosseiramente.** Em PocketNetM-128 são 5.745 ops "não mapeadas", das quais 5.343 são
`CONV_2D`. O problema real são 6 `PADV2` e 27 `ABS`.

Método de diagnóstico: descarte as linhas `More than one subgraph` e olhe só as outras. O
que sobrar são os culpados.

### A flag `-a` derruba essa parede — e não foi medida

O `edgetpu_compiler` tem `-a, --enable_multiple_subgraphs`, marcada como **experimental** e
não usada em nenhum dos pipelines deste projeto. Recompilando com ela:

| modelo | sem `-a` | **com `-a`** | subgrafos | on-chip | streaming |
|---|---|---|---|---|---|
| MixFaceNet-S | 9/443 | **440/443** | 2 | 1,96 MiB | 0 B |
| ShuffleMixFaceNet-XS | 9/514 | **466/514** | 17 | 1,68 MiB | 0 B |
| PocketNetM-128 `_dgc_` | 29/5774 | **5701/5774** | 51 | 5,90 MiB | 0 B |
| PocketNetM-256 `_dgc_` | 29/5774 | **5701/5774** | 51 | 5,96 MiB | 0 B |

E os que continuam falhando, agora com mensagem diferente:

| modelo | erro com `-a` |
|---|---|
| MixFaceNet-XS, -M, ShuffleMixFaceNet-S, -M | `Compilation failed!` sem causa declarada |
| VarGFaceNet `_dgc_` | `Compilation failed due to large activation tensors in model` |
| ShuffleFaceNet 0.5×, 1.5×, 2× `_dgc_` | `Compilation failed!` sem causa declarada |

**Atenção: nada disso foi medido na placa.** Mapear 440 de 443 ops não garante ganho — são
2 subgrafos no MixFaceNet-S, mas **17** no ShuffleMixFaceNet-XS e **51** no PocketNetM, e
cada fronteira entre subgrafos é uma transição CPU↔TPU com transferência de tensor. Pode
sair mais lento que os 81 ms do CPU puro. O fato de caberem no chip sem streaming (0 B nos
quatro) é promissor, mas é promessa, não resultado.

Isso corrige o que este documento afirmava: a parede de subgrafo único **não é da
arquitetura, é do padrão do compilador**. Os quatro modelos que eu havia descartado como
inviáveis merecem uma medição antes do veredito.

**No YuNet a flag não faz nada.** As três variantes, com `-a`, dão números byte a byte
iguais aos de sem a flag: 1 subgrafo, 7,05 MiB on-chip, 2,33 MiB de streaming, 10,81 MiB de
saída. Na variante `out_float32` os 3 `DEQUANTIZE` continuam no CPU (63 na TPU, 3 no CPU),
porque são a cauda do grafo e não fragmentam nada.

Isso delimita para que serve a flag, e é útil: **`-a` resolve fragmentação, não alocação.**
Ela só tem alavanca quando ops sem suporte estão intercaladas no meio da rede, cortando-a em
pedaços. O YuNet não tem op sem suporte — mesmo autorizado a produzir vários subgrafos, o
compilador continua escolhendo um, porque não há o que particionar. Os dois problemas são
ortogonais: um é o grafo, o outro é o espaço que os pesos ocupam no chip.

Uma classe de falha diferente, e mais grave, porque acontece **antes** de qualquer análise
de operador — o modelo não é nem parseado:

```
ERROR: Didn't find op for builtin opcode 'CONV_2D' version '6'.
       An older version of this builtin might be supported.
ERROR: Registration failed.
Invalid model: edgeface_gelutanh_int8.tflite
Model could not be parsed
```

Barra nos dois lados da cadeia: no `edgetpu_compiler` 16.0 **e** no `tflite-runtime` 2.7.0
da placa. Atingiu EdgeFace, PocketNetS-128/256, PocketNetM-128/256, ShuffleFaceNet
0.5×/1.5×/2× e VarGFaceNet.

A origem é o conversor do TensorFlow, que emite `CONV_2D` v6 para convolução agrupada — o
VarGFaceNet tem 49 delas. Os artefatos `_dgc_` são a tentativa de contornar desdobrando as
convoluções agrupadas em convoluções normais: parseiam, mas o desdobramento explode o
modelo (PocketNetM-128 vai de 2,2 MB para 33,4 MB) e o grafo continua fragmentado.

Não é contornável subindo o `tflite-runtime`: a 2.13.0 registra a v6 mas **segfaulta** com o
`libedgetpu` 16.0 desta imagem. E o compilador 16.0 é o último para esta geração de chip.

## Ops de transformer: não há o que mapear

O inventário do EdgeFace int8 (742 ops, `pipeline_gelutanh.log`) contém:

```
MUL 195, ADD 113, FULLY_CONNECTED 86, MEAN 53, SUB 26, RSQRT 26, TRANSPOSE 24,
RESHAPE 23, CONV_2D 22, QUANTIZE 20, TANH 18, STRIDED_SLICE 18, DEQUANTIZE 18,
SQUARE 6, SUM 6, SQRT 6, MAXIMUM 6, DIV 6, BATCH_MATMUL 6, DEPTHWISE_CONV_2D 3,
CONCATENATION 3, SOFTMAX 3
```

Nenhuma de `RSQRT`, `SQUARE`, `SUM`, `SQRT`, `MAXIMUM`, `DIV`, `BATCH_MATMUL`, `TANH`,
`SUB` e `STRIDED_SLICE` aparece na lista de mapeadas em nenhum dos 16 modelos. São
exatamente as peças de LayerNorm (`SUB`/`SQUARE`/`MEAN`/`RSQRT`) e de atenção
(`BATCH_MATMUL`, `SOFTMAX` sobre rank alto).

Isso é estrutural, não uma questão de quantização: **arquitetura com transformer não vai
para esta Edge TPU**, o que inclui EdgeFace e FaceLiVT. O nome `pipeline_gelutanh` registra
uma tentativa de trocar o GELU por aproximação com tanh; não resolve, porque o problema não
é a ativação.

### Mapa do grafo do EdgeFace, op por op

Como o compilador morreu no parser e não emitiu relatório, o mapa vem do flatbuffer direto,
via `map_tflite_graph.py` (saída completa em `edgeface_graph_map.log`). Os três `.tflite`
estão em `PDC/edgeface-edgetpu/artifacts/`.

`edgeface_gelutanh_int8.tflite` — 687 operadores, 1 subgrafo:

| op | n | ver | ranks de saída | veredito |
|---|---|---|---|---|
| `MUL` | 195 | 2 | 3d:9, 4d:186 | ok |
| `ADD` | 113 | 2 | 3d:10, 4d:103 | ok |
| `FULLY_CONNECTED` | 86 | 5 | 2d:2, 3d:12, 4d:72 | ok |
| `MEAN` | 53 | 2 | 3d:6, 4d:47 | ok |
| `SUB` | 26 | 2 | 3d:3, 4d:23 | **sem suporte** |
| `RSQRT` | 26 | 2 | 3d:3, 4d:23 | **sem suporte** |
| `TRANSPOSE` | 24 | 4 | 3d:3, 4d:18, **5d:3** | **rank** |
| `RESHAPE` | 23 | 1 | 2d:1, 3d:6, 4d:13, **5d:3** | **rank** |
| `CONV_2D` | 22 | **6** | 4d:22 | **FATAL, não parseia** |
| `QUANTIZE` | 20 | 1 | 2d:1, 4d:19 | ok |
| `TANH` | 18 | 2 | 4d:18 | **sem suporte** |
| `STRIDED_SLICE` | 18 | 4 | 4d:9, **5d:9** | **sem suporte + rank** |
| `DEQUANTIZE` | 18 | 2 | 4d:18 | **tipo sem suporte** |
| `SQUARE`, `SUM`, `SQRT`, `MAXIMUM`, `DIV`, `BATCH_MATMUL` | 6 cada | 1–2 | 4d | **sem suporte** |
| `DEPTHWISE_CONV_2D` | 3 | 3 | 4d:3 | ok |
| `CONCATENATION` | 3 | 2 | 4d:3 | ok |
| `SOFTMAX` | 3 | 2 | 4d:3 | ok |

**476 das 687 ops (69 %) não têm problema apontado.** O que mata o modelo são 211 ops em
três causas independentes — e a ordem importa, porque só a primeira é fatal:

1. **`CONV_2D` versão 6, 22 instâncias.** Sozinha, encerra o assunto: o parser do compilador
   16.0 rejeita e nada mais é avaliado. As outras duas causas só apareceriam se esta fosse
   resolvida. Note que `FULLY_CONNECTED` está em v5 e passa — o corte é exatamente em 6.
2. **11 operadores que nunca mapearam**, 164 instâncias: `SUB` 26, `RSQRT` 26, `TANH` 18,
   `STRIDED_SLICE` 18, `DEQUANTIZE` 18, e `SQUARE`/`SUM`/`SQRT`/`MAXIMUM`/`DIV`/`BATCH_MATMUL`
   com 6 cada. Os pares (`SUB`, `SQUARE`, `MEAN`, `SUM`, `RSQRT`, `SQRT`, `DIV`, `MAXIMUM`)
   são LayerNorm expandido em aritmética elementar; `BATCH_MATMUL` é a atenção.
3. **Rank 5, 15 instâncias:** `STRIDED_SLICE` 9, `RESHAPE` 3, `TRANSPOSE` 3. O limite é 3
   dimensões internas.

As três variantes falham no mesmo ponto, o que localiza a culpa:

| modelo | ops | `CONV_2D` v6 | sem suporte | rank 5 | ok |
|---|---|---|---|---|---|
| `edgeface_gelutanh_int8` | 687 | 22 | 11 ops / 164 | 15 | 69,3 % |
| `edgeface_xxs_int8` | 512 | 16 | 11 ops / 142 | 15 | 63,5 % |
| `edgeface_xxs_int8_tf213` | 966 | 16 | 11 ops / 152 | 15 | 69,0 % |

O `tf213` é a tentativa de converter com TensorFlow 2.13: **piorou**. O grafo quase dobrou
(512 → 966 ops), apareceram 30 `PACK` que também não mapeiam, e o `CONV_2D` v6 continuou
lá. Trocar a versão do conversor não resolve — a v6 vem da forma da convolução, não da
versão do TensorFlow.

E a lição sobre o `gelutanh`: trocar GELU por tanh **removeu** o `ERF` mas **acrescentou** 18
`TANH`, que também não mapeia. O `SUB`/`RSQRT`/`BATCH_MATMUL` do LayerNorm e da atenção não
foram tocados, e são 78 das 164 instâncias sem suporte. A ativação nunca foi o problema.

## Como os pesos entram no chip, e por que isso decide a latência

O compilador não só decide *quais ops* vão para a TPU: ele decide **quais pesos ficam
cacheados na SRAM do chip** e quais precisam ser transmitidos do host a cada inferência.
Os pesos são embutidos no `_edgetpu.tflite` já no layout do data path, e o relatório do
compilador diz exatamente como ficou a divisão:

```
On-chip memory used for caching model parameters:            X
On-chip memory remaining for caching model parameters:       Y
Off-chip memory used for streaming uncached model parameters: Z
```

O pool é de ~8 MB de SRAM, dos quais ~7,5 MiB sobram para parâmetros (o resto vai para
buffers intermediários e o executável, e por isso `usado + restante` varia de modelo para
modelo). O que couber é carregado **uma vez** e fica; o que não couber (`Z`) atravessa o
PCIe **em toda inferência**.

Medido nos quatro modelos que importam:

| modelo | int8 de entrada | on-chip | sobra | **streaming** | latência |
|---|---|---|---|---|---|
| MobileFaceNet 128d | 1,25 MB | 1,17 MiB | 6,43 MiB | **0 B** | **1,84 ms** |
| MobileFaceNet w600k | 3,70 MiB | 3,56 MiB | 3,93 MiB | 125 KiB | 5,11 ms |
| YuNet 120×160 | 174 KiB | 7,05 MiB | 512 B | **2,33 MiB** | 13,08 ms |
| ArcFace r100 | 65,88 MB | 6,85 MiB | 512 B | **56,00 MiB** | **153 ms** |

**A latência acompanha o streaming, não o tamanho nominal do modelo.** Os 153 ms/face do
ArcFace r100 não são compute: são 56 MiB de peso atravessando o barramento a cada rosto. O
MobileFaceNet de 128 dims, que cabe inteiro com 6,43 MiB de folga, é 83× mais rápido.

### O peso no arquivo não é o peso no chip

Contando os tensores constantes de cada `.tflite` (int8 = 1 byte por parâmetro):

| modelo | params | constantes | on-chip | expansão | streaming |
|---|---|---|---|---|---|
| YuNet 120×160 | **0,083 M** | 0,09 MiB | **7,05 MiB** | **~78×** | 2,33 MiB |
| MobileFaceNet 128d | 0,99 M | 0,97 MiB | 1,17 MiB | 1,2× | 0 B |
| MobileFaceNet w600k | 3,40 M | 3,27 MiB | 3,56 MiB | 1,09× | 125 KiB |
| ArcFace r100 | 65,13 M | 62,18 MiB | 6,85 MiB | — não cabe | 56,00 MiB |

Para os dois MobileFaceNet a conta fecha quase exata: **params em milhões ≈ MiB no chip**,
porque int8 é 1 byte por peso e o relayout custa só 9 a 20 %. No ArcFace também: 62,18 de
constantes menos 6,85 cacheados dá 55,3, que é o 56,00 MiB de streaming reportado.

O YuNet é a exceção: **83 mil parâmetros, 0,09 MiB de peso, e 7,05 MiB ocupados no chip** —
expansão de ~78×, mais 2,33 MiB em streaming. Quase 100× de inflação num modelo que em disco
tem 174 KiB. É o que faz ele mapear 66/66 e ainda perder do CPU: não é limitado por
computação, é limitado por transferência de peso.

**A causa interna não foi determinada.** Testei três hipóteses e as três não fecham a conta
(detalhe em `yunet_graph_map.log`):

| hipótese | teste | resultado |
|---|---|---|
| Depthwise virar conv densa com matriz diagonal | 21 depthwise, 10,2 KiB reais → 0,60 MiB se densos | explica 0,66 dos 9,38 MiB |
| Granularidade fixa por operação | MobileFaceNet tem 70 ops (YuNet 66) e usa 1,17 MiB | refutada |
| Buffers de ativação intermediária | YuNet tem as **menores** ativações: 1,2 MiB contra 4,5 do MobileFaceNet | refutada |

O grafo é 43 convoluções somando 78,8 KiB, sendo o maior tensor de peso 4.096 bytes
(`[64,1,1,64]`), canais entre 16 e 64, e mapas de features que terminam em `[1,1,2,64]` e
`[1,3,5,64]`. Rede estreita, rasa em canais e com resolução final minúscula. O compilador é
fechado e o relatório só dá totais, então a atribuição para dentro do chip não é observável
daqui — mas a direção é clara: **o custo não acompanha o peso, acompanha a forma.**

Um controle confirma que é arquitetura e não quantização: o `model_full_integer_quant.tflite`
do PINTO_model_zoo, quantizado de outra forma e com 91 ops em vez de 66, dá **exatamente os
mesmos** 7,05 MiB on-chip, 2,33 MiB de streaming e 10,81 MiB de saída.

E resolve também por que **não dá para pôr dois modelos na TPU**. O pool é único: YuNet
(7,05 MiB) mais MobileFaceNet w600k (3,56 MiB) somam 10,6 MiB contra ~7,5 MiB disponíveis.
Alternar entre os dois expulsa os parâmetros de um para dar lugar ao outro, a cada frame. É
o que a medição do e2e mostrou: com os dois na TPU o embed vai de 4,51 para 11,46 ms e o
detect de 24,72 para 36,35 (ver o README). Não é contenção de tempo de chip — é recarga de
peso.

### E quando o modelo não cabe nos 8 MB

Nada falha. O ArcFace r100 é o caso medido: 62,18 MiB de constantes, o compilador **compilou
com sucesso**, mapeou **311/311 ops** e reportou **1 subgrafo**. Ele não recusa nem
particiona o grafo — particiona os *parâmetros*: os primeiros 6,85 MiB ficam cacheados e os
55,3 MiB restantes viram `Off-chip memory used for streaming uncached model parameters`,
buscados na RAM do host **a cada inferência**.

O custo disso, medido:

| | ArcFace r100 (56 MiB streaming) | MobileFaceNet w600k (125 KiB) |
|---|---|---|
| Edge TPU | 153 ms/face | 5,11 ms |
| CPU 4 threads | 1090 ms | 191,5 ms |
| ganho da TPU | 7,1× | 37× |

São 56 MiB por inferência em 153 ms, ou seja ~366 MiB/s de banda efetiva no PCIe com a
computação intercalada. **A TPU continua valendo 7×** sobre o CPU — não vira inútil — mas
sai do regime de 5 ms para o de 150 ms. A diferença entre os dois modelos não é arquitetura
nem quantização: é caber ou não caber.

Ordem de grandeza para planejar: com ~7,5 MiB de pool e int8, **cabe um modelo de até ~6–7 M
parâmetros**, desde que o relayout seja eficiente (convolução densa com muitos canais). Se
for estreito e cheio de depthwise, como o YuNet, a conta de parâmetros não serve de guia e
só a compilação diz.

O caminho oficial para modelo grande é **co-compilação / pipelining entre vários Edge TPU**,
segmentando o modelo para que os pesos de cada segmento caibam num chip. Esta placa tem um
único `/dev/apex_0`, então aqui isso não existe.

Regra que sai disso: **olhe `Off-chip memory` antes de comemorar `Number of operations that
will run on Edge TPU`.** Mapear 100 % com streaming alto é pior que mapear 100 % cabendo na
SRAM, e pode ser pior que não usar a TPU.

## Regras práticas

1. **Convolucional puro mapeia; o resto não.** ArcFace r100 e MobileFaceNet chegam a 100 %.
   Qualquer coisa com atenção, LayerNorm ou aritmética elemento-a-elemento exótica, não.
2. **PReLU passa.** Não troque por ReLU preventivamente.
3. **Saída em uint8, não float32.** Só isso levou o YuNet de 62/65 para 66/66.
4. **Nada de rank alto.** Máximo 3 dimensões internas. `channel_shuffle` (reshape/transpose
   em 5D) fragmenta o grafo — o que derruba a família ShuffleNet inteira.
5. **Convolução agrupada é veto duplo:** vira `CONV_2D` v6, que nem parseia; e desdobrar
   explode o tamanho.
6. **Um subgrafo só.** Uma op ruim no meio da rede custa o modelo inteiro, não só ela.
7. **Filtre `More than one subgraph` antes de tirar conclusão** — senão o diagnóstico aponta
   para `CONV_2D` quando o culpado são 6 `PADV2`.
8. **Mapear 100 % não garante que valha a pena.** O YuNet mapeia 66/66 e ainda assim é mais
   rápido no CPU: 13,08 ms na TPU contra 14,93 no CPU, e ponta a ponta o CPU ganha. Para um
   modelo pequeno o custo fixo de transferência come o ganho. Ver o README.
9. **Uma TPU, um modelo.** Dois modelos alternando no mesmo chip recarregam parâmetros a cada
   troca: no e2e o embed foi de 4,51 para 11,46 ms. Ver o README.

## Como reproduzir este levantamento

```bash
# todas as linhas de operador de todos os logs, com o modelo de origem
cd ~/CEIA/PDC
for f in $(find . -name "*_edgetpu.log"); do
  grep -oE "^[A-Z_0-9]+ +[0-9]+ +[A-Za-z].*$" "$f" \
    | sed -E "s|^([A-Z_0-9]+) +([0-9]+) +(.*)$|\1\t\2\t\3\t$(basename $f .log)|"
done | grep -v "More than one subgraph" | sort
```
