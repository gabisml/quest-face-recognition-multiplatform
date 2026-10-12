"""Mapeia o grafo de um .tflite e aponta o que quebraria na compilacao para a Edge TPU.

Existe porque o edgetpu_compiler as vezes nao chega a emitir relatorio por operador: se o
modelo tiver CONV_2D versao 6 ele falha no parser, com "Invalid model / Model could not be
parsed", e nao se descobre mais nada. Foi o que aconteceu com o EdgeFace. Este script le o
flatbuffer direto pelo schema, que nao se importa com versao de op, e reconstroi o
diagnostico que o compilador nao deu.

Tres classes de problema, todas derivadas dos 16 logs em bench/coral/EDGETPU_OPS.md:

  1. VERSAO   builtin_version >= 6 -- barra no parser do compilador 16.0 e no
              tflite-runtime 2.7.0 da placa. Fatal, e mata o modelo inteiro.
  2. OP       operador que nunca apareceu como "Mapped to Edge TPU" em nenhum modelo.
  3. RANK     tensor com mais de 3 dimensoes internas (rank > 4 contando o batch),
              que o compilador recusa com "Tensor has unsupported rank".

Uso: python map_tflite_graph.py MODELO.tflite [--top 25] [--per-op]
"""
import argparse
from collections import Counter, defaultdict
from pathlib import Path

from tflite.BuiltinOperator import BuiltinOperator
from tflite.Model import Model
from tflite.TensorType import TensorType

# Ops que apareceram como "Mapped to Edge TPU" em pelo menos um dos 16 modelos medidos.
# Ver bench/coral/EDGETPU_OPS.md. Nao e a lista oficial da Google: e o que este
# compilador aceitou nestes modelos.
MAPEADAS = {
    "ADD", "CONCATENATION", "CONV_2D", "DEPTHWISE_CONV_2D", "FULLY_CONNECTED",
    "LOGISTIC", "MAX_POOL_2D", "MEAN", "MUL", "PAD", "PRELU", "QUANTIZE", "RELU",
    "RESHAPE", "RESIZE_NEAREST_NEIGHBOR", "SLICE", "SOFTMAX", "SPLIT", "TRANSPOSE",
}
# Recusas explicitas observadas, com a mensagem do compilador.
RECUSADAS = {
    "PADV2": "Operation not supported",
    "ABS": "Operation version not supported",
    "DEQUANTIZE": "Operation is working on an unsupported data type",
}
VERSAO_MAX = 5          # >= 6 nao parseia
RANK_MAX = 4            # batch + 3 dimensoes internas

OP_NAMES = {v: k for k, v in vars(BuiltinOperator).items() if isinstance(v, int)}
TYPE_NAMES = {v: k for k, v in vars(TensorType).items() if isinstance(v, int)}


def load(path):
    buf = bytearray(Path(path).read_bytes())
    return Model.GetRootAsModel(buf, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--top", type=int, default=25, help="quantas ops listar")
    ap.add_argument("--per-op", action="store_true", help="lista cada instancia problematica")
    args = ap.parse_args()

    m = load(args.model)
    path = Path(args.model)
    print(f"modelo   : {path.name}  ({path.stat().st_size / 1e6:.2f} MB)")
    print(f"subgrafos: {m.SubgraphsLength()}   opcodes distintos: {m.OperatorCodesLength()}")

    # versao declarada por opcode
    versoes = {}
    for i in range(m.OperatorCodesLength()):
        oc = m.OperatorCodes(i)
        name = OP_NAMES.get(oc.BuiltinCode(), f"CUSTOM/{oc.BuiltinCode()}")
        cust = oc.CustomCode()
        if cust:
            name = f"CUSTOM:{cust.decode()}"
        versoes[i] = (name, oc.Version())

    contagem = Counter()
    ranks = defaultdict(Counter)          # op -> Counter de rank de saida
    problemas = defaultdict(list)         # (classe, op) -> [detalhes]

    for s in range(m.SubgraphsLength()):
        sg = m.Subgraphs(s)
        for i in range(sg.OperatorsLength()):
            op = sg.Operators(i)
            name, ver = versoes[op.OpcodeIndex()]
            contagem[name] += 1

            rank_max_visto = 0
            dtypes = set()
            for j in range(op.OutputsLength()):
                t = sg.Tensors(op.Outputs(j))
                shape = [t.Shape(k) for k in range(t.ShapeLength())]
                rank_max_visto = max(rank_max_visto, len(shape))
                dtypes.add(TYPE_NAMES.get(t.Type(), str(t.Type())))
                ranks[name][len(shape)] += 1

            if ver > VERSAO_MAX:
                problemas[("VERSAO", name)].append(f"v{ver}")
            if name in RECUSADAS:
                problemas[("OP", name)].append(RECUSADAS[name])
            elif name not in MAPEADAS and not name.startswith("CUSTOM"):
                problemas[("OP", name)].append("nunca mapeou em nenhum modelo medido")
            elif name.startswith("CUSTOM"):
                problemas[("OP", name)].append("operador custom")
            if rank_max_visto > RANK_MAX:
                problemas[("RANK", name)].append(f"rank {rank_max_visto} ({','.join(sorted(dtypes))})")

    total = sum(contagem.values())
    print(f"operadores: {total}\n")

    print(f"{'op':<26} {'n':>5} {'ver':>4}  {'ranks de saida':<18} situacao")
    for name, n in contagem.most_common(args.top):
        ver = next((v for nm, v in versoes.values() if nm == name), "?")
        rk = ",".join(f"{r}d:{c}" for r, c in sorted(ranks[name].items()))
        classes = sorted({c for (c, nm) in problemas if nm == name})
        sit = "+".join(classes) if classes else "ok"
        flag = " " if sit == "ok" else "!"
        print(f"{flag}{name:<25} {n:>5} {ver:>4}  {rk:<18} {sit}")
    if len(contagem) > args.top:
        print(f"  ... e mais {len(contagem) - args.top} ops")

    print("\n=== diagnostico ===")
    fatal = [(c, nm, d) for (c, nm), d in problemas.items() if c == "VERSAO"]
    if fatal:
        print("FATAL no parser (o compilador nem chega a mapear):")
        for _, nm, d in sorted(fatal):
            print(f"  {nm:<24} {len(d):>5} instancias, versao {sorted(set(d))[0]}")
    for classe, titulo in (("OP", "Operador sem suporte na Edge TPU"),
                           ("RANK", "Rank alto (mais de 3 dimensoes internas)")):
        itens = [(nm, d) for (c, nm), d in problemas.items() if c == classe]
        if not itens:
            continue
        print(f"\n{titulo}:")
        for nm, d in sorted(itens, key=lambda x: -len(x[1])):
            print(f"  {nm:<24} {len(d):>5} instancias  {sorted(set(d))[0]}")
            if args.per_op:
                for det in sorted(set(d)):
                    print(f"      {det}")
    if not problemas:
        print("nada barraria: todas as ops mapearam em algum modelo medido, "
              "versoes <= 5 e rank <= 4")

    ok = sum(n for nm, n in contagem.items()
             if not any(nm == x for (_, x) in problemas))
    print(f"\nsem problema apontado: {ok}/{total} ops ({ok / total * 100:.1f}%)")


if __name__ == "__main__":
    main()
