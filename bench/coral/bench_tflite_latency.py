"""Mede latencia de um .tflite na placa, na Edge TPU ou no CPU.

Entrada aleatoria: a latencia e valida, a acuracia nao (por isso nao e reportada
aqui). Para acuracia use mobilefacenet_board_bench.py ou arcface_board_bench.py,
que carregam faces reais e embeddings de referencia.

O numero de invokes se adapta ao modelo: modelos que mapeiam pouco na TPU rodam
quase tudo no CPU e levam centenas de ms por invoke, entao medir 200 vezes seria
inviavel. Depois do warm-up o script estima o custo e escolhe quantas repeticoes
cabem em ~5 s, com no minimo 10.

Uso:
    python bench_tflite_latency.py MODELO_edgetpu.tflite [OUTRO.tflite ...]
    python bench_tflite_latency.py --cpu [--threads 4] MODELO_int8.tflite
"""
import argparse
import time

import numpy as np
import tflite_runtime.interpreter as tfl

BUDGET_S = 5.0


def build(model, use_cpu, threads):
    if use_cpu:
        return tfl.Interpreter(model_path=model, num_threads=threads)
    return tfl.Interpreter(model_path=model,
                           experimental_delegates=[tfl.load_delegate("libedgetpu.so.1")])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="+")
    ap.add_argument("--cpu", action="store_true", help="sem delegate da Edge TPU")
    ap.add_argument("--threads", type=int, default=4, help="threads do CPU (so com --cpu)")
    ap.add_argument("--runs", type=int, default=200, help="teto de invokes")
    args = ap.parse_args()

    tag = f"CPU {args.threads}t" if args.cpu else "TPU"
    for M in args.models:
        name = M.split("/")[-1]
        try:
            it = build(M, args.cpu, args.threads)
            it.allocate_tensors()
        except Exception as e:
            print(f"{name:<44} [{tag}] FALHOU {type(e).__name__}: {str(e)[:120]}")
            continue
        d, o = it.get_input_details()[0], it.get_output_details()[0]
        x = np.random.randint(0, 255, size=d["shape"]).astype(d["dtype"])
        it.set_tensor(d["index"], x)
        try:
            t = time.perf_counter()
            for _ in range(3):
                it.invoke()
            warm = (time.perf_counter() - t) / 3
        except Exception as e:
            print(f"{name:<44} [{tag}] invoke FALHOU {type(e).__name__}: {str(e)[:120]}")
            continue
        n = max(10, min(args.runs, int(BUDGET_S / warm) if warm > 0 else args.runs))
        ts = []
        for _ in range(n):
            t = time.perf_counter()
            it.invoke()
            ts.append(time.perf_counter() - t)
        ts = np.array(ts) * 1000
        print(f"{name:<44} [{tag}] in={list(d['shape'])} out={list(o['shape'])}  "
              f"mean {ts.mean():8.2f} ms  P95 {np.percentile(ts, 95):8.2f}  "
              f"-> {1000 / ts.mean():8.1f} FPS  (n={n})")


if __name__ == "__main__":
    main()
