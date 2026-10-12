"""Custo de recuperar o embedding mais parecido na Coral Dev Board, forca bruta em numpy.

A galeria e (N,512) float32 normalizada, a consulta e um produto interno seguido de
argmax/top-k. Como os embeddings do ArcFace/MobileFaceNet ja saem normalizados, o
cosseno E o produto interno -- nao existe passo de normalizacao por consulta, e a
busca exata cabe num unico GEMV.

Mede latencia por N, com 1 e com 4 threads do OpenBLAS, para 1 consulta (uma face no
frame) e para 4 (o --max-faces tipico). Reporta tambem a RAM da galeria, que na placa
e o limite real: sao 985 MB totais e ~600 MB livres.

Uso: python bench_embedding_search.py [--dim 512] [--topk 5]
"""
import argparse
import os
import time

SIZES = [100, 1_000, 10_000, 100_000]


def bench(gallery, queries, topk, runs):
    import numpy as np
    # aquece o kernel do BLAS
    for _ in range(3):
        gallery @ queries.T
    ts = []
    for _ in range(runs):
        t = time.perf_counter()
        sims = gallery @ queries.T                      # (N, Q)
        if topk == 1:
            sims.argmax(axis=0)
        else:
            np.argpartition(-sims, topk, axis=0)[:topk]
        ts.append(time.perf_counter() - t)
    ts = np.array(ts) * 1000
    return ts.mean(), np.percentile(ts, 95)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dim", type=int, default=512)
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--runs", type=int, default=50)
    ap.add_argument("--threads", type=int, default=None,
                    help="fixa OPENBLAS_NUM_THREADS antes de importar numpy")
    args = ap.parse_args()

    if args.threads:
        os.environ["OPENBLAS_NUM_THREADS"] = str(args.threads)
    import numpy as np

    rng = np.random.default_rng(0)
    thr = os.environ.get("OPENBLAS_NUM_THREADS", "default")
    print(f"dim={args.dim} topk={args.topk} OPENBLAS_NUM_THREADS={thr}")
    print(f"{'N':>8} {'RAM(MB)':>8} {'1 face(ms)':>11} {'P95':>8} "
          f"{'4 faces(ms)':>12} {'P95':>8}")

    for n in SIZES:
        g = rng.standard_normal((n, args.dim), dtype=np.float32)
        g /= np.linalg.norm(g, axis=1, keepdims=True)
        mb = g.nbytes / 1e6
        q1 = g[:1].copy()
        q4 = g[:4].copy()
        m1, p1 = bench(g, q1, args.topk, args.runs)
        m4, p4 = bench(g, q4, args.topk, args.runs)
        print(f"{n:>8} {mb:>8.1f} {m1:>11.3f} {p1:>8.3f} {m4:>12.3f} {p4:>8.3f}")
        del g


if __name__ == "__main__":
    main()
