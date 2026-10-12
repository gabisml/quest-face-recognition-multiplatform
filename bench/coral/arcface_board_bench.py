"""Roda o ArcFace int8 na placa: latencia na Edge TPU, no CPU, e cross-check numerico.

Espera no mesmo diretorio (ficam na placa em /home/mendel/data/arcface-tpu --
ver README.md):
  arcface_r100_int8.tflite           (CPU)
  arcface_r100_int8_edgetpu.tflite   (TPU, se compilou)
  eval_subset_uint8.npy              (N,112,112,3) uint8 ja quantizado no WSL
  emb_ref_int8.npy                   (N,512) embeddings do mesmo int8 rodado no WSL

Uso: python 06_board_bench.py [--runs 30]
"""
import argparse
import time
from pathlib import Path

import numpy as np

try:
    import tflite_runtime.interpreter as tfl
except ImportError:
    import tensorflow.lite as tfl

W = Path(__file__).resolve().parent


def build(model, edgetpu, threads=1):
    kw = {"model_path": str(model), "num_threads": threads}
    if edgetpu:
        kw["experimental_delegates"] = [tfl.load_delegate("libedgetpu.so.1")]
        kw.pop("num_threads")
    it = tfl.Interpreter(**kw)
    it.allocate_tensors()
    return it


def embed(it, X):
    d, o = it.get_input_details()[0], it.get_output_details()[0]
    s, z = o["quantization"]
    out = []
    for i in range(X.shape[0]):
        it.set_tensor(d["index"], X[i:i + 1])
        it.invoke()
        y = it.get_tensor(o["index"])[0].astype(np.float32)
        if s:
            y = (y - z) * s
        out.append(y)
    e = np.stack(out)
    return e / np.linalg.norm(e, axis=1, keepdims=True)


def bench(it, X, runs):
    d = it.get_input_details()[0]
    it.set_tensor(d["index"], X[0:1])
    for _ in range(3):
        it.invoke()
    t = time.perf_counter()
    for _ in range(runs):
        it.invoke()
    return (time.perf_counter() - t) / runs


ap = argparse.ArgumentParser()
ap.add_argument("--runs", type=int, default=30)
args = ap.parse_args()

X = np.load(W / "eval_subset_uint8.npy")
ref = np.load(W / "emb_ref_int8.npy")
print(f"subset: {X.shape} {X.dtype}")

for name, edgetpu, threads in [
    ("arcface_r100_int8_edgetpu.tflite", True, 1),
    ("arcface_r100_int8.tflite", False, 4),
    ("arcface_r100_int8.tflite", False, 1),
]:
    p = W / name
    if not p.exists():
        print(f"\n{name}: ausente, pulando")
        continue
    tag = "EDGE TPU" if edgetpu else f"CPU {threads} thread(s)"
    print(f"\n### {tag}  ({name}, {p.stat().st_size/1e6:.1f} MB)")
    try:
        it = build(p, edgetpu, threads)
    except Exception as e:
        print(f"  falhou ao carregar: {type(e).__name__}: {str(e)[:200]}")
        continue
    dt = bench(it, X, args.runs)
    print(f"  invoke: {dt*1000:.2f} ms  ({1/dt:.2f} FPS)")
    e = embed(it, X)
    cos = (e * ref).sum(1)
    print(f"  cosseno vs mesmo int8 no WSL: mean {cos.mean():.6f} min {cos.min():.6f}")
