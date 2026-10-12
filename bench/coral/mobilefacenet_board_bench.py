"""MobileFaceNet (w600k) na Edge TPU: latencia + acuracia das 490 faces na placa.

Espera no diretorio atual, que na placa e bench/coral/models/ (artefatos gerados
no WSL, nao versionados -- ver README.md):
  mbf_w600k_int8_edgetpu.tflite / mbf_w600k_int8.tflite
  mbf_eval_uint8.npy   (490,112,112,3) uint8 = 245 pares
  mbf_emb_fp32.npy     (490,512) embeddings fp32 de referencia
  mbf_pairs.npy        (2,245) similaridades pos/neg do baseline fp32
  baseline_perm.npy    permutacao que gera os pares negativos

Uso: cd bench/coral/models && python ../mobilefacenet_board_bench.py
"""
import time
import numpy as np
import tflite_runtime.interpreter as tfl

X = np.load("mbf_eval_uint8.npy")
ref = np.load("mbf_emb_fp32.npy")
perm = np.load("baseline_perm.npy")
pos0, neg0 = np.load("mbf_pairs.npy")
P = X.shape[0] // 2
lab = np.concatenate([np.ones(P), np.zeros(P)])


def acc(p, n):
    s = np.concatenate([p, n])
    return max((lab == (s >= t)).mean() for t in np.linspace(s.min(), s.max(), 1000))


for M, tpu, th in [("mbf_w600k_int8_edgetpu.tflite", True, 1),
                   ("mbf_w600k_int8.tflite", False, 4),
                   ("mbf_w600k_int8.tflite", False, 1)]:
    kw = {"model_path": M}
    if tpu:
        kw["experimental_delegates"] = [tfl.load_delegate("libedgetpu.so.1")]
    else:
        kw["num_threads"] = th
    try:
        it = tfl.Interpreter(**kw); it.allocate_tensors()
    except Exception as e:
        print(f"{M} tpu={tpu}: FALHOU {type(e).__name__}: {str(e)[:100]}"); continue
    d, o = it.get_input_details()[0], it.get_output_details()[0]
    s, z = o["quantization"]
    it.set_tensor(d["index"], X[0:1])
    for _ in range(10):
        it.invoke()
    ts = []
    for _ in range(200 if tpu else 30):
        t = time.perf_counter(); it.invoke(); ts.append(time.perf_counter() - t)
    ts = np.array(ts) * 1000
    tag = "EDGE TPU" if tpu else f"CPU {th}t"
    print(f"\n### {tag}: mean {ts.mean():.2f} ms  P95 {np.percentile(ts,95):.2f}  -> {1000/ts.mean():.1f} FPS")
    if not tpu:
        continue
    t0 = time.perf_counter()
    emb = []
    for i in range(X.shape[0]):
        it.set_tensor(d["index"], X[i:i + 1]); it.invoke()
        y = it.get_tensor(o["index"])[0].astype(np.float32)
        emb.append((y - z) * s if s else y)
    e = np.stack(emb); e = e / np.linalg.norm(e, axis=1, keepdims=True)
    print(f"  {X.shape[0]} faces em {time.perf_counter()-t0:.1f} s")
    drift = (e * ref).sum(1)
    pos, neg = (e[:P] * e[P:]).sum(1), (e[:P] * e[P:][perm]).sum(1)
    print(f"  drift vs fp32: mean {drift.mean():.4f} p5 {np.percentile(drift,5):.4f}")
    print(f"  fp32 {acc(pos0,neg0)*100:.2f}%  ->  TPU int8 {acc(pos,neg)*100:.2f}%")
