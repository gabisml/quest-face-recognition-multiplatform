"""YuNet no CPU da Coral Dev Board, via cv2.FaceDetectorYN.

Mede o detect() puro em varias resolucoes de entrada, o escalonamento por thread,
e o custo do resize isolado (que na placa nao e desprezivel). Baseline de CPU
para comparar com o yunet_tflite_tpu.py.

Uso: python bench_yunet_cpu.py [IMAGEM]   (default: face.jpg ao lado do script)
"""
import sys, time
from pathlib import Path

import numpy as np, cv2

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from utils.yunet_inference import detect_faces_yunet, _compute_yunet_input_size, DEFAULT_YUNET_MODEL

MODEL = str(DEFAULT_YUNET_MODEL)
IMAGE = sys.argv[1] if len(sys.argv) > 1 else str(Path(__file__).resolve().parent / "face.jpg")
base = cv2.imread(IMAGE)                               # 512x355, 4 faces reais
if base is None:
    raise SystemExit(f"nao consegui ler a imagem: {IMAGE}")
hd   = cv2.resize(base, (1280, 720), interpolation=cv2.INTER_CUBIC)
print(f"opencv {cv2.__version__}  modelo {MODEL.split('/')[-1]}")
print(f"imagens: base {base.shape[1]}x{base.shape[0]}, hd {hd.shape[1]}x{hd.shape[0]}\n")

def bench_raw(img, long_side, threads, n=20):
    cv2.setNumThreads(threads)
    h, w = img.shape[:2]
    nw, nh, _, _ = _compute_yunet_input_size(w, h, long_side)
    small = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    det = cv2.FaceDetectorYN.create(MODEL, "", (nw, nh), 0.5, 0.3, 5000,
                                    cv2.dnn.DNN_BACKEND_OPENCV, cv2.dnn.DNN_TARGET_CPU)
    for _ in range(3): det.detect(small)
    ts = []
    for _ in range(n):
        t = time.perf_counter(); _, faces = det.detect(small); ts.append(time.perf_counter() - t)
    ts = np.array(ts) * 1000
    nf = 0 if faces is None else len(faces)
    # custo do resize isolado
    t = time.perf_counter()
    for _ in range(20): cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    rs = (time.perf_counter() - t) / 20 * 1000
    return f"{nw}x{nh}", ts.mean(), np.percentile(ts, 95), rs, nf

print("=== detect() puro, 4 threads (imagem 1280x720 -> resize) ===")
print(f"{'long_side':>9} {'shape real':>10} {'mean(ms)':>9} {'P95':>8} {'resize(ms)':>11} {'FPS':>6} {'faces':>6}")
for ls in (640, 480, 320, 160):
    sh, m, p95, rs, nf = bench_raw(hd, ls, 4)
    print(f"{ls:>9} {sh:>10} {m:>9.1f} {p95:>8.1f} {rs:>11.2f} {1000/m:>6.1f} {nf:>6}")

print("\n=== escalonamento por thread (long_side 640, 1280x720) ===")
print(f"{'threads':>7} {'mean(ms)':>9} {'FPS':>6}")
for th in (1, 2, 4):
    sh, m, p95, rs, nf = bench_raw(hd, 640, th)
    print(f"{th:>7} {m:>9.1f} {1000/m:>6.1f}")

print("\n=== imagem nativa 512x355, 4 threads ===")
sh, m, p95, rs, nf = bench_raw(base, 640, 4)
print(f"shape {sh}: {m:.1f} ms ({1000/m:.1f} FPS), {nf} faces")

print("\n=== ponta a ponta pelo detect_faces_yunet() do repo (1280x720, 4 threads) ===")
cv2.setNumThreads(4)
for ls in (640, 320):
    for _ in range(2): detect_faces_yunet(hd, backend="cpu", input_long_side=ls)
    t = time.perf_counter()
    for _ in range(10): r = detect_faces_yunet(hd, backend="cpu", input_long_side=ls)
    dt = (time.perf_counter() - t) / 10 * 1000
    lm = "sim" if r and r[0].get("landmarks") is not None else ("sim" if r and "landmarks" in r[0] else "?")
    print(f"long_side={ls}: {dt:.1f} ms ({1000/dt:.1f} FPS), {len(r)} faces, landmarks={lm}")
    if r: print(f"   chaves do retorno: {sorted(r[0].keys())}")

print("\n=== referencia: custo do resize 1280x720 -> 320x320 (pre-proc da TPU) ===")
t = time.perf_counter()
for _ in range(50): cv2.resize(hd, (320, 320), interpolation=cv2.INTER_AREA)
print(f"{(time.perf_counter()-t)/50*1000:.2f} ms")
