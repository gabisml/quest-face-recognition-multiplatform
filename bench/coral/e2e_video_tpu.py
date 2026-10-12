"""Benchmark e2e em video na Coral Dev Board: deteccao -> alinhamento -> embedding -> busca.

Por que nao usa o bench/e2e_video.py: aquele importa bench.system_monitor (torch) e
pandas, e a placa nao tem nenhum dos dois. Aqui a cadeia inteira e TFLite + numpy + cv2.

Etapas medidas separadamente, por frame:
    decode   cv2.VideoCapture.read()
    detect   YuNet TFLite 120x160 (Edge TPU ou CPU), inclui resize e o decode dos priors
    align    transformacao de similaridade dos 5 pontos -> 112x112, por face
    embed    MobileFaceNet w600k int8 na Edge TPU, por face
    search   produto interno contra a galeria + argmax, por frame

Com --warmup-s e --duration-s a janela fica igual a do resto da suite (bench/config.py:
VIDEO_WARMUP_S=60, MIN_VIDEO_DURATION_S=300), o que torna o numero comparavel com os
outros alvos. Nesse modo tambem sai o perfil termico e a comparacao entre o inicio e o
fim da janela, que e a parte derivavel do Grupo D (ver bench/robustness.py).

O alinhamento replica estimate_arcface_transform de utils/arcface_inference.py. E copia
deliberada: aquele modulo importa utils.local_arcface_backbones, que puxa torch. Os
pontos de referencia e o metodo (LMEDS) sao os mesmos, senao os embeddings nao seriam
comparaveis com o baseline fp32 do WSL.

Uso:
    python e2e_video_tpu.py VIDEO.mp4 --embedder models/mbf_w600k_int8_edgetpu.tflite \\
        --yunet ~/data/yunet-tpu/yunet_120x160_int8_out_uint8.tflite --detector cpu \\
        --warmup-s 60 --duration-s 300 --loop
"""
import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from yunet_tflite_tpu import YuNet  # noqa: E402

import tflite_runtime.interpreter as tfl  # noqa: E402

# Pontos de referencia do ArcFace num recorte 112x112.
ARC_FACE_DST = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041],
], dtype=np.float32)

TEMP_ZONE = "/sys/class/thermal/thermal_zone0/temp"
TEMP_INTERVAL_S = 2.0


def read_temp(path=TEMP_ZONE):
    """Temperatura do SoC em graus C, ou NaN se o sysfs nao existir."""
    try:
        with open(path) as fh:
            return int(fh.read().strip()) / 1000.0
    except (OSError, ValueError):
        return float("nan")


def align(frame_bgr, kps5, size=112):
    m, _ = cv2.estimateAffinePartial2D(np.asarray(kps5, dtype=np.float32),
                                      ARC_FACE_DST, method=cv2.LMEDS)
    if m is None:
        return None
    return cv2.warpAffine(frame_bgr, m, (size, size), borderValue=0.0)


class Embedder:
    """MobileFaceNet int8. Entrada [-1,1] quantizada com o scale/zp do proprio modelo."""

    def __init__(self, model_path, use_tpu=True, threads=4):
        kw = {"model_path": model_path}
        if use_tpu:
            kw["experimental_delegates"] = [tfl.load_delegate("libedgetpu.so.1")]
        else:
            kw["num_threads"] = threads
        self.it = tfl.Interpreter(**kw)
        self.it.allocate_tensors()
        self.d = self.it.get_input_details()[0]
        self.o = self.it.get_output_details()[0]
        self.in_scale, self.in_zp = self.d["quantization"]
        self.out_scale, self.out_zp = self.o["quantization"]
        self.info = np.iinfo(self.d["dtype"])

    def __call__(self, aligned_bgr):
        rgb = cv2.cvtColor(aligned_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
        x = (rgb / 255.0 - 0.5) / 0.5                      # [-1, 1], igual ao baseline fp32
        if self.in_scale:
            x = np.clip(np.round(x / self.in_scale + self.in_zp),
                        self.info.min, self.info.max)
        self.it.set_tensor(self.d["index"], x[None].astype(self.d["dtype"]))
        self.it.invoke()
        y = self.it.get_tensor(self.o["index"])[0].astype(np.float32)
        if self.out_scale:
            y = (y - self.out_zp) * self.out_scale
        n = np.linalg.norm(y)
        return y / n if n else y


def grab(cap, loop):
    """Le um frame, reiniciando o video se acabar e loop estiver ligado."""
    t = time.perf_counter()
    ok, frame = cap.read()
    dec = (time.perf_counter() - t) * 1000
    if not ok and loop:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        t = time.perf_counter()
        ok, frame = cap.read()
        dec = (time.perf_counter() - t) * 1000
    return ok, frame, dec


def one_frame(cap, det, emb, gallery, max_faces, loop):
    """Um frame completo. Devolve (ok, timings_ms, n_embeddings)."""
    ok, frame, dec = grab(cap, loop)
    if not ok:
        return False, None, 0

    t = time.perf_counter()
    faces = det.detect(frame)[:max_faces]
    dt_det = (time.perf_counter() - t) * 1000

    embs, dt_ali, dt_emb = [], 0.0, 0.0
    for f in faces:
        t = time.perf_counter()
        crop = align(frame, f["kps"])
        dt_ali += (time.perf_counter() - t) * 1000
        if crop is None:
            continue
        t = time.perf_counter()
        embs.append(emb(crop))
        dt_emb += (time.perf_counter() - t) * 1000

    t = time.perf_counter()
    if embs:
        sims = gallery @ np.stack(embs).T
        sims.argmax(axis=0)
    dt_sea = (time.perf_counter() - t) * 1000

    return True, (dec, dt_det, dt_ali, dt_emb, dt_sea), len(embs)


def stats(ms):
    a = np.asarray(ms)
    if not a.size:
        return "        -        -        -"
    return f"{a.mean():8.2f} {np.percentile(a, 95):8.2f} {a.sum() / 1000:8.2f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--yunet", required=True)
    ap.add_argument("--embedder", required=True)
    ap.add_argument("--detector", choices=["tpu", "cpu"], default="cpu")
    ap.add_argument("--embedder-device", choices=["tpu", "cpu"], default="tpu")
    ap.add_argument("--conf", type=float, default=0.6)
    ap.add_argument("--max-faces", type=int, default=4)
    ap.add_argument("--gallery-size", type=int, default=50)
    ap.add_argument("--dim", type=int, default=512)
    ap.add_argument("--warmup-s", type=float, default=0.0,
                    help="warm-up por tempo; sobrepoe --warmup-frames")
    ap.add_argument("--duration-s", type=float, default=0.0,
                    help="janela medida por tempo; sobrepoe --frames")
    ap.add_argument("--warmup-frames", type=int, default=15)
    ap.add_argument("--frames", type=int, default=300)
    ap.add_argument("--loop", action="store_true", help="reinicia o video se acabar antes")
    args = ap.parse_args()

    det = YuNet(args.yunet, use_edgetpu=args.detector == "tpu", conf_th=args.conf)
    emb = Embedder(args.embedder, use_tpu=args.embedder_device == "tpu")
    rng = np.random.default_rng(0)
    gallery = rng.standard_normal((args.gallery_size, args.dim), dtype=np.float32)
    gallery /= np.linalg.norm(gallery, axis=1, keepdims=True)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"nao abriu o video: {args.video}")
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"video    : {Path(args.video).name} {w}x{h} @ {cap.get(cv2.CAP_PROP_FPS):.0f} fps")
    print(f"detector : YuNet {det.w}x{det.h} em {args.detector.upper()}")
    print(f"embedder : {Path(args.embedder).name} em {args.embedder_device.upper()}")
    print(f"galeria  : {args.gallery_size} x {args.dim} ({gallery.nbytes / 1024:.0f} KB)")
    if args.warmup_s or args.duration_s:
        print(f"janela   : warm-up {args.warmup_s:.0f} s + {args.duration_s:.0f} s medidos"
              f"{', com loop' if args.loop else ''}")

    # --- warm-up, descartado ---
    temp_pre = read_temp()
    if args.warmup_s > 0:
        t0 = time.perf_counter()
        n_warm = 0
        while time.perf_counter() - t0 < args.warmup_s:
            ok, _, _ = one_frame(cap, det, emb, gallery, args.max_faces, args.loop)
            if not ok:
                break
            n_warm += 1
        print(f"warm-up  : {n_warm} frames em {time.perf_counter() - t0:.1f} s, "
              f"temp {temp_pre:.1f} -> {read_temp():.1f} C")
    else:
        for _ in range(args.warmup_frames):
            if not one_frame(cap, det, emb, gallery, args.max_faces, args.loop)[0]:
                break

    # --- janela medida ---
    t_dec, t_det, t_ali, t_emb, t_sea, t_at = [], [], [], [], [], []
    n_faces = 0
    temp_start = read_temp()
    temp_peak = temp_start
    next_temp = time.perf_counter() + TEMP_INTERVAL_S
    t_wall = time.perf_counter()

    while True:
        now = time.perf_counter()
        if args.duration_s > 0:
            if now - t_wall >= args.duration_s:
                break
        elif args.frames and len(t_dec) >= args.frames:
            break
        ok, tm, n = one_frame(cap, det, emb, gallery, args.max_faces, args.loop)
        if not ok:
            break
        t_dec.append(tm[0]); t_det.append(tm[1]); t_ali.append(tm[2])
        t_emb.append(tm[3]); t_sea.append(tm[4])
        t_at.append(now - t_wall)
        n_faces += n
        if now >= next_temp:
            temp_peak = max(temp_peak, read_temp())
            next_temp = now + TEMP_INTERVAL_S

    cap.release()
    wall = time.perf_counter() - t_wall
    temp_end = read_temp()
    temp_peak = max(temp_peak, temp_end)
    n_frames = len(t_dec)
    if not n_frames:
        raise SystemExit("nenhum frame medido")
    per_frame = (np.array(t_dec) + np.array(t_det) + np.array(t_ali)
                 + np.array(t_emb) + np.array(t_sea))

    print(f"\nframes medidos: {n_frames}  faces: {n_faces} ({n_faces / n_frames:.2f}/frame)")
    print(f"{'etapa':<10} {'mean(ms)':>8} {'P95':>8} {'total(s)':>8}")
    for name, v in (("decode", t_dec), ("detect", t_det), ("align", t_ali),
                    ("embed", t_emb), ("search", t_sea)):
        print(f"{name:<10} {stats(v)}")
    print(f"{'soma':<10} {per_frame.mean():8.2f} {np.percentile(per_frame, 95):8.2f} "
          f"{per_frame.sum() / 1000:8.2f}")

    print(f"\nlatencia  p50 {np.median(per_frame):.2f} ms  p95 "
          f"{np.percentile(per_frame, 95):.2f}  max {per_frame.max():.2f}")
    for budget in (33.0, 66.0, 100.0):
        print(f"  frac_in_{int(budget)}ms: {(per_frame <= budget).mean():.3f}")
    print(f"\nfps_mean (soma das etapas): {1000 / per_frame.mean():.2f}")
    print(f"fps_wall (relogio de parede): {n_frames / wall:.2f}  em {wall:.1f} s")

    # inicio vs fim da janela: a parte derivavel do Grupo D
    if n_frames >= 40:
        d = max(10, n_frames // 10)
        at = np.array(t_at)
        fps_ini = d / (at[d - 1] - at[0]) if at[d - 1] > at[0] else float("nan")
        fps_fim = d / (at[-1] - at[-d]) if at[-1] > at[-d] else float("nan")
        print(f"\nfps 1o decil {fps_ini:.2f} -> ultimo decil {fps_fim:.2f} "
              f"(razao {fps_fim / fps_ini:.3f})")
        print(f"latencia p50 1o decil {np.median(per_frame[:d]):.2f} -> "
              f"ultimo {np.median(per_frame[-d:]):.2f} ms")
    print(f"temp SoC  {temp_start:.1f} -> {temp_end:.1f} C  "
          f"(delta {temp_end - temp_start:+.1f}, pico {temp_peak:.1f})")


if __name__ == "__main__":
    main()
