"""Reescala um video para uma resolucao fixa, ANTES do benchmark.

O resize fica fora da medicao de proposito: no e2e_video_tpu.py o que se mede e
decode + detect + align + embed + search, e um resize por frame poluiria o numero
do decode. Entao o arquivo ja chega na resolucao alvo.

Nao usa ffmpeg (a imagem da placa nao tem), so cv2.VideoWriter com mp4v.

Uso: python resize_video.py ENTRADA.mp4 SAIDA.mp4 [--width 640] [--height 480] [--frames 900]
"""
import argparse
import time
from pathlib import Path

import cv2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--frames", type=int, default=0, help="0 = video inteiro")
    ap.add_argument("--interpolation", default="area", choices=["area", "linear", "cubic"])
    args = ap.parse_args()

    interp = {"area": cv2.INTER_AREA, "linear": cv2.INTER_LINEAR,
              "cubic": cv2.INTER_CUBIC}[args.interpolation]

    cap = cv2.VideoCapture(args.src)
    if not cap.isOpened():
        raise SystemExit(f"nao abriu: {args.src}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    sw, sh = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    want = args.frames or total
    print(f"{Path(args.src).name}: {sw}x{sh} {fps:.0f} fps {total} frames")
    print(f"  -> {args.width}x{args.height}, {want} frames, interp={args.interpolation}")
    if abs(sw / sh - args.width / args.height) > 0.01:
        print(f"  aviso: proporcao muda ({sw / sh:.3f} -> {args.width / args.height:.3f}), "
              f"a imagem fica distorcida")

    out = cv2.VideoWriter(args.dst, cv2.VideoWriter_fourcc(*"mp4v"), fps,
                          (args.width, args.height))
    if not out.isOpened():
        raise SystemExit("VideoWriter nao abriu (codec mp4v indisponivel?)")

    n, t0 = 0, time.perf_counter()
    while n < want:
        ok, frame = cap.read()
        if not ok:
            break
        out.write(cv2.resize(frame, (args.width, args.height), interpolation=interp))
        n += 1
        if n % 200 == 0:
            print(f"  {n}/{want}  {n / (time.perf_counter() - t0):.1f} frames/s")
    cap.release()
    out.release()
    print(f"  pronto: {n} frames em {time.perf_counter() - t0:.1f} s "
          f"-> {Path(args.dst).stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
