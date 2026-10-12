from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import time
from typing import Any, Optional

import cv2
import numpy as np

from utils.emotion_inference import DEFAULT_EMOTION_MODEL, EmotionClassifier
from utils.face_pipeline import (
    DEFAULT_YUNET_MODEL,
    detect_faces,
    expand_bbox,
)


EMOTION_COLORS = {
    "anger": (60, 60, 220),
    "disgust": (60, 160, 90),
    "fear": (140, 70, 160),
    "happiness": (40, 200, 230),
    "neutral": (200, 200, 200),
    "sadness": (200, 130, 60),
    "surprise": (40, 220, 255),
}
DEFAULT_COLOR = (0, 200, 0)


def resolve_runtime_device(requested_device: str | None) -> str | None:
    if requested_device:
        return requested_device
    if sys.platform == "darwin":
        return "mps"
    return None


def cuda_is_active(device: str | None) -> bool:
    """True when inference will actually run on a CUDA GPU (fp16 only helps there)."""
    if device is not None:
        return str(device).startswith("cuda")
    try:
        import torch

        return torch.cuda.is_available()
    except Exception:
        return False


def parse_source(source: str) -> Any:
    """Return an int (webcam index) when the source is purely numeric, else the path."""
    if source.isdigit():
        return int(source)
    return source


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inferencia de emocao em video (YuNet + YOLO classify), com display em tempo real."
    )
    parser.add_argument("source", type=str, help="Caminho do video ou indice da webcam (ex.: 0).")
    parser.add_argument("--emotion-model", type=str, default=str(DEFAULT_EMOTION_MODEL))
    parser.add_argument("--emotion-imgsz", type=int, default=224)
    parser.add_argument("--min-emotion-score", type=float, default=0.0,
                        help="Esconde o rotulo de emocao abaixo deste score.")
    parser.add_argument("--padding-ratio", type=float, default=0.15,
                        help="Padding ao redor do bbox antes de classificar a emocao.")
    # Detector (YuNet)
    parser.add_argument("--detector", type=str, default="yunet", choices=["yunet", "mtcnn"])
    parser.add_argument("--yunet-model", type=str, default=str(DEFAULT_YUNET_MODEL))
    parser.add_argument("--yunet-nms", type=float, default=0.3)
    parser.add_argument("--yunet-top-k", type=int, default=5000)
    parser.add_argument("--yunet-backend", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--conf", type=float, default=0.6)
    parser.add_argument("--max-faces", type=int, default=10)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--precision", type=str, default="fp16", choices=["fp16", "fp32"],
                        help="Precisao da inferencia (fp16 e mais rapido em GPU).")
    # Optional identity recognition
    parser.add_argument("--recognize", action="store_true",
                        help="Tambem identifica a pessoa via ArcFace + Qdrant.")
    parser.add_argument("--collection", type=str, default="Rostos")
    parser.add_argument("--qdrant-url", type=str, default="http://localhost:6333")
    parser.add_argument("--qdrant-api-key", type=str, default=None)
    parser.add_argument("--qdrant-timeout", type=float, default=15.0)
    parser.add_argument("--score-threshold", type=float, default=0.45)
    parser.add_argument("--compile-embedder", action="store_true",
                        help="Ativa torch.compile no ArcFace/EdgeFace. Desligado por padrao: "
                             "com numero de rostos variavel ele recompila e causa travadas periodicas.")
    # Video flow
    parser.add_argument("--frame-skip", type=int, default=1,
                        help="Processa 1 a cada N frames (>=1).")
    parser.add_argument("--detect-interval", type=int, default=1,
                        help="Re-detecta a cada N frames processados; reaproveita bboxes nos demais.")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--output-video", type=str, default=None)
    parser.add_argument("--no-display", action="store_true")
    args = parser.parse_args()

    if args.frame_skip < 1:
        parser.error("--frame-skip deve ser maior ou igual a 1.")
    if args.detect_interval < 1:
        parser.error("--detect-interval deve ser maior ou igual a 1.")
    if args.max_frames is not None and args.max_frames < 1:
        parser.error("--max-frames deve ser maior ou igual a 1.")
    if args.start_frame < 0:
        parser.error("--start-frame nao pode ser negativo.")
    return args


def build_recognizer(args: argparse.Namespace):
    from utils.face_pipeline import FaceRecognizer

    return FaceRecognizer(
        qdrant_url=args.qdrant_url,
        qdrant_api_key=args.qdrant_api_key,
        qdrant_timeout=args.qdrant_timeout,
        collection_name=args.collection,
        embedder_precision=args.precision,
        conf=args.conf,
        score_threshold=args.score_threshold,
        device=args.device,
        max_faces=args.max_faces,
        detector_backend=args.detector,
        yunet_model=args.yunet_model,
        yunet_nms=args.yunet_nms,
        yunet_top_k=args.yunet_top_k,
        yunet_backend=args.yunet_backend,
        arcface_compile=args.compile_embedder,
        edgeface_compile=args.compile_embedder,
    )


def detect_bboxes(frame_bgr: np.ndarray, args: argparse.Namespace) -> list[dict[str, Any]]:
    detections = detect_faces(
        image=frame_bgr,
        detector_backend=args.detector,
        device=args.device,
        conf=args.conf,
        max_det=args.max_faces,
        yunet_model=args.yunet_model,
        yunet_nms=args.yunet_nms,
        yunet_top_k=args.yunet_top_k,
        yunet_backend=args.yunet_backend,
    )
    return [{"bbox": [int(v) for v in d["bbox"]], "score": float(d["score"])} for d in detections]


def crop_faces(
    frame_bgr: np.ndarray,
    faces: list[dict[str, Any]],
    padding_ratio: float,
) -> list[np.ndarray]:
    h, w = frame_bgr.shape[:2]
    crops: list[np.ndarray] = []
    for face in faces:
        x1, y1, x2, y2 = expand_bbox(face["bbox"], w, h, padding_ratio)
        crops.append(frame_bgr[y1:y2, x1:x2])
    return crops


def annotate(
    frame_bgr: np.ndarray,
    faces: list[dict[str, Any]],
    min_emotion_score: float,
) -> None:
    for face in faces:
        x1, y1, x2, y2 = face["bbox"]
        emotion = face.get("emotion")
        emo_score = face.get("emotion_score")
        name = face.get("name")
        color = EMOTION_COLORS.get(emotion, DEFAULT_COLOR)

        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, 2)

        lines: list[str] = []
        if name:
            id_score = face.get("identity_score")
            lines.append(f"{name} {id_score:.2f}" if id_score is not None else name)
        if emotion and emo_score is not None and emo_score >= min_emotion_score:
            lines.append(f"{emotion} {emo_score:.2f}")
        elif emotion is None:
            lines.append("?")

        for i, text in enumerate(lines):
            ts, baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            ty2 = max(0, y1 - 4) - i * (ts[1] + baseline + 8)
            ty1 = ty2 - ts[1] - baseline - 6
            cv2.rectangle(frame_bgr, (x1, ty1), (x1 + ts[0] + 10, ty2), color, -1)
            cv2.putText(
                frame_bgr, text, (x1 + 5, ty2 - baseline - 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA,
            )


def draw_status(frame_bgr: np.ndarray, text: str) -> None:
    cv2.putText(
        frame_bgr, text, (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA,
    )


def main() -> None:
    args = parse_args()
    args.device = resolve_runtime_device(args.device)

    use_half = args.precision == "fp16" and cuda_is_active(args.device)
    if args.precision == "fp16" and not use_half:
        print("Aviso: fp16 pedido mas sem CUDA ativa; usando fp32.")
        args.precision = "fp32"

    emotion = EmotionClassifier(
        model_path=args.emotion_model,
        device=args.device,
        imgsz=args.emotion_imgsz,
        half=use_half,
    )
    print(
        f"Modelo de emocao: {args.emotion_model} | classes: {list(emotion.names.values())} | "
        f"precisao: {args.precision}"
    )

    recognizer = build_recognizer(args) if args.recognize else None

    cap = cv2.VideoCapture(parse_source(args.source))
    if not cap.isOpened():
        raise RuntimeError(f"Nao foi possivel abrir a fonte de video: {args.source}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    if args.start_frame:
        cap.set(cv2.CAP_PROP_POS_FRAMES, args.start_frame)

    writer: Optional[cv2.VideoWriter] = None
    raw_index = args.start_frame - 1
    processed = 0
    last_faces: list[dict[str, Any]] = []

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            raw_index += 1
            if (raw_index - args.start_frame) % args.frame_skip != 0:
                continue
            if args.max_frames is not None and processed >= args.max_frames:
                break

            t0 = time.perf_counter()
            do_detect = (processed % args.detect_interval == 0) or not last_faces

            if do_detect:
                if recognizer is not None:
                    recs = recognizer.recognize_frame(frame)
                    faces = [{
                        "bbox": [int(v) for v in r["bbox"]],
                        "name": (r.get("name") if r.get("recognized") else None),
                        "identity_score": r.get("score"),
                    } for r in recs]
                else:
                    faces = detect_bboxes(frame, args)

                crops = crop_faces(frame, faces, args.padding_ratio)
                emo_results = emotion.classify_batch(crops)
                for face, emo in zip(faces, emo_results):
                    face["emotion"] = emo["emotion"]
                    face["emotion_score"] = emo["score"]
                last_faces = faces
            else:
                faces = last_faces

            infer_ms = (time.perf_counter() - t0) * 1000.0
            annotate(frame, faces, args.min_emotion_score)
            fps = 1000.0 / infer_ms if infer_ms > 0 else 0.0
            status = f"Frame {raw_index}"
            if total_frames > 0:
                status += f"/{total_frames - 1}"
            status += f" | faces: {len(faces)} | {infer_ms:.0f} ms ({fps:.1f} FPS)"
            draw_status(frame, status)

            if args.output_video:
                if writer is None:
                    h, w = frame.shape[:2]
                    out_fps = src_fps if src_fps > 0 else 25.0
                    out_fps = max(1.0, out_fps / args.frame_skip)
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = cv2.VideoWriter(args.output_video, fourcc, out_fps, (w, h))
                    if not writer.isOpened():
                        raise RuntimeError(f"Nao foi possivel criar {args.output_video}.")
                writer.write(frame)

            if not args.no_display:
                cv2.imshow("Emotion", frame)
                if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
                    print("Interrompido pelo usuario.")
                    break

            processed += 1
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if not args.no_display:
            cv2.destroyAllWindows()

    print(f"Frames processados: {processed}")
    if args.output_video:
        print(f"Video anotado salvo em: {args.output_video}")


if __name__ == "__main__":
    main()
