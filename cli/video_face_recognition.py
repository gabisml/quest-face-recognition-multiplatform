from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import time
from typing import Any

import cv2
import numpy as np
from imutils.video import FileVideoStream

from utils.drawing import draw_recognitions
from utils.edgeface_inference import DEFAULT_EDGEFACE_CHECKPOINT, DEFAULT_EDGEFACE_MODEL
from utils.facelivt_inference import DEFAULT_FACELIVT_CHECKPOINT, DEFAULT_FACELIVT_MODEL
from utils.face_pipeline import DEFAULT_ARCFACE_WEIGHT, DEFAULT_YUNET_MODEL, FaceRecognizer
from utils.identity_matching import group_matches_by_identity
from utils.pipeline_artifacts import save_pipeline_presentation_artifacts


def resolve_runtime_device(requested_device: str | None) -> str | None:
    if requested_device:
        return requested_device
    if sys.platform == "darwin":
        return "mps"
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reconhecimento facial em arquivos de video usando Qdrant."
    )
    parser.add_argument("video_path", type=str)
    parser.add_argument("--collection", type=str, default="Rostos")
    parser.add_argument("--qdrant-url", type=str, default="http://localhost:6333")
    parser.add_argument("--qdrant-api-key", type=str, default=None)
    parser.add_argument("--qdrant-timeout", type=float, default=15.0)
    parser.add_argument("--arcface-weight", type=str, default=str(DEFAULT_ARCFACE_WEIGHT))
    parser.add_argument("--arcface-network", type=str, default="r100")
    parser.add_argument("--embedder", type=str, default="arcface", choices=["arcface", "edgeface", "facelivt", "onnx", "tflite"])
    parser.add_argument("--onnx-model", type=str, default=None,
                   help="modelo .onnx para --embedder-backend onnx")
    parser.add_argument("--tflite-model", type=str, default=None,
                   help="modelo .tflite para --embedder-backend tflite")
    parser.add_argument("--tflite-cpu", action="store_true",
                   help="desliga o delegate da Edge TPU no backend tflite")
    parser.add_argument("--detector", type=str, default="mtcnn", choices=["mtcnn", "yunet"])
    parser.add_argument("--yunet-model", type=str, default=str(DEFAULT_YUNET_MODEL))
    parser.add_argument("--yunet-nms", type=float, default=0.3)
    parser.add_argument("--yunet-top-k", type=int, default=5000)
    parser.add_argument("--yunet-backend", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--embedder-precision", type=str, default="fp16", choices=["fp32", "fp16"])
    parser.add_argument("--edgeface-model", type=str, default=DEFAULT_EDGEFACE_MODEL)
    parser.add_argument("--edgeface-checkpoint", type=str, default=str(DEFAULT_EDGEFACE_CHECKPOINT))
    parser.add_argument("--no-edgeface-compile", action="store_true")
    parser.add_argument("--no-arcface-compile", action="store_true")
    parser.add_argument("--facelivt-model", type=str, default=DEFAULT_FACELIVT_MODEL)
    parser.add_argument("--facelivt-checkpoint", type=str, default=str(DEFAULT_FACELIVT_CHECKPOINT))
    parser.add_argument("--no-facelivt-compile", action="store_true")
    parser.add_argument("--conf", type=float, default=None)
    parser.add_argument("--score-threshold", type=float, default=0.45)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--frame-skip", type=int, default=1)
    parser.add_argument("--max-faces", type=int, default=5)
    parser.add_argument("--detect-width", type=int, default=640)
    parser.add_argument("--disable-alignment", action="store_true")
    parser.add_argument("--show-stage-timings", action="store_true")
    parser.add_argument("--output-video", type=str, default=None)
    parser.add_argument("--no-display", action="store_true")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--require-alignment", action="store_true")
    parser.add_argument("--detect-interval", type=int, default=3)
    parser.add_argument("--disable-temporal-cache", action="store_true")
    parser.add_argument("--temporal-iou-threshold", type=float, default=0.9)
    parser.add_argument("--candidate-limit", type=int, default=5)
    parser.add_argument("--score-margin", type=float, default=0.10)
    parser.add_argument("--min-identity-votes", type=int, default=1)
    parser.add_argument("--save-pipeline-dir", type=str, default=None)
    parser.add_argument("--save-pipeline-frame", type=int, default=None)
    parser.add_argument("--save-pipeline-face-index", type=int, default=0)
    parser.add_argument("--save-pipeline-candidate-limit", type=int, default=5)
    parser.add_argument("--use-fast-mtcnn", action="store_true")
    parser.add_argument("--fast-mtcnn-stride", type=int, default=3)
    parser.add_argument("--fast-mtcnn-resize", type=float, default=1.0)
    parser.add_argument("--fast-mtcnn-batch-size", type=int, default=60)
    return parser.parse_args()


def build_writer(
    output_path: str, cap: cv2.VideoCapture, frame_shape: tuple[int, int, int]
) -> cv2.VideoWriter:
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 20.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or frame_shape[1]
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or frame_shape[0]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_file), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Nao foi possivel criar o arquivo de saida {output_file}.")
    return writer


def maybe_save_pipeline_artifacts(
    recognizer: FaceRecognizer,
    args: argparse.Namespace,
    video_path: Path,
    current_frame_index: int,
) -> tuple[bool, str | None]:
    try:
        debug_face = recognizer.get_debug_face(args.save_pipeline_face_index)
        raw_matches = recognizer.qdrant.query(
            collection_name=args.collection,
            vector=debug_face["embedding"],
            limit=max(1, args.save_pipeline_candidate_limit),
            score_threshold=args.score_threshold,
        )
        identity_matches = group_matches_by_identity(
            raw_matches, limit=args.save_pipeline_candidate_limit
        )
        best_identity_match = identity_matches[0] if identity_matches else None
        artifact_paths = save_pipeline_presentation_artifacts(
            output_dir=Path(args.save_pipeline_dir),
            source_image_rgb=debug_face["image_rgb"],
            detections=debug_face["detections"],
            selected_detection={
                "bbox": debug_face["bbox"],
                "score": debug_face["detection_score"],
            },
            face_region=debug_face["face_region"],
            aligned_face_bgr=debug_face["aligned_face_bgr"],
            alignment_info=debug_face["alignment_info"],
            alignment_source=debug_face["alignment_source"],
            embedding=debug_face["embedding"],
            raw_matches=raw_matches,
            identity_matches=identity_matches,
            recognized_name=best_identity_match.get("name") if best_identity_match else None,
            entity_id=best_identity_match.get("entity_id") if best_identity_match else None,
            score=best_identity_match.get("score") if best_identity_match else None,
            isolate_method="crop",
            metadata={
                "pipeline_source": "video_face_recognition.py",
                "video_path": str(video_path.resolve()),
                "frame_index": current_frame_index,
                "collection": args.collection,
            },
        )
        print({"pipeline_artifacts": artifact_paths, "frame": current_frame_index})
        return True, None
    except Exception as exc:
        return False, str(exc)


def render_frame(
    frame: np.ndarray,
    current_frame_index: int,
    total_frames: int,
    recognitions: list[dict[str, Any]],
    inference_ms: float | None,
    show_stage_timings: bool,
    recognizer: FaceRecognizer,
) -> np.ndarray:
    frame_to_draw = frame.copy()
    draw_recognitions(frame_to_draw, recognitions)
    frame_ms = recognizer.last_timing.get("total_ms", 0.0)
    fps = 1000.0 / frame_ms if frame_ms > 0 else 0.0

    status = f"Frame: {current_frame_index}"
    if total_frames > 0:
        status += f"/{total_frames - 1}"
    status += f" | Faces: {len(recognitions)}"
    if inference_ms is not None:
        status += f" | inferencia: {inference_ms:.1f} ms"
    status += f" | frame: {frame_ms:.1f} ms ({fps:.1f} FPS)"
    cv2.putText(
        frame_to_draw,
        status,
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    if show_stage_timings and recognizer.last_timing:
        timing = recognizer.last_timing
        detail = (
            f"det {timing.get('detection_ms', 0.0):.1f} | "
            f"align {timing.get('alignment_ms', 0.0):.1f} | "
            f"emb {timing.get('embedding_ms', 0.0):.1f} | "
            f"busca {timing.get('search_ms', 0.0):.1f} ms"
        )
        cv2.putText(
            frame_to_draw,
            detail,
            (20, 55),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    return frame_to_draw


def main() -> None:
    args = parse_args()
    if args.frame_skip < 1:
        raise ValueError("--frame-skip deve ser maior ou igual a 1.")
    if args.start_frame < 0:
        raise ValueError("--start-frame nao pode ser negativo.")
    if args.max_frames is not None and args.max_frames < 1:
        raise ValueError("--max-frames deve ser maior ou igual a 1.")
    if args.detect_interval < 1:
        raise ValueError("--detect-interval deve ser maior ou igual a 1.")
    if not 0.0 <= args.temporal_iou_threshold <= 1.0:
        raise ValueError("--temporal-iou-threshold deve ficar entre 0 e 1.")
    if args.candidate_limit < 1:
        raise ValueError("--candidate-limit deve ser maior ou igual a 1.")
    if args.score_margin < 0.0:
        raise ValueError("--score-margin nao pode ser negativo.")
    if args.min_identity_votes < 1:
        raise ValueError("--min-identity-votes deve ser maior ou igual a 1.")
    if args.fast_mtcnn_stride < 1:
        raise ValueError("--fast-mtcnn-stride deve ser maior ou igual a 1.")
    if args.fast_mtcnn_resize <= 0:
        raise ValueError("--fast-mtcnn-resize deve ser maior que 0.")
    if args.fast_mtcnn_batch_size < 1:
        raise ValueError("--fast-mtcnn-batch-size deve ser maior ou igual a 1.")
    if args.use_fast_mtcnn and args.detector != "mtcnn":
        raise ValueError("--use-fast-mtcnn so pode ser usado com --detector mtcnn.")
    if args.conf is None:
        args.conf = 0.5 if args.detector == "yunet" else 0.95

    args.device = resolve_runtime_device(args.device)
    if args.device == "mps" and args.embedder_precision == "fp16":
        print("Device mps detectado; usando fp32 para compatibilidade com o embedder.")
        args.embedder_precision = "fp32"

    recognizer = FaceRecognizer(
        qdrant_url=args.qdrant_url,
        qdrant_api_key=args.qdrant_api_key,
        qdrant_timeout=args.qdrant_timeout,
        collection_name=args.collection,
        arcface_weight=args.arcface_weight,
        arcface_network=args.arcface_network,
        embedder_precision=args.embedder_precision,
        conf=args.conf,
        score_threshold=args.score_threshold,
        device=args.device,
        require_alignment=args.require_alignment,
        max_faces=args.max_faces,
        detect_width=None if args.detect_width == 0 else args.detect_width,
        align_faces=not args.disable_alignment,
        temporal_cache=not args.disable_temporal_cache,
        temporal_iou_threshold=args.temporal_iou_threshold,
        candidate_limit=args.candidate_limit,
        score_margin=args.score_margin,
        min_identity_votes=args.min_identity_votes,
        embedder_backend=args.embedder,
        edgeface_model_name=args.edgeface_model,
        edgeface_checkpoint=args.edgeface_checkpoint,
        edgeface_compile=not args.no_edgeface_compile,
        arcface_compile=not args.no_arcface_compile,
        facelivt_model_name=args.facelivt_model,
        facelivt_checkpoint=args.facelivt_checkpoint,
        facelivt_compile=not args.no_facelivt_compile,
        detect_interval=args.detect_interval,
        detector_backend=args.detector,
        yunet_model=args.yunet_model,
        yunet_nms=args.yunet_nms,
        yunet_top_k=args.yunet_top_k,
        yunet_backend=args.yunet_backend,
        use_fast_mtcnn=args.use_fast_mtcnn,
        fast_mtcnn_stride=args.fast_mtcnn_stride,
        fast_mtcnn_resize=args.fast_mtcnn_resize,
    )

    video_path = Path(args.video_path)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Nao foi possivel abrir o video {video_path}.")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if args.start_frame:
        cap.set(cv2.CAP_PROP_POS_FRAMES, args.start_frame)

    ok, first_frame = cap.read()
    if not ok or first_frame is None:
        cap.release()
        raise RuntimeError(f"O video {video_path} nao retornou frames validos.")

    writer = build_writer(args.output_video, cap, first_frame.shape) if args.output_video else None
    cap.release()

    stream = FileVideoStream(str(video_path)).start()
    display_enabled = not args.no_display
    pipeline_saved = False
    pipeline_export_error: str | None = None
    frames_processed = 0
    current_frame_index = -1
    batch_frames: list[np.ndarray] = []
    batch_indices: list[int] = []

    def flush_batch() -> bool:
        nonlocal batch_frames, batch_indices, frames_processed
        nonlocal pipeline_saved, pipeline_export_error, display_enabled

        if not batch_frames:
            return True

        if args.use_fast_mtcnn:
            if recognizer.fast_mtcnn is None:
                raise RuntimeError("FastMTCNN nao foi inicializado no recognizer.")
            detection_start = time.perf_counter()
            detections_by_frame = recognizer.fast_mtcnn.detect(
                [frame[:, :, ::-1] for frame in batch_frames],
                conf=args.conf,
                max_det=args.max_faces,
            )
            detection_ms = (time.perf_counter() - detection_start) * 1000.0 / max(1, len(batch_frames))
        else:
            detections_by_frame = [None] * len(batch_frames)
            detection_ms = 0.0

        for batch_frame, batch_index, precomputed_detections in zip(
            batch_frames, batch_indices, detections_by_frame
        ):
            should_export_pipeline = (
                args.save_pipeline_dir is not None
                and not pipeline_saved
                and (
                    args.save_pipeline_frame is None
                    or batch_index == args.save_pipeline_frame
                )
            )
            should_process_frame = frames_processed % args.frame_skip == 0 or should_export_pipeline
            if not should_process_frame:
                frames_processed += 1
                continue

            inference_start = time.perf_counter()
            try:
                if precomputed_detections is None:
                    recognitions = recognizer.recognize_frame(batch_frame)
                else:
                    recognitions = recognizer.recognize_frame_with_detections(
                        batch_frame,
                        precomputed_detections,
                        detection_ms=detection_ms,
                    )
            except Exception as exc:
                recognitions = []
                cv2.putText(
                    batch_frame,
                    f"Erro: {exc}",
                    (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )
            inference_ms = (time.perf_counter() - inference_start) * 1000.0

            if should_export_pipeline and recognitions:
                pipeline_saved, pipeline_export_error = maybe_save_pipeline_artifacts(
                    recognizer,
                    args,
                    video_path,
                    batch_index,
                )
            elif should_export_pipeline and args.save_pipeline_frame is not None and not recognitions:
                pipeline_export_error = (
                    f"Nenhum rosto encontrado no frame {batch_index} para exportar a pipeline."
                )

            rendered = render_frame(
                batch_frame,
                batch_index,
                total_frames,
                recognitions,
                inference_ms,
                args.show_stage_timings,
                recognizer,
            )
            if writer is not None:
                writer.write(rendered)

            if display_enabled:
                try:
                    cv2.imshow("Reconhecimento Facial em Video", rendered)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (27, ord("q")):
                        batch_frames = []
                        batch_indices = []
                        return False
                except cv2.error:
                    display_enabled = False
                    print("OpenCV sem suporte a janela neste ambiente; continuando sem display.")
            elif frames_processed % max(1, args.frame_skip * 10) == 0:
                print(
                    {
                        "frame": batch_index,
                        "faces": len(recognitions),
                        "inferencia_ms_total": round(inference_ms, 1),
                    }
                )

            frames_processed += 1

        batch_frames = []
        batch_indices = []
        return True

    try:
        while True:
            if args.max_frames is not None and frames_processed >= args.max_frames:
                break

            frame = stream.read()
            if frame is None:
                break

            current_frame_index += 1
            if current_frame_index < args.start_frame:
                continue

            batch_frames.append(frame)
            batch_indices.append(current_frame_index)
            if len(batch_frames) >= args.fast_mtcnn_batch_size:
                if not flush_batch():
                    break

        flush_batch()
    finally:
        stream.stop()
        if writer is not None:
            writer.release()
        if display_enabled:
            try:
                cv2.destroyAllWindows()
            except cv2.error:
                pass

    if args.save_pipeline_dir and not pipeline_saved:
        detail = f" Detalhe: {pipeline_export_error}" if pipeline_export_error else ""
        print(f"Nao foi possivel salvar a pipeline em {args.save_pipeline_dir}.{detail}")


if __name__ == "__main__":
    main()
