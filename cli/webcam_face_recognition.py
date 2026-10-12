from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
from collections import deque
import shutil
import subprocess
import time
from typing import Any, Protocol

import cv2
import numpy as np

from utils.drawing import draw_recognitions
from utils.edgeface_inference import DEFAULT_EDGEFACE_CHECKPOINT, DEFAULT_EDGEFACE_MODEL
from utils.face_pipeline import DEFAULT_ARCFACE_WEIGHT, FaceRecognizer
from utils.yunet_inference import DEFAULT_YUNET_MODEL


BYTES_IN_MB = 1024 * 1024


class VideoSource(Protocol):
    def read(self) -> tuple[bool, Any]:
        ...

    def release(self) -> None:
        ...


class FpsMeter:
    def __init__(self, window_size: int = 30) -> None:
        self.frame_times_ms: deque[float] = deque(maxlen=max(1, window_size))

    def update(self, frame_ms: float) -> tuple[float, float]:
        self.frame_times_ms.append(frame_ms)
        average_ms = sum(self.frame_times_ms) / len(self.frame_times_ms)
        fps = 1000.0 / average_ms if average_ms > 0 else 0.0
        return average_ms, fps


class RealSenseColorCapture:
    def __init__(self, width: int, height: int, fps: int, serial: str | None = None) -> None:
        try:
            import pyrealsense2 as rs
        except ImportError as exc:
            raise RuntimeError(
                "pyrealsense2 nao esta instalado. Instale as dependencias com "
                "`uv pip install -r requirements.txt`."
            ) from exc

        self.rs = rs
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        if serial:
            self.config.enable_device(serial)
        self.config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        try:
            self.profile = self.pipeline.start(self.config)
        except RuntimeError as exc:
            raise RuntimeError(
                "Nao foi possivel iniciar a RealSense com o stream RGB solicitado: "
                f"{width}x{height} BGR8 @ {fps}Hz. "
                "Para a L515, tente 1280x720 @ 30Hz ou confira os modos com "
                "`rs-enumerate-devices`."
            ) from exc

    def read(self) -> tuple[bool, Any]:
        frames = self.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            return False, None
        return True, np.asanyarray(color_frame.get_data())

    def release(self) -> None:
        self.pipeline.stop()


def mps_is_available() -> bool:
    try:
        import torch
    except ImportError:
        return False
    return bool(
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    )


def resolve_runtime_device(requested_device: str | None) -> str | None:
    if requested_device:
        if requested_device == "mps" and not mps_is_available():
            raise RuntimeError(
                "MPS foi solicitado, mas nao esta disponivel neste ambiente. "
                "Use --device cpu ou instale um build do torch com suporte a MPS."
            )
        return requested_device
    if sys.platform == "darwin" and mps_is_available():
        return "mps"
    return None


def configure_torch_runtime(device: str | None, matmul_precision: str = "high") -> None:
    if not device or not device.startswith("cuda"):
        return
    try:
        import torch

        torch.set_float32_matmul_precision(matmul_precision)
    except Exception:
        pass


def _safe_float(value: str | None) -> float | None:
    if value is None:
        return None
    cleaned = value.strip()
    if not cleaned or cleaned.upper() in {"N/A", "[N/A]"}:
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None


def _safe_int(value: str | None) -> int | None:
    number = _safe_float(value)
    return None if number is None else int(number)


def _format_value(value: float | int | str | None, suffix: str = "", precision: int = 0) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, str):
        return value
    return f"{value:.{precision}f}{suffix}"


def draw_text_lines(
    frame: np.ndarray,
    lines: list[str],
    origin: tuple[int, int],
    font_scale: float = 0.55,
    color: tuple[int, int, int] = (255, 255, 255),
    line_height: int = 22,
) -> None:
    x, y = origin
    for index, line in enumerate(lines):
        position = (x, y + index * line_height)
        cv2.putText(
            frame,
            line,
            position,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 0, 0),
            4,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            line,
            position,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            color,
            1,
            cv2.LINE_AA,
        )


class GpuTelemetryMonitor:
    QUERY_FIELDS = [
        "index",
        "name",
        "driver_version",
        "utilization.gpu",
        "utilization.memory",
        "memory.used",
        "memory.total",
        "temperature.gpu",
        "power.draw",
        "power.limit",
        "clocks.sm",
        "clocks.mem",
        "fan.speed",
        "pstate",
    ]

    def __init__(self, device: str | None, sample_interval: float = 1.0) -> None:
        self.device = device or ""
        self.sample_interval = max(0.1, sample_interval)
        self.has_nvidia_smi = shutil.which("nvidia-smi") is not None
        self.gpu_index = self._resolve_gpu_index()
        self.last_sample: dict[str, Any] = {
            "available": False,
            "reason": "nvidia-smi nao encontrado" if not self.has_nvidia_smi else None,
        }
        self._last_query_at = 0.0
        self._last_energy_at: float | None = None
        self.energy_wh = 0.0

    def _resolve_gpu_index(self) -> int:
        if self.device.startswith("cuda:"):
            return int(self.device.split(":", 1)[1])
        try:
            import torch

            if torch.cuda.is_available():
                return int(torch.cuda.current_device())
        except Exception:
            pass
        return 0

    def _sample_torch_memory(self, sample: dict[str, Any]) -> None:
        try:
            import torch

            if torch.cuda.is_available():
                sample["torch_allocated_mb"] = torch.cuda.memory_allocated(self.gpu_index) / BYTES_IN_MB
                sample["torch_reserved_mb"] = torch.cuda.memory_reserved(self.gpu_index) / BYTES_IN_MB
        except Exception:
            sample["torch_allocated_mb"] = None
            sample["torch_reserved_mb"] = None

    def sample(self) -> dict[str, Any]:
        now = time.perf_counter()
        if now - self._last_query_at < self.sample_interval:
            return self.last_sample

        self._last_query_at = now
        sample: dict[str, Any] = {
            "available": False,
            "gpu_index": self.gpu_index,
            "name": None,
            "driver_version": None,
            "gpu_util_percent": None,
            "memory_util_percent": None,
            "memory_used_mb": None,
            "memory_total_mb": None,
            "temperature_c": None,
            "power_draw_w": None,
            "power_limit_w": None,
            "energy_wh": self.energy_wh,
            "clock_sm_mhz": None,
            "clock_mem_mhz": None,
            "fan_speed_percent": None,
            "pstate": None,
            "torch_allocated_mb": None,
            "torch_reserved_mb": None,
        }

        self._sample_torch_memory(sample)

        if not self.has_nvidia_smi:
            sample["reason"] = "nvidia-smi nao encontrado"
            self.last_sample = sample
            return sample

        result = subprocess.run(
            [
                "nvidia-smi",
                f"--query-gpu={','.join(self.QUERY_FIELDS)}",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            sample["reason"] = result.stderr.strip() or "falha ao executar nvidia-smi"
            self.last_sample = sample
            return sample

        for line in result.stdout.splitlines():
            parts = [part.strip() for part in line.split(",")]
            if len(parts) != len(self.QUERY_FIELDS):
                continue
            index = _safe_int(parts[0])
            if index != self.gpu_index:
                continue

            memory_used = _safe_float(parts[5])
            memory_total = _safe_float(parts[6])
            power_draw = _safe_float(parts[8])
            if power_draw is not None:
                if self._last_energy_at is not None:
                    self.energy_wh += power_draw * (now - self._last_energy_at) / 3600.0
                self._last_energy_at = now

            sample.update(
                {
                    "available": True,
                    "gpu_index": index,
                    "name": parts[1],
                    "driver_version": parts[2],
                    "gpu_util_percent": _safe_float(parts[3]),
                    "memory_util_percent": _safe_float(parts[4]),
                    "memory_used_mb": memory_used,
                    "memory_total_mb": memory_total,
                    "temperature_c": _safe_float(parts[7]),
                    "power_draw_w": power_draw,
                    "power_limit_w": _safe_float(parts[9]),
                    "energy_wh": self.energy_wh,
                    "clock_sm_mhz": _safe_int(parts[10]),
                    "clock_mem_mhz": _safe_int(parts[11]),
                    "fan_speed_percent": _safe_float(parts[12]),
                    "pstate": parts[13] if parts[13] else None,
                    "reason": None,
                }
            )
            break

        self.last_sample = sample
        return sample

    @staticmethod
    def format_overlay(sample: dict[str, Any]) -> list[str]:
        if not sample.get("available"):
            reason = sample.get("reason") or "GPU indisponivel"
            return [f"GPU: {reason}"]

        memory_used = sample.get("memory_used_mb")
        memory_total = sample.get("memory_total_mb")
        memory_percent = (
            (memory_used / memory_total) * 100.0
            if memory_used is not None and memory_total
            else None
        )
        power_limit = sample.get("power_limit_w")
        power_text = _format_value(sample.get("power_draw_w"), "W", 1)
        if power_limit is not None:
            power_text += f"/{_format_value(power_limit, 'W', 0)}"

        return [
            f"GPU{sample.get('gpu_index')}: {sample.get('name')}",
            (
                f"Uso {_format_value(sample.get('gpu_util_percent'), '%')} | "
                f"Mem {_format_value(memory_used, 'MB')}/{_format_value(memory_total, 'MB')} "
                f"({_format_value(memory_percent, '%')})"
            ),
            (
                f"Temp {_format_value(sample.get('temperature_c'), 'C')} | "
                f"Power {power_text} | Energia {_format_value(sample.get('energy_wh'), 'Wh', 4)}"
            ),
            (
                f"Clock SM {_format_value(sample.get('clock_sm_mhz'), 'MHz')} | "
                f"MemClock {_format_value(sample.get('clock_mem_mhz'), 'MHz')} | "
                f"Fan {_format_value(sample.get('fan_speed_percent'), '%')} | "
                f"P-state {sample.get('pstate') or 'N/A'}"
            ),
            (
                f"Torch alloc {_format_value(sample.get('torch_allocated_mb'), 'MB')} | "
                f"reserved {_format_value(sample.get('torch_reserved_mb'), 'MB')} | "
                f"Driver {sample.get('driver_version') or 'N/A'}"
            ),
        ]


def open_camera(camera_source: str | int) -> cv2.VideoCapture:
    if sys.platform == "darwin" and isinstance(camera_source, int):
        cap = cv2.VideoCapture(camera_source, cv2.CAP_AVFOUNDATION)
        if cap.isOpened():
            return cap
        cap.release()
    return cv2.VideoCapture(camera_source)


def open_video_source(args: argparse.Namespace, camera_source: str | int) -> VideoSource:
    if isinstance(camera_source, str) and camera_source.lower() in {"realsense", "rs", "l515"}:
        return RealSenseColorCapture(
            width=args.realsense_width,
            height=args.realsense_height,
            fps=args.realsense_fps,
            serial=args.realsense_serial,
        )

    cap = open_camera(camera_source)
    if not cap.isOpened():
        raise RuntimeError(f"Nao foi possivel abrir a camera: {args.camera}")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    return cap


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reconhecimento facial em tempo real com webcam/IP e Qdrant."
    )
    parser.add_argument(
        "--camera",
        type=str,
        default="0",
        help="Fonte de video: indice de webcam (ex: 0), URL de camera IP ou `realsense` para Intel RealSense.",
    )
    parser.add_argument("--realsense-width", type=int, default=1280)
    parser.add_argument("--realsense-height", type=int, default=720)
    parser.add_argument("--realsense-fps", type=int, default=30)
    parser.add_argument("--realsense-serial", type=str, default=None)
    parser.add_argument("--collection", type=str, default="Rostos")
    parser.add_argument("--qdrant-url", type=str, default="http://localhost:6333")
    parser.add_argument("--qdrant-api-key", type=str, default=None)
    parser.add_argument("--qdrant-timeout", type=float, default=15.0)
    parser.add_argument("--arcface-weight", type=str, default=str(DEFAULT_ARCFACE_WEIGHT))
    parser.add_argument("--arcface-network", type=str, default="r100")
    parser.add_argument("--embedder", type=str, default="arcface", choices=["arcface", "edgeface"])
    parser.add_argument("--embedder-precision", type=str, default="fp16", choices=["fp32", "fp16"])
    parser.add_argument("--edgeface-model", type=str, default=DEFAULT_EDGEFACE_MODEL)
    parser.add_argument("--edgeface-checkpoint", type=str, default=str(DEFAULT_EDGEFACE_CHECKPOINT))
    parser.add_argument(
        "--edgeface-compile",
        dest="no_edgeface_compile",
        action="store_false",
        help="Ativa torch.compile para EdgeFace. Pode travar na primeira face enquanto compila.",
    )
    parser.add_argument("--no-edgeface-compile", dest="no_edgeface_compile", action="store_true", default=True)
    parser.add_argument(
        "--arcface-compile",
        dest="no_arcface_compile",
        action="store_false",
        help="Ativa torch.compile para ArcFace. Pode travar na primeira face enquanto compila.",
    )
    parser.add_argument("--no-arcface-compile", dest="no_arcface_compile", action="store_true", default=True)
    parser.add_argument("--detector", type=str, default="mtcnn", choices=["mtcnn", "yunet"])
    parser.add_argument("--yunet-model", type=str, default=str(DEFAULT_YUNET_MODEL))
    parser.add_argument("--yunet-nms", type=float, default=0.3)
    parser.add_argument("--yunet-top-k", type=int, default=5000)
    parser.add_argument("--yunet-backend", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--conf", type=float, default=0.90)
    parser.add_argument("--score-threshold", type=float, default=0.45)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--frame-skip", type=int, default=2)
    parser.add_argument("--max-faces", type=int, default=5)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--detect-width", type=int, default=960)
    parser.add_argument("--disable-alignment", action="store_true")
    parser.add_argument("--detect-interval", type=int, default=3)
    parser.add_argument("--disable-temporal-cache", action="store_true")
    parser.add_argument("--temporal-iou-threshold", type=float, default=0.9)
    parser.add_argument("--candidate-limit", type=int, default=5)
    parser.add_argument("--score-margin", type=float, default=0.10)
    parser.add_argument("--min-identity-votes", type=int, default=1)
    parser.add_argument("--show-stage-timings", action="store_true")
    parser.add_argument("--fps-window", type=int, default=30)
    parser.add_argument(
        "--torch-matmul-precision",
        type=str,
        default="high",
        choices=["highest", "high", "medium"],
        help="Precisao interna de matmul float32 no torch. `high` evita aviso de TF32 e melhora desempenho em CUDA.",
    )
    parser.add_argument(
        "--show-gpu-stats",
        action="store_true",
        default=True,
        help="Mostra telemetria da GPU no overlay e no log sem display. Ativo por padrao.",
    )
    parser.add_argument(
        "--hide-gpu-stats",
        dest="show_gpu_stats",
        action="store_false",
        help="Oculta telemetria da GPU.",
    )
    parser.add_argument(
        "--gpu-stats-interval",
        type=float,
        default=1.0,
        help="Intervalo em segundos para consultar nvidia-smi.",
    )
    parser.add_argument("--no-display", action="store_true")
    parser.add_argument("--output-video", type=str, default=None)
    parser.add_argument("--require-alignment", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
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
    if args.fps_window < 1:
        raise ValueError("--fps-window deve ser maior ou igual a 1.")

    args.device = resolve_runtime_device(args.device)
    if args.device == "mps" and args.embedder_precision == "fp16":
        print("Device mps detectado; usando fp32 para compatibilidade com o embedder.")
        args.embedder_precision = "fp32"
    configure_torch_runtime(args.device, matmul_precision=args.torch_matmul_precision)

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
        detect_interval=args.detect_interval,
        detector_backend=args.detector,
        yunet_model=args.yunet_model,
        yunet_nms=args.yunet_nms,
        yunet_top_k=args.yunet_top_k,
        yunet_backend=args.yunet_backend,
    )
    gpu_monitor = (
        GpuTelemetryMonitor(device=recognizer.device, sample_interval=args.gpu_stats_interval)
        if args.show_gpu_stats
        else None
    )

    # Aceita índice numérico (webcam USB), URL (câmera IP) ou "realsense".
    camera_source: str | int = args.camera
    try:
        camera_source = int(args.camera)
    except ValueError:
        pass

    cap = open_video_source(args, camera_source)

    ok, frame = cap.read()
    if not ok or frame is None:
        cap.release()
        raise RuntimeError(f"A camera abriu mas nao retornou frames: {args.camera}")

    frame_index = 0
    fps_meter = FpsMeter(window_size=args.fps_window)
    last_recognitions: list[dict[str, Any]] = []
    last_duration_ms: float | None = None
    last_error_message: str | None = None
    last_gpu_sample: dict[str, Any] | None = None
    display_enabled = not args.no_display
    writer = None

    if args.output_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.output_video, fourcc, 20.0, (frame.shape[1], frame.shape[0]))

    try:
        while True:
            frame_start = time.perf_counter()

            if frame_index > 0:
                ok, frame = cap.read()
                if not ok or frame is None:
                    raise RuntimeError("Falha ao capturar frame da webcam.")

            if frame_index % max(1, args.frame_skip) == 0:
                start_time = time.perf_counter()
                try:
                    last_recognitions = recognizer.recognize_frame(frame)
                    last_error_message = None
                except Exception as exc:
                    last_recognitions = []
                    last_error_message = f"{type(exc).__name__}: {exc}"
                    if frame_index % max(1, args.frame_skip * 10) == 0:
                        print({"webcam_error": last_error_message})
                last_duration_ms = (time.perf_counter() - start_time) * 1000.0

            if gpu_monitor is not None:
                last_gpu_sample = gpu_monitor.sample()

            draw_recognitions(frame, last_recognitions)

            frame_ms = (time.perf_counter() - frame_start) * 1000.0
            smooth_frame_ms, smooth_fps = fps_meter.update(frame_ms)

            status = f"Faces: {len(last_recognitions)}"
            if last_duration_ms is not None:
                status += f" | inferencia: {last_duration_ms:.1f} ms"
            status += (
                f" | frame: {smooth_frame_ms:.1f} ms ({smooth_fps:.1f} FPS)"
                f" | {recognizer.device}/{args.embedder_precision}"
            )
            cv2.putText(
                frame, status, (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA,
            )

            if last_error_message:
                cv2.putText(
                    frame,
                    last_error_message[:100],
                    (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )

            if args.show_stage_timings and recognizer.last_timing:
                timing = recognizer.last_timing
                detail = (
                    f"det {timing.get('detection_ms', 0.0):.1f} | "
                    f"align {timing.get('alignment_ms', 0.0):.1f} | "
                    f"emb {timing.get('embedding_ms', 0.0):.1f} | "
                    f"busca {timing.get('search_ms', 0.0):.1f} ms"
                )
                cv2.putText(
                    frame, detail, (20, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA,
                )

            if args.show_gpu_stats and last_gpu_sample is not None:
                gpu_y = 120 if args.show_stage_timings and recognizer.last_timing else 90
                draw_text_lines(
                    frame,
                    GpuTelemetryMonitor.format_overlay(last_gpu_sample),
                    origin=(20, gpu_y),
                    font_scale=0.52,
                    color=(220, 255, 220),
                    line_height=21,
                )

            if writer is not None:
                writer.write(frame)

            if display_enabled:
                try:
                    cv2.imshow("Reconhecimento Facial", frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (27, ord("q")):
                        break
                except cv2.error:
                    display_enabled = False
                    print("OpenCV sem suporte a janela neste ambiente; continuando sem display.")
            elif frame_index % max(1, args.frame_skip * 10) == 0:
                names = [item.get("name") or "Desconhecido" for item in last_recognitions]
                print({
                    "faces": len(last_recognitions),
                    "names": names,
                    "inferencia_ms": None if last_duration_ms is None else round(last_duration_ms, 1),
                    "frame_ms_smooth": round(smooth_frame_ms, 1),
                    "fps_smooth": round(smooth_fps, 1),
                    "device": recognizer.device,
                    "embedder_precision": args.embedder_precision,
                    "arcface_compile": not args.no_arcface_compile,
                    "stage_timings_ms": {
                        key: round(value, 1) for key, value in recognizer.last_timing.items()
                    },
                    "gpu": last_gpu_sample,
                })

            frame_index += 1
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if display_enabled:
            try:
                cv2.destroyAllWindows()
            except cv2.error:
                pass


if __name__ == "__main__":
    main()
