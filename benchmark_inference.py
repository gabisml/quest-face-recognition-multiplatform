"""Benchmark comparativo do pipeline de reconhecimento facial.

Compara duas variantes da pipeline atual:
1. MTCNN + ArcFace
2. YuNet + ArcFace

Para imagem, mede cada etapa separadamente:
- deteccao
- extracao de regiao facial
- alinhamento
- embedding
- busca no Qdrant

Para video, executa a pipeline completa frame a frame com cada detector e
resume as medias por etapa.
"""

from __future__ import annotations

import argparse
import base64
import gc
import os
import shutil
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import psutil
import torch
from PIL import Image

from utils.arcface_inference import ArcFaceEmbedder, FastMTCNN, align_face, align_face_crop, detect_faces_mtcnn, get_torch_device
from utils.face_pipeline import DEFAULT_ARCFACE_WEIGHT, FaceRecognizer, extract_face_region
from utils.identity_matching import select_best_identity_match
from utils.qdrant import QdrantStore
from utils.yunet_inference import DEFAULT_YUNET_MODEL, detect_faces_yunet


PROJECT_ROOT = Path(__file__).resolve().parent
BYTES_IN_MB = 1024**2
BYTES_IN_GB = 1024**3
TABLE_HEADER = f"{'Etapa':<28} {'Media':>8} {'Mediana':>8} {'Min':>8} {'P95':>8} {'P99':>8} {'Std':>8}"
TABLE_SEP = "-" * len(TABLE_HEADER)
HARDWARE_TABLE_HEADER = f"{'Recurso':<28} {'Media':>10} {'Max':>10} {'Ultimo':>10}"
HARDWARE_TABLE_SEP = "-" * len(HARDWARE_TABLE_HEADER)


@dataclass(frozen=True)
class DetectorConfig:
    key: str
    label: str


DETECTOR_CONFIGS = [
    DetectorConfig(key="mtcnn", label="MTCNN"),
    DetectorConfig(key="yunet", label="YuNet"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark comparativo MTCNN vs YuNet")
    parser.add_argument("image_path", nargs="?", default=None, help="Imagem para benchmark")
    parser.add_argument("--video-path", type=str, default=None, help="Video para benchmark")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--rounds", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--detect-width", type=int, default=640)
    parser.add_argument("--max-faces", type=int, default=5)
    parser.add_argument("--conf-mtcnn", type=float, default=0.90)
    parser.add_argument("--conf-yunet", type=float, default=0.50)
    parser.add_argument("--use-fast-mtcnn", action="store_true")
    parser.add_argument("--fast-mtcnn-stride", type=int, default=3)
    parser.add_argument("--fast-mtcnn-resize", type=float, default=1.0)
    parser.add_argument("--fast-mtcnn-batch-size", type=int, default=60)
    parser.add_argument("--arcface-weight", type=str, default=str(DEFAULT_ARCFACE_WEIGHT))
    parser.add_argument("--arcface-network", type=str, default="r100")
    parser.add_argument("--embedder-precision", type=str, default="fp16", choices=["fp16", "fp32"])
    parser.add_argument("--yunet-model", type=str, default=str(DEFAULT_YUNET_MODEL))
    parser.add_argument("--yunet-nms", type=float, default=0.3)
    parser.add_argument("--yunet-top-k", type=int, default=5000)
    parser.add_argument("--yunet-backend", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--collection", type=str, default="Rostos")
    parser.add_argument("--qdrant-url", type=str, default="http://localhost:6333")
    parser.add_argument("--qdrant-api-key", type=str, default=None)
    parser.add_argument("--qdrant-timeout", type=float, default=15.0)
    parser.add_argument("--score-threshold", type=float, default=0.45)
    parser.add_argument("--candidate-limit", type=int, default=5)
    parser.add_argument("--score-margin", type=float, default=0.10)
    parser.add_argument("--min-identity-votes", type=int, default=1)
    parser.add_argument("--require-alignment", action="store_true")
    parser.add_argument("--disable-alignment", action="store_true")
    parser.add_argument("--detect-interval", type=int, default=1)
    parser.add_argument("--disable-temporal-cache", action="store_true")
    parser.add_argument("--temporal-iou-threshold", type=float, default=0.9)
    parser.add_argument("--video-max-frames", type=int, default=120)
    parser.add_argument("--video-start-frame", type=int, default=0)
    parser.add_argument(
        "--hardware-stats-interval",
        type=float,
        default=1.0,
        help="Intervalo em segundos para amostrar CPU/RAM/GPU durante o benchmark.",
    )
    parser.add_argument(
        "--disable-hardware-stats",
        action="store_true",
        help="Desativa a coleta de CPU/RAM/GPU/energia.",
    )
    parser.add_argument(
        "--html-output",
        type=str,
        default=str(PROJECT_ROOT / "output" / "benchmarks" / "benchmark_inference.html"),
        help="Caminho do relatorio HTML do benchmark.",
    )
    args = parser.parse_args()

    if bool(args.image_path) == bool(args.video_path):
        parser.error("Informe exatamente um entre image_path e --video-path.")
    if args.fast_mtcnn_stride < 1:
        parser.error("--fast-mtcnn-stride deve ser maior ou igual a 1.")
    if args.fast_mtcnn_resize <= 0:
        parser.error("--fast-mtcnn-resize deve ser maior que 0.")
    if args.fast_mtcnn_batch_size < 1:
        parser.error("--fast-mtcnn-batch-size deve ser maior ou igual a 1.")
    if args.batch_size < 1:
        parser.error("--batch-size deve ser maior ou igual a 1.")
    if args.rounds < 1:
        parser.error("--rounds deve ser maior ou igual a 1.")
    if args.warmup < 0:
        parser.error("--warmup nao pode ser negativo.")
    if args.hardware_stats_interval <= 0:
        parser.error("--hardware-stats-interval deve ser maior que 0.")
    return args


def safe_float(value: str) -> float | None:
    value = value.strip()
    if not value or value.upper() in {"N/A", "[N/A]"}:
        return None
    try:
        return float(value)
    except ValueError:
        try:
            return float(value.split()[0])
        except (IndexError, ValueError):
            return None


def safe_mean(values: list[float | None]) -> float | None:
    valid = [value for value in values if value is not None]
    return float(np.mean(valid)) if valid else None


def safe_max(values: list[float | None]) -> float | None:
    valid = [value for value in values if value is not None]
    return float(np.max(valid)) if valid else None


def format_optional(value: float | None, digits: int = 1) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}"


class HardwareMonitor:
    def __init__(self, device: str, sample_interval: float = 1.0, enabled: bool = True) -> None:
        self.device = device
        self.sample_interval = sample_interval
        self.enabled = enabled
        self.has_nvidia_smi = shutil.which("nvidia-smi") is not None
        self.process = psutil.Process()
        self.pid = os.getpid()
        self.logical_cpus = psutil.cpu_count(logical=True) or 1
        self.gpu_index = 0
        if device.startswith("cuda:"):
            self.gpu_index = int(device.split(":", 1)[1])
        elif device.startswith("cuda") and torch.cuda.is_available():
            self.gpu_index = torch.cuda.current_device()
        self.samples: list[dict[str, float | None]] = []
        self.energy_wh = 0.0
        self._last_sample_at: float | None = None
        self._last_power_at: float | None = None
        psutil.cpu_percent(interval=None)
        self.process.cpu_percent(interval=None)

    def maybe_sample(self, *, force: bool = False) -> None:
        if not self.enabled:
            return
        now = time.perf_counter()
        if not force and self._last_sample_at is not None and now - self._last_sample_at < self.sample_interval:
            return
        self.samples.append(self._sample(now))
        self._last_sample_at = now

    def _sample(self, now: float) -> dict[str, float | None]:
        memory = psutil.virtual_memory()
        process_memory = self.process.memory_info()
        process_cpu_raw = float(self.process.cpu_percent(interval=None))
        sample: dict[str, float | None] = {
            "timestamp_s": now,
            "cpu_percent": float(psutil.cpu_percent(interval=None)),
            "process_cpu_percent": process_cpu_raw / float(self.logical_cpus),
            "process_cpu_raw_percent": process_cpu_raw,
            "ram_percent": float(memory.percent),
            "ram_used_gb": float(memory.used / BYTES_IN_GB),
            "process_rss_mb": float(process_memory.rss / BYTES_IN_MB),
            "gpu_util_percent": None,
            "gpu_mem_util_percent": None,
            "gpu_memory_used_mb": None,
            "gpu_memory_total_mb": None,
            "gpu_power_w": None,
            "gpu_energy_wh": self.energy_wh,
            "gpu_process_memory_mb": None,
            "gpu_accounted_util_percent": None,
            "gpu_accounted_mem_util_percent": None,
            "gpu_accounted_max_memory_mb": None,
            "gpu_accounted_time_ms": None,
            "torch_allocated_mb": None,
            "torch_reserved_mb": None,
        }

        if self.device.startswith("cuda") and torch.cuda.is_available():
            sample["torch_allocated_mb"] = float(torch.cuda.memory_allocated(self.gpu_index) / BYTES_IN_MB)
            sample["torch_reserved_mb"] = float(torch.cuda.memory_reserved(self.gpu_index) / BYTES_IN_MB)

        if self.has_nvidia_smi and self.device.startswith("cuda"):
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=index,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode == 0:
                for line in result.stdout.splitlines():
                    parts = [part.strip() for part in line.split(",")]
                    if len(parts) != 6:
                        continue
                    gpu_index = safe_float(parts[0])
                    if gpu_index is None or int(gpu_index) != self.gpu_index:
                        continue
                    memory_used = safe_float(parts[3])
                    memory_total = safe_float(parts[4])
                    power_w = safe_float(parts[5])
                    if power_w is not None and self._last_power_at is not None:
                        self.energy_wh += power_w * (now - self._last_power_at) / 3600.0
                    if power_w is not None:
                        self._last_power_at = now
                    sample.update(
                        {
                            "gpu_util_percent": safe_float(parts[1]),
                            "gpu_mem_util_percent": (
                                (memory_used / memory_total) * 100.0
                                if memory_used is not None and memory_total and memory_total > 0
                                else None
                            ),
                            "gpu_memory_used_mb": memory_used,
                            "gpu_memory_total_mb": memory_total,
                            "gpu_power_w": power_w,
                            "gpu_energy_wh": self.energy_wh,
                        }
                    )
                    break

            compute_result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-compute-apps=pid,used_memory",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            if compute_result.returncode == 0:
                process_memory = 0.0
                found_process = False
                for line in compute_result.stdout.splitlines():
                    parts = [part.strip() for part in line.split(",")]
                    if len(parts) != 2:
                        continue
                    pid = safe_float(parts[0])
                    if pid is None or int(pid) != self.pid:
                        continue
                    memory_mb = safe_float(parts[1])
                    if memory_mb is not None:
                        process_memory += memory_mb
                        found_process = True
                if found_process:
                    sample["gpu_process_memory_mb"] = process_memory

            accounting_result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-accounted-apps=pid,gpu_util,mem_util,max_memory_usage,time",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            if accounting_result.returncode == 0:
                for line in accounting_result.stdout.splitlines():
                    parts = [part.strip() for part in line.split(",")]
                    if len(parts) != 5:
                        continue
                    pid = safe_float(parts[0])
                    if pid is None or int(pid) != self.pid:
                        continue
                    sample.update(
                        {
                            "gpu_accounted_util_percent": safe_float(parts[1]),
                            "gpu_accounted_mem_util_percent": safe_float(parts[2]),
                            "gpu_accounted_max_memory_mb": safe_float(parts[3]),
                            "gpu_accounted_time_ms": safe_float(parts[4]),
                        }
                    )
                    break

        return sample

    def summary(self, elapsed_s: float) -> dict[str, float | None]:
        samples = self.samples
        if not self.enabled or not samples:
            return {"enabled": 0.0, "samples": 0.0, "elapsed_s": elapsed_s}
        return {
            "enabled": 1.0,
            "samples": float(len(samples)),
            "elapsed_s": elapsed_s,
            "cpu_percent_mean": safe_mean([sample.get("cpu_percent") for sample in samples]),
            "cpu_percent_max": safe_max([sample.get("cpu_percent") for sample in samples]),
            "cpu_percent_last": samples[-1].get("cpu_percent"),
            "process_cpu_percent_mean": safe_mean([sample.get("process_cpu_percent") for sample in samples]),
            "process_cpu_percent_max": safe_max([sample.get("process_cpu_percent") for sample in samples]),
            "process_cpu_percent_last": samples[-1].get("process_cpu_percent"),
            "process_cpu_raw_percent_mean": safe_mean([sample.get("process_cpu_raw_percent") for sample in samples]),
            "process_cpu_raw_percent_max": safe_max([sample.get("process_cpu_raw_percent") for sample in samples]),
            "process_cpu_raw_percent_last": samples[-1].get("process_cpu_raw_percent"),
            "ram_percent_mean": safe_mean([sample.get("ram_percent") for sample in samples]),
            "ram_percent_max": safe_max([sample.get("ram_percent") for sample in samples]),
            "ram_percent_last": samples[-1].get("ram_percent"),
            "ram_used_gb_mean": safe_mean([sample.get("ram_used_gb") for sample in samples]),
            "ram_used_gb_max": safe_max([sample.get("ram_used_gb") for sample in samples]),
            "ram_used_gb_last": samples[-1].get("ram_used_gb"),
            "process_rss_mb_mean": safe_mean([sample.get("process_rss_mb") for sample in samples]),
            "process_rss_mb_max": safe_max([sample.get("process_rss_mb") for sample in samples]),
            "process_rss_mb_last": samples[-1].get("process_rss_mb"),
            "gpu_util_percent_mean": safe_mean([sample.get("gpu_util_percent") for sample in samples]),
            "gpu_util_percent_max": safe_max([sample.get("gpu_util_percent") for sample in samples]),
            "gpu_util_percent_last": samples[-1].get("gpu_util_percent"),
            "gpu_mem_util_percent_mean": safe_mean([sample.get("gpu_mem_util_percent") for sample in samples]),
            "gpu_mem_util_percent_max": safe_max([sample.get("gpu_mem_util_percent") for sample in samples]),
            "gpu_mem_util_percent_last": samples[-1].get("gpu_mem_util_percent"),
            "gpu_memory_used_mb_mean": safe_mean([sample.get("gpu_memory_used_mb") for sample in samples]),
            "gpu_memory_used_mb_max": safe_max([sample.get("gpu_memory_used_mb") for sample in samples]),
            "gpu_memory_used_mb_last": samples[-1].get("gpu_memory_used_mb"),
            "gpu_memory_total_mb_last": samples[-1].get("gpu_memory_total_mb"),
            "gpu_power_w_mean": safe_mean([sample.get("gpu_power_w") for sample in samples]),
            "gpu_power_w_max": safe_max([sample.get("gpu_power_w") for sample in samples]),
            "gpu_power_w_last": samples[-1].get("gpu_power_w"),
            "gpu_energy_wh": self.energy_wh,
            "gpu_process_memory_mb_mean": safe_mean([sample.get("gpu_process_memory_mb") for sample in samples]),
            "gpu_process_memory_mb_max": safe_max([sample.get("gpu_process_memory_mb") for sample in samples]),
            "gpu_process_memory_mb_last": samples[-1].get("gpu_process_memory_mb"),
            "gpu_accounted_util_percent_mean": safe_mean([sample.get("gpu_accounted_util_percent") for sample in samples]),
            "gpu_accounted_util_percent_max": safe_max([sample.get("gpu_accounted_util_percent") for sample in samples]),
            "gpu_accounted_util_percent_last": samples[-1].get("gpu_accounted_util_percent"),
            "gpu_accounted_mem_util_percent_mean": safe_mean([sample.get("gpu_accounted_mem_util_percent") for sample in samples]),
            "gpu_accounted_mem_util_percent_max": safe_max([sample.get("gpu_accounted_mem_util_percent") for sample in samples]),
            "gpu_accounted_mem_util_percent_last": samples[-1].get("gpu_accounted_mem_util_percent"),
            "gpu_accounted_max_memory_mb_mean": safe_mean([sample.get("gpu_accounted_max_memory_mb") for sample in samples]),
            "gpu_accounted_max_memory_mb_max": safe_max([sample.get("gpu_accounted_max_memory_mb") for sample in samples]),
            "gpu_accounted_max_memory_mb_last": samples[-1].get("gpu_accounted_max_memory_mb"),
            "gpu_accounted_time_ms_mean": safe_mean([sample.get("gpu_accounted_time_ms") for sample in samples]),
            "gpu_accounted_time_ms_max": safe_max([sample.get("gpu_accounted_time_ms") for sample in samples]),
            "gpu_accounted_time_ms_last": samples[-1].get("gpu_accounted_time_ms"),
            "torch_allocated_mb_mean": safe_mean([sample.get("torch_allocated_mb") for sample in samples]),
            "torch_allocated_mb_max": safe_max([sample.get("torch_allocated_mb") for sample in samples]),
            "torch_allocated_mb_last": samples[-1].get("torch_allocated_mb"),
            "torch_reserved_mb_mean": safe_mean([sample.get("torch_reserved_mb") for sample in samples]),
            "torch_reserved_mb_max": safe_max([sample.get("torch_reserved_mb") for sample in samples]),
            "torch_reserved_mb_last": samples[-1].get("torch_reserved_mb"),
        }


def print_hardware_summary(summary: dict[str, float | None]) -> None:
    if not summary or not summary.get("enabled"):
        print("Telemetria hardware: desativada")
        return
    print(HARDWARE_TABLE_HEADER)
    print(HARDWARE_TABLE_SEP)
    rows = [
        ("CPU sistema (%)", "cpu_percent", 1),
        ("CPU processo total (%)", "process_cpu_percent", 1),
        ("RAM sistema (%)", "ram_percent", 1),
        ("RAM sistema (GB)", "ram_used_gb", 2),
        ("RSS processo (MB)", "process_rss_mb", 0),
        ("GPU uso (%)", "gpu_util_percent", 1),
        ("GPU memoria (%)", "gpu_mem_util_percent", 1),
        ("GPU memoria usada (MB)", "gpu_memory_used_mb", 0),
        ("GPU memoria processo (MB)", "gpu_process_memory_mb", 0),
        ("GPU potencia (W)", "gpu_power_w", 1),
        ("GPU uso processo acct (%)", "gpu_accounted_util_percent", 1),
        ("GPU mem processo acct (%)", "gpu_accounted_mem_util_percent", 1),
        ("GPU max processo acct MB", "gpu_accounted_max_memory_mb", 0),
        ("GPU tempo processo acct ms", "gpu_accounted_time_ms", 0),
        ("Torch alocada (MB)", "torch_allocated_mb", 0),
        ("Torch reservada (MB)", "torch_reserved_mb", 0),
    ]
    for label, key, digits in rows:
        print(
            f"{label:<28} "
            f"{format_optional(summary.get(f'{key}_mean'), digits):>10} "
            f"{format_optional(summary.get(f'{key}_max'), digits):>10} "
            f"{format_optional(summary.get(f'{key}_last'), digits):>10}"
        )
    gpu_total = summary.get("gpu_memory_total_mb_last")
    if gpu_total is not None:
        print(f"GPU memoria total: {gpu_total:.0f} MB")
    print(f"GPU energia estimada: {format_optional(summary.get('gpu_energy_wh'), 4)} Wh")


def _triplet(summary: dict[str, float | None], key: str, digits: int) -> str:
    return (
        f"{format_optional(summary.get(f'{key}_mean'), digits)} / "
        f"{format_optional(summary.get(f'{key}_max'), digits)} / "
        f"{format_optional(summary.get(f'{key}_last'), digits)}"
    )


HARDWARE_LIST_ROWS = [
    ("CPU sistema (%)", "cpu_percent", 1),
    ("CPU processo total (%)", "process_cpu_percent", 1),
    ("RAM sistema (%)", "ram_percent", 1),
    ("RAM sistema (GB)", "ram_used_gb", 2),
    ("RSS processo (MB)", "process_rss_mb", 0),
    ("GPU uso (%)", "gpu_util_percent", 1),
    ("GPU memoria (%)", "gpu_mem_util_percent", 1),
    ("GPU memoria usada (MB)", "gpu_memory_used_mb", 0),
    ("GPU memoria processo (MB)", "gpu_process_memory_mb", 0),
    ("GPU potencia (W)", "gpu_power_w", 1),
    ("GPU uso processo acct (%)", "gpu_accounted_util_percent", 1),
    ("GPU mem processo acct (%)", "gpu_accounted_mem_util_percent", 1),
    ("GPU max processo acct MB", "gpu_accounted_max_memory_mb", 0),
    ("GPU tempo processo acct ms", "gpu_accounted_time_ms", 0),
    ("Torch alocada (MB)", "torch_allocated_mb", 0),
    ("Torch reservada (MB)", "torch_reserved_mb", 0),
]


VIDEO_STAGE_ORDER = [
    ("read_ms", "Leitura do frame"),
    ("detection_ms", "Deteccao"),
    ("alignment_ms", "Alinhamento"),
    ("embedding_ms", "Geracao de embedding (ArcFace)"),
    ("search_ms", "Busca vetorial (Qdrant)"),
    ("total_ms", "Total pipeline (recognizer)"),
]


def print_hardware_usage_list(summary: dict[str, float | None]) -> None:
    if not summary or not summary.get("enabled"):
        print("  - Telemetria hardware: desativada")
        return
    print(f"  - Tempo de parede: {format_optional(summary.get('elapsed_s'), 2)} s")
    print(f"  - Amostras hardware: {int(summary.get('samples') or 0)}")
    for label, key, digits in HARDWARE_LIST_ROWS:
        print(f"  - {label} (media / max / ultimo): {_triplet(summary, key, digits)}")
    gpu_total = summary.get("gpu_memory_total_mb_last")
    if gpu_total is not None:
        print(f"  - GPU memoria total: {gpu_total:.0f} MB")
    print(f"  - GPU energia estimada: {format_optional(summary.get('gpu_energy_wh'), 4)} Wh")


def print_stage_metrics_list(label: str, stats: dict[str, float]) -> None:
    print(f"\n  {label}:")
    for metric in ("media", "mediana", "min", "p95", "p99", "std"):
        print(f"    - {metric}: {stats[metric]:.3f} ms")


def print_inference_table(stage_summaries: dict[str, dict[str, float]], fps_mean: float) -> None:
    label_w = 32
    col_w = 11
    headers = ("p50", "media", "p95", "p99")
    print(f"  {'Etapa':<{label_w}}" + "".join(f"{h:>{col_w}}" for h in headers) + "  unid.")
    print(f"  {'-' * (label_w + col_w * 4 + 7)}")
    for stage_key, stage_label in VIDEO_STAGE_ORDER:
        stats = stage_summaries.get(stage_key)
        if stats is None:
            continue
        row = (
            f"  {stage_label:<{label_w}}"
            f"{stats['mediana']:>{col_w}.3f}"
            f"{stats['media']:>{col_w}.3f}"
            f"{stats['p95']:>{col_w}.3f}"
            f"{stats['p99']:>{col_w}.3f}"
            f"  ms"
        )
        print(row)
    print(f"  {'-' * (label_w + col_w * 4 + 7)}")
    print(f"  {'FPS (media)':<{label_w}}{fps_mean:>{col_w}.2f}{'':>{col_w * 3}}  frames/s")


def plot_resources_chart(video_results: list[dict[str, Any]], output_path: Path) -> Path | None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    valid = [
        result
        for result in video_results
        if result.get("ok") and result.get("hardware_samples")
    ]
    if not valid:
        return None
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(len(valid), 1, figsize=(9.0, 3.2 * len(valid)), squeeze=False)
    for idx, result in enumerate(valid):
        ax = axes[idx, 0]
        samples = result["hardware_samples"]
        if not samples:
            continue
        t0 = samples[0].get("timestamp_s") or 0.0
        ts = [(s.get("timestamp_s") or t0) - t0 for s in samples]
        cpu = [s.get("cpu_percent") for s in samples]
        gpu = [s.get("gpu_util_percent") for s in samples]
        ax.plot(ts, cpu, label="CPU (%)", color="#1f77b4", marker="o", markersize=3, linewidth=1.2)
        ax.plot(ts, gpu, label="GPU (%)", color="#d62728", marker="s", markersize=3, linewidth=1.2)
        ax.set_title(f"Uso de CPU e GPU - {result['detector'].label}")
        ax.set_xlabel("Tempo (s)")
        ax.set_ylabel("Uso (%)")
        ax.set_ylim(0, 105)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)
    return output_path


def print_fps_summary(fps_summary: dict[str, float]) -> None:
    print(
        "FPS: "
        f"media={fps_summary['media']:.2f} "
        f"mediana={fps_summary['mediana']:.2f} "
        f"min={fps_summary['min']:.2f}"
    )


def bench_cpu(fn, rounds: int, warmup: int = 0) -> np.ndarray:
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(rounds):
        start = time.perf_counter()
        fn()
        times.append((time.perf_counter() - start) * 1000.0)
    return np.asarray(times, dtype=np.float64)


def bench_gpu(fn, rounds: int, warmup: int) -> np.ndarray:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(rounds)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(rounds)]
    for index in range(rounds):
        start_events[index].record()
        fn()
        end_events[index].record()
    torch.cuda.synchronize()
    return np.asarray([s.elapsed_time(e) for s, e in zip(start_events, end_events)], dtype=np.float64)


def bench_gpu_or_cpu(fn, rounds: int, warmup: int, use_cuda: bool) -> np.ndarray:
    if use_cuda:
        return bench_gpu(fn, rounds, warmup)
    return bench_cpu(fn, rounds, warmup=warmup)


def summarize(values: np.ndarray) -> dict[str, float]:
    if values.size == 0:
        return {key: 0.0 for key in ("media", "mediana", "min", "max", "std", "p95", "p99")}
    return {
        "media": float(np.mean(values)),
        "mediana": float(np.median(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "std": float(np.std(values)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
    }


def summarize_fps(total_ms_values: np.ndarray) -> dict[str, float]:
    valid = total_ms_values[total_ms_values > 0]
    if valid.size == 0:
        return {key: 0.0 for key in ("media", "mediana", "min")}
    fps_values = 1000.0 / valid
    return {
        "media": float(np.mean(fps_values)),
        "mediana": float(np.median(fps_values)),
        "min": float(np.min(fps_values)),
    }


HTML_STAGE_ORDER_VIDEO = [
    ("read_ms", "Leitura do frame"),
    ("detection_ms", "Deteccao"),
    ("alignment_ms", "Alinhamento"),
    ("embedding_ms", "Geracao de embedding (ArcFace)"),
    ("search_ms", "Busca vetorial (Qdrant)"),
    ("total_ms", "Total pipeline (recognizer)"),
]


HTML_STAGE_ORDER_IMAGE = [
    ("detection_ms", "Deteccao"),
    ("extraction_ms", "Extracao de regioes"),
    ("alignment_ms", "Alinhamento"),
    ("embedding_ms", "Geracao de embedding (ArcFace)"),
    ("search_ms", "Busca vetorial (Qdrant)"),
    ("total_ms", "Total pipeline"),
]


def html_escape(value: object) -> str:
    return (
        str(value)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def html_optional(value: float | None, digits: int = 1) -> str:
    if value is None:
        return "&mdash;"
    return f"{value:.{digits}f}"


HTML_CSS = """
:root { color-scheme: light; }
body {
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
  margin: 2rem auto;
  max-width: 1080px;
  padding: 0 1rem 3rem;
  color: #1f2933;
  background: #f7f9fc;
}
h1 {
  margin-bottom: 0.25rem;
  border-bottom: 3px solid #1f77b4;
  padding-bottom: 0.5rem;
}
h2 {
  color: #1f4d80;
  margin-top: 2.5rem;
  border-bottom: 1px solid #d0d7de;
  padding-bottom: 0.3rem;
}
h3 { color: #444; margin-top: 1.5rem; }
.meta { color: #57606a; margin: 0 0 1.5rem; font-size: 0.92rem; }
.hardware { list-style: none; padding: 0; }
.hardware li {
  padding: 0.45rem 0;
  border-bottom: 1px dashed #d0d7de;
  display: flex;
  gap: 1rem;
}
.hardware li strong {
  flex: 0 0 200px;
  color: #57606a;
  font-weight: 600;
}
.summary {
  display: flex;
  flex-wrap: wrap;
  gap: 1.25rem;
  margin: 0.75rem 0 1rem;
}
.summary .card {
  background: #fff;
  border: 1px solid #d0d7de;
  border-radius: 8px;
  padding: 0.7rem 1rem;
  min-width: 160px;
  box-shadow: 0 1px 2px rgba(15,23,42,0.04);
}
.summary .card .label { color: #57606a; font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.04em; }
.summary .card .value { font-size: 1.25rem; font-weight: 600; margin-top: 0.2rem; }
table.metrics {
  border-collapse: collapse;
  width: 100%;
  background: #fff;
  border-radius: 8px;
  overflow: hidden;
  box-shadow: 0 1px 2px rgba(15,23,42,0.04);
  font-variant-numeric: tabular-nums;
}
table.metrics th, table.metrics td {
  padding: 0.55rem 0.8rem;
  border-bottom: 1px solid #eef2f6;
  text-align: right;
}
table.metrics th { background: #f0f4f9; color: #1f4d80; font-weight: 600; }
table.metrics th:first-child, table.metrics td:first-child { text-align: left; }
table.metrics tr.fps td { background: #f6fbff; font-weight: 600; }
.chart { background: #fff; border: 1px solid #d0d7de; padding: 0.5rem; border-radius: 8px; }
.chart img { display: block; width: 100%; height: auto; }
.footer { color: #888; font-size: 0.85rem; margin-top: 3rem; padding-top: 1rem; border-top: 1px solid #d0d7de; }
"""


def _stage_metric_row(stage_label: str, stats: dict[str, float]) -> str:
    cells = "".join(
        f"<td>{stats[key]:.3f}</td>"
        for key in ("mediana", "media", "p95", "p99")
    )
    return f"<tr><td>{html_escape(stage_label)}</td>{cells}</tr>"


def _inference_table_html(stage_summaries: dict[str, dict[str, float]], fps_mean: float, mode: str) -> str:
    order = HTML_STAGE_ORDER_VIDEO if mode == "video" else HTML_STAGE_ORDER_IMAGE
    body_rows = []
    for stage_key, stage_label in order:
        stats = stage_summaries.get(stage_key)
        if stats is None:
            continue
        body_rows.append(_stage_metric_row(stage_label, stats))
    body_rows.append(
        '<tr class="fps">'
        '<td>FPS (media)</td>'
        f'<td colspan="4">{fps_mean:.2f} frames/s</td>'
        '</tr>'
    )
    head = (
        "<thead><tr>"
        "<th>Etapa</th>"
        "<th>p50 (ms)</th>"
        "<th>media (ms)</th>"
        "<th>p95 (ms)</th>"
        "<th>p99 (ms)</th>"
        "</tr></thead>"
    )
    return f'<table class="metrics">{head}<tbody>{"".join(body_rows)}</tbody></table>'


def _summary_cards_html(result: dict[str, Any]) -> str:
    cards: list[tuple[str, str]] = [
        ("Frames", str(result.get("frames", len(result.get("detections", []))))),
        ("Faces/frame (media)", f"{result.get('mean_faces_per_frame', 0.0):.2f}"),
    ]
    fwf = result.get("frames_without_faces")
    if fwf is not None:
        cards.append(("Frames sem rosto", str(int(fwf))))
    fragments = [
        f'<div class="card"><div class="label">{html_escape(label)}</div>'
        f'<div class="value">{html_escape(value)}</div></div>'
        for label, value in cards
    ]
    return f'<div class="summary">{"".join(fragments)}</div>'


def write_html_report(
    results: list[dict[str, Any]],
    args: argparse.Namespace,
    mode: str,
    device: str,
    precision: str,
    chart_path: Path | None = None,
) -> Path:
    output_path = Path(args.html_output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    valid_results = [result for result in results if result.get("ok")]
    source = args.image_path if mode == "imagem" else args.video_path
    source_name = Path(source).name if source else mode
    hardware_info = get_hardware_info(device, precision)
    generated_at = datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

    body_parts: list[str] = []
    body_parts.append(
        '<h1>Benchmark de inferencia</h1>'
        f'<p class="meta">Modo: <strong>{html_escape(mode)}</strong> &middot; '
        f'Fonte: <strong>{html_escape(source_name)}</strong> &middot; '
        f'Gerado em {html_escape(generated_at)}</p>'
    )

    body_parts.append("<h2>Hardware</h2>")
    body_parts.append('<ul class="hardware">')
    for label, key in [
        ("CPU", "cpu"),
        ("RAM", "ram"),
        ("GPU", "gpu"),
        ("Driver NVIDIA", "driver"),
        ("Device / precisao", "device"),
    ]:
        body_parts.append(
            f"<li><strong>{html_escape(label)}</strong><span>{html_escape(hardware_info.get(key, '--'))}</span></li>"
        )
    body_parts.append("</ul>")

    if mode == "video" and chart_path is not None and chart_path.exists():
        body_parts.append("<h2>Uso de CPU e GPU</h2>")
        encoded = base64.b64encode(chart_path.read_bytes()).decode("ascii")
        body_parts.append(
            '<div class="chart">'
            f'<img src="data:image/png;base64,{encoded}" alt="Uso de CPU e GPU">'
            '</div>'
        )

    if not valid_results:
        body_parts.append('<h2>Resultados</h2><p>Sem resultados validos.</p>')
    else:
        for result in valid_results:
            detector_label = result["detector"].label
            body_parts.append(f"<h2>{html_escape(detector_label)}</h2>")
            body_parts.append(_summary_cards_html(result))
            body_parts.append("<h3>Tempo de inferencia + FPS</h3>")
            body_parts.append(
                _inference_table_html(
                    result["stage_summaries"],
                    result["fps_summary"]["media"],
                    mode,
                )
            )

    if len(valid_results) >= 2 and mode == "video":
        mtcnn_total = valid_results[0]["stage_summaries"]["total_ms"]["media"]
        yunet_total = valid_results[1]["stage_summaries"]["total_ms"]["media"]
        faster = "YuNet" if yunet_total < mtcnn_total else "MTCNN"
        speedup = max(mtcnn_total, yunet_total) / min(mtcnn_total, yunet_total)
        body_parts.append(
            '<h2>Comparativo</h2>'
            f'<p>{html_escape(faster)} foi <strong>{speedup:.2f}x</strong> mais rapido no total medio.</p>'
        )

    body_parts.append(
        '<div class="footer">benchmark_inference.py &middot; '
        f"device={html_escape(device)} &middot; precisao={html_escape(precision)}</div>"
    )

    html = (
        "<!DOCTYPE html>\n"
        '<html lang="pt-br">\n'
        "<head>\n"
        '  <meta charset="utf-8">\n'
        f"  <title>Benchmark inferencia - {html_escape(source_name)}</title>\n"
        f"  <style>{HTML_CSS}</style>\n"
        "</head>\n"
        "<body>\n"
        + "\n".join(body_parts)
        + "\n</body>\n</html>\n"
    )
    output_path.write_text(html, encoding="utf-8")
    return output_path


def get_cpu_name() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8", errors="ignore").splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return "CPU desconhecida"


def get_hardware_info(device: str, precision: str) -> dict[str, str]:
    cpu_name = get_cpu_name()
    logical_cpus = psutil.cpu_count(logical=True) or 0
    physical_cpus = psutil.cpu_count(logical=False) or 0
    ram_gb = psutil.virtual_memory().total / BYTES_IN_GB
    info = {
        "cpu": f"{cpu_name} ({physical_cpus} cores fisicos, {logical_cpus} threads)",
        "ram": f"{ram_gb:.1f} GB",
        "device": f"{device}, {precision}",
        "gpu": "--",
        "driver": "--",
    }
    if device.startswith("cuda") and torch.cuda.is_available():
        gpu_index = 0
        if device.startswith("cuda:"):
            gpu_index = int(device.split(":", 1)[1])
        props = torch.cuda.get_device_properties(gpu_index)
        info["gpu"] = f"{torch.cuda.get_device_name(gpu_index)} ({props.total_memory / BYTES_IN_GB:.1f} GB)"
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0 and result.stdout.strip():
            info["driver"] = result.stdout.splitlines()[0].strip()
    return info


def print_stats(label: str, values: np.ndarray) -> dict[str, float]:
    stats = summarize(values)
    print(
        f"{label:<28} "
        f"{stats['media']:>8.3f}  "
        f"{stats['mediana']:>8.3f}  "
        f"{stats['min']:>8.3f}  "
        f"{stats['p95']:>8.3f}  "
        f"{stats['p99']:>8.3f}  "
        f"{stats['std']:>8.3f}"
    )
    return stats


def resolve_precision(requested_precision: str, device: str) -> str:
    if requested_precision == "fp16" and not device.startswith("cuda"):
        return "fp32"
    return requested_precision


def prepare_detection_frame(frame_bgr: np.ndarray, detect_width: int | None) -> tuple[np.ndarray, float, float]:
    if not detect_width or detect_width <= 0 or frame_bgr.shape[1] <= detect_width:
        return frame_bgr, 1.0, 1.0
    scale = detect_width / float(frame_bgr.shape[1])
    resized_height = max(1, int(round(frame_bgr.shape[0] * scale)))
    det_frame = cv2.resize(frame_bgr, (detect_width, resized_height), interpolation=cv2.INTER_LINEAR)
    return det_frame, frame_bgr.shape[1] / float(det_frame.shape[1]), frame_bgr.shape[0] / float(det_frame.shape[0])


def scale_detections(detections: list[dict[str, Any]], scale_x: float, scale_y: float, width: int, height: int) -> list[dict[str, Any]]:
    if scale_x == 1.0 and scale_y == 1.0:
        return detections
    scaled: list[dict[str, Any]] = []
    for item in detections:
        entry = {**item}
        x1, y1, x2, y2 = item["bbox"]
        entry["bbox"] = [
            max(0, min(width, int(round(x1 * scale_x)))),
            max(0, min(height, int(round(y1 * scale_y)))),
            max(0, min(width, int(round(x2 * scale_x)))),
            max(0, min(height, int(round(y2 * scale_y)))),
        ]
        if item.get("kps") is not None:
            kps = item["kps"].copy()
            kps[:, 0] *= scale_x
            kps[:, 1] *= scale_y
            entry["kps"] = kps
        scaled.append(entry)
    return scaled


def detect_with_backend(detector_key: str, frame_bgr: np.ndarray, args: argparse.Namespace, device: str) -> list[dict[str, Any]]:
    det_frame, scale_x, scale_y = prepare_detection_frame(frame_bgr, args.detect_width)
    if detector_key == "mtcnn":
        if args.use_fast_mtcnn:
            fast_mtcnn_device = device if device != "mps" else "cpu"
            detector = FastMTCNN(
                stride=args.fast_mtcnn_stride,
                resize=args.fast_mtcnn_resize,
                margin=0,
                post_process=False,
                select_largest=False,
                keep_all=True,
                device=fast_mtcnn_device,
            )
            detections = detector.detect([det_frame[:, :, ::-1]], conf=args.conf_mtcnn, max_det=args.max_faces)[0]
        else:
            detections = detect_faces_mtcnn(det_frame, device=device, conf=args.conf_mtcnn, max_det=args.max_faces)
    else:
        detections = detect_faces_yunet(
            image=det_frame,
            model_path=args.yunet_model,
            conf=args.conf_yunet,
            nms_threshold=args.yunet_nms,
            top_k=args.yunet_top_k,
            max_det=args.max_faces,
            backend=args.yunet_backend,
        )
    return scale_detections(detections, scale_x, scale_y, frame_bgr.shape[1], frame_bgr.shape[0])


def benchmark_detection(detector: DetectorConfig, frame_bgr: np.ndarray, args: argparse.Namespace, device: str) -> tuple[np.ndarray, list[dict[str, Any]]]:
    detections = detect_with_backend(detector.key, frame_bgr, args, device)
    if not detections:
        raise RuntimeError(f"Nenhum rosto detectado com {detector.label}.")
    times = bench_cpu(lambda: detect_with_backend(detector.key, frame_bgr, args, device), rounds=args.rounds, warmup=args.warmup)
    return times, detections


def benchmark_extraction(image_rgb: np.ndarray, detections: list[dict[str, Any]], rounds: int, warmup: int) -> tuple[np.ndarray, list[dict[str, Any]]]:
    face_regions = [extract_face_region(image_rgb=image_rgb, mask=None, bbox=item["bbox"]) for item in detections]
    times = bench_cpu(
        lambda: [extract_face_region(image_rgb=image_rgb, mask=None, bbox=item["bbox"]) for item in detections],
        rounds=rounds,
        warmup=warmup,
    )
    return times, face_regions


def benchmark_alignment(image_bgr: np.ndarray, face_regions: list[dict[str, Any]], detections: list[dict[str, Any]], device: str, rounds: int, warmup: int) -> tuple[np.ndarray, list[np.ndarray], str]:
    def align_one(index: int) -> np.ndarray:
        detection = detections[index]
        if detection.get("kps") is not None:
            return align_face(image_bgr, detection["kps"], image_size=112)
        crop_bgr = face_regions[index]["crop_rgb"][:, :, ::-1].copy()
        aligned_face, _, _ = align_face_crop(
            face_crop_bgr=crop_bgr,
            fallback_crop_bgr=crop_bgr,
            device=device,
            allow_unaligned_fallback=True,
        )
        return aligned_face

    aligned_faces = [align_one(index) for index in range(len(detections))]
    if not aligned_faces:
        raise RuntimeError("Nenhuma face alinhada disponivel para benchmark.")
    times = bench_cpu(lambda: [align_one(index) for index in range(len(detections))], rounds=rounds, warmup=warmup)
    alignment_mode = "keypoints" if any(item.get("kps") is not None for item in detections) else "crop_fallback"
    return times, aligned_faces, alignment_mode


def build_embedding_batch(aligned_faces_bgr: list[np.ndarray], batch_size: int) -> list[np.ndarray]:
    if not aligned_faces_bgr:
        raise ValueError("aligned_faces_bgr nao pode ser vazio.")
    batch = []
    for index in range(batch_size):
        batch.append(aligned_faces_bgr[index % len(aligned_faces_bgr)])
    return batch


def benchmark_embedding(aligned_faces_bgr: list[np.ndarray], args: argparse.Namespace, device: str, compile_model: bool, use_cuda: bool) -> tuple[np.ndarray, ArcFaceEmbedder, list[np.ndarray]]:
    precision = resolve_precision(args.embedder_precision, device)
    embedder = ArcFaceEmbedder(
        weight_path=args.arcface_weight,
        network=args.arcface_network,
        device=device,
        precision=precision,
        compile_model=compile_model,
    )
    batch_faces = build_embedding_batch(aligned_faces_bgr, args.batch_size)
    reference_embeddings = embedder.embed_batch(batch_faces)

    def run_embedding() -> None:
        embedder.embed_batch(batch_faces)

    times = bench_gpu_or_cpu(run_embedding, rounds=args.rounds, warmup=args.warmup, use_cuda=use_cuda)
    return times, embedder, reference_embeddings


def benchmark_search(embeddings: list[np.ndarray], args: argparse.Namespace, rounds: int, warmup: int) -> tuple[np.ndarray, list[list[dict[str, Any]]]]:
    store = QdrantStore(
        url=args.qdrant_url,
        api_key=args.qdrant_api_key,
        timeout=args.qdrant_timeout,
    )
    reference_matches = store.query_batch(
        collection_name=args.collection,
        vectors=embeddings,
        limit=args.candidate_limit,
        score_threshold=args.score_threshold,
    )

    def run_search() -> None:
        store.query_batch(
            collection_name=args.collection,
            vectors=embeddings,
            limit=args.candidate_limit,
            score_threshold=args.score_threshold,
        )

    times = bench_cpu(run_search, rounds=rounds, warmup=warmup)
    return times, reference_matches


def summarize_recognition(matches: list[list[dict[str, Any]]], args: argparse.Namespace) -> list[dict[str, Any]]:
    results = []
    for face_matches in matches:
        best_match, grouped = select_best_identity_match(
            face_matches,
            limit=args.candidate_limit,
            min_score=args.score_threshold,
            score_margin=args.score_margin,
            min_match_count=args.min_identity_votes,
        )
        top = grouped[0] if grouped else None
        results.append(
            {
                "recognized": best_match is not None,
                "name": best_match.get("name") if best_match else None,
                "score": best_match.get("score") if best_match else None,
                "top_candidate_name": top.get("name") if top else None,
                "top_candidate_score": top.get("score") if top else None,
                "top_candidate_match_count": top.get("match_count") if top else 0,
            }
        )
    return results


def benchmark_image_variant(detector: DetectorConfig, image_rgb: np.ndarray, frame_bgr: np.ndarray, args: argparse.Namespace, device: str, compile_model: bool, use_cuda: bool) -> dict[str, Any]:
    hardware_monitor = HardwareMonitor(
        device=device,
        sample_interval=args.hardware_stats_interval,
        enabled=not args.disable_hardware_stats,
    )
    variant_started_at = time.perf_counter()
    hardware_monitor.maybe_sample(force=True)
    detection_times, detections = benchmark_detection(detector, frame_bgr, args, device)
    hardware_monitor.maybe_sample(force=True)
    extraction_times, face_regions = benchmark_extraction(image_rgb, detections, args.rounds, args.warmup)
    hardware_monitor.maybe_sample(force=True)
    alignment_times, aligned_faces, alignment_mode = benchmark_alignment(frame_bgr, face_regions, detections, device, args.rounds, args.warmup)
    hardware_monitor.maybe_sample(force=True)
    embedding_times, embedder, embeddings = benchmark_embedding(aligned_faces, args, device, compile_model, use_cuda)
    hardware_monitor.maybe_sample(force=True)
    search_times, matches = benchmark_search(embeddings, args, args.rounds, args.warmup)
    hardware_monitor.maybe_sample(force=True)
    recognition = summarize_recognition(matches, args)

    gc.collect()
    if use_cuda:
        torch.cuda.empty_cache()

    stage_summaries = {
        "detection_ms": summarize(detection_times),
        "extraction_ms": summarize(extraction_times),
        "alignment_ms": summarize(alignment_times),
        "embedding_ms": summarize(embedding_times),
        "search_ms": summarize(search_times),
    }
    stage_summaries["total_ms"] = {
        key: stage_summaries["detection_ms"][key]
        + stage_summaries["extraction_ms"][key]
        + stage_summaries["alignment_ms"][key]
        + stage_summaries["embedding_ms"][key]
        + stage_summaries["search_ms"][key]
        for key in stage_summaries["detection_ms"]
    }
    total_times = detection_times + extraction_times + alignment_times + embedding_times + search_times
    return {
        "detector": detector,
        "ok": True,
        "detections": detections,
        "alignment_mode": alignment_mode,
        "embedder_precision": embedder.precision,
        "recognition": recognition,
        "stage_summaries": stage_summaries,
        "fps_summary": summarize_fps(total_times),
        "hardware_summary": hardware_monitor.summary(time.perf_counter() - variant_started_at),
    }


def benchmark_video_variant(detector: DetectorConfig, args: argparse.Namespace, device: str, compile_model: bool) -> dict[str, Any]:
    hardware_monitor = HardwareMonitor(
        device=device,
        sample_interval=args.hardware_stats_interval,
        enabled=not args.disable_hardware_stats,
    )
    variant_started_at = time.perf_counter()
    precision = resolve_precision(args.embedder_precision, device)
    recognizer = FaceRecognizer(
        qdrant_url=args.qdrant_url,
        qdrant_api_key=args.qdrant_api_key,
        qdrant_timeout=args.qdrant_timeout,
        collection_name=args.collection,
        arcface_weight=args.arcface_weight,
        arcface_network=args.arcface_network,
        embedder_precision=precision,
        conf=args.conf_mtcnn if detector.key == "mtcnn" else args.conf_yunet,
        score_threshold=args.score_threshold,
        device=device,
        require_alignment=args.require_alignment,
        max_faces=args.max_faces,
        detect_width=None if args.detect_width == 0 else args.detect_width,
        align_faces=not args.disable_alignment,
        temporal_cache=not args.disable_temporal_cache,
        temporal_iou_threshold=args.temporal_iou_threshold,
        candidate_limit=args.candidate_limit,
        score_margin=args.score_margin,
        min_identity_votes=args.min_identity_votes,
        embedder_backend="arcface",
        arcface_compile=compile_model,
        detect_interval=args.detect_interval,
        use_fast_mtcnn=args.use_fast_mtcnn if detector.key == "mtcnn" else False,
        fast_mtcnn_stride=args.fast_mtcnn_stride,
        fast_mtcnn_resize=args.fast_mtcnn_resize,
        detector_backend=detector.key,
        yunet_model=args.yunet_model,
        yunet_nms=args.yunet_nms,
        yunet_top_k=args.yunet_top_k,
        yunet_backend=args.yunet_backend,
    )

    cap = cv2.VideoCapture(str(args.video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Nao foi possivel abrir o video {args.video_path}.")
    if args.video_start_frame:
        cap.set(cv2.CAP_PROP_POS_FRAMES, args.video_start_frame)

    stage_times: dict[str, list[float]] = {
        "read_ms": [],
        "detection_ms": [],
        "alignment_ms": [],
        "embedding_ms": [],
        "search_ms": [],
        "total_ms": [],
    }
    face_counts: list[int] = []
    processed_frames = 0
    frames_without_faces = 0

    try:
        hardware_monitor.maybe_sample(force=True)
        while True:
            if args.video_max_frames is not None and processed_frames >= args.video_max_frames:
                break
            read_start = time.perf_counter()
            ok, frame = cap.read()
            read_ms = (time.perf_counter() - read_start) * 1000.0
            if not ok or frame is None:
                break

            recognitions = recognizer.recognize_frame(frame)
            timing = dict(recognizer.last_timing)
            stage_times["read_ms"].append(read_ms)
            for key in ("detection_ms", "alignment_ms", "embedding_ms", "search_ms", "total_ms"):
                stage_times[key].append(float(timing.get(key, 0.0)))
            face_counts.append(len(recognitions))
            if not recognitions:
                frames_without_faces += 1
            processed_frames += 1
            hardware_monitor.maybe_sample()
        hardware_monitor.maybe_sample(force=True)
    finally:
        cap.release()

    total_times = np.asarray(stage_times["total_ms"], dtype=np.float64)
    return {
        "detector": detector,
        "ok": True,
        "frames": processed_frames,
        "frames_without_faces": frames_without_faces,
        "mean_faces_per_frame": float(np.mean(face_counts)) if face_counts else 0.0,
        "stage_summaries": {key: summarize(np.asarray(values, dtype=np.float64)) for key, values in stage_times.items()},
        "fps_summary": summarize_fps(total_times),
        "hardware_summary": hardware_monitor.summary(time.perf_counter() - variant_started_at),
        "hardware_samples": list(hardware_monitor.samples),
    }


def print_comparison_table(title: str, variants: list[dict[str, Any]], stage_keys: list[tuple[str, str]]) -> None:
    print(title)
    for variant in variants:
        print()
        print(f"[{variant['detector'].label}]")
        print(TABLE_HEADER)
        print(TABLE_SEP)
        for stage_key, stage_label in stage_keys:
            print_stats(stage_label, np.asarray([variant["stage_summaries"][stage_key]["media"]]))


def main() -> None:
    args = parse_args()
    device = get_torch_device(args.device)
    use_cuda = device.startswith("cuda")
    compile_model = not args.no_compile
    precision = resolve_precision(args.embedder_precision, device)

    if args.image_path:
        print(f"Imagem:        {args.image_path}")
    else:
        print(f"Video:         {args.video_path}")
    print(f"Device:        {device}")
    print(f"Precision:     {precision}")
    print(f"torch.compile: {compile_model}")
    print(f"Qdrant:        {args.qdrant_url} / colecao {args.collection}")
    if use_cuda:
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"GPU:           {gpu_name} ({gpu_mem:.1f} GB)")
    print()

    if args.image_path:
        image = Image.open(args.image_path).convert("RGB")
        image_rgb = np.asarray(image)
        frame_bgr = image_rgb[:, :, ::-1].copy()
        print(f"Resolucao:     {image_rgb.shape[1]}x{image_rgb.shape[0]}")
        print()

        results = []
        for detector in DETECTOR_CONFIGS:
            print(f"Benchmarking {detector.label}...")
            try:
                result = benchmark_image_variant(detector, image_rgb, frame_bgr, args, device, compile_model, use_cuda)
            except Exception as exc:
                result = {
                    "detector": detector,
                    "ok": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            results.append(result)

        print(f"\n{'=' * 90}")
        print("BENCHMARK DE IMAGEM - COMPARATIVO MTCNN VS YUNET")
        print(f"{'=' * 90}\n")
        for result in results:
            detector = result["detector"]
            print(f"[{detector.label}]")
            if not result.get("ok", False):
                print(f"Falhou: {result['error']}\n")
                continue
            print(f"Rostos detectados: {len(result['detections'])}")
            print(f"Modo de alinhamento: {result['alignment_mode']}")
            print(f"Precisao embedding: {result['embedder_precision']}")
            if result["recognition"]:
                first = result["recognition"][0]
                print(
                    "Reconhecimento exemplo: "
                    f"recognized={first['recognized']} "
                    f"name={first['name'] or first['top_candidate_name']} "
                    f"score={first['score'] or first['top_candidate_score']}"
                )
            print(TABLE_HEADER)
            print(TABLE_SEP)
            stage_order = [
                ("detection_ms", "Deteccao"),
                ("extraction_ms", "Extracao"),
                ("alignment_ms", f"Alinhamento ({result['alignment_mode']})"),
                ("embedding_ms", "Embedding ArcFace"),
                ("search_ms", "Busca Qdrant"),
                ("total_ms", "Total estimado"),
            ]
            for stage_key, stage_label in stage_order:
                stats = result["stage_summaries"][stage_key]
                print(
                    f"{stage_label:<28} "
                    f"{stats['media']:>8.3f}  "
                    f"{stats['mediana']:>8.3f}  "
                    f"{stats['min']:>8.3f}  "
                    f"{stats['p95']:>8.3f}  "
                    f"{stats['p99']:>8.3f}  "
                    f"{stats['std']:>8.3f}"
                )
            print_fps_summary(result["fps_summary"])
            print_hardware_summary(result.get("hardware_summary", {}))
            print()

        valid_results = [item for item in results if item.get("ok")]
        if len(valid_results) >= 2:
            mtcnn_total = valid_results[0]["stage_summaries"]["total_ms"]["media"]
            yunet_total = valid_results[1]["stage_summaries"]["total_ms"]["media"]
            faster = "YuNet" if yunet_total < mtcnn_total else "MTCNN"
            speedup = max(mtcnn_total, yunet_total) / min(mtcnn_total, yunet_total)
            print(f"Comparativo final: {faster} foi {speedup:.2f}x mais rapido no total medio.")
        html_path = write_html_report(results, args, "imagem", device, precision)
        print(f"Relatorio HTML: {html_path}")
        return

    video_results = []
    for detector in DETECTOR_CONFIGS:
        print(f"Benchmarking video com {detector.label}...")
        try:
            video_results.append(benchmark_video_variant(detector, args, device, compile_model))
        except Exception as exc:
            video_results.append(
                {
                    "detector": detector,
                    "ok": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    print(f"\n{'=' * 90}")
    print("BENCHMARK DE VIDEO - COMPARATIVO MTCNN VS YUNET")
    print(f"{'=' * 90}\n")

    hardware_info = get_hardware_info(device, precision)
    print("Hardware")
    print("--------")
    print(f"  - CPU: {hardware_info['cpu']}")
    print(f"  - RAM: {hardware_info['ram']}")
    print(f"  - GPU: {hardware_info['gpu']}")
    print(f"  - Driver NVIDIA: {hardware_info['driver']}")
    print(f"  - Device / precisao: {hardware_info['device']}")
    print()

    for result in video_results:
        detector = result["detector"]
        print(f"[{detector.label}]")
        if not result.get("ok", False):
            print(f"  Falhou: {result['error']}\n")
            continue
        print(f"  - Frames processados: {result['frames']}")
        print(f"  - Frames sem rostos reconhecidos: {result['frames_without_faces']}")
        print(f"  - Faces/frame (media): {result['mean_faces_per_frame']:.2f}")

        print("\n  Tempo de inferencia + FPS")
        print("  -------------------------")
        print_inference_table(result["stage_summaries"], result["fps_summary"]["media"])
        print()

    valid_video_results = [item for item in video_results if item.get("ok")]
    chart_path = plot_resources_chart(
        valid_video_results,
        Path(args.html_output).parent / "benchmark_resources.png",
    )
    if chart_path is not None:
        print(f"Grafico de CPU/GPU: {chart_path}")
    elif valid_video_results:
        print("Grafico de CPU/GPU: nao gerado (sem amostras de hardware).")

    if len(valid_video_results) >= 2:
        mtcnn_total = valid_video_results[0]["stage_summaries"]["total_ms"]["media"]
        yunet_total = valid_video_results[1]["stage_summaries"]["total_ms"]["media"]
        faster = "YuNet" if yunet_total < mtcnn_total else "MTCNN"
        speedup = max(mtcnn_total, yunet_total) / min(mtcnn_total, yunet_total)
        print(f"Comparativo final: {faster} foi {speedup:.2f}x mais rapido no total medio.")

    html_path = write_html_report(video_results, args, "video", device, precision, chart_path)
    print(f"Relatorio HTML: {html_path}")


if __name__ == "__main__":
    main()
