"""Constants, hardware registry, and result paths."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_ROOT = PROJECT_ROOT / "results"
SUBSET_PATH = PROJECT_ROOT / "bench" / "benchmark_subset.json"

STAGES = ("read", "detection", "alignment", "embedding", "search", "total")

DEFAULT_K = 500
DEFAULT_N = 50
DEFAULT_WARMUP = 50
SYSTEM_SAMPLE_INTERVAL_S = 0.5
STEADY_STATE_WARMUP_S = 300.0
VIDEO_WARMUP_S = 60.0
MIN_VIDEO_DURATION_S = 300.0

LATENCY_BUDGETS_MS = (33.0, 66.0, 100.0)


@dataclass(frozen=True)
class HardwareSpec:
    hardware_id: str
    label: str
    category: str
    has_nvidia_gpu: bool
    is_jetson: bool
    is_rpi: bool
    is_coral: bool
    expected_governor_cmd: Optional[str]
    cpu_model_canonical: str
    cpu_arch: str  # "x86_64" or "aarch64"
    cpu_cores: int
    cpu_threads: int


HARDWARE_REGISTRY: dict[str, HardwareSpec] = {
    "rtx3090": HardwareSpec(
        hardware_id="rtx3090",
        label="RTX 3090",
        category="Desktop GPU",
        has_nvidia_gpu=True,
        is_jetson=False,
        is_rpi=False,
        is_coral=False,
        expected_governor_cmd="sudo cpupower frequency-set -g performance",
        cpu_model_canonical="AMD Ryzen 5 5600X (Zen 3)",
        cpu_arch="x86_64",
        cpu_cores=6,
        cpu_threads=12,
    ),
    "rtx4060": HardwareSpec(
        hardware_id="rtx4060",
        label="RTX 4060 (NUC)",
        category="Mini-PC GPU",
        has_nvidia_gpu=True,
        is_jetson=False,
        is_rpi=False,
        is_coral=False,
        expected_governor_cmd="sudo cpupower frequency-set -g performance",
        cpu_model_canonical="Intel Core Ultra 7 155H (Meteor Lake)",
        cpu_arch="x86_64",
        cpu_cores=16,
        cpu_threads=22,
    ),
    "dgx_spark": HardwareSpec(
        hardware_id="dgx_spark",
        label="DGX Spark",
        category="Desktop/Workstation",
        has_nvidia_gpu=True,
        is_jetson=False,
        is_rpi=False,
        is_coral=False,
        expected_governor_cmd="sudo cpupower frequency-set -g performance",
        # GB10 Grace-Blackwell Superchip — 10 Cortex-X925 + 10 Cortex-A725.
        cpu_model_canonical="NVIDIA GB10 Grace (10× Cortex-X925 + 10× Cortex-A725)",
        cpu_arch="aarch64",
        cpu_cores=20,
        cpu_threads=20,
    ),
    "jetson_agx_orin": HardwareSpec(
        hardware_id="jetson_agx_orin",
        label="Jetson AGX Orin",
        category="Edge GPU (high)",
        has_nvidia_gpu=True,
        is_jetson=True,
        is_rpi=False,
        is_coral=False,
        expected_governor_cmd="sudo nvpmodel -m 0 && sudo jetson_clocks",
        cpu_model_canonical="Arm Cortex-A78AE v8.2 (12 cores)",
        cpu_arch="aarch64",
        cpu_cores=12,
        cpu_threads=12,
    ),
    "jetson_orin_nano": HardwareSpec(
        hardware_id="jetson_orin_nano",
        label="Jetson Orin Nano",
        category="Edge GPU (low)",
        has_nvidia_gpu=True,
        is_jetson=True,
        is_rpi=False,
        is_coral=False,
        expected_governor_cmd="sudo nvpmodel -m 0 && sudo jetson_clocks",
        cpu_model_canonical="Arm Cortex-A78AE v8.2 (6 cores)",
        cpu_arch="aarch64",
        cpu_cores=6,
        cpu_threads=6,
    ),
    "rpi5": HardwareSpec(
        hardware_id="rpi5",
        label="Raspberry Pi 5",
        category="SBC CPU",
        has_nvidia_gpu=False,
        is_jetson=False,
        is_rpi=True,
        is_coral=False,
        expected_governor_cmd="sudo cpufreq-set -g performance",
        cpu_model_canonical="Broadcom BCM2712 — 4× Arm Cortex-A76 @ 2.4 GHz",
        cpu_arch="aarch64",
        cpu_cores=4,
        cpu_threads=4,
    ),
    "rpi4": HardwareSpec(
        hardware_id="rpi4",
        label="Raspberry Pi 4",
        category="SBC CPU",
        has_nvidia_gpu=False,
        is_jetson=False,
        is_rpi=True,
        is_coral=False,
        expected_governor_cmd="sudo cpufreq-set -g performance",
        cpu_model_canonical="Broadcom BCM2711 — 4× Arm Cortex-A72 @ 1.5–1.8 GHz",
        cpu_arch="aarch64",
        cpu_cores=4,
        cpu_threads=4,
    ),
    "coral_dev": HardwareSpec(
        hardware_id="coral_dev",
        label="Coral Dev Board",
        category="SBC + Edge TPU",
        has_nvidia_gpu=False,
        is_jetson=False,
        is_rpi=False,
        is_coral=True,
        expected_governor_cmd="sudo cpufreq-set -g performance",
        cpu_model_canonical="NXP i.MX 8M — 4× Arm Cortex-A53 @ 1.5 GHz",
        cpu_arch="aarch64",
        cpu_cores=4,
        cpu_threads=4,
    ),
}


def get_hardware_spec(hardware_id: str) -> HardwareSpec:
    if hardware_id not in HARDWARE_REGISTRY:
        valid = ", ".join(sorted(HARDWARE_REGISTRY.keys()))
        raise ValueError(f"hardware_id desconhecido: {hardware_id!r}. Validos: {valid}")
    return HARDWARE_REGISTRY[hardware_id]


def result_dir(hardware_id: str, timestamp: str) -> Path:
    return RESULTS_ROOT / hardware_id / timestamp
