"""Hardware identification and system_info.json snapshot."""

from __future__ import annotations

import json
import platform
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from bench.config import HardwareSpec, get_hardware_spec


@dataclass
class SystemInfo:
    hardware_id: str
    hostname: str
    kernel: str
    arch: str
    python_version: str
    cpu_model_canonical: Optional[str] = None
    cpu_model_detected: Optional[str] = None
    cpu_cores_canonical: Optional[int] = None
    cpu_cores_detected: Optional[int] = None
    cpu_threads_canonical: Optional[int] = None
    cpu_threads_detected: Optional[int] = None
    cpu_max_freq_mhz: Optional[float] = None
    soc_model: Optional[str] = None
    torch_version: Optional[str] = None
    cuda_version: Optional[str] = None
    cudnn_version: Optional[int] = None
    driver_version: Optional[str] = None
    gpu_model: Optional[str] = None
    qdrant_version: Optional[str] = None
    detector_backend: Optional[str] = None
    detector_model_path: Optional[str] = None
    embedder_backend: Optional[str] = None
    embedder_weight_path: Optional[str] = None
    power_mode: Optional[str] = None
    cpu_governor: Optional[str] = None
    extra: dict = field(default_factory=dict)
    steady_state_start_utc: Optional[str] = None
    steady_state_end_utc: Optional[str] = None

    def to_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.__dict__, indent=2, sort_keys=True))


def _safe_run(cmd: list[str]) -> Optional[str]:
    if shutil.which(cmd[0]) is None:
        return None
    try:
        out = subprocess.run(
            cmd, capture_output=True, text=True, timeout=5.0, check=False
        )
        return out.stdout.strip() or None
    except Exception:
        return None


def _read_first_line(path: str) -> Optional[str]:
    try:
        return Path(path).read_text().strip().splitlines()[0]
    except Exception:
        return None


def detect_cpu_governor() -> Optional[str]:
    return _read_first_line("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor")


def detect_cpu_info() -> dict:
    """Best-effort CPU model + core/thread count + max freq detection.

    Strategy:
        1. /proc/cpuinfo `model name`            (x86 — friendly name like "AMD Ryzen 5 5600X 6-Core Processor")
        2. lscpu `Model name`                    (most systems — same info)
        3. /sys/firmware/devicetree/base/model   (ARM SBCs — "Raspberry Pi 5 Model B Rev 1.0")
        4. /proc/cpuinfo `Hardware`/`Model`      (RPi-specific fallback)

    Returns a dict with optional keys: cpu_model_detected, soc_model,
    cpu_cores_detected, cpu_threads_detected, cpu_max_freq_mhz.
    """
    out: dict = {
        "cpu_model_detected": None,
        "soc_model": None,
        "cpu_cores_detected": None,
        "cpu_threads_detected": None,
        "cpu_max_freq_mhz": None,
    }

    # /proc/cpuinfo — `model name` (x86) or last-resort `Hardware`.
    try:
        cpuinfo_text = Path("/proc/cpuinfo").read_text()
        for line in cpuinfo_text.splitlines():
            if line.startswith("model name") and ":" in line:
                out["cpu_model_detected"] = line.split(":", 1)[1].strip()
                break
        if out["cpu_model_detected"] is None:
            for key in ("Hardware", "Model"):
                for line in cpuinfo_text.splitlines():
                    if line.startswith(key) and ":" in line:
                        out["cpu_model_detected"] = line.split(":", 1)[1].strip()
                        break
                if out["cpu_model_detected"] is not None:
                    break
    except Exception:
        pass

    # /sys/firmware/devicetree/base/model — friendliest on Jetson/RPi/Coral.
    try:
        dt_model = Path("/sys/firmware/devicetree/base/model").read_bytes()
        out["soc_model"] = dt_model.decode(errors="ignore").rstrip("\x00").strip() or None
    except Exception:
        pass

    # lscpu — robust extraction of cores/threads/freq, plus a fallback for model.
    lscpu_text = _safe_run(["lscpu"])
    if lscpu_text:
        for line in lscpu_text.splitlines():
            if ":" not in line:
                continue
            k, v = (s.strip() for s in line.split(":", 1))
            if k in ("Model name", "BIOS Model name") and not out["cpu_model_detected"]:
                out["cpu_model_detected"] = v
            elif k in ("CPU(s)",) and out["cpu_threads_detected"] is None:
                try:
                    out["cpu_threads_detected"] = int(v)
                except ValueError:
                    pass
            elif k in ("Core(s) per socket",) and out["cpu_cores_detected"] is None:
                try:
                    sockets_line = next(
                        (l for l in lscpu_text.splitlines() if l.startswith("Socket(s)")),
                        "Socket(s): 1",
                    )
                    sockets = int(sockets_line.split(":", 1)[1].strip())
                    out["cpu_cores_detected"] = int(v) * sockets
                except (ValueError, StopIteration):
                    pass
            elif k in ("CPU max MHz",) and out["cpu_max_freq_mhz"] is None:
                try:
                    out["cpu_max_freq_mhz"] = float(v)
                except ValueError:
                    pass

    # Final fallback for thread count.
    if out["cpu_threads_detected"] is None:
        try:
            import os as _os

            out["cpu_threads_detected"] = _os.cpu_count()
        except Exception:
            pass

    return out


def detect_jetson_power_mode() -> Optional[str]:
    out = _safe_run(["nvpmodel", "-q"])
    if not out:
        return None
    for line in out.splitlines():
        line = line.strip()
        if line and not line.startswith("NV"):
            return line
    return out.splitlines()[-1] if out else None


def detect_nvidia_driver_and_gpu() -> tuple[Optional[str], Optional[str]]:
    try:
        import pynvml

        pynvml.nvmlInit()
        driver = pynvml.nvmlSystemGetDriverVersion()
        if isinstance(driver, bytes):
            driver = driver.decode()
        h = pynvml.nvmlDeviceGetHandleByIndex(0)
        name = pynvml.nvmlDeviceGetName(h)
        if isinstance(name, bytes):
            name = name.decode()
        pynvml.nvmlShutdown()
        return driver, name
    except Exception:
        return None, None


def detect_qdrant_version(qdrant_url: str = "http://localhost:6333") -> Optional[str]:
    try:
        import urllib.request

        with urllib.request.urlopen(qdrant_url, timeout=2.0) as r:
            data = json.loads(r.read().decode())
            return data.get("version") or data.get("title")
    except Exception:
        return None


def build_system_info(
    hardware_id: str,
    detector_backend: Optional[str] = None,
    detector_model_path: Optional[str] = None,
    embedder_backend: Optional[str] = None,
    embedder_weight_path: Optional[str] = None,
    qdrant_url: str = "http://localhost:6333",
) -> SystemInfo:
    spec = get_hardware_spec(hardware_id)
    cpu_info = detect_cpu_info()
    info = SystemInfo(
        hardware_id=hardware_id,
        hostname=platform.node(),
        kernel=platform.release(),
        arch=platform.machine(),
        python_version=platform.python_version(),
        cpu_model_canonical=spec.cpu_model_canonical,
        cpu_model_detected=cpu_info.get("cpu_model_detected"),
        cpu_cores_canonical=spec.cpu_cores,
        cpu_cores_detected=cpu_info.get("cpu_cores_detected"),
        cpu_threads_canonical=spec.cpu_threads,
        cpu_threads_detected=cpu_info.get("cpu_threads_detected"),
        cpu_max_freq_mhz=cpu_info.get("cpu_max_freq_mhz"),
        soc_model=cpu_info.get("soc_model"),
        cpu_governor=detect_cpu_governor(),
        detector_backend=detector_backend,
        detector_model_path=detector_model_path,
        embedder_backend=embedder_backend,
        embedder_weight_path=embedder_weight_path,
        qdrant_version=detect_qdrant_version(qdrant_url),
    )

    try:
        import torch

        info.torch_version = torch.__version__
        if torch.cuda.is_available():
            info.cuda_version = torch.version.cuda
            try:
                info.cudnn_version = int(torch.backends.cudnn.version() or 0)
            except Exception:
                info.cudnn_version = None
    except Exception:
        pass

    if spec.has_nvidia_gpu:
        driver, gpu = detect_nvidia_driver_and_gpu()
        info.driver_version = driver
        info.gpu_model = gpu

    if spec.is_jetson:
        info.power_mode = detect_jetson_power_mode()
        # /sys/kernel/debug e root-only: Path.exists() levanta PermissionError
        # (nao retorna False) quando rodando sem sudo. Trata como indeterminado.
        try:
            info.extra["jetson_clocks_applied"] = Path("/sys/kernel/debug/clk").exists()
        except OSError:
            info.extra["jetson_clocks_applied"] = None

    if spec.is_rpi:
        info.extra["vcgencmd_throttled_initial"] = _safe_run(["vcgencmd", "get_throttled"])
        info.extra["vcgencmd_measure_temp_initial"] = _safe_run(["vcgencmd", "measure_temp"])

    return info


def get_hardware_label(hardware_id: str) -> str:
    return get_hardware_spec(hardware_id).label


__all__ = [
    "SystemInfo",
    "HardwareSpec",
    "build_system_info",
    "detect_cpu_governor",
    "detect_jetson_power_mode",
    "detect_nvidia_driver_and_gpu",
    "get_hardware_label",
    "get_hardware_spec",
]
