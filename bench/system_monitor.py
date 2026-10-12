"""Background sampling thread that records system metrics to parquet."""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Optional

import pandas as pd

from bench.config import SYSTEM_SAMPLE_INTERVAL_S
from bench.probes import CompositeProbe, assert_no_throttling, build_probes


class SystemMonitor:
    """Samples a CompositeProbe at a fixed interval on a background thread.

    Usage:
        mon = SystemMonitor(hardware_id="rpi5")
        mon.start()
        # ... run workload ...
        df = mon.stop()
        df.to_parquet("system_metrics.parquet")
    """

    def __init__(
        self,
        hardware_id: str,
        interval_s: float = SYSTEM_SAMPLE_INTERVAL_S,
        probe: Optional[CompositeProbe] = None,
    ) -> None:
        self.hardware_id = hardware_id
        self.interval_s = float(interval_s)
        self.probe = probe or build_probes(hardware_id)
        self._rows: list[dict[str, float]] = []
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._t0: Optional[float] = None

    def _loop(self) -> None:
        next_tick = time.perf_counter()
        while not self._stop.is_set():
            now = time.perf_counter()
            sample = self.probe.sample()
            # Somente o que vira metrica no artigo: potencia (energia por integracao) e
            # temperatura (robustez termica). Utilizacao de CPU/GPU, RAM e VRAM geram ruido
            # de amostragem sem entrar em nenhuma tabela, e a propria coleta custa ciclos.
            # cpu_freq_mhz e gpu_freq_mhz entram porque a bancada mede com clocks
            # dinamicos: sem a serie de frequencia, variacao de P95 entre rodadas
            # fica sem instrumento que a explique.
            PERMITIDAS = ("gpu_power_w", "board_power_w", "board_power_soma_trilhos",
                          "gpu_temp_c", "cpu_temp_c", "tj_temp_c",
                          "cpu_freq_mhz", "gpu_freq_mhz",
                          "throttled_now", "throttled_ever")
            sample = {k: v for k, v in sample.items() if k in PERMITIDAS}
            sample["t_s"] = now - (self._t0 or now)
            sample["wall_ns"] = time.time_ns()
            self._rows.append(sample)
            next_tick += self.interval_s
            sleep = next_tick - time.perf_counter()
            if sleep > 0:
                self._stop.wait(timeout=sleep)
            else:
                next_tick = time.perf_counter()

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("SystemMonitor already started")
        self._t0 = time.perf_counter()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self, *, check_throttling: bool = True) -> pd.DataFrame:
        if self._thread is None:
            raise RuntimeError("SystemMonitor not started")
        self._stop.set()
        self._thread.join(timeout=5.0)
        try:
            if check_throttling:
                assert_no_throttling(self.probe)
        finally:
            self.probe.close()
        df = pd.DataFrame(self._rows)
        df["hardware_id"] = self.hardware_id
        return df

    def torch_memory_snapshot(self) -> dict[str, float]:
        """Capture peak torch CUDA allocations. Returns NaN on non-CUDA systems."""
        try:
            import torch

            if not torch.cuda.is_available():
                return {"torch_alloc_max_gb": float("nan"), "torch_reserved_max_gb": float("nan")}
            alloc = torch.cuda.max_memory_allocated() / (1024 ** 3)
            reserved = torch.cuda.max_memory_reserved() / (1024 ** 3)
            return {
                "torch_alloc_max_gb": float(alloc),
                "torch_reserved_max_gb": float(reserved),
            }
        except ImportError:
            return {"torch_alloc_max_gb": float("nan"), "torch_reserved_max_gb": float("nan")}


def reset_torch_peak_memory() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except ImportError:
        pass


def write_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


__all__ = ["SystemMonitor", "reset_torch_peak_memory", "write_parquet"]
