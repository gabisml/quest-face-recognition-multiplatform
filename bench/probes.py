"""Resource probes — sample CPU/GPU/RAM/temp/power as floats (or NaN).

Each probe exposes:
    .sample() -> dict[str, float]   # NaN if unsupported on the sensor.
    .close()                        # release any handles.

Composition of probes is done by `build_probes(hardware_id)` which picks
the right combination based on the hardware spec.
"""

from __future__ import annotations

import logging
import math
import re
import shutil
import subprocess
import threading
from pathlib import Path
from typing import Optional

import psutil

from bench.config import HardwareSpec, get_hardware_spec

LOG = logging.getLogger("bench.probes")
NAN = float("nan")
_BYTES_GB = 1024 ** 3
_BYTES_MB = 1024 ** 2


class CpuRamProbe:
    """psutil-based CPU% / RAM probe — works everywhere."""

    def __init__(self) -> None:
        psutil.cpu_percent(interval=None)
        # Temperatura de CPU vale em qualquer alvo, nao so nos que tem sonda
        # propria: sem ela nao ha evidencia de estabilidade termica fora das
        # Jetson. psutil cobre x86 (coretemp/k10temp) e o sysfs cobre o resto.
        self._zona = _ThermalZoneProbe()
        # Com clocks dinamicos em toda a bancada, a frequencia e o unico
        # instrumento capaz de explicar variacao de P95 entre rodadas.
        self._freq = _CpuFreqReader()

    def _temp_cpu(self) -> float:
        try:
            leituras = psutil.sensors_temperatures()
        except Exception:
            leituras = {}
        for chave in ("coretemp", "k10temp", "cpu_thermal", "acpitz", "zenpower"):
            if chave in leituras and leituras[chave]:
                return float(max(s.current for s in leituras[chave] if s.current))
        z = self._zona.sample()
        return float(z) if z is not None else NAN

    def sample(self) -> dict[str, float]:
        cpu = psutil.cpu_percent(interval=None)
        vm = psutil.virtual_memory()
        return {
            "cpu_pct": float(cpu),
            "ram_used_gb": float(vm.used) / _BYTES_GB,
            "cpu_temp_c": self._temp_cpu(),
            "cpu_freq_mhz": self._freq.sample(),
        }

    def close(self) -> None:
        pass


class NvmlGpuProbe:
    """pynvml-based probe for desktop/workstation NVIDIA GPUs.

    On Jetsons, pynvml does not report power reliably — prefer TegrastatsProbe.
    """

    def __init__(self) -> None:
        import pynvml

        self._nvml = pynvml
        pynvml.nvmlInit()
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)

    def sample(self) -> dict[str, float]:
        nvml = self._nvml
        out: dict[str, float] = {}
        try:
            util = nvml.nvmlDeviceGetUtilizationRates(self.handle)
            out["gpu_util_pct"] = float(util.gpu)
        except Exception:
            out["gpu_util_pct"] = NAN
        try:
            mem = nvml.nvmlDeviceGetMemoryInfo(self.handle)
            out["vram_used_gb"] = float(mem.used) / _BYTES_GB
        except Exception:
            out["vram_used_gb"] = NAN
        try:
            out["gpu_temp_c"] = float(
                nvml.nvmlDeviceGetTemperature(self.handle, nvml.NVML_TEMPERATURE_GPU)
            )
        except Exception:
            out["gpu_temp_c"] = NAN
        try:
            out["gpu_power_w"] = float(nvml.nvmlDeviceGetPowerUsage(self.handle)) / 1000.0
        except Exception:
            out["gpu_power_w"] = NAN
        try:
            # Clock grafico (SM). E o que o GPU Boost varia; o de memoria fica
            # praticamente fixo e nao explicaria outlier de latencia.
            out["gpu_freq_mhz"] = float(
                nvml.nvmlDeviceGetClockInfo(self.handle, nvml.NVML_CLOCK_GRAPHICS)
            )
        except Exception:
            out["gpu_freq_mhz"] = NAN
        return out

    def close(self) -> None:
        try:
            self._nvml.nvmlShutdown()
        except Exception:
            pass


class _ThermalZoneProbe:
    """Read /sys/class/thermal/thermal_zone*/temp (CPU temp on RPi, Coral)."""

    def __init__(self) -> None:
        self.paths = sorted(Path("/sys/class/thermal").glob("thermal_zone*/temp"))

    def sample(self) -> Optional[float]:
        for p in self.paths:
            try:
                v = int(p.read_text().strip())
                return float(v) / 1000.0
            except Exception:
                continue
        return None


class _CpuFreqReader:
    """Frequencia de CPU pelo sysfs de cpufreq, em MHz.

    Le scaling_cur_freq de cada nucleo. Prefere-se o sysfs ao psutil.cpu_freq()
    porque ele foi verificado nos quatro drivers em uso na bancada (intel_pstate,
    cpufreq-dt, imx8mq-cpufreq e tegra194), enquanto o psutil devolve None em
    parte das placas ARM; o psutil fica como reserva.

    Devolve o MAXIMO entre os nucleos, nao a media: o pipeline e dominado por uma
    thread, e a media sobre 12 a 22 nucleos quase todos ociosos esconderia a
    frequencia do nucleo que esta de fato trabalhando.
    """

    def __init__(self) -> None:
        self.paths = sorted(
            Path("/sys/devices/system/cpu").glob("cpu[0-9]*/cpufreq/scaling_cur_freq")
        )

    def sample(self) -> float:
        melhor = NAN
        for p in self.paths:
            try:
                mhz = int(p.read_text().strip()) / 1000.0
            except Exception:
                continue
            if math.isnan(melhor) or mhz > melhor:
                melhor = mhz
        if not math.isnan(melhor):
            return melhor
        try:
            f = psutil.cpu_freq()
            return float(f.current) if f else NAN
        except Exception:
            return NAN


class _GpuDevfreqReader:
    """Frequencia da GPU integrada pelo devfreq, em MHz (Jetson).

    O caminho e /sys/class/devfreq/<endereco>.gpu/cur_freq, em Hz. Nao se usa o
    GR3D_FREQ do tegrastats porque aquele campo e utilizacao em porcento, nao
    frequencia.
    """

    def __init__(self) -> None:
        self.paths = sorted(Path("/sys/class/devfreq").glob("*.gpu/cur_freq"))

    def sample(self) -> float:
        for p in self.paths:
            try:
                return int(p.read_text().strip()) / 1e6
            except Exception:
                continue
        return NAN


class VcgencmdProbe:
    """Raspberry Pi probe via vcgencmd. Detects throttling.

    Throttling history bits (from `vcgencmd get_throttled`):
        bit 0  Under-voltage detected
        bit 1  Arm frequency capped
        bit 2  Currently throttled
        bit 3  Soft temperature limit active
        bit 16 Under-voltage has occurred
        bit 17 Arm frequency capping has occurred
        bit 18 Throttling has occurred
        bit 19 Soft temperature limit has occurred
    """

    HISTORICO = 0b1111 << 16
    AGORA = 0b1110

    def __init__(self) -> None:
        self._available = shutil.which("vcgencmd") is not None
        self._thermal_zone = _ThermalZoneProbe()
        self.throttled_observed: bool = False
        # Os bits 16-19 sao acumulados desde o boot e so zeram no proximo reinicio.
        # A Pi 5 costuma acender o 16 (sub-tensao) no proprio power-on com fonte que
        # nao seja a de 27 W; contar isso como throttling reprovava toda rodada da
        # placa, mesmo com o SoC a 53 C e o clock em 1,6 GHz. A primeira amostra vira
        # linha de base: so conta o que ACENDER durante a janela.
        self.historico_inicial: Optional[int] = None
        self._historico_avisado: bool = False

    def _call(self, *args: str) -> Optional[str]:
        if not self._available:
            return None
        try:
            out = subprocess.run(
                ["vcgencmd", *args],
                capture_output=True,
                text=True,
                timeout=2.0,
                check=False,
            )
            return out.stdout.strip() or None
        except Exception:
            return None

    def sample(self) -> dict[str, float]:
        temp_c = NAN
        if self._available:
            raw = self._call("measure_temp")
            if raw and "=" in raw:
                m = re.search(r"([0-9.]+)", raw)
                if m:
                    try:
                        temp_c = float(m.group(1))
                    except ValueError:
                        pass
        if math.isnan(temp_c):
            zone = self._thermal_zone.sample()
            if zone is not None:
                temp_c = zone

        throttled_now = NAN
        throttled_ever = NAN
        raw_th = self._call("get_throttled") if self._available else None
        if raw_th and "0x" in raw_th:
            try:
                value = int(raw_th.split("0x")[1], 16)
                if self.historico_inicial is None:
                    self.historico_inicial = value & self.HISTORICO
                novo = (value & self.HISTORICO) & ~self.historico_inicial
                throttled_now = float(bool(value & self.AGORA))
                # A coluna da serie passa a significar "acendeu durante a janela".
                # O que ja vinha do boot vai para o log, uma vez, na primeira amostra.
                throttled_ever = float(bool(novo))
                if self.historico_inicial and not self._historico_avisado:
                    self._historico_avisado = True
                    LOG.warning(
                        "vcgencmd get_throttled ja trazia 0x%05x antes da janela "
                        "(bit 16 sub-tensao, 17 cap de clock, 18 throttling, 19 limite "
                        "termico brando; acumulados desde o boot). Nao reprova a rodada, "
                        "mas bit 16 aceso costuma ser fonte fraca -- confira a PSU.",
                        self.historico_inicial,
                    )
                if value & self.AGORA or novo:
                    self.throttled_observed = True
            except Exception:
                pass

        return {
            "cpu_temp_c": temp_c,
            "throttled_now": throttled_now,
            "throttled_ever": throttled_ever,
            # SBC has no NVIDIA GPU — report NaN columns so schema matches.
            "gpu_util_pct": NAN,
            "vram_used_gb": NAN,
            "gpu_temp_c": NAN,
            "gpu_power_w": NAN,
            "gpu_freq_mhz": NAN,
        }

    def close(self) -> None:
        pass


class TegrastatsProbe:
    """Jetson-specific probe via tegrastats stdout parsing.

    Spawns a background subprocess that streams a line every interval ms.
    Each .sample() returns the most recent parsed row (or NaNs if not yet).
    """

    LINE_RE_GPU = re.compile(r"GR3D_FREQ\s+([0-9]+)%")
    # Os trilhos tem nomes distintos por placa: a AGX Orin emite VDD_GPU_SOC,
    # VDD_CPU_CV e VIN_SYS_5V0; o Orin Nano emite VDD_IN, VDD_CPU_GPU_CV e
    # VDD_SOC. Casar generico e classificar depois evita um regex por placa.
    LINE_RE_TRILHO = re.compile(r"\b(V(?:DD|IN)_[A-Z0-9_]+)\s+([0-9]+)mW")
    # L4T R36 emite as temperaturas em minusculas (cpu@54.7C, gpu@49.4C, tj@54.8C).
    # Os regexes anteriores exigiam maiusculas e por isso nunca casavam -- as
    # colunas de temperatura sairam vazias em toda rodada das duas Jetson.
    LINE_RE_GPU_TEMP = re.compile(r"\bgpu@([0-9.]+)C", re.IGNORECASE)
    LINE_RE_CPU_TEMP = re.compile(r"\bcpu@([0-9.]+)C", re.IGNORECASE)
    LINE_RE_TJ_TEMP = re.compile(r"\btj@([0-9.]+)C", re.IGNORECASE)
    LINE_RE_RAM = re.compile(r"RAM\s+([0-9]+)/([0-9]+)MB")

    def __init__(self, interval_ms: int = 500) -> None:
        self._gpu_freq = _GpuDevfreqReader()
        if shutil.which("tegrastats") is None:
            raise RuntimeError("tegrastats not found in PATH; install jetson-stats")
        self._proc = subprocess.Popen(
            ["tegrastats", "--interval", str(interval_ms)],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )
        self._lock = threading.Lock()
        self._latest: dict[str, float] = {
            "gpu_util_pct": NAN,
            "vram_used_gb": NAN,
            "gpu_temp_c": NAN,
            "cpu_temp_c": NAN,
            "gpu_power_w": NAN,
            "board_power_w": NAN,
        }
        self._stop = threading.Event()
        self._reader = threading.Thread(target=self._read_loop, daemon=True)
        self._reader.start()

    def _read_loop(self) -> None:
        assert self._proc.stdout is not None
        for line in self._proc.stdout:
            if self._stop.is_set():
                break
            parsed = self._parse(line)
            with self._lock:
                self._latest.update(parsed)

    def _parse(self, line: str) -> dict[str, float]:
        out: dict[str, float] = {}
        m = self.LINE_RE_GPU.search(line)
        if m:
            out["gpu_util_pct"] = float(m.group(1))

        trilhos = {nome: float(mw) / 1000.0 for nome, mw in self.LINE_RE_TRILHO.findall(line)}
        if trilhos:
            # Trilho da GPU: o unico que nomeia GPU. Na AGX vem somado ao SoC, no
            # Orin Nano somado a CPU -- a placa nao expoe separado, e o nome do
            # trilho fica registrado em gpu_power_rail para nao virar comparacao
            # entre grandezas diferentes sem aviso.
            for nome, w in trilhos.items():
                if "GPU" in nome:
                    out["gpu_power_w"] = w
                    break
            # Potencia da placa: onde existe trilho de entrada unico (VDD_IN no
            # Orin Nano) usa-se ele. A AGX Orin nao tem, e a soma dos trilhos e a
            # aproximacao usual.
            entrada = next((w for nome, w in trilhos.items()
                            if nome in ("VDD_IN", "VIN_SYS_5V0_IN", "VDD_SYS_IN")), None)
            out["board_power_w"] = entrada if entrada is not None else sum(trilhos.values())
            out["board_power_soma_trilhos"] = 0.0 if entrada is not None else 1.0

        m = self.LINE_RE_GPU_TEMP.search(line)
        if m:
            out["gpu_temp_c"] = float(m.group(1))
        m = self.LINE_RE_CPU_TEMP.search(line)
        if m:
            out["cpu_temp_c"] = float(m.group(1))
        m = self.LINE_RE_TJ_TEMP.search(line)
        if m:
            out["tj_temp_c"] = float(m.group(1))
        m = self.LINE_RE_RAM.search(line)
        if m:
            # RAM used MB (shared CPU+GPU on Jetson)
            out["vram_used_gb"] = float(m.group(1)) / 1024.0
        return out

    def sample(self) -> dict[str, float]:
        with self._lock:
            snapshot = dict(self._latest)
        # As duas potencias saem como colunas separadas. Antes o trilho da placa
        # sobrescrevia o da GPU e nao era devolvido, o que fazia a mesma coluna
        # significar potencia total num alvo e parcial no outro.
        return {
            "gpu_util_pct": snapshot.get("gpu_util_pct", NAN),
            "vram_used_gb": snapshot.get("vram_used_gb", NAN),
            "gpu_temp_c": snapshot.get("gpu_temp_c", NAN),
            "cpu_temp_c": snapshot.get("cpu_temp_c", NAN),
            "tj_temp_c": snapshot.get("tj_temp_c", NAN),
            "gpu_power_w": snapshot.get("gpu_power_w", NAN),
            "board_power_w": snapshot.get("board_power_w", NAN),
            "board_power_soma_trilhos": snapshot.get("board_power_soma_trilhos", NAN),
            "gpu_freq_mhz": self._gpu_freq.sample(),
        }

    def close(self) -> None:
        self._stop.set()
        try:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        except Exception:
            pass


class CoralProbe:
    """Coral Dev Board: no NVIDIA GPU; CPU/RAM via psutil, SoC temp via sysfs."""

    def __init__(self) -> None:
        self._thermal_zone = _ThermalZoneProbe()

    def sample(self) -> dict[str, float]:
        zone = self._thermal_zone.sample()
        return {
            "cpu_temp_c": float(zone) if zone is not None else NAN,
            "gpu_util_pct": NAN,
            "vram_used_gb": NAN,
            "gpu_temp_c": NAN,
            "gpu_power_w": NAN,
            "gpu_freq_mhz": NAN,
        }

    def close(self) -> None:
        pass


class CompositeProbe:
    """Compose CPU/RAM probe with the hardware-specific accelerator probe."""

    def __init__(self, probes: list) -> None:
        self.probes = probes

    def sample(self) -> dict[str, float]:
        merged: dict[str, float] = {}
        for p in self.probes:
            try:
                merged.update(p.sample())
            except Exception:
                continue
        return merged

    def close(self) -> None:
        for p in self.probes:
            try:
                p.close()
            except Exception:
                pass


def build_probes(hardware_id: str) -> CompositeProbe:
    spec: HardwareSpec = get_hardware_spec(hardware_id)
    probes: list = [CpuRamProbe()]

    if spec.is_jetson:
        try:
            probes.append(TegrastatsProbe())
        except RuntimeError:
            # Fallback: NVML on Jetson (limited but better than nothing).
            try:
                probes.append(NvmlGpuProbe())
            except Exception:
                pass
    elif spec.is_rpi:
        probes.append(VcgencmdProbe())
    elif spec.is_coral:
        probes.append(CoralProbe())
    elif spec.has_nvidia_gpu:
        try:
            probes.append(NvmlGpuProbe())
        except Exception:
            pass

    return CompositeProbe(probes)


def assert_no_throttling(probe: CompositeProbe) -> None:
    """Walk the composite probes for a VcgencmdProbe; raise if it saw throttling."""
    for p in probe.probes:
        if isinstance(p, VcgencmdProbe) and p.throttled_observed:
            raise RuntimeError(
                "Raspberry Pi throttling detected during measurement window "
                f"(historico antes da janela: 0x{p.historico_inicial or 0:05x}). "
                "Result is contaminated — cool device, re-run."
            )


__all__ = [
    "build_probes",
    "assert_no_throttling",
    "CompositeProbe",
    "CpuRamProbe",
    "NvmlGpuProbe",
    "VcgencmdProbe",
    "TegrastatsProbe",
    "CoralProbe",
]
