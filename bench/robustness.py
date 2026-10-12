"""Grupo D (parte derivável) — throttling e burst-vs-sustained sobre o e2e_video.

Estas duas métricas são pós-processamento dos traços que o e2e_video já grava:
    - e2e_per_frame.parquet  → latency_ms + t_s (tempo de cada frame na medição)
    - system_metrics.parquet → gpu_temp_c/cpu_temp_c + gpu_power_w ao longo do tempo
      (+ throttled_now/throttled_ever no Raspberry Pi, quando existem)

As outras duas métricas do Grupo D (drift de embeddings e degradação
FP32→FP16→INT8) NÃO estão aqui: elas exigem rodar o modelo em cada precisão,
não dá pra derivar de traço. Quem mede isso é precision_accuracy/, que roda a
matriz dataset × precisão e reporta acurácia por k-fold.

Burst vs sustained: com o protocolo atual (warm-up de 60 s descartado), o
per_frame só tem a fase sustentada — então aqui comparo o INÍCIO com o FIM
dessa fase (detecta queda por aquecimento). "Burst" de verdade (clocks em
boost, no primeiro segundo a frio) exige rodar o e2e com --warmup-s 0.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from bench.config import RESULTS_ROOT
from bench.aggregate import discover_runs


def _fps_in_window(t_s: np.ndarray, lo: float, hi: float) -> float:
    """FPS = (nº de frames - 1) / (duração), contando só frames em [lo, hi]."""
    tw = t_s[(t_s >= lo) & (t_s <= hi)]
    if tw.size < 2:
        return float("nan")
    span = float(tw.max() - tw.min())
    return (tw.size - 1) / span if span > 0 else float("nan")


def burst_vs_sustained(per: pd.DataFrame, head_s: float = 10.0, tail_s: float = 30.0) -> dict:
    """Compara FPS/latência no início vs no fim da fase medida.

    ratio_fps < 1  ⇒  a máquina desacelerou ao longo da medição (suspeita de
    throttling térmico). ratio_fps ≈ 1 ⇒ desempenho estável.
    """
    t = per["t_s"].to_numpy(dtype=float)
    lat = per["latency_ms"].to_numpy(dtype=float)
    t0, t1 = float(t.min()), float(t.max())

    fps_burst = _fps_in_window(t, t0, t0 + head_s)
    fps_sustained = _fps_in_window(t, t1 - tail_s, t1)

    head_mask = (t >= t0) & (t <= t0 + head_s)
    tail_mask = (t >= t1 - tail_s) & (t <= t1)
    lat_burst_p50 = float(np.median(lat[head_mask])) if head_mask.any() else float("nan")
    lat_sustained_p50 = float(np.median(lat[tail_mask])) if tail_mask.any() else float("nan")

    ratio = fps_sustained / fps_burst if fps_burst and fps_burst > 0 else float("nan")
    return {
        "fps_burst": fps_burst,
        "fps_sustained": fps_sustained,
        "fps_ratio_sustained_over_burst": ratio,
        "lat_burst_p50_ms": lat_burst_p50,
        "lat_sustained_p50_ms": lat_sustained_p50,
        # <0,95 = perdeu >5% de FPS do começo ao fim da janela.
        "slowdown_flag": bool(ratio < 0.95) if not np.isnan(ratio) else False,
    }


def throttling_report(sysm: pd.DataFrame, edge_s: float = 10.0) -> dict:
    """Aquecimento e sinais de throttling ao longo da medição.

    temp_delta_c > 0 e crescente indica que o dispositivo não saturou
    termicamente; combinado com slowdown_flag do burst_vs_sustained, é a
    assinatura de throttling. No Raspberry Pi, throttled_ever vem dos bits
    do vcgencmd (fonte de verdade direta).
    """
    def _temp_series(df: pd.DataFrame) -> np.ndarray:
        gpu = df["gpu_temp_c"] if "gpu_temp_c" in df else pd.Series(dtype=float)
        cpu = df["cpu_temp_c"] if "cpu_temp_c" in df else pd.Series(dtype=float)
        s = gpu.where(gpu.notna() & (gpu != 0), cpu)
        return s.to_numpy(dtype=float)

    t = sysm["t_s"].to_numpy(dtype=float)
    temp = _temp_series(sysm)
    t0, t1 = float(t.min()), float(t.max())

    def _window_mean(lo: float, hi: float) -> float:
        seg = temp[(t >= lo) & (t <= hi)]
        seg = seg[np.isfinite(seg)]
        return float(seg.mean()) if seg.size else float("nan")

    temp_start = _window_mean(t0, t0 + edge_s)
    temp_end = _window_mean(t1 - edge_s, t1)
    temp_peak = float(np.nanmax(temp)) if np.isfinite(temp).any() else float("nan")

    out = {
        "temp_start_c": temp_start,
        "temp_end_c": temp_end,
        "temp_delta_c": (temp_end - temp_start) if np.isfinite(temp_start) and np.isfinite(temp_end) else float("nan"),
        "temp_peak_c": temp_peak,
        "temp_sensor_ok": bool(np.isfinite(temp).any()),
    }
    # Raspberry Pi: bits de throttling do vcgencmd (verdade direta do HW).
    for col in ("throttled_now", "throttled_ever"):
        if col in sysm.columns:
            vals = sysm[col].to_numpy(dtype=float)
            out[col] = bool(np.nanmax(vals) >= 1.0) if np.isfinite(vals).any() else False
    return out


def robustness_for_run(run_dir: Path, hardware_id: str) -> list[dict]:
    perf_path = run_dir / "e2e_per_frame.parquet"
    sysm_path = run_dir / "system_metrics.parquet"
    if not perf_path.exists():
        return []
    perf = pd.read_parquet(perf_path)
    sysm = pd.read_parquet(sysm_path) if sysm_path.exists() else pd.DataFrame()

    rows: list[dict] = []
    for video, g in perf.groupby("video"):
        row = {"hardware_id": hardware_id, "video": video, "run_dir": str(run_dir), "n_frames": int(len(g))}
        row.update(burst_vs_sustained(g))
        if not sysm.empty:
            row.update(throttling_report(sysm))
        rows.append(row)
    return rows


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--results-root", type=Path, default=RESULTS_ROOT)
    p.add_argument("--out-dir", type=Path, default=RESULTS_ROOT / "_aggregated")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    runs = discover_runs(args.results_root)
    all_rows: list[dict] = []
    for hw_id, run_dirs in runs.items():
        for run in run_dirs:
            all_rows.extend(robustness_for_run(run, hw_id))
    if not all_rows:
        print(f"Nenhum e2e_per_frame encontrado em {args.results_root}")
        return
    df = pd.DataFrame(all_rows)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / "robustness_aggregated.csv"
    df.to_csv(out, index=False)
    cols = ["hardware_id", "fps_burst", "fps_sustained", "fps_ratio_sustained_over_burst",
            "slowdown_flag", "temp_delta_c", "temp_peak_c"]
    print(df[[c for c in cols if c in df.columns]].to_string(index=False))
    print(f"\nEscrito em {out}")


if __name__ == "__main__":
    main()
