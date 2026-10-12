"""Grupo B — métricas de eficiência energética, derivadas do e2e_video.

A suíte só mede potência INSTANTÂNEA (Watts, amostrada a cada 0,5 s pelo
SystemMonitor). Energia é a integral dessa potência no tempo. Este módulo
faz essa conta e deriva as 4 métricas do grupo, por (hardware, vídeo).

Entradas (por rodada, results/<hw>/<timestamp>/):
    - system_metrics.parquet  → gpu_power_w (W) + wall_ns (relógio absoluto)
    - e2e_per_frame.parquet    → 1 linha/frame medido: latency_ms + timestamp_ns
    - e2e_video.csv            → fps_mean, latency_median_ms, n_frames por vídeo

Métricas:
    - power_mean_w  : potência média SÓ na janela de medição [W]
    - energy_j      : energia integrada na janela (trapézio) [J = W·s]
    - j_per_frame   : energia por frame [J]              ← PRINCIPAL
    - fps_per_watt  : FPS por watt = 1 / j_per_frame     ← complementar
    - edp_j_s       : energy-delay product por frame [J·s] ← secundária

Por que restringir à "janela de medição": o SystemMonitor liga ANTES do
warm-up (60 s), então a média de potência do system_aggregated.csv inclui o
aquecimento. Os frames em e2e_per_frame.parquet são só pós-warm-up. Alinhamos
os dois pelo relógio absoluto (wall_ns no monitor == timestamp_ns no frame,
ambos time.time_ns()) e integramos a potência só entre o 1º e o último frame.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from bench.config import RESULTS_ROOT
from bench.aggregate import discover_runs


def integrate_energy_j(power_w: np.ndarray, t_s: np.ndarray) -> float:
    """Energia = ∫ P dt via regra do trapézio. P em W, t em s (monotônico) → J."""
    mask = ~np.isnan(power_w)
    if mask.sum() < 2:
        return float("nan")
    return float(np.trapz(power_w[mask], t_s[mask]))


def energy_metrics_for_run(run_dir: Path, hardware_id: str) -> list[dict]:
    sysm_path = run_dir / "system_metrics.parquet"
    perf_path = run_dir / "e2e_per_frame.parquet"
    summ_path = run_dir / "e2e_video.csv"
    if not (sysm_path.exists() and perf_path.exists() and summ_path.exists()):
        return []

    sysm = pd.read_parquet(sysm_path)
    perf = pd.read_parquet(perf_path)
    summ = pd.read_csv(summ_path)
    if "gpu_power_w" not in sysm.columns:
        return []

    rows: list[dict] = []
    for video, g in perf.groupby("video"):
        # Janela de medição deste vídeo (relógio absoluto dos frames).
        t0_ns = int(g["timestamp_ns"].min())
        t1_ns = int(g["timestamp_ns"].max())
        win = sysm[(sysm["wall_ns"] >= t0_ns) & (sysm["wall_ns"] <= t1_ns)].sort_values("t_s")

        power = win["gpu_power_w"].to_numpy(dtype=float)
        t = win["t_s"].to_numpy(dtype=float)
        n_frames = int(len(g))

        srow = summ[summ["video"] == video]
        fps = float(srow["fps_mean"].iloc[0]) if not srow.empty else float("nan")
        lat_p50_s = float(srow["latency_median_ms"].iloc[0]) / 1000.0 if not srow.empty else float("nan")

        # --- potência média sob carga (só na janela) ---
        power_mean_w = float(np.nanmean(power)) if power.size else float("nan")

        # --- energia integrada (trapézio) ---
        energy_j = integrate_energy_j(power, t)

        # --- J/frame (PRINCIPAL) ---
        #   método rigoroso: energia da janela / nº de frames
        j_per_frame = energy_j / n_frames if n_frames else float("nan")
        #   cross-check trivial: P_médio / FPS  (= W / (frames/s) = J/frame)
        j_per_frame_check = power_mean_w / fps if fps and fps > 0 else float("nan")

        # --- FPS/W (complementar) — é o recíproco de J/frame ---
        fps_per_watt = fps / power_mean_w if power_mean_w and power_mean_w > 0 else float("nan")

        # --- EDP por frame (secundária): energia/frame × atraso/frame [J·s] ---
        edp_j_s = j_per_frame * lat_p50_s

        rows.append(
            {
                "hardware_id": hardware_id,
                "video": video,
                "run_dir": str(run_dir),
                "n_frames": n_frames,
                "n_power_samples": int((~np.isnan(power)).sum()),
                "fps_mean": fps,
                "power_mean_w": power_mean_w,
                "energy_j": energy_j,
                "j_per_frame": j_per_frame,
                "j_per_frame_check": j_per_frame_check,
                "fps_per_watt": fps_per_watt,
                "edp_j_s": edp_j_s,
                "power_sensor_ok": bool((~np.isnan(power)).any()),
            }
        )
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
            all_rows.extend(energy_metrics_for_run(run, hw_id))

    if not all_rows:
        print(f"Nenhum e2e_video com system_metrics encontrado em {args.results_root}")
        return

    df = pd.DataFrame(all_rows)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / "energy_aggregated.csv"
    df.to_csv(out, index=False)
    print(df.to_string(index=False))
    print(f"\nEscrito em {out}")

    missing = df[~df["power_sensor_ok"]]
    if not missing.empty:
        print("\nAVISO — sem sensor de potência (J/frame não medível por software):")
        for _, r in missing.iterrows():
            print(f"  {r['hardware_id']} / {r['video']}")


if __name__ == "__main__":
    main()
