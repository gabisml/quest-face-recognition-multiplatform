"""Aggregate parquet outputs into CSV summaries.

Two outputs:
    stage_aggregated.csv   — (hardware, stage) -> median_ms, p95_ms, max_ms
    system_aggregated.csv  — (hardware) -> cpu/ram/gpu/vram/temp/power means+peaks

If multiple runs exist for the same hardware (the protocol asks for 3 replications),
the script computes the median of each per-run aggregated value across runs.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from bench.config import RESULTS_ROOT, STAGES


def _safe_median(values: list[float]) -> float:
    arr = np.asarray([v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))], dtype=float)
    return float(np.median(arr)) if arr.size else float("nan")


def _safe_max(values: list[float]) -> float:
    arr = np.asarray([v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))], dtype=float)
    return float(np.max(arr)) if arr.size else float("nan")


def discover_runs(results_root: Path) -> dict[str, list[Path]]:
    """Map hardware_id -> list of run directories sorted by timestamp."""
    runs: dict[str, list[Path]] = {}
    if not results_root.exists():
        return runs
    for hw_dir in sorted(results_root.iterdir()):
        if not hw_dir.is_dir():
            continue
        run_dirs = sorted([p for p in hw_dir.iterdir() if p.is_dir()])
        if run_dirs:
            runs[hw_dir.name] = run_dirs
    return runs


def aggregate_stage_two_level(raw_df: pd.DataFrame, hardware_id: str) -> list[dict]:
    """Re-aggregate from raw rows in case the per-run CSV is missing or stale."""
    rows: list[dict] = []
    for stage in STAGES:
        sub = raw_df[raw_df["stage"] == stage]
        if sub.empty:
            rows.append(
                {
                    "hardware_id": hardware_id,
                    "stage": stage,
                    "median_ms": float("nan"),
                    "p95_ms": float("nan"),
                    "max_ms": float("nan"),
                    "n_images": 0,
                    "n_reps": 0,
                }
            )
            continue
        per_image = sub.groupby("image_id")["latency_ns"].median()
        per_image_ms = per_image.to_numpy() / 1_000_000.0
        rows.append(
            {
                "hardware_id": hardware_id,
                "stage": stage,
                "median_ms": float(np.median(per_image_ms)),
                "p95_ms": float(np.percentile(per_image_ms, 95)),
                "max_ms": float(np.max(per_image_ms)),
                "n_images": int(per_image_ms.shape[0]),
                "n_reps": int(sub.groupby("image_id").size().median()),
            }
        )
    return rows


def aggregate_stage(runs: dict[str, list[Path]]) -> pd.DataFrame:
    rows: list[dict] = []
    for hw_id, run_dirs in runs.items():
        per_run: dict[str, list[dict]] = {stage: [] for stage in STAGES}
        for run in run_dirs:
            agg_csv = run / "stage_aggregated.csv"
            raw_parquet = run / "stage_raw.parquet"
            if agg_csv.exists():
                df = pd.read_csv(agg_csv)
            elif raw_parquet.exists():
                df = pd.DataFrame(aggregate_stage_two_level(pd.read_parquet(raw_parquet), hw_id))
            else:
                continue
            for _, r in df.iterrows():
                per_run[r["stage"]].append(r.to_dict())

        for stage in STAGES:
            entries = per_run[stage]
            if not entries:
                continue
            medians = [e.get("median_ms", float("nan")) for e in entries]
            p95s = [e.get("p95_ms", float("nan")) for e in entries]
            maxes = [e.get("max_ms", float("nan")) for e in entries]
            n_images = max((int(e.get("n_images", 0)) for e in entries), default=0)
            n_reps = max((int(e.get("n_reps", 0)) for e in entries), default=0)
            spread = max(medians) - min(medians) if len(medians) > 1 else 0.0
            base = float(np.median(medians)) if medians else 0.0
            high_variance = base > 0 and (spread / base) > 0.20

            rows.append(
                {
                    "hardware_id": hw_id,
                    "stage": stage,
                    "median_ms": _safe_median(medians),
                    "p95_ms": _safe_median(p95s),
                    "max_ms": _safe_median(maxes),
                    "n_runs": len(entries),
                    "n_images": n_images,
                    "n_reps": n_reps,
                    "between_run_spread_pct": float(spread / base * 100.0) if base else 0.0,
                    "high_variance_flag": bool(high_variance),
                }
            )
    return pd.DataFrame(rows)


def _read_system_info_cpu(run: Path) -> dict:
    """Pull CPU fields from system_info.json into the aggregated row."""
    info_path = run / "system_info.json"
    if not info_path.exists():
        return {}
    try:
        info = json.loads(info_path.read_text())
    except Exception:
        return {}
    return {
        "cpu_model_canonical": info.get("cpu_model_canonical"),
        "cpu_model_detected": info.get("cpu_model_detected"),
        "cpu_cores_canonical": info.get("cpu_cores_canonical"),
        "cpu_threads_canonical": info.get("cpu_threads_canonical"),
        "cpu_max_freq_mhz": info.get("cpu_max_freq_mhz"),
        "cpu_governor": info.get("cpu_governor"),
    }


def aggregate_system(runs: dict[str, list[Path]]) -> pd.DataFrame:
    rows: list[dict] = []
    for hw_id, run_dirs in runs.items():
        per_run_summary: list[dict] = []
        cpu_meta: dict = {}
        for run in run_dirs:
            sys_parquet = run / "system_metrics.parquet"
            if not sys_parquet.exists():
                continue
            df = pd.read_parquet(sys_parquet)
            torch_mem_path = run / "torch_memory.json"
            torch_mem = {}
            if torch_mem_path.exists():
                try:
                    torch_mem = json.loads(torch_mem_path.read_text())
                except Exception:
                    torch_mem = {}
            # First run wins for static CPU metadata; later runs only override if missing.
            for k, v in _read_system_info_cpu(run).items():
                if v is not None and cpu_meta.get(k) in (None, ""):
                    cpu_meta[k] = v

            cpu = df["cpu_pct"] if "cpu_pct" in df else pd.Series([np.nan])
            ram = df["ram_used_gb"] if "ram_used_gb" in df else pd.Series([np.nan])
            gpu_util = df["gpu_util_pct"] if "gpu_util_pct" in df else pd.Series([np.nan])
            vram = df["vram_used_gb"] if "vram_used_gb" in df else pd.Series([np.nan])
            gpu_temp = df["gpu_temp_c"] if "gpu_temp_c" in df else pd.Series([np.nan])
            cpu_temp = df["cpu_temp_c"] if "cpu_temp_c" in df else pd.Series([np.nan])
            power = df["gpu_power_w"] if "gpu_power_w" in df else pd.Series([np.nan])

            # Use the first non-NaN of (gpu_temp, cpu_temp) as the canonical temp.
            temp_series = gpu_temp.where(gpu_temp.notna() & (gpu_temp != 0), cpu_temp)

            per_run_summary.append(
                {
                    "cpu_mean_pct": float(np.nanmean(cpu)) if len(cpu) else float("nan"),
                    "cpu_max_pct": float(np.nanmax(cpu)) if len(cpu) else float("nan"),
                    "ram_mean_gb": float(np.nanmean(ram)) if len(ram) else float("nan"),
                    "ram_max_gb": float(np.nanmax(ram)) if len(ram) else float("nan"),
                    "gpu_mean_pct": float(np.nanmean(gpu_util)) if len(gpu_util) else float("nan"),
                    "vram_max_gb": float(np.nanmax(vram)) if len(vram) else float("nan"),
                    "temp_mean_c": float(np.nanmean(temp_series)) if len(temp_series) else float("nan"),
                    "power_mean_w": float(np.nanmean(power)) if len(power) else float("nan"),
                    "torch_alloc_max_gb": float(torch_mem.get("torch_alloc_max_gb", float("nan"))),
                    "torch_reserved_max_gb": float(torch_mem.get("torch_reserved_max_gb", float("nan"))),
                }
            )

        if not per_run_summary:
            continue
        # Fall back to the canonical registry entry if system_info.json was missing.
        try:
            from bench.config import get_hardware_spec

            spec = get_hardware_spec(hw_id)
            cpu_meta.setdefault("cpu_model_canonical", spec.cpu_model_canonical)
            cpu_meta.setdefault("cpu_cores_canonical", spec.cpu_cores)
            cpu_meta.setdefault("cpu_threads_canonical", spec.cpu_threads)
        except ValueError:
            pass
        row: dict[str, object] = {
            "hardware_id": hw_id,
            "n_runs": len(per_run_summary),
            **cpu_meta,
        }
        for key in (
            "cpu_mean_pct",
            "cpu_max_pct",
            "ram_mean_gb",
            "ram_max_gb",
            "gpu_mean_pct",
            "vram_max_gb",
            "temp_mean_c",
            "power_mean_w",
            "torch_alloc_max_gb",
            "torch_reserved_max_gb",
        ):
            values = [s[key] for s in per_run_summary]
            if key.endswith("_max_pct") or key.endswith("_max_gb"):
                row[key] = _safe_max(values)
            else:
                row[key] = _safe_median(values)
        rows.append(row)

    return pd.DataFrame(rows)


def aggregate_e2e(runs: dict[str, list[Path]]) -> pd.DataFrame:
    rows: list[dict] = []
    for hw_id, run_dirs in runs.items():
        for run in run_dirs:
            e2e_csv = run / "e2e_video.csv"
            if not e2e_csv.exists():
                continue
            df = pd.read_csv(e2e_csv)
            df["run_dir"] = str(run)
            rows.append(df)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--results-root", type=Path, default=RESULTS_ROOT)
    p.add_argument("--out-dir", type=Path, default=RESULTS_ROOT / "_aggregated")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    runs = discover_runs(args.results_root)
    if not runs:
        print(f"No runs found under {args.results_root}")
        return

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stage_df = aggregate_stage(runs)
    system_df = aggregate_system(runs)
    e2e_df = aggregate_e2e(runs)

    stage_df.to_csv(args.out_dir / "stage_aggregated.csv", index=False)
    system_df.to_csv(args.out_dir / "system_aggregated.csv", index=False)
    if not e2e_df.empty:
        e2e_df.to_csv(args.out_dir / "e2e_video.csv", index=False)

    print(f"Aggregated {sum(len(v) for v in runs.values())} runs across {len(runs)} hardwares")
    print(f"Outputs in {args.out_dir}")

    flagged = stage_df[stage_df.get("high_variance_flag", False) == True]  # noqa: E712
    if not flagged.empty:
        print("\nWARNING — high between-run variance (>20%):")
        for _, r in flagged.iterrows():
            print(
                f"  {r['hardware_id']:>20s}  {r['stage']:>10s}  "
                f"spread={r['between_run_spread_pct']:.1f}%"
            )


if __name__ == "__main__":
    main()
