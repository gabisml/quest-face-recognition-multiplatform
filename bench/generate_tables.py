"""Generate the five LaTeX table fragments from aggregated CSVs.

Inputs:
    <results>/_aggregated/system_aggregated.csv
    <results>/_aggregated/stage_aggregated.csv

Outputs (under --out-dir, default same directory):
    table1_cpu_ram.tex
    table2_gpu_vram_temp_power.tex
    table3_latency_median.tex
    table4_latency_p95.tex
    table5_latency_max.tex

Cells default to `--` when the corresponding sensor is unavailable
(NaN in CSV) — matches the protocol's `--` for SBCs without an NVIDIA GPU.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd

from bench.config import HARDWARE_REGISTRY, RESULTS_ROOT, STAGES, get_hardware_spec

STAGE_LABELS = {
    "read": "Leitura",
    "detection": "Detecção",
    "alignment": "Alinhamento",
    "embedding": "Embedding",
    "search": "Busca",
    "total": "Total",
}


def _fmt(value: object, fmt: str = "{:.2f}", na: str = "--") -> str:
    if value is None:
        return na
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return na
    try:
        return fmt.format(float(value))
    except (TypeError, ValueError):
        return na


def _ordered_hardwares(present: list[str]) -> list[str]:
    order = list(HARDWARE_REGISTRY.keys())
    return [hw for hw in order if hw in present] + [hw for hw in present if hw not in order]


def _hardware_label(hw_id: str) -> str:
    try:
        return get_hardware_spec(hw_id).label
    except ValueError:
        return hw_id


def render_table1(system_df: pd.DataFrame) -> str:
    hardwares = _ordered_hardwares(system_df["hardware_id"].unique().tolist())
    lines = [
        r"\begin{tabular}{llcccc}",
        r"\toprule",
        r"Hardware & CPU & CPU médio (\%) & CPU pico (\%) & RAM média (GB) & RAM pico (GB) \\",
        r"\midrule",
    ]
    for hw in hardwares:
        row = system_df[system_df["hardware_id"] == hw].iloc[0]
        cpu_name = row.get("cpu_model_canonical")
        if not isinstance(cpu_name, str) or not cpu_name:
            try:
                cpu_name = get_hardware_spec(hw).cpu_model_canonical
            except ValueError:
                cpu_name = "--"
        lines.append(
            f"{_hardware_label(hw)} & "
            f"{cpu_name} & "
            f"{_fmt(row.get('cpu_mean_pct'), '{:.1f}')} & "
            f"{_fmt(row.get('cpu_max_pct'), '{:.1f}')} & "
            f"{_fmt(row.get('ram_mean_gb'))} & "
            f"{_fmt(row.get('ram_max_gb'))} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(lines)


def render_table2(system_df: pd.DataFrame) -> str:
    hardwares = _ordered_hardwares(system_df["hardware_id"].unique().tolist())
    lines = [
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Hardware & GPU médio (\%) & VRAM pico (GB) & Temp média (°C) & Potência média (W) \\",
        r"\midrule",
    ]
    for hw in hardwares:
        row = system_df[system_df["hardware_id"] == hw].iloc[0]
        lines.append(
            f"{_hardware_label(hw)} & "
            f"{_fmt(row.get('gpu_mean_pct'), '{:.1f}')} & "
            f"{_fmt(row.get('vram_max_gb'))} & "
            f"{_fmt(row.get('temp_mean_c'), '{:.1f}')} & "
            f"{_fmt(row.get('power_mean_w'), '{:.1f}')} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(lines)


def render_stage_table(stage_df: pd.DataFrame, column: str, caption_metric: str) -> str:
    hardwares = _ordered_hardwares(stage_df["hardware_id"].unique().tolist())
    pivot = stage_df.pivot_table(
        index="hardware_id", columns="stage", values=column, aggfunc="first"
    )

    header_cells = " & ".join(STAGE_LABELS[s] for s in STAGES)
    lines = [
        r"\begin{tabular}{l" + "c" * len(STAGES) + r"}",
        r"\toprule",
        f"Hardware & {header_cells} \\\\",
        r"\midrule",
    ]
    for hw in hardwares:
        if hw not in pivot.index:
            continue
        row = pivot.loc[hw]
        cells = " & ".join(_fmt(row.get(stage)) for stage in STAGES)
        lines.append(f"{_hardware_label(hw)} & {cells} \\\\")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--aggregated-dir", type=Path, default=RESULTS_ROOT / "_aggregated")
    p.add_argument("--out-dir", type=Path, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir or args.aggregated_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    sys_csv = args.aggregated_dir / "system_aggregated.csv"
    stg_csv = args.aggregated_dir / "stage_aggregated.csv"

    if not sys_csv.exists() or not stg_csv.exists():
        raise FileNotFoundError(
            f"Aggregated CSVs not found in {args.aggregated_dir}. "
            "Run `python -m bench.aggregate` first."
        )

    system_df = pd.read_csv(sys_csv)
    stage_df = pd.read_csv(stg_csv)

    (out_dir / "table1_cpu_ram.tex").write_text(render_table1(system_df) + "\n")
    (out_dir / "table2_gpu_vram_temp_power.tex").write_text(render_table2(system_df) + "\n")
    (out_dir / "table3_latency_median.tex").write_text(
        render_stage_table(stage_df, "median_ms", "mediana") + "\n"
    )
    (out_dir / "table4_latency_p95.tex").write_text(
        render_stage_table(stage_df, "p95_ms", "P95") + "\n"
    )
    (out_dir / "table5_latency_max.tex").write_text(
        render_stage_table(stage_df, "max_ms", "máximo") + "\n"
    )
    print(f"Wrote 5 .tex fragments to {out_dir}")


if __name__ == "__main__":
    main()
