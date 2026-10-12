"""Benchmark suite for the face-recognition pipeline.

Modules:
    config          Hardware IDs, constants, paths.
    hardware        Hardware identification and capability probes.
    probes          CPU/GPU/temp/power samplers.
    system_monitor  Background sampling thread.
    sync            CUDA synchronization helpers (fail-loud).
    artifacts       Pre-computed deterministic stage inputs.
    microbench      Benchmark 1 (per-stage microbenchmark).
    e2e_video       Benchmark 2 (end-to-end on video).
    aggregate       Parquet -> CSV aggregation.
    generate_tables CSV -> LaTeX table fragments.
"""
