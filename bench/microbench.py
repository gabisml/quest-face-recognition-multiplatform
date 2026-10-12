"""Benchmark 1 — per-stage microbenchmark with two-level aggregation.

Protocol (non-negotiable):
    - K fixed images, N effective repetitions per stage after warm-up.
    - Pre-computed artifacts feed each stage so downstream stages see
      the same input on every hardware.
    - torch.cuda.synchronize() before and after every measured region on
      CUDA devices (loud failure if synchronize cannot run).
    - Per-image median over N reps → then median/p95/max across K medians.
    - 50 warm-up iterations discarded; iteration 51 logs MEASUREMENT_BEGIN.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np
import pandas as pd

from bench.artifacts import PrecomputedArtifacts, load_artifacts
from bench.config import (
    DEFAULT_K,
    DEFAULT_N,
    DEFAULT_WARMUP,
    SUBSET_PATH,
    STAGES,
    STEADY_STATE_WARMUP_S,
    get_hardware_spec,
    result_dir,
)
from bench.hardware import build_system_info
from bench.sync import cuda_sync, is_cuda_device
from bench.system_monitor import SystemMonitor, reset_torch_peak_memory

LOG = logging.getLogger("bench.microbench")


# ---------------------------------------------------------------------------
# Stage closures — each returns a no-arg callable that, when invoked, runs
# the stage on a fixed input. Outer code times the call.
# ---------------------------------------------------------------------------


def make_read_stage(image_path: str) -> Callable[[], np.ndarray]:
    """Disk read + JPEG decode. CPU-only — no CUDA sync needed."""
    def _run() -> np.ndarray:
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"cv2.imread returned None for {image_path}")
        return img
    return _run


def make_detection_stage(
    image_bgr: np.ndarray,
    detector_callable: Callable,
    device: str,
) -> Callable[[], list]:
    """Detection on a fixed decoded BGR image."""
    def _run() -> list:
        cuda_sync(device)
        out = detector_callable(image_bgr)
        cuda_sync(device)
        return out
    return _run


def make_alignment_stage(
    image_bgr: np.ndarray,
    kps: Optional[list[list[float]]],
    bbox: list[float],
) -> Callable[[], np.ndarray]:
    """Similarity-transform alignment from 5 landmarks. CPU-only."""
    from utils.arcface_inference import align_face, resize_face_crop
    from utils.face_pipeline import extract_face_region

    if kps is not None:
        kps_np = np.asarray(kps, dtype=np.float32)

        def _run() -> np.ndarray:
            return align_face(image_bgr, kps_np, image_size=112)

        return _run

    bbox_int = [int(round(v)) for v in bbox]

    def _run_fallback() -> np.ndarray:
        region = extract_face_region(
            image_rgb=image_bgr[:, :, ::-1], mask=None, bbox=bbox_int
        )
        return resize_face_crop(region["crop_rgb"][:, :, ::-1].copy())

    return _run_fallback


def make_embedding_stage(
    aligned_bgr: np.ndarray,
    embedder,
    device: str,
) -> Callable[[], np.ndarray]:
    """ArcFace forward on a fixed 112x112 BGR crop."""
    def _run() -> np.ndarray:
        cuda_sync(device)
        emb = embedder.embed_aligned_face(aligned_bgr)
        cuda_sync(device)
        return emb
    return _run


def make_search_stage(
    embedding: np.ndarray,
    store,
    collection_name: str,
    limit: int,
) -> Callable[[], list]:
    """Consulta de um vetor no store de embeddings. Sem GPU.

    O custo depende do backend, nao do hardware: com Qdrant e dominado pelo round-trip
    cliente-servidor; com o backend numpy e um GEMV em processo. Ver utils/vector_store.py.
    """
    vec = list(map(float, embedding))

    def _run() -> list:
        return store.query_batch(
            collection_name=collection_name,
            vectors=[vec],
            limit=limit,
            score_threshold=None,
        )

    return _run


# ---------------------------------------------------------------------------
# Measurement loop
# ---------------------------------------------------------------------------


def _time_ns_callable(fn: Callable[[], object]) -> int:
    t0 = time.perf_counter_ns()
    fn()
    t1 = time.perf_counter_ns()
    return t1 - t0


def measure_stage(
    stage_name: str,
    builder: Callable[[int], Callable[[], object]],
    image_ids: list[str],
    n_reps: int,
    warmup: int,
    device: str,
    hardware_id: str,
) -> list[dict]:
    """Run K images × N reps for one stage, returning raw rows.

    `builder(i)` returns a callable that runs the stage on the i-th image.
    """
    rows: list[dict] = []

    if not image_ids:
        return rows

    first_runner = builder(0)
    LOG.info("[%s][%s] warm-up: %d iterations", hardware_id, stage_name, warmup)
    for w in range(warmup):
        first_runner()
        if w + 1 == warmup:
            LOG.info(
                "[%s][%s] MEASUREMENT_BEGIN at iter %d", hardware_id, stage_name, warmup + 1
            )

    for image_idx, image_id in enumerate(image_ids):
        runner = builder(image_idx)
        # One untimed pump to ensure caches/JIT settled for this specific input.
        runner()
        for rep_idx in range(n_reps):
            latency_ns = _time_ns_callable(runner)
            rows.append(
                {
                    "hardware_id": hardware_id,
                    "stage": stage_name,
                    "image_id": image_id,
                    "repetition_idx": rep_idx,
                    "latency_ns": latency_ns,
                    "timestamp": time.time_ns(),
                }
            )
    return rows


def aggregate_stage_rows(
    rows: list[dict],
    stage_name: str,
    hardware_id: str,
    n_reps: int,
) -> dict:
    """Two-level aggregation: per-image median over reps, then median/p95/max across images."""
    if not rows:
        return {
            "hardware_id": hardware_id,
            "stage": stage_name,
            "median_ms": float("nan"),
            "p95_ms": float("nan"),
            "max_ms": float("nan"),
            "n_images": 0,
            "n_reps": n_reps,
        }
    df = pd.DataFrame(rows)
    per_image = df.groupby("image_id")["latency_ns"].median()
    per_image_ms = per_image.to_numpy() / 1_000_000.0
    return {
        "hardware_id": hardware_id,
        "stage": stage_name,
        "median_ms": float(np.median(per_image_ms)),
        "p95_ms": float(np.percentile(per_image_ms, 95)),
        "max_ms": float(np.max(per_image_ms)),
        "n_images": int(per_image_ms.shape[0]),
        "n_reps": n_reps,
    }


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def run_microbenchmark(
    hardware_id: str,
    artifacts: PrecomputedArtifacts,
    detector_callable: Callable,
    embedder,
    store,
    collection_name: str,
    candidate_limit: int,
    device: str,
    n_reps: int = DEFAULT_N,
    warmup: int = DEFAULT_WARMUP,
    skip_steady_state_warmup: bool = False,
    skip_search: bool = False,
) -> dict[str, list[dict]]:
    spec = get_hardware_spec(hardware_id)
    if spec.has_nvidia_gpu and not is_cuda_device(device):
        LOG.warning(
            "hardware %s has an NVIDIA GPU but device=%r — CUDA sync will be skipped.",
            hardware_id,
            device,
        )

    if not skip_steady_state_warmup:
        LOG.info("Steady-state warm-up: %.0f s on first image", STEADY_STATE_WARMUP_S)
        warmup_runner = make_embedding_stage(artifacts.aligned_bgr[0], embedder, device)
        t_end = time.perf_counter() + STEADY_STATE_WARMUP_S
        while time.perf_counter() < t_end:
            warmup_runner()

    image_ids = artifacts.image_ids
    raw_rows: dict[str, list[dict]] = {stage: [] for stage in STAGES}

    # --- read --------------------------------------------------------------
    raw_rows["read"] = measure_stage(
        "read",
        lambda i: make_read_stage(artifacts.image_paths[i]),
        image_ids,
        n_reps,
        warmup,
        device="cpu",
        hardware_id=hardware_id,
    )

    # --- detection ---------------------------------------------------------
    raw_rows["detection"] = measure_stage(
        "detection",
        lambda i: make_detection_stage(artifacts.raw_bgr[i], detector_callable, device),
        image_ids,
        n_reps,
        warmup,
        device=device,
        hardware_id=hardware_id,
    )

    # --- alignment ---------------------------------------------------------
    raw_rows["alignment"] = measure_stage(
        "alignment",
        lambda i: make_alignment_stage(
            artifacts.raw_bgr[i],
            artifacts.detections[i].get("kps"),
            artifacts.detections[i]["bbox"],
        ),
        image_ids,
        n_reps,
        warmup,
        device="cpu",
        hardware_id=hardware_id,
    )

    # --- embedding ---------------------------------------------------------
    raw_rows["embedding"] = measure_stage(
        "embedding",
        lambda i: make_embedding_stage(artifacts.aligned_bgr[i], embedder, device),
        image_ids,
        n_reps,
        warmup,
        device=device,
        hardware_id=hardware_id,
    )

    # --- search ------------------------------------------------------------
    if not skip_search:
        raw_rows["search"] = measure_stage(
            "search",
            lambda i: make_search_stage(
                artifacts.embeddings[i], store, collection_name, candidate_limit
            ),
            image_ids,
            n_reps,
            warmup,
            device="cpu",
            hardware_id=hardware_id,
        )
    else:
        raw_rows["search"] = []

    # --- total (sum of per-image medians) ----------------------------------
    summed_rows: list[dict] = []
    for image_idx, image_id in enumerate(image_ids):
        for rep_idx in range(n_reps):
            total_ns = 0
            for stage_name in ("read", "detection", "alignment", "embedding", "search"):
                stage_rows = raw_rows[stage_name]
                if not stage_rows:
                    continue
                ns = stage_rows[image_idx * n_reps + rep_idx]["latency_ns"]
                total_ns += ns
            summed_rows.append(
                {
                    "hardware_id": hardware_id,
                    "stage": "total",
                    "image_id": image_id,
                    "repetition_idx": rep_idx,
                    "latency_ns": total_ns,
                    "timestamp": time.time_ns(),
                }
            )
    raw_rows["total"] = summed_rows

    return raw_rows


def write_outputs(
    hardware_id: str,
    rows_by_stage: dict[str, list[dict]],
    n_reps: int,
    out_dir: Path,
    monitor_df: Optional[pd.DataFrame] = None,
    steady_state_start_utc: Optional[str] = None,
    steady_state_end_utc: Optional[str] = None,
    extra_info: Optional[dict] = None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict] = []
    for rows in rows_by_stage.values():
        all_rows.extend(rows)
    df = pd.DataFrame(all_rows)
    df.to_parquet(out_dir / "stage_raw.parquet", index=False)

    agg = [
        aggregate_stage_rows(rows_by_stage[stage], stage, hardware_id, n_reps)
        for stage in STAGES
    ]
    pd.DataFrame(agg).to_csv(out_dir / "stage_aggregated.csv", index=False)

    if monitor_df is not None:
        monitor_df.to_parquet(out_dir / "system_metrics.parquet", index=False)

    info = build_system_info(hardware_id, **(extra_info or {}))
    info.steady_state_start_utc = steady_state_start_utc
    info.steady_state_end_utc = steady_state_end_utc
    info.to_json(out_dir / "system_info.json")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--hardware-id", required=True)
    p.add_argument("--artifacts", type=Path, required=True)
    p.add_argument("--subset", type=Path, default=SUBSET_PATH)
    p.add_argument("--k", type=int, default=DEFAULT_K)
    p.add_argument("--n", type=int, default=DEFAULT_N)
    p.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--detector-backend", type=str, default="yunet", choices=["yunet", "mtcnn"])
    p.add_argument("--detector-conf", type=float, default=0.5)
    p.add_argument("--embedder-backend", type=str, default="arcface", choices=["arcface", "edgeface"])
    p.add_argument("--arcface-network", type=str, default="r100")
    p.add_argument("--embedder-precision", type=str, default="fp16")
    p.add_argument("--collection", type=str, default="Rostos")
    p.add_argument("--candidate-limit", type=int, default=5)
    p.add_argument("--vector-store", type=str, default="numpy", choices=["numpy", "qdrant"],
                   help="backend da etapa de busca; use o mesmo em todos os alvos para "
                        "que a etapa seja comparavel entre plataformas")
    p.add_argument("--gallery", type=str, default=None,
                   help="prefixo da galeria para --vector-store numpy (.npy + .json)")
    p.add_argument("--qdrant-url", type=str, default="http://localhost:6333")
    p.add_argument("--qdrant-api-key", type=str, default=None)
    p.add_argument("--skip-steady-state-warmup", action="store_true")
    p.add_argument("--skip-search", action="store_true",
                   help="Disable Qdrant stage (offline benchmarks).")
    p.add_argument("--out-dir", type=Path, default=None,
                   help="Override results dir (default: results/<hardware>/<timestamp>).")
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = parse_args()

    from utils.arcface_inference import get_torch_device

    device = args.device or get_torch_device()
    LOG.info("hardware=%s device=%s", args.hardware_id, device)

    if is_cuda_device(device):
        import torch

        torch.set_float32_matmul_precision("high")

    artifacts = load_artifacts(args.artifacts)
    if len(artifacts) > args.k:
        artifacts = PrecomputedArtifacts(
            image_ids=artifacts.image_ids[: args.k],
            image_paths=artifacts.image_paths[: args.k],
            image_hashes=artifacts.image_hashes[: args.k],
            raw_bgr=artifacts.raw_bgr[: args.k],
            detections=artifacts.detections[: args.k],
            aligned_bgr=artifacts.aligned_bgr[: args.k],
            embeddings=artifacts.embeddings[: args.k],
        )
    LOG.info("Loaded %d artifacts", len(artifacts))

    # Build a thin detector callable that mirrors the production detector.
    from utils.face_pipeline import detect_faces
    from utils.yunet_inference import DEFAULT_YUNET_MODEL

    def detector_callable(image_bgr: np.ndarray) -> list[dict]:
        return detect_faces(
            image=image_bgr,
            detector_backend=args.detector_backend,
            device=device,
            conf=args.detector_conf,
            max_det=5,
            yunet_model=str(DEFAULT_YUNET_MODEL),
        )

    # Embedder
    if args.embedder_backend == "edgeface":
        from utils.edgeface_inference import (
            DEFAULT_EDGEFACE_CHECKPOINT,
            DEFAULT_EDGEFACE_MODEL,
            EdgeFaceEmbedder,
        )

        embedder = EdgeFaceEmbedder(
            checkpoint_path=str(DEFAULT_EDGEFACE_CHECKPOINT),
            model_name=DEFAULT_EDGEFACE_MODEL,
            device=device,
            precision=args.embedder_precision,
            compile_model=True,
        )
    else:
        from utils.arcface_inference import ArcFaceEmbedder
        from utils.face_pipeline import DEFAULT_ARCFACE_WEIGHT

        embedder = ArcFaceEmbedder(
            weight_path=str(DEFAULT_ARCFACE_WEIGHT),
            network=args.arcface_network,
            device=device,
            precision=args.embedder_precision,
            compile_model=True,
        )

    # O backend e selecionavel porque a etapa de busca e dominada por ele, nao pelo
    # hardware: o Qdrant paga round-trip cliente-servidor (~3 ms) enquanto o numpy resolve
    # em processo (~0,1 ms). Para comparar plataformas use o MESMO backend em todas; o
    # Qdrant vira uma medicao separada, de custo de implantacao.
    store = None
    if not args.skip_search:
        from utils.vector_store import build_vector_store

        store = build_vector_store(
            args.vector_store,
            qdrant_url=args.qdrant_url,
            qdrant_api_key=args.qdrant_api_key,
            gallery_path=args.gallery,
        )

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = args.out_dir or result_dir(args.hardware_id, timestamp)

    monitor = SystemMonitor(args.hardware_id)
    reset_torch_peak_memory()
    monitor.start()
    steady_start_utc = datetime.now(timezone.utc).isoformat()

    try:
        rows_by_stage = run_microbenchmark(
            hardware_id=args.hardware_id,
            artifacts=artifacts,
            detector_callable=detector_callable,
            embedder=embedder,
            store=store,
            collection_name=args.collection,
            candidate_limit=args.candidate_limit,
            device=device,
            n_reps=args.n,
            warmup=args.warmup,
            skip_steady_state_warmup=args.skip_steady_state_warmup,
            skip_search=args.skip_search,
        )
    finally:
        steady_end_utc = datetime.now(timezone.utc).isoformat()
        monitor_df = monitor.stop(check_throttling=True)

    write_outputs(
        hardware_id=args.hardware_id,
        rows_by_stage=rows_by_stage,
        n_reps=args.n,
        out_dir=out_dir,
        monitor_df=monitor_df,
        steady_state_start_utc=steady_start_utc,
        steady_state_end_utc=steady_end_utc,
        extra_info={
            "detector_backend": args.detector_backend,
            "embedder_backend": args.embedder_backend,
            "vector_store": args.vector_store,
            "gallery": args.gallery,
            "qdrant_url": args.qdrant_url,
        },
    )

    torch_mem = monitor.torch_memory_snapshot()
    (out_dir / "torch_memory.json").write_text(json.dumps(torch_mem, indent=2))
    LOG.info("Wrote outputs to %s", out_dir)


if __name__ == "__main__":
    main()
