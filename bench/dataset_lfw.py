"""Automated LFW (marcelohaps/lfw) provisioning for the benchmark suite.

Subcommands:
    download     Download the dataset from HuggingFace into data/lfw/.
    build-subset Pick K deterministic images, write bench/benchmark_subset.json.
    enroll       Compute one ArcFace embedding per identity and upsert to Qdrant.
    artifacts    Pre-compute bboxes / aligned faces / embeddings on this hardware.
    prepare      One-shot: download → build-subset → enroll → artifacts.

The LFW HuggingFace dataset (marcelohaps/lfw) ships as an imagefolder:
    lfw/
    ├── pairs.csv
    └── train/
        ├── images/<shard>/<file>.jpg
        └── metadata.csv  (file_name, label, label_name, identity, image_num, ...)

13 233 imagens, 5 749 identidades — 4 069 com apenas 1 imagem; o restante tem
2+ imagens, o que permite enrolar uma e fazer queries com as demais.
"""

from __future__ import annotations

import argparse
import logging
import random
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from bench.artifacts import SUBSET_PATH, file_sha256, relativize_to_hf_cache

if TYPE_CHECKING:  # so para a anotacao "pandas.DataFrame" resolver
    import pandas

LOG = logging.getLogger("bench.dataset_lfw")

DEFAULT_REPO = "marcelohaps/lfw"
DEFAULT_COLLECTION = "Rostos"
DEFAULT_ARCFACE_DIM = 512


# ---------------------------------------------------------------------------
# download
# ---------------------------------------------------------------------------


def download_lfw(
    repo_id: str = DEFAULT_REPO,
    revision: Optional[str] = None,
    token: Optional[str] = None,
    cache_dir: Optional[Path] = None,
) -> Path:
    """Snapshot-download the dataset to the HF cache; return the snapshot path.

    Uses the standard HuggingFace cache (~/.cache/huggingface/hub/) for
    content-addressed storage — subsequent runs hit the cache instead of
    re-downloading. To override the location, set HF_HOME or pass cache_dir.

    Anonymous downloads of marcelohaps/lfw (13k+ files) can hit the HF rate
    limit (HTTP 429). If you have a HuggingFace token, set it via:
        huggingface-cli login
    or pass it explicitly via `token=...`.
    """
    from huggingface_hub import snapshot_download

    LOG.info("snapshot_download(%s, repo_type='dataset') ...", repo_id)
    path = snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        revision=revision,
        token=token,
        cache_dir=str(cache_dir) if cache_dir else None,
    )
    LOG.info("Snapshot ready at %s", path)
    return Path(path)


# ---------------------------------------------------------------------------
# metadata + subset selection
# ---------------------------------------------------------------------------


def _find_metadata_csv(lfw_dir: Path) -> Path:
    candidates = list(lfw_dir.rglob("metadata.csv"))
    if not candidates:
        raise FileNotFoundError(f"metadata.csv not found under {lfw_dir}. "
                                "Run `download` first.")
    # Prefer the deepest match (likely train/metadata.csv).
    return max(candidates, key=lambda p: len(p.parts))


def load_metadata(lfw_dir: Path):
    """Return (DataFrame, image_root) — image paths are relative to image_root."""
    import pandas as pd

    metadata_csv = _find_metadata_csv(lfw_dir)
    df = pd.read_csv(metadata_csv)
    image_root = metadata_csv.parent
    return df, image_root


def select_subset(
    metadata,
    k: int,
    seed: int = 1337,
    min_images_per_identity: int = 2,
) -> "pandas.DataFrame":
    """Pick K images deterministically, biased to identities with ≥ N images.

    Why prefer multi-image identities: enables enrollment + query split for
    the Qdrant search stage to exercise real top-k retrieval (rather than
    a coin-toss against a single registered face).
    """

    counts = metadata.groupby("identity")["file_name"].count()
    multi = metadata[metadata["identity"].isin(counts[counts >= min_images_per_identity].index)]
    if len(multi) >= k:
        pool = multi
    else:
        LOG.warning(
            "Only %d images from identities with ≥%d images; falling back to full pool.",
            len(multi),
            min_images_per_identity,
        )
        pool = metadata
    if len(pool) < k:
        raise RuntimeError(f"Need K={k} images but pool has only {len(pool)}.")

    rng = random.Random(seed)
    indices = rng.sample(list(pool.index), k)
    subset = pool.loc[indices].copy().reset_index(drop=True)
    return subset


def write_subset_manifest(
    subset,
    image_root: Path,
    output: Path = SUBSET_PATH,
    seed: int = 1337,
) -> dict:
    """Write the K-image manifest in the format consumed by bench.artifacts."""
    import json

    entries = []
    for _, row in subset.iterrows():
        rel = row["file_name"]
        abs_path = (image_root / rel).resolve()
        if not abs_path.exists():
            raise FileNotFoundError(f"Image listed in metadata not on disk: {abs_path}")
        # image_id includes identity + image_num for reproducibility across reruns.
        image_id = f"{row['identity']}_{int(row['image_num']):04d}"
        entries.append(
            {
                "image_id": image_id,
                # Cache-relative so the manifest is portable across machines;
                # bench.artifacts.resolve_image_path rebases it onto the local
                # HF cache at consume time.
                "path": relativize_to_hf_cache(abs_path),
                "sha256": file_sha256(abs_path),
                "identity": row["identity"],
                "image_num": int(row["image_num"]),
            }
        )
    manifest = {
        "seed": seed,
        "k": len(entries),
        "source_dataset": DEFAULT_REPO,
        "images": entries,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2))
    LOG.info("Wrote subset manifest with %d images to %s", len(entries), output)
    return manifest


# ---------------------------------------------------------------------------
# Qdrant enrollment
# ---------------------------------------------------------------------------


def select_enrollment_identities(
    metadata,
    n_identities: int,
    seed: int = 1337,
    min_images_per_identity: int = 2,
):
    """Pick N identities with ≥2 images; return one reference image per identity."""

    counts = metadata.groupby("identity")["file_name"].count()
    eligible = counts[counts >= min_images_per_identity].index.tolist()
    if len(eligible) < n_identities:
        raise RuntimeError(
            f"Asked for {n_identities} identities with ≥{min_images_per_identity} "
            f"images but only {len(eligible)} qualify."
        )
    rng = random.Random(seed)
    chosen = rng.sample(eligible, n_identities)
    # Pick the first (lowest image_num) image per identity as the reference.
    refs = (
        metadata[metadata["identity"].isin(chosen)]
        .sort_values(["identity", "image_num"])
        .groupby("identity")
        .head(1)
        .reset_index(drop=True)
    )
    return refs


def enroll_in_qdrant(
    refs,
    image_root: Path,
    embedder,
    qdrant_url: str = "http://localhost:6333",
    qdrant_api_key: Optional[str] = None,
    collection_name: str = DEFAULT_COLLECTION,
    recreate: bool = True,
    embedding_dim: int = DEFAULT_ARCFACE_DIM,
) -> int:
    """Embed each reference image and upsert to Qdrant. Returns # enrolled."""
    import cv2
    import numpy as np

    from utils.arcface_inference import align_face, resize_face_crop
    from utils.face_pipeline import detect_faces, extract_face_region
    from utils.qdrant import QdrantStore
    from utils.yunet_inference import DEFAULT_YUNET_MODEL

    qdrant = QdrantStore(url=qdrant_url, api_key=qdrant_api_key)
    qdrant.create_collection(
        collection_name=collection_name,
        vector_size=embedding_dim,
        distance="cosine",
        recreate=recreate,
    )

    enrolled = 0
    skipped = 0
    for idx, row in refs.iterrows():
        path = (image_root / row["file_name"]).resolve()
        img_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img_bgr is None:
            LOG.warning("[skip] cv2.imread failed for %s", path)
            skipped += 1
            continue

        detections = detect_faces(
            image=img_bgr,
            detector_backend="yunet",
            conf=0.5,
            max_det=5,
            yunet_model=str(DEFAULT_YUNET_MODEL),
        )
        if not detections:
            LOG.warning("[skip] no face detected in %s", path)
            skipped += 1
            continue
        primary = max(detections, key=lambda d: (d.get("area", 0), d.get("score", 0)))

        kps = primary.get("kps")
        if kps is not None:
            aligned = align_face(img_bgr, kps, image_size=112)
        else:
            region = extract_face_region(
                image_rgb=img_bgr[:, :, ::-1], mask=None, bbox=primary["bbox"]
            )
            aligned = resize_face_crop(region["crop_rgb"][:, :, ::-1].copy())

        embedding = embedder.embed_aligned_face(aligned)
        emb = np.asarray(embedding, dtype=np.float32).tolist()

        qdrant.index_point(
            collection_name=collection_name,
            point_id=int(idx) + 1,  # Qdrant requires positive int or UUID
            vector=emb,
            payload={
                "name": str(row["identity"]),
                "entity_id": f"lfw:{row['identity']}",
                "source": "lfw_bench_enrollment",
                "source_filename": row.get("source_filename"),
            },
        )
        enrolled += 1

    LOG.info("Enrolled %d identities (skipped %d) into collection %r",
             enrolled, skipped, collection_name)
    return enrolled


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _resolve_data_dir(args: argparse.Namespace) -> Path:
    """If --data-dir already has metadata.csv, use it; else snapshot-download."""
    if args.data_dir:
        candidate = Path(args.data_dir)
        if list(candidate.rglob("metadata.csv")):
            return candidate
    return download_lfw(
        repo_id=getattr(args, "repo", DEFAULT_REPO),
        token=getattr(args, "token", None),
        cache_dir=getattr(args, "cache_dir", None),
    )


def cmd_download(args: argparse.Namespace) -> None:
    path = download_lfw(
        repo_id=args.repo, token=args.token, cache_dir=args.cache_dir
    )
    print(path)


def cmd_build_subset(args: argparse.Namespace) -> None:
    data_dir = _resolve_data_dir(args)
    df, image_root = load_metadata(data_dir)
    subset = select_subset(df, k=args.k, seed=args.seed)
    write_subset_manifest(subset, image_root, output=args.output, seed=args.seed)


def cmd_enroll(args: argparse.Namespace) -> None:
    from utils.arcface_inference import ArcFaceEmbedder, get_torch_device
    from utils.face_pipeline import DEFAULT_ARCFACE_WEIGHT

    data_dir = _resolve_data_dir(args)
    df, image_root = load_metadata(data_dir)
    refs = select_enrollment_identities(df, n_identities=args.n, seed=args.seed)
    device = args.device or get_torch_device()
    embedder = ArcFaceEmbedder(
        weight_path=str(DEFAULT_ARCFACE_WEIGHT),
        network=args.arcface_network,
        device=device,
        precision=args.embedder_precision,
        compile_model=False,  # one-shot enrollment doesn't need torch.compile.
    )
    enroll_in_qdrant(
        refs=refs,
        image_root=image_root,
        embedder=embedder,
        qdrant_url=args.qdrant_url,
        qdrant_api_key=args.qdrant_api_key,
        collection_name=args.collection,
        recreate=not args.no_recreate,
    )


def cmd_artifacts(args: argparse.Namespace) -> None:
    import json

    from bench.artifacts import precompute_artifacts
    from utils.arcface_inference import ArcFaceEmbedder, get_torch_device
    from utils.face_pipeline import DEFAULT_ARCFACE_WEIGHT, detect_faces
    from utils.yunet_inference import DEFAULT_YUNET_MODEL

    manifest = json.loads(Path(args.subset).read_text())
    device = args.device or get_torch_device()
    embedder = ArcFaceEmbedder(
        weight_path=str(DEFAULT_ARCFACE_WEIGHT),
        network=args.arcface_network,
        device=device,
        precision=args.embedder_precision,
        compile_model=False,
    )

    class _RecognizerStub:
        def __init__(self, emb):
            self.embedder = emb
            self.device = device

    def detector(img_bgr):
        return detect_faces(
            image=img_bgr,
            detector_backend="yunet",
            device=device,
            conf=0.5,
            max_det=5,
            yunet_model=str(DEFAULT_YUNET_MODEL),
        )

    precompute_artifacts(
        recognizer=_RecognizerStub(embedder),
        detector_callable=detector,
        image_entries=manifest["images"],
        save_path=args.output,
    )
    LOG.info("Wrote artifacts to %s", args.output)


def cmd_prepare(args: argparse.Namespace) -> None:
    LOG.info("=== STEP 1/4: download ===")
    data_dir = download_lfw(
        repo_id=args.repo, token=args.token, cache_dir=args.cache_dir
    )
    args.data_dir = data_dir  # downstream cmd_enroll / cmd_artifacts can find it

    LOG.info("=== STEP 2/4: build-subset ===")
    df, image_root = load_metadata(data_dir)
    subset = select_subset(df, k=args.k, seed=args.seed)
    write_subset_manifest(subset, image_root, output=args.subset_output, seed=args.seed)

    if not args.skip_enroll:
        LOG.info("=== STEP 3/4: enroll ===")
        cmd_enroll(args)
    else:
        LOG.info("=== STEP 3/4: enroll (skipped) ===")

    LOG.info("=== STEP 4/4: artifacts ===")
    args.subset = args.subset_output
    cmd_artifacts(args)
    LOG.info("LFW benchmark setup complete.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    def add_data_args(p):
        p.add_argument("--data-dir", type=Path, default=None,
                       help="Local LFW dir (skip download). If absent, snapshot-downloads.")
        p.add_argument("--repo", default=DEFAULT_REPO)
        p.add_argument("--token", default=None, help="HF token (else uses env/CLI login)")
        p.add_argument("--cache-dir", type=Path, default=None,
                       help="Override HF cache dir (else uses HF_HOME or ~/.cache/huggingface).")

    p_dl = sub.add_parser("download", help="Download dataset from HF (returns cache path)")
    p_dl.add_argument("--repo", default=DEFAULT_REPO)
    p_dl.add_argument("--token", default=None, help="HF token (else uses env/CLI login)")
    p_dl.add_argument("--cache-dir", type=Path, default=None)
    p_dl.set_defaults(func=cmd_download)

    p_sub = sub.add_parser("build-subset", help="Write K-image manifest")
    add_data_args(p_sub)
    p_sub.add_argument("--k", type=int, default=500)
    p_sub.add_argument("--seed", type=int, default=1337)
    p_sub.add_argument("--output", type=Path, default=SUBSET_PATH)
    p_sub.set_defaults(func=cmd_build_subset)

    p_en = sub.add_parser("enroll", help="Enroll N identities into Qdrant")
    add_data_args(p_en)
    p_en.add_argument("--n", type=int, default=100)
    p_en.add_argument("--seed", type=int, default=1337)
    p_en.add_argument("--collection", default=DEFAULT_COLLECTION)
    p_en.add_argument("--qdrant-url", default="http://localhost:6333")
    p_en.add_argument("--qdrant-api-key", default=None)
    p_en.add_argument("--no-recreate", action="store_true",
                      help="Keep existing collection; otherwise it is recreated.")
    p_en.add_argument("--device", default=None)
    p_en.add_argument("--arcface-network", default="r100")
    p_en.add_argument("--embedder-precision", default="fp16")
    p_en.set_defaults(func=cmd_enroll)

    p_art = sub.add_parser("artifacts", help="Pre-compute stage inputs on this HW")
    p_art.add_argument("--subset", type=Path, default=SUBSET_PATH)
    p_art.add_argument("--output", type=Path, default=Path("bench/artifacts.npz"))
    p_art.add_argument("--device", default=None)
    p_art.add_argument("--arcface-network", default="r100")
    p_art.add_argument("--embedder-precision", default="fp16")
    p_art.set_defaults(func=cmd_artifacts)

    p_prep = sub.add_parser("prepare", help="One-shot: download → subset → enroll → artifacts")
    p_prep.add_argument("--data-dir", type=Path, default=None,
                        help="Reuse a local LFW dir; otherwise snapshot-downloads.")
    p_prep.add_argument("--repo", default=DEFAULT_REPO)
    p_prep.add_argument("--token", default=None, help="HF token (else uses env/CLI login)")
    p_prep.add_argument("--cache-dir", type=Path, default=None)
    p_prep.add_argument("--k", type=int, default=500)
    p_prep.add_argument("--n", type=int, default=100,
                        help="Identities to enroll into Qdrant.")
    p_prep.add_argument("--seed", type=int, default=1337)
    p_prep.add_argument("--subset-output", type=Path, default=SUBSET_PATH)
    p_prep.add_argument("--output", type=Path, default=Path("bench/artifacts.npz"))
    p_prep.add_argument("--collection", default=DEFAULT_COLLECTION)
    p_prep.add_argument("--qdrant-url", default="http://localhost:6333")
    p_prep.add_argument("--qdrant-api-key", default=None)
    p_prep.add_argument("--no-recreate", action="store_true")
    p_prep.add_argument("--device", default=None)
    p_prep.add_argument("--arcface-network", default="r100")
    p_prep.add_argument("--embedder-precision", default="fp16")
    p_prep.add_argument("--skip-enroll", action="store_true",
                        help="Skip Qdrant enrollment (e.g., offline benchmark).")
    p_prep.set_defaults(func=cmd_prepare)

    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
