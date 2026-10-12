"""Deterministic, hardware-agnostic stage inputs.

The reference hardware (typically rtx3090) runs the pipeline once to
produce, for each image in the benchmark subset:
    - the raw decoded BGR (input to detection)
    - the canonical detection result (bbox + 5 landmarks + score)
    - the aligned 112x112 BGR crop (input to embedding)
    - the 512-d ArcFace embedding (input to vector search)

These artifacts are saved as a single .npz so every other hardware
consumes the same input at every stage, preventing tiny numerical
differences from contaminating downstream timings.

The benchmark_subset.json file pins the exact image set used.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

from bench.config import SUBSET_PATH


@dataclass
class PrecomputedArtifacts:
    image_ids: list[str]
    image_paths: list[str]
    image_hashes: list[str]
    raw_bgr: list[np.ndarray]
    detections: list[dict]  # one dict per image: {bbox, kps, score}
    aligned_bgr: list[np.ndarray]  # 112x112x3 uint8 each
    embeddings: np.ndarray  # (N, 512) float32

    def __len__(self) -> int:
        return len(self.image_ids)


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def hf_hub_cache() -> Path:
    """Local HuggingFace hub cache root (honours HF_HOME / HF_HUB_CACHE)."""
    from huggingface_hub.constants import HF_HUB_CACHE

    return Path(HF_HUB_CACHE)


def relativize_to_hf_cache(abs_path: Path) -> str:
    """Make an image path portable across machines.

    LFW images live in the content-addressed HF cache, so the tail
    `datasets--<repo>/blobs/<sha>` is identical on every machine — only the
    cache root differs. We store that tail; paths outside the cache are kept
    absolute (best effort).
    """
    abs_path = Path(abs_path)
    try:
        return str(abs_path.relative_to(hf_hub_cache()))
    except ValueError:
        return str(abs_path)


def resolve_image_path(stored: str) -> Path:
    """Resolve a manifest/artifact image path to a local absolute path.

    Cache-relative paths (the portable form written by `relativize_to_hf_cache`)
    are rebased onto the local HF hub cache. Absolute paths are returned
    unchanged, for backwards compatibility with manifests built before this.
    """
    p = Path(stored)
    if p.is_absolute():
        return p
    return hf_hub_cache() / p


def discover_images(roots: Iterable[Path], extensions: tuple[str, ...] = (".jpg", ".jpeg", ".png")) -> list[Path]:
    paths: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        for ext in extensions:
            paths.extend(sorted(root.rglob(f"*{ext}")))
    return paths


def build_subset(
    roots: Iterable[Path],
    k: int,
    seed: int = 1337,
    output: Path = SUBSET_PATH,
) -> dict:
    """Pick K images deterministically and save the manifest."""
    candidates = discover_images(roots)
    if len(candidates) < k:
        raise RuntimeError(
            f"Found only {len(candidates)} images under {[str(r) for r in roots]}; "
            f"need at least K={k}."
        )
    rng = random.Random(seed)
    selected = rng.sample(candidates, k)
    manifest = {
        "seed": seed,
        "k": k,
        "images": [
            {
                "image_id": p.stem,
                "path": str(p.relative_to(p.parents[len(p.parts) - 1])) if False else str(p),
                "sha256": file_sha256(p),
            }
            for p in selected
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2))
    return manifest


def load_subset(path: Path = SUBSET_PATH) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(
            f"Subset manifest not found at {path}. Run "
            "`python -m bench.artifacts build-subset ...` first."
        )
    return json.loads(path.read_text())["images"]


def precompute_artifacts(
    recognizer,
    detector_callable,
    image_entries: list[dict],
    save_path: Path,
) -> PrecomputedArtifacts:
    """Run detection + alignment + embedding on each image; save .npz.

    detector_callable(image_bgr) -> list[dict] with keys {bbox, kps, score, area}
    recognizer is a FaceRecognizer instance — we use its embedder directly.
    """
    from utils.face_pipeline import extract_face_region
    from utils.arcface_inference import align_face, resize_face_crop

    image_ids: list[str] = []
    image_paths: list[str] = []
    image_hashes: list[str] = []
    raw_bgr: list[np.ndarray] = []
    detections_kept: list[dict] = []
    aligned_bgr: list[np.ndarray] = []
    embeddings_list: list[np.ndarray] = []

    for entry in image_entries:
        path = resolve_image_path(entry["path"])
        if not path.exists():
            continue
        img_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img_bgr is None:
            continue

        det_results = detector_callable(img_bgr)
        if not det_results:
            continue
        primary = max(det_results, key=lambda d: (d.get("area", 0), d.get("score", 0)))

        kps = primary.get("kps")
        if kps is not None:
            aligned = align_face(img_bgr, kps, image_size=112)
        else:
            region = extract_face_region(image_rgb=img_bgr[:, :, ::-1], mask=None, bbox=primary["bbox"])
            aligned = resize_face_crop(region["crop_rgb"][:, :, ::-1].copy())

        emb = recognizer.embedder.embed_aligned_face(aligned)
        if isinstance(emb, np.ndarray):
            emb_np = emb.astype(np.float32)
        else:
            emb_np = np.asarray(emb, dtype=np.float32)

        image_ids.append(entry["image_id"])
        image_paths.append(entry["path"])  # portable (cache-relative) form
        image_hashes.append(entry["sha256"])
        raw_bgr.append(img_bgr)
        detections_kept.append(
            {
                "bbox": list(map(float, primary["bbox"])),
                "kps": kps.tolist() if isinstance(kps, np.ndarray) else None,
                "score": float(primary.get("score", 0.0)),
            }
        )
        aligned_bgr.append(aligned.astype(np.uint8))
        embeddings_list.append(emb_np)

    if not image_ids:
        raise RuntimeError("No usable images after detection; cannot precompute artifacts.")

    embeddings = np.stack(embeddings_list, axis=0)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        save_path,
        image_ids=np.array(image_ids),
        image_paths=np.array(image_paths),
        image_hashes=np.array(image_hashes),
        detections=np.array(json.dumps(detections_kept)),
        aligned_bgr=np.stack(aligned_bgr, axis=0),
        embeddings=embeddings,
    )
    # raw_bgr varies in size, save side-by-side as individual .npy files in a dir.
    raw_dir = save_path.with_suffix("")
    raw_dir.mkdir(parents=True, exist_ok=True)
    for img_id, arr in zip(image_ids, raw_bgr):
        np.save(raw_dir / f"{img_id}.npy", arr)

    return PrecomputedArtifacts(
        image_ids=image_ids,
        image_paths=image_paths,
        image_hashes=image_hashes,
        raw_bgr=raw_bgr,
        detections=detections_kept,
        aligned_bgr=aligned_bgr,
        embeddings=embeddings,
    )


def load_artifacts(save_path: Path) -> PrecomputedArtifacts:
    data = np.load(save_path, allow_pickle=False)
    image_ids = list(map(str, data["image_ids"]))
    image_paths = [str(resolve_image_path(p)) for p in map(str, data["image_paths"])]
    image_hashes = list(map(str, data["image_hashes"]))
    detections = json.loads(str(data["detections"]))
    aligned = data["aligned_bgr"]
    embeddings = data["embeddings"].astype(np.float32)

    raw_dir = save_path.with_suffix("")
    raw_bgr = [np.load(raw_dir / f"{img_id}.npy") for img_id in image_ids]
    aligned_list = [aligned[i] for i in range(aligned.shape[0])]

    return PrecomputedArtifacts(
        image_ids=image_ids,
        image_paths=image_paths,
        image_hashes=image_hashes,
        raw_bgr=raw_bgr,
        detections=detections,
        aligned_bgr=aligned_list,
        embeddings=embeddings,
    )


def cli_build_subset() -> None:
    """`python -m bench.artifacts build-subset ROOTS... --k K --seed S`"""
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("roots", nargs="+", type=Path)
    parser.add_argument("--k", type=int, default=500)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--output", type=Path, default=SUBSET_PATH)
    args = parser.parse_args()
    manifest = build_subset(args.roots, args.k, args.seed, args.output)
    print(f"Wrote {len(manifest['images'])} images to {args.output}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "build-subset":
        sys.argv = [sys.argv[0]] + sys.argv[2:]
        cli_build_subset()
    else:
        print("Usage: python -m bench.artifacts build-subset <roots>... [--k K] [--seed S]")
        sys.exit(1)
