from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import json
from typing import Any

from utils.edgeface_inference import DEFAULT_EDGEFACE_CHECKPOINT, DEFAULT_EDGEFACE_MODEL
from utils.face_pipeline import (
    DEFAULT_ARCFACE_WEIGHT,
    DEFAULT_YUNET_MODEL,
    compute_primary_face_embedding,
)
from utils.identity_matching import select_best_identity_match
from utils.pipeline_artifacts import save_pipeline_presentation_artifacts
from utils.qdrant import QdrantStore


def recognize_face(args: argparse.Namespace) -> dict[str, Any]:
    face_data = compute_primary_face_embedding(
        image_path=args.image_path,
        arcface_weight=args.arcface_weight,
        arcface_network=args.arcface_network,
        conf=args.conf,
        device=args.device,
        require_alignment=args.require_alignment,
        embedder_backend=args.embedder,
        edgeface_model_name=args.edgeface_model,
        edgeface_checkpoint=args.edgeface_checkpoint,
        edgeface_compile=not args.no_edgeface_compile,
        arcface_compile=not args.no_arcface_compile,
        detector_backend=args.detector,
        yunet_model=args.yunet_model,
        yunet_nms=args.yunet_nms,
        yunet_top_k=args.yunet_top_k,
        yunet_backend=args.yunet_backend,
    )

    qdrant = QdrantStore(
        url=args.qdrant_url,
        api_key=args.qdrant_api_key,
        timeout=args.qdrant_timeout,
    )
    raw_matches = qdrant.query(
        collection_name=args.collection,
        vector=face_data["embedding"],
        limit=max(args.limit, args.candidate_limit),
        score_threshold=args.score_threshold,
    )
    best_identity_match, grouped_identity_matches = select_best_identity_match(
        raw_matches,
        limit=args.limit,
        min_score=args.score_threshold,
        score_margin=args.score_margin,
        min_match_count=args.min_identity_votes,
    )
    identity_matches = grouped_identity_matches[: args.limit]
    recognized_name = best_identity_match.get("name") if best_identity_match else None
    artifact_paths = None

    if args.save_pipeline_dir:
        artifact_paths = save_pipeline_presentation_artifacts(
            output_dir=Path(args.save_pipeline_dir),
            source_image_rgb=face_data["image"],
            detections=face_data["detections"],
            selected_detection=face_data["primary_face"],
            face_region=face_data["face_region"],
            aligned_face_bgr=face_data["aligned_face_bgr"],
            alignment_info=face_data["alignment_info"],
            alignment_source=face_data["alignment_source"],
            embedding=face_data["embedding"],
            raw_matches=raw_matches,
            identity_matches=identity_matches,
            recognized_name=recognized_name,
            entity_id=best_identity_match.get("entity_id") if best_identity_match else None,
            score=best_identity_match.get("score") if best_identity_match else None,
            isolate_method="crop",
            metadata={
                "pipeline_source": "recognize_face.py",
                "image_path": str(Path(args.image_path).resolve()),
                "collection": args.collection,
            },
        )

    return {
        "recognized": best_identity_match is not None and recognized_name is not None,
        "name": recognized_name,
        "entity_id": best_identity_match.get("entity_id") if best_identity_match else None,
        "score": best_identity_match.get("score") if best_identity_match else None,
        "match_id": best_identity_match.get("best_match_id") if best_identity_match else None,
        "alignment_source": face_data["alignment_source"],
        "alignment_warning": face_data["alignment_info"].get("warning"),
        "identity_matches": identity_matches,
        "matches": raw_matches,
        "artifacts": artifact_paths,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reconhece um rosto em uma imagem consultando a colecao do Qdrant."
    )
    parser.add_argument("image_path", type=str)
    parser.add_argument("--collection", type=str, default="Rostos")
    parser.add_argument("--qdrant-url", type=str, default="http://localhost:6333")
    parser.add_argument("--qdrant-api-key", type=str, default=None)
    parser.add_argument("--qdrant-timeout", type=float, default=15.0)
    parser.add_argument("--arcface-weight", type=str, default=str(DEFAULT_ARCFACE_WEIGHT))
    parser.add_argument("--arcface-network", type=str, default="r100")
    parser.add_argument("--embedder", type=str, default="arcface", choices=["arcface", "edgeface"])
    parser.add_argument("--edgeface-model", type=str, default=DEFAULT_EDGEFACE_MODEL)
    parser.add_argument("--edgeface-checkpoint", type=str, default=str(DEFAULT_EDGEFACE_CHECKPOINT))
    parser.add_argument("--no-edgeface-compile", action="store_true")
    parser.add_argument("--no-arcface-compile", action="store_true")
    parser.add_argument("--detector", type=str, default="mtcnn", choices=["mtcnn", "yunet"])
    parser.add_argument("--yunet-model", type=str, default=str(DEFAULT_YUNET_MODEL))
    parser.add_argument("--yunet-nms", type=float, default=0.3)
    parser.add_argument("--yunet-top-k", type=int, default=5000)
    parser.add_argument("--yunet-backend", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--conf", type=float, default=0.95)
    parser.add_argument("--score-threshold", type=float, default=0.45)
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--candidate-limit", type=int, default=5)
    parser.add_argument("--score-margin", type=float, default=0.10)
    parser.add_argument("--min-identity-votes", type=int, default=1)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--require-alignment", action="store_true")
    parser.add_argument("--save-pipeline-dir", type=str, default=None)
    args = parser.parse_args()
    if args.candidate_limit < 1:
        parser.error("--candidate-limit deve ser maior ou igual a 1.")
    if args.score_margin < 0.0:
        parser.error("--score-margin nao pode ser negativo.")
    if args.min_identity_votes < 1:
        parser.error("--min-identity-votes deve ser maior ou igual a 1.")
    return args


if __name__ == "__main__":
    result = recognize_face(parse_args())
    print(json.dumps(result, ensure_ascii=True, indent=2))
