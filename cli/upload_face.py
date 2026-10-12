from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import copy
import json
import uuid
from datetime import datetime
from typing import Any

from PIL import Image
import numpy as np

from utils.edgeface_inference import DEFAULT_EDGEFACE_CHECKPOINT, DEFAULT_EDGEFACE_MODEL
from utils.face_pipeline import (
    DEFAULT_ARCFACE_WEIGHT,
    DEFAULT_YUNET_MODEL,
    compute_primary_face_embedding,
)
from utils.qdrant import QdrantStore


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output" / "uploads"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def save_upload_artifacts(
    output_dir: Path,
    source_image: Image.Image,
    face_region: dict[str, Any],
    aligned_face_bgr: np.ndarray,
    metadata: dict[str, Any],
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)

    source_path = output_dir / "source_image.png"
    segmented_crop_path = output_dir / "segmented_face.png"
    original_crop_path = output_dir / "original_face.png"
    mask_path = output_dir / "face_mask.png"
    aligned_path = output_dir / "aligned_face.png"
    metadata_path = output_dir / "metadata.json"

    source_image.save(source_path)
    Image.fromarray(face_region["segmented_crop_rgb"]).save(segmented_crop_path)
    Image.fromarray(face_region["crop_rgb"]).save(original_crop_path)
    Image.fromarray(face_region["crop_mask"].astype(np.uint8) * 255, mode="L").save(mask_path)
    Image.fromarray(aligned_face_bgr[:, :, ::-1].copy()).save(aligned_path)

    with metadata_path.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, ensure_ascii=True, indent=2)

    return {
        "source_image": str(source_path),
        "segmented_crop": str(segmented_crop_path),
        "original_crop": str(original_crop_path),
        "mask": str(mask_path),
        "aligned_face": str(aligned_path),
        "metadata": str(metadata_path),
    }


def upload_face(args: argparse.Namespace) -> dict[str, Any]:
    point_id = args.point_id or uuid.uuid4().hex
    entity_id = args.entity_id or args.name or point_id
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
    device = face_data["device"]
    image = face_data["image"]
    primary_face = face_data["primary_face"]
    face_region = face_data["face_region"]
    aligned_face_bgr = face_data["aligned_face_bgr"]
    alignment_info = face_data["alignment_info"]
    alignment_source = face_data["alignment_source"]
    embedding = face_data["embedding"]

    timestamp = datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
    upload_dir = Path(args.output_dir) / str(entity_id) / point_id
    metadata = {
        "point_id": point_id,
        "entity_id": entity_id,
        "name": args.name,
        "collection": args.collection,
        "source_image_path": str(Path(args.image_path).resolve()),
        "created_at": timestamp,
        "device": device,
        "embedding_size": len(embedding),
        "detector": args.detector,
        "yunet_backend": primary_face.get("yunet_backend"),
        "detection": {
            k: v.tolist() if isinstance(v, np.ndarray) else v
            for k, v in primary_face.items()
        },
        "crop_bbox": face_region["crop_bbox"],
        "alignment_source": alignment_source,
        "alignment_bbox": alignment_info["bbox"],
        "alignment_score": alignment_info["score"],
        "alignment_kps": alignment_info["kps"].tolist(),
        "alignment_warning": alignment_info.get("warning"),
    }
    artifact_paths = save_upload_artifacts(
        output_dir=upload_dir,
        source_image=image,
        face_region=face_region,
        aligned_face_bgr=aligned_face_bgr,
        metadata=metadata,
    )

    payload = {
        "point_id": point_id,
        "entity_id": entity_id,
        "name": args.name,
        "source_image_path": str(Path(args.image_path).resolve()),
        "created_at": timestamp,
        "detection_score": primary_face["score"],
        "detection_bbox": primary_face["bbox"],
        "crop_bbox": face_region["crop_bbox"],
        "alignment_source": alignment_source,
        "alignment_warning": alignment_info.get("warning"),
        "detector": args.detector,
        "yunet_backend": primary_face.get("yunet_backend"),
        "artifacts": artifact_paths,
    }

    qdrant = QdrantStore(
        url=args.qdrant_url,
        api_key=args.qdrant_api_key,
        timeout=args.qdrant_timeout,
    )
    qdrant.create_collection(
        collection_name=args.collection,
        vector_size=len(embedding),
        distance="cosine",
    )
    qdrant.index_point(
        collection_name=args.collection,
        point_id=point_id,
        vector=embedding,
        payload=payload,
    )

    return {
        "point_id": point_id,
        "entity_id": entity_id,
        "name": args.name,
        "image_path": str(Path(args.image_path).resolve()),
        "collection": args.collection,
        "embedding_size": len(embedding),
        "artifacts": artifact_paths,
    }


def is_supported_image(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def collect_people_images(people_dir: Path) -> list[tuple[str, list[Path]]]:
    entities: list[tuple[str, list[Path]]] = []
    for entity_dir in sorted(path for path in people_dir.iterdir() if path.is_dir()):
        image_paths = sorted(path for path in entity_dir.iterdir() if is_supported_image(path))
        if image_paths:
            entities.append((entity_dir.name, image_paths))
    return entities


def collect_entity_images(entity_dir: Path) -> list[Path]:
    return sorted(path for path in entity_dir.iterdir() if is_supported_image(path))


def upload_entity_directory(args: argparse.Namespace) -> dict[str, Any]:
    entity_dir = Path(args.entity_dir).resolve()
    if not entity_dir.is_dir():
        raise ValueError(f"Diretorio invalido para --entity-dir: {entity_dir}")

    entity_name = entity_dir.name
    image_paths = collect_entity_images(entity_dir)
    if not image_paths:
        raise ValueError(f"Nenhuma imagem suportada encontrada em {entity_dir}")

    uploaded: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []

    for image_path in image_paths:
        image_args = copy.deepcopy(args)
        image_args.image_path = str(image_path)
        image_args.entity_id = entity_name
        image_args.name = entity_name
        image_args.point_id = None

        try:
            result = upload_face(image_args)
            uploaded.append(result)
        except Exception as exc:
            error = {
                "entity_id": entity_name,
                "name": entity_name,
                "image_path": str(image_path),
                "error": str(exc),
            }
            errors.append(error)
            if args.fail_fast:
                raise RuntimeError(json.dumps(error, ensure_ascii=True)) from exc

    return {
        "entity_dir": str(entity_dir),
        "entity_id": entity_name,
        "name": entity_name,
        "collection": args.collection,
        "images_found": len(image_paths),
        "images_uploaded": len(uploaded),
        "errors_count": len(errors),
        "uploaded": uploaded,
        "errors": errors,
    }


def upload_people_directory(args: argparse.Namespace) -> dict[str, Any]:
    people_dir = Path(args.people_dir).resolve()
    if not people_dir.is_dir():
        raise ValueError(f"Diretorio invalido para --people-dir: {people_dir}")

    entities = collect_people_images(people_dir)
    if not entities:
        raise ValueError(f"Nenhuma imagem suportada encontrada em {people_dir}")

    uploaded: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []

    for entity_name, image_paths in entities:
        for image_path in image_paths:
            image_args = copy.deepcopy(args)
            image_args.image_path = str(image_path)
            image_args.entity_id = entity_name
            image_args.name = entity_name
            image_args.point_id = None

            try:
                result = upload_face(image_args)
                uploaded.append(result)
            except Exception as exc:
                error = {
                    "entity_id": entity_name,
                    "name": entity_name,
                    "image_path": str(image_path),
                    "error": str(exc),
                }
                errors.append(error)
                if args.fail_fast:
                    raise RuntimeError(json.dumps(error, ensure_ascii=True)) from exc

    return {
        "people_dir": str(people_dir),
        "collection": args.collection,
        "entities_found": len(entities),
        "images_uploaded": len(uploaded),
        "errors_count": len(errors),
        "uploaded": uploaded,
        "errors": errors,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Faz o upload de um rosto: detecta, segmenta, alinha, extrai embedding e indexa no Qdrant."
    )
    parser.add_argument("image_path", nargs="?", default=None, help="Caminho da imagem.")
    parser.add_argument("--people-dir", type=str, default=None)
    parser.add_argument("--entity-dir", type=str, default=None)
    parser.add_argument("--point-id", type=str, default=None)
    parser.add_argument("--person-id", dest="point_id", type=str, default=None)
    parser.add_argument("--entity-id", type=str, default=None)
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--collection", type=str, default="Rostos")
    parser.add_argument("--qdrant-url", type=str, default="http://localhost:6333")
    parser.add_argument("--qdrant-api-key", type=str, default=None)
    parser.add_argument("--qdrant-timeout", type=float, default=15.0)
    parser.add_argument("--arcface-weight", type=str, default=str(DEFAULT_ARCFACE_WEIGHT))
    parser.add_argument("--arcface-network", type=str, default="r100")
    parser.add_argument("--embedder", type=str, default="arcface", choices=["arcface", "edgeface", "facelivt", "onnx", "tflite"])
    parser.add_argument("--onnx-model", type=str, default=None,
                   help="modelo .onnx para --embedder-backend onnx")
    parser.add_argument("--tflite-model", type=str, default=None,
                   help="modelo .tflite para --embedder-backend tflite")
    parser.add_argument("--tflite-cpu", action="store_true",
                   help="desliga o delegate da Edge TPU no backend tflite")
    parser.add_argument("--edgeface-model", type=str, default=DEFAULT_EDGEFACE_MODEL)
    parser.add_argument("--edgeface-checkpoint", type=str, default=str(DEFAULT_EDGEFACE_CHECKPOINT))
    parser.add_argument("--no-edgeface-compile", action="store_true")
    parser.add_argument("--no-arcface-compile", action="store_true")
    parser.add_argument("--detector", type=str, default="yunet", choices=["mtcnn", "yunet"])
    parser.add_argument("--yunet-model", type=str, default=str(DEFAULT_YUNET_MODEL))
    parser.add_argument("--yunet-nms", type=float, default=0.3)
    parser.add_argument("--yunet-top-k", type=int, default=5000)
    parser.add_argument("--yunet-backend", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--conf", type=float, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--require-alignment", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    args = parser.parse_args()

    provided_inputs = [bool(args.image_path), bool(args.people_dir), bool(args.entity_dir)]
    if sum(provided_inputs) != 1:
        parser.error("Informe exatamente um entre image_path, --entity-dir ou --people-dir.")
    if (args.people_dir is not None or args.entity_dir is not None) and args.point_id is not None:
        parser.error("--point-id nao pode ser usado junto com --entity-dir ou --people-dir.")
    if args.conf is None:
        args.conf = 0.5 if args.detector == "yunet" else 0.95
    return args


if __name__ == "__main__":
    cli_args = parse_args()
    if cli_args.entity_dir:
        result = upload_entity_directory(cli_args)
    elif cli_args.people_dir:
        result = upload_people_directory(cli_args)
    else:
        result = upload_face(cli_args)
    print(json.dumps(result, ensure_ascii=True, indent=2))
