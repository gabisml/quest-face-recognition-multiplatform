from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
from PIL import Image, ImageOps

from utils.arcface_inference import (
    ArcFaceEmbedder,
    FastMTCNN,
    align_face,
    align_face_crop,
    detect_faces_mtcnn,
    get_torch_device,
    resize_face_crop,
)
from utils.edgeface_inference import (
    DEFAULT_EDGEFACE_CHECKPOINT,
    DEFAULT_EDGEFACE_MODEL,
    EdgeFaceEmbedder,
)
from utils.facelivt_inference import (
    DEFAULT_FACELIVT_CHECKPOINT,
    DEFAULT_FACELIVT_MODEL,
    FaceLivTEmbedder,
)
from utils.identity_matching import select_best_identity_match
from utils.onnx_embedder import OnnxEmbedder
from utils.tflite_embedder import TFLiteEmbedder
from utils.vector_store import build_vector_store
from utils.yunet_inference import (
    DEFAULT_YUNET_INPUT_LONG_SIDE,
    DEFAULT_YUNET_MODEL,
    detect_faces_yunet,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
_PACKAGED_ARCFACE_WEIGHT = PROJECT_ROOT / "models" / "ms1mv3_arcface_r100_fp16" / "backbone.pth"
_LOCAL_ARCFACE_WEIGHT = PROJECT_ROOT / "models" / "backbone.pth"
DEFAULT_ARCFACE_WEIGHT = (
    _PACKAGED_ARCFACE_WEIGHT if _PACKAGED_ARCFACE_WEIGHT.exists() else _LOCAL_ARCFACE_WEIGHT
)
DETECTOR_BACKENDS = {"mtcnn", "yunet"}


def normalize_detector_backend(detector_backend: str) -> str:
    normalized = detector_backend.lower().strip()
    if normalized not in DETECTOR_BACKENDS:
        raise ValueError(
            f"Detector invalido: {detector_backend}. Use 'mtcnn' ou 'yunet'."
        )
    return normalized


def detect_faces(
    image: Any,
    detector_backend: str = "mtcnn",
    device: Optional[str] = None,
    conf: float = 0.9,
    max_det: int = 10,
    yunet_model: str = str(DEFAULT_YUNET_MODEL),
    yunet_nms: float = 0.3,
    yunet_top_k: int = 5000,
    yunet_backend: str = "auto",
    yunet_input_long_side: int = DEFAULT_YUNET_INPUT_LONG_SIDE,
) -> list[dict[str, Any]]:
    backend = normalize_detector_backend(detector_backend)
    if backend == "yunet":
        return detect_faces_yunet(
            image=image,
            model_path=yunet_model,
            conf=conf,
            nms_threshold=yunet_nms,
            top_k=yunet_top_k,
            max_det=max_det,
            backend=yunet_backend,
            input_long_side=yunet_input_long_side,
        )
    return detect_faces_mtcnn(image, device=device, conf=conf, max_det=max_det)


def bbox_iou(box_a: list[int], box_b: list[int]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    if inter_area <= 0:
        return 0.0

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union_area = area_a + area_b - inter_area
    if union_area <= 0:
        return 0.0
    return inter_area / float(union_area)


def expand_bbox(
    bbox: list[int],
    image_width: int,
    image_height: int,
    padding_ratio: float = 0.2,
) -> list[int]:
    x1, y1, x2, y2 = bbox
    pad = int(max(x2 - x1, y2 - y1) * padding_ratio)
    return [
        max(0, x1 - pad),
        max(0, y1 - pad),
        min(image_width, x2 + pad),
        min(image_height, y2 + pad),
    ]


def extract_face_region(
    image_rgb: np.ndarray,
    mask: Optional[np.ndarray],
    bbox: list[int],
    padding_ratio: float = 0.2,
) -> dict[str, Any]:
    height, width = image_rgb.shape[:2]
    crop_bbox = expand_bbox(
        bbox,
        image_width=width,
        image_height=height,
        padding_ratio=padding_ratio,
    )
    x1, y1, x2, y2 = crop_bbox

    crop_rgb = image_rgb[y1:y2, x1:x2].copy()
    if mask is None:
        crop_mask = np.ones(crop_rgb.shape[:2], dtype=np.uint8)
        segmented_crop_rgb = crop_rgb
    else:
        crop_mask = (mask[y1:y2, x1:x2] > 0).astype(np.uint8)
        if not crop_mask.any():
            crop_mask = np.ones(crop_rgb.shape[:2], dtype=np.uint8)
            segmented_crop_rgb = crop_rgb
        else:
            segmented_crop_rgb = crop_rgb.copy()
            segmented_crop_rgb[crop_mask == 0] = 0

    return {
        "crop_bbox": crop_bbox,
        "crop_rgb": crop_rgb,
        "segmented_crop_rgb": segmented_crop_rgb,
        "crop_mask": crop_mask,
    }


def compute_primary_face_embedding(
    image_path: str,
    arcface_weight: str = str(DEFAULT_ARCFACE_WEIGHT),
    arcface_network: str = "r100",
    embedder_precision: str = "fp16",
    conf: float = 0.9,
    device: Optional[str] = None,
    require_alignment: bool = False,
    embedder_backend: str = "arcface",
    edgeface_model_name: str = DEFAULT_EDGEFACE_MODEL,
    edgeface_checkpoint: str = str(DEFAULT_EDGEFACE_CHECKPOINT),
    edgeface_dir: Optional[str] = None,
    edgeface_compile: bool = True,
    arcface_compile: bool = True,
    facelivt_model_name: str = DEFAULT_FACELIVT_MODEL,
    facelivt_checkpoint: str = str(DEFAULT_FACELIVT_CHECKPOINT),
    facelivt_compile: bool = True,
    onnx_model: Optional[str] = None,
    tflite_model: Optional[str] = None,
    tflite_use_tpu: Optional[bool] = None,
    detector_backend: str = "mtcnn",
    yunet_model: str = str(DEFAULT_YUNET_MODEL),
    yunet_nms: float = 0.3,
    yunet_top_k: int = 5000,
    yunet_backend: str = "auto",
) -> dict[str, Any]:
    resolved_device = get_torch_device(device)

    image = ImageOps.exif_transpose(Image.open(image_path)).convert("RGB")
    image_rgb = np.asarray(image)
    image_bgr = image_rgb[:, :, ::-1].copy()

    detections = detect_faces(
        image=image_bgr if normalize_detector_backend(detector_backend) == "yunet" else image,
        detector_backend=detector_backend,
        device=resolved_device,
        conf=conf,
        max_det=10,
        yunet_model=yunet_model,
        yunet_nms=yunet_nms,
        yunet_top_k=yunet_top_k,
        yunet_backend=yunet_backend,
    )
    if not detections:
        raise ValueError("Nenhum rosto detectado na imagem.")
    primary_face = max(detections, key=lambda d: (d["area"], d["score"]))

    face_region = extract_face_region(
        image_rgb=image_rgb,
        mask=None,
        bbox=primary_face["bbox"],
    )

    if primary_face.get("kps") is not None:
        aligned_face_bgr = align_face(image_bgr, primary_face["kps"], image_size=112)
        alignment_info = {
            "bbox": [float(v) for v in primary_face["bbox"]],
            "kps": primary_face["kps"],
            "score": primary_face["score"],
        }
        alignment_source = "mtcnn_keypoints"
    else:
        original_crop_bgr = face_region["crop_rgb"][:, :, ::-1].copy()
        aligned_face_bgr, alignment_info, alignment_source = align_face_crop(
            face_crop_bgr=original_crop_bgr,
            fallback_crop_bgr=original_crop_bgr,
            device=resolved_device,
            allow_unaligned_fallback=not require_alignment,
        )

    if embedder_backend == "onnx":
        embedder = OnnxEmbedder(
            model_path=onnx_model,
            device=resolved_device,
            precision=embedder_precision,
        )
    elif embedder_backend == "tflite":
        embedder = TFLiteEmbedder(
            model_path=tflite_model,
            device=resolved_device,
            use_tpu=tflite_use_tpu,
            precision=embedder_precision,
        )
    elif embedder_backend == "edgeface":
        embedder = EdgeFaceEmbedder(
            checkpoint_path=edgeface_checkpoint,
            model_name=edgeface_model_name,
            edgeface_dir=edgeface_dir,
            device=resolved_device,
            precision=embedder_precision,
            compile_model=edgeface_compile,
        )
    elif embedder_backend == "facelivt":
        embedder = FaceLivTEmbedder(
            checkpoint_path=facelivt_checkpoint,
            model_name=facelivt_model_name,
            device=resolved_device,
            precision=embedder_precision,
            compile_model=facelivt_compile,
        )
    else:
        embedder = ArcFaceEmbedder(
            weight_path=arcface_weight,
            network=arcface_network,
            device=resolved_device,
            precision=embedder_precision,
            compile_model=arcface_compile,
        )
    embedding = embedder.embed_aligned_face(aligned_face_bgr)

    return {
        "device": resolved_device,
        "image": image,
        "image_rgb": image_rgb,
        "detections": detections,
        "primary_face": primary_face,
        "face_region": face_region,
        "aligned_face_bgr": aligned_face_bgr,
        "alignment_info": alignment_info,
        "alignment_source": alignment_source,
        "embedding": embedding,
    }


class FaceRecognizer:
    def __init__(
        self,
        qdrant_url: str = "http://localhost:6333",
        qdrant_api_key: Optional[str] = None,
        qdrant_timeout: float = 15.0,
        vector_store: str = "qdrant",
        gallery_path: Optional[str] = None,
        collection_name: str = "Rostos",
        arcface_weight: str = str(DEFAULT_ARCFACE_WEIGHT),
        arcface_network: str = "r100",
        embedder_precision: str = "fp16",
        conf: float = 0.9,
        score_threshold: Optional[float] = None,
        device: Optional[str] = None,
        require_alignment: bool = False,
        max_faces: int = 5,
        detect_width: Optional[int] = None,
        align_faces: bool = True,
        temporal_cache: bool = True,
        temporal_iou_threshold: float = 0.9,
        candidate_limit: int = 5,
        score_margin: float = 0.0,
        min_identity_votes: int = 1,
        embedder_backend: str = "arcface",
        edgeface_model_name: str = DEFAULT_EDGEFACE_MODEL,
        edgeface_checkpoint: str = str(DEFAULT_EDGEFACE_CHECKPOINT),
        edgeface_dir: Optional[str] = None,
        edgeface_compile: bool = True,
        arcface_compile: bool = True,
        facelivt_model_name: str = DEFAULT_FACELIVT_MODEL,
        facelivt_checkpoint: str = str(DEFAULT_FACELIVT_CHECKPOINT),
        facelivt_compile: bool = True,
        onnx_model: Optional[str] = None,
        tflite_model: Optional[str] = None,
        tflite_use_tpu: Optional[bool] = None,
        detect_interval: int = 1,
        use_fast_mtcnn: bool = False,
        fast_mtcnn_stride: int = 3,
        fast_mtcnn_resize: float = 1.0,
        detector_backend: str = "mtcnn",
        yunet_model: str = str(DEFAULT_YUNET_MODEL),
        yunet_nms: float = 0.3,
        yunet_top_k: int = 5000,
        yunet_backend: str = "auto",
        yunet_input_long_side: int = DEFAULT_YUNET_INPUT_LONG_SIDE,
    ) -> None:
        self.device = get_torch_device(device)
        self.collection_name = collection_name
        self.conf = conf
        self.score_threshold = score_threshold
        self.require_alignment = require_alignment
        self.max_faces = max_faces
        self.detect_width = detect_width
        self.align_faces = align_faces
        self.temporal_cache = temporal_cache
        self.temporal_iou_threshold = temporal_iou_threshold
        self.candidate_limit = max(1, candidate_limit)
        self.score_margin = max(0.0, score_margin)
        self.min_identity_votes = max(1, min_identity_votes)
        self.detect_interval = max(1, detect_interval)
        self.use_fast_mtcnn = use_fast_mtcnn
        self.fast_mtcnn_stride = max(1, fast_mtcnn_stride)
        self.fast_mtcnn_resize = fast_mtcnn_resize
        self.detector_backend = normalize_detector_backend(detector_backend)
        self.yunet_model = yunet_model
        self.yunet_nms = yunet_nms
        self.yunet_top_k = yunet_top_k
        self.yunet_backend = yunet_backend
        self.yunet_input_long_side = yunet_input_long_side
        self._frame_counter: int = 0
        self.last_timing: dict[str, float] = {}
        self.last_debug_frame_rgb: Optional[np.ndarray] = None
        self.last_detections: list[dict[str, Any]] = []
        self.last_debug_faces: list[dict[str, Any]] = []
        self.temporal_face_cache: list[dict[str, Any]] = []
        self._cached_recognitions: list[dict[str, Any]] = []

        if embedder_backend == "onnx":
            self.embedder = OnnxEmbedder(
                model_path=onnx_model,
                device=self.device,
                precision=embedder_precision,
            )
        elif embedder_backend == "tflite":
            self.embedder = TFLiteEmbedder(
                model_path=tflite_model,
                device=self.device,
                use_tpu=tflite_use_tpu,
                precision=embedder_precision,
            )
        elif embedder_backend == "edgeface":
            self.embedder = EdgeFaceEmbedder(
                checkpoint_path=edgeface_checkpoint,
                model_name=edgeface_model_name,
                edgeface_dir=edgeface_dir,
                device=self.device,
                precision=embedder_precision,
                compile_model=edgeface_compile,
            )
        elif embedder_backend == "facelivt":
            self.embedder = FaceLivTEmbedder(
                checkpoint_path=facelivt_checkpoint,
                model_name=facelivt_model_name,
                device=self.device,
                precision=embedder_precision,
                compile_model=facelivt_compile,
            )
        else:
            self.embedder = ArcFaceEmbedder(
                weight_path=arcface_weight,
                network=arcface_network,
                device=self.device,
                precision=embedder_precision,
                compile_model=arcface_compile,
            )

        # O backend de busca e selecionavel porque a Coral Dev Board nao roda o container
        # do Qdrant (985 MB de RAM). Os dois expoem o mesmo query_batch, entao o resto do
        # pipeline nao muda. Ver utils/vector_store.py para a ressalva metodologica: o
        # backend domina a etapa de busca, nao o hardware.
        self.vector_store_kind = vector_store
        self.qdrant = build_vector_store(
            vector_store,
            qdrant_url=qdrant_url,
            qdrant_api_key=qdrant_api_key,
            qdrant_timeout=qdrant_timeout,
            gallery_path=gallery_path,
        )
        self.fast_mtcnn: Optional[FastMTCNN] = None
        if self.use_fast_mtcnn:
            if self.detector_backend != "mtcnn":
                raise ValueError("use_fast_mtcnn so pode ser usado com detector_backend='mtcnn'.")
            fast_mtcnn_device = self.device if self.device != "mps" else "cpu"
            self.fast_mtcnn = FastMTCNN(
                stride=self.fast_mtcnn_stride,
                resize=self.fast_mtcnn_resize,
                margin=0,
                post_process=False,
                select_largest=False,
                keep_all=True,
                device=fast_mtcnn_device,
            )

    @staticmethod
    def _scale_detections(
        detections: list[dict[str, Any]],
        scale_x: float,
        scale_y: float,
        frame_width: int,
        frame_height: int,
    ) -> list[dict[str, Any]]:
        scaled: list[dict[str, Any]] = []
        for item in detections:
            x1, y1, x2, y2 = item["bbox"]
            entry = {
                **item,
                "bbox": [
                    max(0, min(frame_width, int(round(x1 * scale_x)))),
                    max(0, min(frame_height, int(round(y1 * scale_y)))),
                    max(0, min(frame_width, int(round(x2 * scale_x)))),
                    max(0, min(frame_height, int(round(y2 * scale_y)))),
                ],
            }
            if "kps" in item and item["kps"] is not None:
                kps = item["kps"].copy()
                kps[:, 0] *= scale_x
                kps[:, 1] *= scale_y
                entry["kps"] = kps
            scaled.append(entry)
        return scaled

    def _prepare_detection_frame(
        self,
        frame_bgr: np.ndarray,
    ) -> tuple[np.ndarray, float, float]:
        if self.detect_width is None or self.detect_width <= 0:
            return frame_bgr, 1.0, 1.0

        frame_height, frame_width = frame_bgr.shape[:2]
        if frame_width <= self.detect_width:
            return frame_bgr, 1.0, 1.0

        try:
            import cv2
        except ImportError:
            return frame_bgr, 1.0, 1.0

        scale = self.detect_width / float(frame_width)
        resized_width = self.detect_width
        resized_height = max(1, int(round(frame_height * scale)))
        resized = cv2.resize(
            frame_bgr,
            (resized_width, resized_height),
            interpolation=cv2.INTER_LINEAR,
        )
        return resized, frame_width / float(resized_width), frame_height / float(resized_height)

    def _recognize_from_rgb(
        self,
        image_rgb: np.ndarray,
        detections: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, float]]:
        per_face: list[dict[str, Any]] = []
        alignment_ms = 0.0

        any_has_kps = self.align_faces and any(
            "kps" in d and d["kps"] is not None for d in detections
        )
        image_bgr = image_rgb[:, :, ::-1] if any_has_kps else None

        # Etapa 1: alinhamento de todas as faces
        for detection in detections:
            face_region = extract_face_region(
                image_rgb=image_rgb,
                mask=None,
                bbox=detection["bbox"],
            )

            align_start = time.perf_counter()
            if not self.align_faces:
                original_crop_bgr = face_region["crop_rgb"][:, :, ::-1].copy()
                aligned_face_bgr = resize_face_crop(original_crop_bgr)
                height, width = original_crop_bgr.shape[:2]
                alignment_info = {
                    "bbox": [0.0, 0.0, float(width), float(height)],
                    "kps": np.empty((0, 2), dtype=np.float32),
                    "score": None,
                    "warning": "alignment_disabled",
                }
                alignment_source = "disabled"
            elif "kps" in detection and detection["kps"] is not None:
                kps = detection["kps"]
                aligned_face_bgr = align_face(image_bgr, kps, image_size=112)
                alignment_info = {
                    "bbox": [float(v) for v in detection["bbox"]],
                    "kps": kps,
                    "score": detection["score"],
                }
                alignment_source = "mtcnn_keypoints"
            else:
                original_crop_bgr = face_region["crop_rgb"][:, :, ::-1].copy()
                segmented_crop_bgr = face_region["segmented_crop_rgb"][:, :, ::-1].copy()
                aligned_face_bgr, alignment_info, alignment_source = align_face_crop(
                    face_crop_bgr=segmented_crop_bgr,
                    fallback_crop_bgr=original_crop_bgr,
                    device=self.device,
                    allow_unaligned_fallback=not self.require_alignment,
                )
            alignment_ms += (time.perf_counter() - align_start) * 1000.0

            per_face.append(
                {
                    "detection": detection,
                    "face_region": face_region,
                    "aligned_face_bgr": aligned_face_bgr,
                    "alignment_info": alignment_info,
                    "alignment_source": alignment_source,
                }
            )

        # Etapa 2: embedding em batch
        embedding_start = time.perf_counter()
        embeddings = self.embedder.embed_batch([f["aligned_face_bgr"] for f in per_face])
        embedding_ms = (time.perf_counter() - embedding_start) * 1000.0

        # Etapa 3: busca em batch
        search_start = time.perf_counter()
        batch_results = self.qdrant.query_batch(
            collection_name=self.collection_name,
            vectors=embeddings,
            limit=self.candidate_limit,
            score_threshold=self.score_threshold,
        )
        search_ms = (time.perf_counter() - search_start) * 1000.0

        # Etapa 4: montar reconhecimentos
        recognitions = []
        debug_faces = []
        for face, embedding, matches in zip(per_face, embeddings, batch_results):
            detection = face["detection"]
            best_identity_match, identity_matches = select_best_identity_match(
                matches,
                limit=self.candidate_limit,
                min_score=self.score_threshold,
                score_margin=self.score_margin,
                min_match_count=self.min_identity_votes,
            )
            top_identity_match = identity_matches[0] if identity_matches else None
            payload = best_identity_match.get("matches", [{}])[0].get("payload") if best_identity_match else None
            name = None
            entity_id = None
            if isinstance(payload, dict):
                name = payload.get("name")
                entity_id = payload.get("entity_id")
            if best_identity_match is not None:
                entity_id = best_identity_match.get("entity_id") or entity_id
                if name is None:
                    name = best_identity_match.get("name")

            recognitions.append(
                {
                    "bbox": detection["bbox"],
                    "detection_score": detection["score"],
                    "entity_id": entity_id,
                    "name": name,
                    "recognized": best_identity_match is not None and name is not None,
                    "score": best_identity_match.get("score") if best_identity_match else None,
                    "match_id": best_identity_match.get("best_match_id") if best_identity_match else None,
                    "top_candidate_name": (
                        top_identity_match.get("name") if top_identity_match else None
                    ),
                    "top_candidate_entity_id": (
                        top_identity_match.get("entity_id") if top_identity_match else None
                    ),
                    "top_candidate_score": (
                        top_identity_match.get("score") if top_identity_match else None
                    ),
                    "top_candidate_match_count": (
                        top_identity_match.get("match_count") if top_identity_match else 0
                    ),
                    "identity_match_count": (
                        best_identity_match.get("match_count") if best_identity_match else 0
                    ),
                    "identity_score_gap": (
                        None
                        if len(identity_matches) < 2
                        else (
                            None
                            if identity_matches[0].get("score") is None or identity_matches[1].get("score") is None
                            else identity_matches[0]["score"] - identity_matches[1]["score"]
                        )
                    ),
                    "alignment_source": face["alignment_source"],
                    "alignment_warning": face["alignment_info"].get("warning"),
                }
            )
            debug_faces.append(
                {
                    "bbox": detection["bbox"],
                    "detection_score": detection["score"],
                    "face_region": face["face_region"],
                    "aligned_face_bgr": face["aligned_face_bgr"].copy(),
                    "alignment_info": face["alignment_info"],
                    "alignment_source": face["alignment_source"],
                    "embedding": embedding,
                    "matches": matches,
                    "identity_matches": identity_matches,
                }
            )

        return recognitions, debug_faces, {
            "alignment_ms": alignment_ms,
            "embedding_ms": embedding_ms,
            "search_ms": search_ms,
        }

    def _clone_temporal_recognition(
        self,
        detection: dict[str, Any],
        cached_item: dict[str, Any],
        image_rgb: np.ndarray,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        face_region = extract_face_region(
            image_rgb=image_rgb,
            mask=None,
            bbox=detection["bbox"],
        )

        cached_recognition = cached_item["recognition"]
        cached_debug_face = cached_item["debug_face"]
        recognition = {
            **cached_recognition,
            "bbox": detection["bbox"],
            "detection_score": detection["score"],
            "reused_from_previous_frame": True,
        }
        debug_face = {
            **cached_debug_face,
            "bbox": detection["bbox"],
            "detection_score": detection["score"],
            "face_region": face_region,
            "reused_from_previous_frame": True,
        }
        return recognition, debug_face

    def _match_temporal_cache(self, detection: dict[str, Any], used_cache_indices: set[int]) -> Optional[int]:
        best_index: Optional[int] = None
        best_iou = self.temporal_iou_threshold

        for cache_index, cached_item in enumerate(self.temporal_face_cache):
            if cache_index in used_cache_indices:
                continue

            iou = bbox_iou(detection["bbox"], cached_item["bbox"])
            if iou >= best_iou:
                best_iou = iou
                best_index = cache_index

        return best_index

    def _update_temporal_cache(
        self,
        recognitions: list[dict[str, Any]],
        debug_faces: list[dict[str, Any]],
    ) -> None:
        cache: list[dict[str, Any]] = []
        for recognition, debug_face in zip(recognitions, debug_faces):
            cache.append(
                {
                    "bbox": list(recognition["bbox"]),
                    "recognition": {**recognition},
                    "debug_face": {**debug_face},
                }
            )
        self.temporal_face_cache = cache

    def get_debug_face(self, face_index: int = 0) -> dict[str, Any]:
        if self.last_debug_frame_rgb is None or not self.last_debug_faces:
            raise ValueError("Nenhum artefato de debug disponivel. Execute o reconhecimento antes.")

        if face_index < 0 or face_index >= len(self.last_debug_faces):
            raise IndexError(
                f"face_index invalido: {face_index}. Faces disponiveis: {len(self.last_debug_faces)}."
            )

        debug_face = self.last_debug_faces[face_index]
        return {
            "image_rgb": self.last_debug_frame_rgb,
            "detections": self.last_detections,
            **debug_face,
        }

    def recognize_image(self, image: Image.Image) -> list[dict[str, Any]]:
        image_rgb = np.asarray(image.convert("RGB"))
        total_start = time.perf_counter()
        detection_start = time.perf_counter()
        image_bgr = image_rgb[:, :, ::-1].copy()
        detections = detect_faces(
            image=image_bgr if self.detector_backend == "yunet" else image,
            detector_backend=self.detector_backend,
            device=self.device,
            conf=self.conf,
            max_det=self.max_faces,
            yunet_model=self.yunet_model,
            yunet_nms=self.yunet_nms,
            yunet_top_k=self.yunet_top_k,
            yunet_backend=self.yunet_backend,
            yunet_input_long_side=self.yunet_input_long_side,
        )
        detection_ms = (time.perf_counter() - detection_start) * 1000.0
        if not detections:
            self.last_debug_frame_rgb = image_rgb.copy()
            self.last_detections = []
            self.last_debug_faces = []
            self.temporal_face_cache = []
            self.last_timing = {
                "detection_ms": detection_ms,
                "alignment_ms": 0.0,
                "embedding_ms": 0.0,
                "search_ms": 0.0,
                "total_ms": (time.perf_counter() - total_start) * 1000.0,
            }
            return []

        self.last_debug_frame_rgb = image_rgb.copy()
        self.last_detections = detections
        self.last_timing = {"detection_ms": detection_ms}
        recognitions, debug_faces, stage_timing = self._recognize_from_rgb(
            image_rgb=image_rgb,
            detections=detections,
        )
        self.last_timing.update(stage_timing)
        self.last_debug_faces = debug_faces
        self.temporal_face_cache = []
        self.last_timing["total_ms"] = (time.perf_counter() - total_start) * 1000.0
        return recognitions

    def recognize_frame(self, frame_bgr: np.ndarray) -> list[dict[str, Any]]:
        total_start = time.perf_counter()
        self._frame_counter += 1

        should_detect = (
            self._frame_counter % self.detect_interval == 1
            or self.detect_interval == 1
            or not self._cached_recognitions
        )
        if not should_detect:
            self.last_timing = {
                "detection_ms": 0.0,
                "alignment_ms": 0.0,
                "embedding_ms": 0.0,
                "search_ms": 0.0,
                "reused_faces": float(len(self._cached_recognitions)),
                "new_faces": 0.0,
                "skipped_detection": 1.0,
                "total_ms": (time.perf_counter() - total_start) * 1000.0,
            }
            return self._cached_recognitions

        detection_frame, scale_x, scale_y = self._prepare_detection_frame(frame_bgr)

        detection_start = time.perf_counter()
        detections = detect_faces(
            image=detection_frame,
            detector_backend=self.detector_backend,
            device=self.device,
            conf=self.conf,
            max_det=self.max_faces,
            yunet_model=self.yunet_model,
            yunet_nms=self.yunet_nms,
            yunet_top_k=self.yunet_top_k,
            yunet_backend=self.yunet_backend,
            yunet_input_long_side=self.yunet_input_long_side,
        )
        detection_ms = (time.perf_counter() - detection_start) * 1000.0
        self.last_timing = {"detection_ms": detection_ms}

        if not detections:
            self.last_debug_frame_rgb = None
            self.last_detections = []
            self.last_debug_faces = []
            self.temporal_face_cache = []
            self._cached_recognitions = []
            self.last_timing.update(
                {
                    "alignment_ms": 0.0,
                    "embedding_ms": 0.0,
                    "search_ms": 0.0,
                    "total_ms": (time.perf_counter() - total_start) * 1000.0,
                }
            )
            return []

        if scale_x != 1.0 or scale_y != 1.0:
            detections = self._scale_detections(
                detections=detections,
                scale_x=scale_x,
                scale_y=scale_y,
                frame_width=frame_bgr.shape[1],
                frame_height=frame_bgr.shape[0],
            )

        image_rgb = frame_bgr[:, :, ::-1]
        self.last_debug_frame_rgb = image_rgb
        self.last_detections = detections
        recognitions: list[Optional[dict[str, Any]]] = [None] * len(detections)
        debug_faces: list[Optional[dict[str, Any]]] = [None] * len(detections)
        unmatched_indices: list[int] = []
        unmatched_detections: list[dict[str, Any]] = []
        used_cache_indices: set[int] = set()
        reused_faces = 0

        if self.temporal_cache and self.temporal_face_cache:
            for detection_index, detection in enumerate(detections):
                cache_index = self._match_temporal_cache(detection, used_cache_indices)
                if cache_index is None:
                    unmatched_indices.append(detection_index)
                    unmatched_detections.append(detection)
                    continue

                recognition, debug_face = self._clone_temporal_recognition(
                    detection=detection,
                    cached_item=self.temporal_face_cache[cache_index],
                    image_rgb=image_rgb,
                )
                recognitions[detection_index] = recognition
                debug_faces[detection_index] = debug_face
                used_cache_indices.add(cache_index)
                reused_faces += 1
        else:
            unmatched_indices = list(range(len(detections)))
            unmatched_detections = detections

        stage_timing = {
            "alignment_ms": 0.0,
            "embedding_ms": 0.0,
            "search_ms": 0.0,
        }
        if unmatched_detections:
            new_recognitions, new_debug_faces, stage_timing = self._recognize_from_rgb(
                image_rgb=image_rgb,
                detections=unmatched_detections,
            )
            for detection_index, recognition, debug_face in zip(
                unmatched_indices,
                new_recognitions,
                new_debug_faces,
            ):
                recognition["reused_from_previous_frame"] = False
                debug_face["reused_from_previous_frame"] = False
                recognitions[detection_index] = recognition
                debug_faces[detection_index] = debug_face

        final_recognitions = [item for item in recognitions if item is not None]
        final_debug_faces = [item for item in debug_faces if item is not None]
        self.last_timing.update(stage_timing)
        self.last_timing["reused_faces"] = float(reused_faces)
        self.last_timing["new_faces"] = float(len(unmatched_detections))
        self.last_debug_faces = final_debug_faces
        self._update_temporal_cache(final_recognitions, final_debug_faces)
        self._cached_recognitions = final_recognitions
        self.last_timing["total_ms"] = (time.perf_counter() - total_start) * 1000.0
        return final_recognitions

    def recognize_frame_with_detections(
        self,
        frame_bgr: np.ndarray,
        detections: list[dict[str, Any]],
        detection_ms: float = 0.0,
    ) -> list[dict[str, Any]]:
        total_start = time.perf_counter()
        self._frame_counter += 1
        self.last_timing = {"detection_ms": detection_ms}

        if not detections:
            self.last_debug_frame_rgb = None
            self.last_detections = []
            self.last_debug_faces = []
            self.temporal_face_cache = []
            self._cached_recognitions = []
            self.last_timing.update(
                {
                    "alignment_ms": 0.0,
                    "embedding_ms": 0.0,
                    "search_ms": 0.0,
                    "total_ms": (time.perf_counter() - total_start) * 1000.0,
                }
            )
            return []

        image_rgb = frame_bgr[:, :, ::-1]
        self.last_debug_frame_rgb = image_rgb
        self.last_detections = detections
        recognitions: list[Optional[dict[str, Any]]] = [None] * len(detections)
        debug_faces: list[Optional[dict[str, Any]]] = [None] * len(detections)
        unmatched_indices: list[int] = []
        unmatched_detections: list[dict[str, Any]] = []
        used_cache_indices: set[int] = set()
        reused_faces = 0

        if self.temporal_cache and self.temporal_face_cache:
            for detection_index, detection in enumerate(detections):
                cache_index = self._match_temporal_cache(detection, used_cache_indices)
                if cache_index is None:
                    unmatched_indices.append(detection_index)
                    unmatched_detections.append(detection)
                    continue

                recognition, debug_face = self._clone_temporal_recognition(
                    detection=detection,
                    cached_item=self.temporal_face_cache[cache_index],
                    image_rgb=image_rgb,
                )
                recognitions[detection_index] = recognition
                debug_faces[detection_index] = debug_face
                used_cache_indices.add(cache_index)
                reused_faces += 1
        else:
            unmatched_indices = list(range(len(detections)))
            unmatched_detections = detections

        stage_timing = {
            "alignment_ms": 0.0,
            "embedding_ms": 0.0,
            "search_ms": 0.0,
        }
        if unmatched_detections:
            new_recognitions, new_debug_faces, stage_timing = self._recognize_from_rgb(
                image_rgb=image_rgb,
                detections=unmatched_detections,
            )
            for detection_index, recognition, debug_face in zip(
                unmatched_indices,
                new_recognitions,
                new_debug_faces,
            ):
                recognition["reused_from_previous_frame"] = False
                debug_face["reused_from_previous_frame"] = False
                recognitions[detection_index] = recognition
                debug_faces[detection_index] = debug_face

        final_recognitions = [item for item in recognitions if item is not None]
        final_debug_faces = [item for item in debug_faces if item is not None]
        self.last_timing.update(stage_timing)
        self.last_timing["reused_faces"] = float(reused_faces)
        self.last_timing["new_faces"] = float(len(unmatched_detections))
        self.last_debug_faces = final_debug_faces
        self._update_temporal_cache(final_recognitions, final_debug_faces)
        self._cached_recognitions = final_recognitions
        self.last_timing["total_ms"] = (time.perf_counter() - total_start) * 1000.0
        return final_recognitions


__all__ = [
    "DEFAULT_ARCFACE_WEIGHT",
    "DEFAULT_YUNET_MODEL",
    "FaceRecognizer",
    "compute_primary_face_embedding",
    "detect_faces",
    "expand_bbox",
    "extract_face_region",
]
