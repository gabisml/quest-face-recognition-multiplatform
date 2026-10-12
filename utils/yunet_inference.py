from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_YUNET_MODEL = PROJECT_ROOT / "models" / "yunet" / "face_detection_yunet_2023mar.onnx"
YUNET_BACKENDS = {"auto", "cpu", "cuda"}

DEFAULT_YUNET_INPUT_LONG_SIDE = 640
# Lado minimo aceito. Abaixo disso o backend CUDA do DNN segfalta; ver a guarda em
# detect_faces_yunet.
MIN_YUNET_INPUT_SIDE = 64

_DETECTOR_CACHE: dict[tuple[str, int, int], Any] = {}


def _compute_yunet_input_size(orig_w: int, orig_h: int, long_side: int) -> tuple[int, int, float, float]:
    """Resize so the longest side equals long_side, rounded to multiples of 32.

    Com long_side <= 0 o quadro vai ao detector na resolucao nativa, sem reamostrar.
    O arredondamento abaixo nunca preserva 800x600, que e a resolucao do ChokePoint:
    a altura sempre cairia para 576. O YuNet e totalmente convolucional e aceita
    qualquer tamanho, inclusive fora do multiplo de 32 -- o proprio OpenCV monta as
    grades a partir do tamanho real das saidas.

    Returns (new_w, new_h, scale_x, scale_y) where scale_* maps original -> resized.
    """
    if long_side <= 0:
        return orig_w, orig_h, 1.0, 1.0
    long_side = max(32, (long_side // 32) * 32)
    scale = min(1.0, long_side / max(orig_w, orig_h))
    new_w = max(32, (int(round(orig_w * scale)) // 32) * 32)
    new_h = max(32, (int(round(orig_h * scale)) // 32) * 32)
    return new_w, new_h, new_w / orig_w, new_h / orig_h


def normalize_yunet_backend(backend: str) -> str:
    normalized = backend.lower().strip()
    if normalized not in YUNET_BACKENDS:
        raise ValueError(f"Backend YuNet invalido: {backend}. Use auto, cpu ou cuda.")
    return normalized


def _resolve_opencv_dnn_backend(cv2: Any, backend: str) -> tuple[str, int, int]:
    normalized = normalize_yunet_backend(backend)
    cuda_available = (
        hasattr(cv2, "cuda")
        and hasattr(cv2.cuda, "getCudaEnabledDeviceCount")
        and cv2.cuda.getCudaEnabledDeviceCount() > 0
    )
    if normalized == "cuda" or (normalized == "auto" and cuda_available):
        if not hasattr(cv2.dnn, "DNN_BACKEND_CUDA"):
            if normalized == "cuda":
                raise RuntimeError("OpenCV instalado nao possui backend DNN CUDA.")
        else:
            return "cuda", cv2.dnn.DNN_BACKEND_CUDA, cv2.dnn.DNN_TARGET_CUDA

    return "cpu", cv2.dnn.DNN_BACKEND_OPENCV, cv2.dnn.DNN_TARGET_CPU


def _get_or_create_detector(
    cv2: Any,
    model_path: str,
    size: tuple[int, int],
    conf: float,
    nms_threshold: float,
    top_k: int,
    backend_id: int,
    target_id: int,
) -> Any:
    key = (model_path, backend_id, target_id)
    detector = _DETECTOR_CACHE.get(key)
    if detector is None:
        detector = cv2.FaceDetectorYN.create(
            model_path, "", size, conf, nms_threshold, top_k, backend_id, target_id,
        )
        _DETECTOR_CACHE[key] = detector
        return detector

    detector.setInputSize(size)
    detector.setScoreThreshold(conf)
    detector.setNMSThreshold(nms_threshold)
    detector.setTopK(top_k)
    return detector


def detect_faces_yunet(
    image: Any,
    model_path: str | Path = DEFAULT_YUNET_MODEL,
    conf: float = 0.5,
    nms_threshold: float = 0.3,
    top_k: int = 5000,
    max_det: int = 10,
    backend: str = "auto",
    input_long_side: int = DEFAULT_YUNET_INPUT_LONG_SIDE,
) -> list[dict[str, Any]]:
    """Detect faces with OpenCV YuNet and return bbox + 5 ArcFace landmarks."""
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("opencv-python nao esta instalado.") from exc

    if not hasattr(cv2, "FaceDetectorYN"):
        raise RuntimeError("OpenCV instalado nao possui cv2.FaceDetectorYN.")

    resolved_model_path = Path(model_path)
    if not resolved_model_path.exists():
        raise FileNotFoundError(f"Modelo YuNet nao encontrado: {resolved_model_path}")

    if isinstance(image, np.ndarray):
        image_bgr = image.copy()
    else:
        image_rgb = np.asarray(image.convert("RGB"))
        image_bgr = image_rgb[:, :, ::-1].copy()

    height, width = image_bgr.shape[:2]

    # Guarda contra entrada minuscula. Com imagens muito pequenas o backend CUDA do DNN
    # provoca falha de segmentacao em vez de erro tratavel (reproduzido com 39x51, que o
    # redimensionamento leva a 32x32), e o backend de CPU devolve zero deteccoes de
    # qualquer forma, pois a rede nao tem resolucao para operar. Devolver lista vazia e o
    # comportamento correto e evita derrubar o processo inteiro por um quadro degenerado.
    if min(height, width) < MIN_YUNET_INPUT_SIDE:
        return []

    resolved_backend, backend_id, target_id = _resolve_opencv_dnn_backend(cv2, backend)
    model_key = str(resolved_model_path)

    new_w, new_h, scale_x, scale_y = _compute_yunet_input_size(width, height, input_long_side)
    if (new_w, new_h) != (width, height):
        resized_bgr = cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
    else:
        resized_bgr = image_bgr

    size = (new_w, new_h)
    conf_f = float(conf)
    nms_f = float(nms_threshold)
    top_k_i = int(top_k)
    try:
        detector = _get_or_create_detector(
            cv2, model_key, size, conf_f, nms_f, top_k_i, backend_id, target_id,
        )
    except Exception:
        if normalize_yunet_backend(backend) != "auto" or resolved_backend != "cuda":
            raise
        resolved_backend, backend_id, target_id = (
            "cpu",
            cv2.dnn.DNN_BACKEND_OPENCV,
            cv2.dnn.DNN_TARGET_CPU,
        )
        detector = _get_or_create_detector(
            cv2, model_key, size, conf_f, nms_f, top_k_i, backend_id, target_id,
        )
    _, faces = detector.detect(resized_bgr)
    if faces is None or len(faces) == 0:
        return []

    inv_x = 1.0 / scale_x
    inv_y = 1.0 / scale_y
    detections: list[dict[str, Any]] = []
    for index, face in enumerate(faces):
        x, y, w, h = (float(face[0]) * inv_x, float(face[1]) * inv_y,
                      float(face[2]) * inv_x, float(face[3]) * inv_y)
        score = float(face[14])
        x1 = max(0, min(width, int(round(x))))
        y1 = max(0, min(height, int(round(y))))
        x2 = max(0, min(width, int(round(x + w))))
        y2 = max(0, min(height, int(round(y + h))))
        area = max(0, x2 - x1) * max(0, y2 - y1)
        if area <= 0:
            continue

        kps_raw = np.asarray(face[4:14], dtype=np.float32).reshape(5, 2)
        kps = kps_raw.copy()
        kps[:, 0] *= inv_x
        kps[:, 1] *= inv_y
        detections.append(
            {
                "index": index,
                "bbox": [x1, y1, x2, y2],
                "score": score,
                "area": area,
                "kps": kps,
                "detector": "yunet",
                "yunet_backend": resolved_backend,
            }
        )

    detections.sort(key=lambda item: item["score"], reverse=True)
    return detections[:max_det]


__all__ = [
    "DEFAULT_YUNET_MODEL",
    "YUNET_BACKENDS",
    "detect_faces_yunet",
    "normalize_yunet_backend",
]
