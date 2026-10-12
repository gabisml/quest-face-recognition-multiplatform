from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

import numpy as np


# utils.local_arcface_backbones importa torch no topo, entao ele NAO pode ser importado
# aqui em nivel de modulo: isso faria `import utils.face_pipeline` exigir torch em toda
# plataforma, inclusive nas que rodam ONNX ou TFLite justamente por nao comportarem torch.
# O Raspberry Pi 5 e o caso concreto: a wheel aarch64 do PyPI arrasta 1,5 GB de pilha CUDA
# e nao cabe no cartao. O import fica dentro de _get_arcface_get_model, que so e chamado
# quando o backend arcface e efetivamente instanciado.


ARC_FACE_DST = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)


@lru_cache(maxsize=1)
@lru_cache(maxsize=1)
def _get_arcface_get_model():
    from utils.local_arcface_backbones import get_model as get_local_arcface_model

    return get_local_arcface_model


def get_torch_device(device: Optional[str] = None) -> str:
    if device is not None:
        return device

    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("torch nao esta instalado.") from exc

    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_model_precision(
    precision: str,
    device: str,
) -> str:
    normalized = precision.lower().strip()
    if normalized not in {"fp16", "fp32"}:
        raise ValueError(f"Precisao invalida: {precision}. Use 'fp16' ou 'fp32'.")
    if normalized == "fp16" and not str(device).startswith("cuda"):
        return "fp32"
    return normalized


def self_torch_no_grad(fn):
    def wrapper(self, *args, **kwargs):
        with self.torch.no_grad():
            return fn(self, *args, **kwargs)

    return wrapper


def resize_face_crop(
    image_bgr: np.ndarray,
    image_size: int = 112,
) -> np.ndarray:
    try:
        import cv2

        return cv2.resize(image_bgr, (image_size, image_size))
    except ImportError:
        from PIL import Image

        image_rgb = image_bgr[:, :, ::-1]
        resized_rgb = np.asarray(
            Image.fromarray(image_rgb).resize((image_size, image_size))
        )
        return resized_rgb[:, :, ::-1].copy()


def estimate_arcface_transform(
    landmarks5: np.ndarray,
    image_size: int = 112,
) -> np.ndarray:
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("opencv-python nao esta instalado.") from exc

    if landmarks5.shape != (5, 2):
        raise ValueError("Os landmarks devem ter shape (5, 2).")

    if image_size % 112 == 0:
        ratio = float(image_size) / 112.0
        diff_x = 0.0
    elif image_size % 128 == 0:
        ratio = float(image_size) / 128.0
        diff_x = 8.0 * ratio
    else:
        raise ValueError("image_size deve ser multiplo de 112 ou 128.")

    dst = ARC_FACE_DST * ratio
    dst[:, 0] += diff_x
    matrix, _ = cv2.estimateAffinePartial2D(
        landmarks5.astype(np.float32),
        dst.astype(np.float32),
        method=cv2.LMEDS,
    )
    if matrix is None:
        raise ValueError("Nao foi possivel estimar a transformacao facial.")
    return matrix


class ArcFaceEmbedder:
    def __init__(
        self,
        weight_path: str,
        network: str = "r100",
        device: Optional[str] = None,
        precision: str = "fp32",
        compile_model: bool = True,
    ) -> None:
        try:
            import cv2
            import torch
            import torch.nn.functional as functional
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "Dependencias do ArcFace nao estao disponiveis. "
                "Instale torch e opencv-python no ambiente ativo."
            ) from exc

        get_model = _get_arcface_get_model()

        self.cv2 = cv2
        self.torch = torch
        self.functional = functional
        self.device = get_torch_device(device)
        self.precision = resolve_model_precision(precision, self.device)
        self.use_fp16 = self.precision == "fp16"

        self.model = get_model(network, fp16=self.use_fp16).to(self.device)
        resolved_weight_path = Path(weight_path)
        if resolved_weight_path.is_dir():
            resolved_weight_path = resolved_weight_path / "backbone.pth"
        if not resolved_weight_path.exists():
            raise FileNotFoundError(
                f"Peso ArcFace nao encontrado em {resolved_weight_path}."
            )

        state_dict = torch.load(resolved_weight_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()

        if compile_model and self.device.startswith("cuda"):
            self.model = torch.compile(self.model, mode="reduce-overhead")

    def _preprocess_face(self, aligned_face_bgr: np.ndarray) -> np.ndarray:
        face = aligned_face_bgr
        if face.shape[:2] != (112, 112):
            face = self.cv2.resize(face, (112, 112))
        face = self.cv2.cvtColor(face, self.cv2.COLOR_BGR2RGB)
        return np.transpose(face, (2, 0, 1))

    @self_torch_no_grad
    def embed_aligned_face(
        self,
        aligned_face_bgr: np.ndarray,
        l2_normalize: bool = True,
    ) -> np.ndarray:
        return self.embed_batch([aligned_face_bgr], l2_normalize=l2_normalize)[0]

    @self_torch_no_grad
    def embed_batch(
        self,
        aligned_faces_bgr: list[np.ndarray],
        l2_normalize: bool = True,
    ) -> list[np.ndarray]:
        """Processa múltiplas faces em um único forward pass do ArcFace."""
        if not aligned_faces_bgr:
            return []

        arrays = np.stack([self._preprocess_face(f) for f in aligned_faces_bgr], axis=0)
        tensor = self.torch.from_numpy(arrays).float().to(self.device)
        tensor.div_(255).sub_(0.5).div_(0.5)

        try:
            embeddings = self.model(tensor)
        except RuntimeError:
            embeddings = self.model.to("cpu")(tensor.to("cpu"))
            self.model.to(self.device)
        if l2_normalize:
            embeddings = self.functional.normalize(embeddings, p=2, dim=1)

        result = embeddings.detach().cpu().numpy()
        return [result[i] for i in range(result.shape[0])]


@lru_cache(maxsize=2)
def _get_mtcnn_detector(device: str):
    try:
        from facenet_pytorch import MTCNN
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "facenet-pytorch nao esta disponivel. "
            "Instale com: pip install facenet-pytorch"
        ) from exc
    return MTCNN(
        margin=0,
        post_process=False,
        select_largest=False,
        keep_all=True,
        device=device,
    )


class FastMTCNN:
    """Fast MTCNN implementation with strided batched detection."""

    def __init__(self, stride: int, resize: float = 1.0, *args, **kwargs) -> None:
        if stride < 1:
            raise ValueError("stride deve ser maior ou igual a 1.")
        if resize <= 0:
            raise ValueError("resize deve ser maior que 0.")

        try:
            from facenet_pytorch import MTCNN
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "facenet-pytorch nao esta disponivel. "
                "Instale com: pip install facenet-pytorch"
            ) from exc

        self.stride = stride
        self.resize = resize
        self.mtcnn = MTCNN(*args, **kwargs)

    @staticmethod
    def _normalize_frame(frame: Any) -> np.ndarray:
        if isinstance(frame, np.ndarray):
            return frame
        if hasattr(frame, "convert"):
            return np.asarray(frame.convert("RGB"))
        raise TypeError("Frame invalido para FastMTCNN.")

    def _resize_frames(self, frames: list[np.ndarray]) -> list[np.ndarray]:
        if self.resize == 1.0:
            return frames

        try:
            import cv2
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError("opencv-python nao esta instalado.") from exc

        resized_frames: list[np.ndarray] = []
        for frame in frames:
            resized_frames.append(
                cv2.resize(
                    frame,
                    (
                        max(1, int(frame.shape[1] * self.resize)),
                        max(1, int(frame.shape[0] * self.resize)),
                    ),
                )
            )
        return resized_frames

    def detect(
        self,
        frames: list[Any],
        conf: float = 0.9,
        max_det: int = 10,
    ) -> list[list[dict[str, Any]]]:
        if not frames:
            return []

        normalized_frames = [self._normalize_frame(frame) for frame in frames]
        resized_frames = self._resize_frames(normalized_frames)
        sampled_frames = resized_frames[:: self.stride]
        boxes, probs, points = self.mtcnn.detect(sampled_frames, landmarks=True)

        detections_by_frame: list[list[dict[str, Any]]] = []
        for frame_index, frame in enumerate(resized_frames):
            sampled_index = min(frame_index // self.stride, len(sampled_frames) - 1)
            frame_boxes = None if boxes is None else boxes[sampled_index]
            frame_probs = None if probs is None else probs[sampled_index]
            frame_points = None if points is None else points[sampled_index]
            frame_detections: list[dict[str, Any]] = []

            if frame_boxes is None or frame_probs is None:
                detections_by_frame.append(frame_detections)
                continue

            for det_index, (box, prob) in enumerate(zip(frame_boxes, frame_probs)):
                if prob is None or float(prob) < conf:
                    continue

                x1, y1, x2, y2 = (
                    int(round(box[0])),
                    int(round(box[1])),
                    int(round(box[2])),
                    int(round(box[3])),
                )
                x1 = max(0, min(frame.shape[1], x1))
                y1 = max(0, min(frame.shape[0], y1))
                x2 = max(0, min(frame.shape[1], x2))
                y2 = max(0, min(frame.shape[0], y2))
                if x2 <= x1 or y2 <= y1:
                    continue

                kps = None
                if frame_points is not None and det_index < len(frame_points):
                    kps = np.asarray(frame_points[det_index], dtype=np.float32)

                frame_detections.append(
                    {
                        "index": det_index,
                        "bbox": [x1, y1, x2, y2],
                        "score": float(prob),
                        "area": max(0, x2 - x1) * max(0, y2 - y1),
                        "kps": kps,
                    }
                )

            frame_detections.sort(key=lambda item: item["score"], reverse=True)
            detections_by_frame.append(frame_detections[:max_det])

        if self.resize != 1.0:
            scale_x = 1.0 / self.resize
            scale_y = 1.0 / self.resize
            scaled_detections: list[list[dict[str, Any]]] = []
            for original_frame, detections in zip(normalized_frames, detections_by_frame):
                scaled_frame: list[dict[str, Any]] = []
                for item in detections:
                    scaled_item = {
                        **item,
                        "bbox": [
                            max(0, min(original_frame.shape[1], int(round(item["bbox"][0] * scale_x)))),
                            max(0, min(original_frame.shape[0], int(round(item["bbox"][1] * scale_y)))),
                            max(0, min(original_frame.shape[1], int(round(item["bbox"][2] * scale_x)))),
                            max(0, min(original_frame.shape[0], int(round(item["bbox"][3] * scale_y)))),
                        ],
                    }
                    if item.get("kps") is not None:
                        kps = item["kps"].copy()
                        kps[:, 0] *= scale_x
                        kps[:, 1] *= scale_y
                        scaled_item["kps"] = kps
                    scaled_frame.append(scaled_item)
                scaled_detections.append(scaled_frame)
            return scaled_detections

        return detections_by_frame

    def __call__(self, frames: list[Any]) -> list[np.ndarray]:
        normalized_frames = [self._normalize_frame(frame) for frame in frames]
        detections_by_frame = self.detect(normalized_frames)
        faces: list[np.ndarray] = []
        for frame, detections in zip(normalized_frames, detections_by_frame):
            for detection in detections:
                x1, y1, x2, y2 = detection["bbox"]
                faces.append(frame[y1:y2, x1:x2])
        return faces


def detect_faces_mtcnn(
    image: Any,
    device: Optional[str] = None,
    conf: float = 0.9,
    max_det: int = 10,
) -> list[dict[str, Any]]:
    """Detecta rostos com MTCNN e retorna bbox + 5 keypoints faciais.

    Aceita BGR np.ndarray ou PIL Image.
    Retorna lista no mesmo formato de detect_faces_with_model.
    """
    from PIL import Image as PILImage

    resolved_device = get_torch_device(device)
    if resolved_device == "mps":
        resolved_device = "cpu"
    mtcnn = _get_mtcnn_detector(device=resolved_device)

    if isinstance(image, np.ndarray):
        image_rgb = image[:, :, ::-1].copy() if image.ndim == 3 else image
        pil_image = PILImage.fromarray(image_rgb.astype(np.uint8))
    else:
        pil_image = image.convert("RGB")

    boxes, probs, points = mtcnn.detect(pil_image, landmarks=True)

    if boxes is None or len(boxes) == 0:
        return []

    detections: list[dict[str, Any]] = []
    for idx, (box, prob, pts) in enumerate(zip(boxes, probs, points)):
        if prob < conf:
            continue
        x1, y1, x2, y2 = (
            int(round(box[0])), int(round(box[1])),
            int(round(box[2])), int(round(box[3])),
        )
        detections.append({
            "index": idx,
            "bbox": [x1, y1, x2, y2],
            "score": float(prob),
            "area": max(0, x2 - x1) * max(0, y2 - y1),
            "kps": pts.astype(np.float32),
        })

    detections.sort(key=lambda d: d["score"], reverse=True)
    return detections[:max_det]


@lru_cache(maxsize=2)
def _get_mtcnn(
    device: str,
    image_size: int,
):
    try:
        from facenet_pytorch import MTCNN
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "facenet-pytorch nao esta disponivel para alinhamento facial. "
            "Instale com: pip install facenet-pytorch"
        ) from exc

    return MTCNN(
        image_size=image_size,
        margin=0,
        post_process=False,
        select_largest=True,
        keep_all=True,
        device=device,
    )


def detect_face_landmarks(
    image_bgr: np.ndarray,
    device: Optional[str] = None,
    image_size: int = 112,
) -> dict[str, Any]:
    resolved_device = get_torch_device(device)
    if resolved_device == "mps":
        resolved_device = "cpu"
    mtcnn = _get_mtcnn(device=resolved_device, image_size=image_size)
    image_rgb = image_bgr[:, :, ::-1]
    boxes, probs, points = mtcnn.detect(image_rgb, landmarks=True)
    if boxes is None or points is None or len(boxes) == 0:
        raise ValueError("Nao foi possivel detectar landmarks faciais com MTCNN.")

    best_index = max(
        range(len(boxes)),
        key=lambda idx: float((boxes[idx][2] - boxes[idx][0]) * (boxes[idx][3] - boxes[idx][1])),
    )

    return {
        "bbox": [float(value) for value in boxes[best_index].tolist()],
        "kps": np.asarray(points[best_index], dtype=np.float32),
        "score": float(probs[best_index]) if probs is not None else None,
    }


def align_face(
    image_bgr: np.ndarray,
    landmarks5: np.ndarray,
    image_size: int = 112,
) -> np.ndarray:
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("opencv-python nao esta instalado.") from exc

    matrix = estimate_arcface_transform(landmarks5, image_size=image_size)
    return cv2.warpAffine(image_bgr, matrix, (image_size, image_size), borderValue=0.0)


def align_face_crop(
    face_crop_bgr: np.ndarray,
    fallback_crop_bgr: Optional[np.ndarray] = None,
    device: Optional[str] = None,
    image_size: int = 112,
    allow_unaligned_fallback: bool = True,
) -> tuple[np.ndarray, dict[str, Any], str]:
    last_error: Optional[Exception] = None

    try:
        info = detect_face_landmarks(
            image_bgr=face_crop_bgr,
            device=device,
            image_size=image_size,
        )
        aligned = align_face(face_crop_bgr, info["kps"], image_size=image_size)
        return aligned, info, "segmented_crop"
    except (ImportError, ValueError) as exc:
        last_error = exc

    if fallback_crop_bgr is not None:
        try:
            info = detect_face_landmarks(
                image_bgr=fallback_crop_bgr,
                device=device,
                image_size=image_size,
            )
            aligned = align_face(fallback_crop_bgr, info["kps"], image_size=image_size)
            return aligned, info, "original_crop"
        except (ImportError, ValueError) as exc:
            last_error = exc

    if not allow_unaligned_fallback:
        raise last_error if last_error is not None else ValueError(
            "Nao foi possivel alinhar a face."
        )

    source_crop = fallback_crop_bgr if fallback_crop_bgr is not None else face_crop_bgr
    aligned = resize_face_crop(source_crop, image_size=image_size)
    height, width = source_crop.shape[:2]
    info = {
        "bbox": [0.0, 0.0, float(width), float(height)],
        "kps": np.empty((0, 2), dtype=np.float32),
        "score": None,
        "warning": str(last_error) if last_error is not None else None,
    }
    return aligned, info, "unaligned_resize_fallback"


__all__ = [
    "ArcFaceEmbedder",
    "FastMTCNN",
    "align_face",
    "align_face_crop",
    "detect_face_landmarks",
    "detect_faces_mtcnn",
    "get_torch_device",
    "resolve_model_precision",
    "resize_face_crop",
]
