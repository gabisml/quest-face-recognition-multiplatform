from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from utils.arcface_inference import (
    get_torch_device,
    resolve_model_precision,
    self_torch_no_grad,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_FACELIVT_MODEL = "facelivtv2_xs"
DEFAULT_FACELIVT_CHECKPOINT = PROJECT_ROOT / "models" / "facelivt2" / "facelivtv2-xs.pt"


class FaceLivTEmbedder:
    """Embedder usando FaceLiVT v2 (lightweight ViT), interface compativel com ArcFaceEmbedder."""

    def __init__(
        self,
        checkpoint_path: str,
        model_name: str = DEFAULT_FACELIVT_MODEL,
        device: Optional[str] = None,
        precision: str = "fp32",
        compile_model: bool = True,
        num_classes: int = 512,
    ) -> None:
        try:
            import cv2
            import torch
            import torch.nn.functional as functional
        except ImportError as exc:
            raise ImportError(
                "Dependencias do FaceLiVT nao estao disponiveis. "
                "Verifique se torch, timm e opencv estao instalados."
            ) from exc

        from utils.facelivt_arch import build_facelivt

        self.cv2 = cv2
        self.torch = torch
        self.functional = functional
        self.device = get_torch_device(device)
        self.precision = resolve_model_precision(precision, self.device)
        self.use_fp16 = self.precision == "fp16"

        self.model = build_facelivt(model_name, num_classes=num_classes).to(self.device)
        state_dict = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(state_dict)
        self.model.eval()

        if compile_model and self.device.startswith("cuda"):
            self.model = torch.compile(self.model, mode="reduce-overhead")

    def _run_model(self, tensor):
        if self.use_fp16:
            with self.torch.autocast(device_type="cuda", dtype=self.torch.float16):
                return self.model(tensor)
        return self.model(tensor)

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
        """Processa multiplas faces em um unico forward pass do FaceLiVT."""
        if not aligned_faces_bgr:
            return []

        arrays = np.stack([self._preprocess_face(f) for f in aligned_faces_bgr], axis=0)
        tensor = self.torch.from_numpy(arrays).float().to(self.device)
        tensor.div_(255).sub_(0.5).div_(0.5)

        embeddings = self._run_model(tensor)
        if l2_normalize:
            embeddings = self.functional.normalize(embeddings, p=2, dim=1)

        result = embeddings.detach().cpu().numpy()
        return [result[i] for i in range(result.shape[0])]


__all__ = [
    "DEFAULT_FACELIVT_CHECKPOINT",
    "DEFAULT_FACELIVT_MODEL",
    "FaceLivTEmbedder",
]
