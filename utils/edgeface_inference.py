from __future__ import annotations

import importlib.util
import sys
from functools import lru_cache
from pathlib import Path
from typing import Optional

import numpy as np

from utils.arcface_inference import (
    get_torch_device,
    resolve_model_precision,
    self_torch_no_grad,
)


# O EdgeFace foi incorporado ao repositório pai `sam3`, então os artefatos
# locais agora vivem dentro de `sam3/edge-face/edgeface`.
EDGEFACE_DIR = Path(__file__).resolve().parent.parent / "edge-face" / "edgeface"
DEFAULT_EDGEFACE_MODEL = "edgeface_xs_gamma_06"
DEFAULT_EDGEFACE_CHECKPOINT = EDGEFACE_DIR / "checkpoints" / "edgeface_xs_gamma_06.pt"


def _ensure_edgeface_imports(edgeface_dir: Optional[str] = None) -> None:
    path = str(edgeface_dir or EDGEFACE_DIR)
    if path not in sys.path:
        sys.path.insert(0, path)


@lru_cache(maxsize=4)
def _get_edgeface_get_model(edgeface_dir: str):
    package_dir = Path(edgeface_dir) / "backbones"
    spec = importlib.util.spec_from_file_location(
        f"sam3_edgeface_backbones_{abs(hash(edgeface_dir))}",
        package_dir / "__init__.py",
        submodule_search_locations=[str(package_dir)],
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Nao foi possivel carregar o pacote local do EdgeFace em {package_dir}.")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.get_model


class EdgeFaceEmbedder:
    """Embedder usando EdgeFace (edgenext-based), interface compativel com ArcFaceEmbedder."""

    def __init__(
        self,
        checkpoint_path: str,
        model_name: str = DEFAULT_EDGEFACE_MODEL,
        edgeface_dir: Optional[str] = None,
        device: Optional[str] = None,
        precision: str = "fp32",
        compile_model: bool = True,
    ) -> None:
        _ensure_edgeface_imports(edgeface_dir)

        try:
            import cv2
            import torch
            import torch.nn.functional as functional
        except ImportError as exc:
            raise ImportError(
                "Dependencias do EdgeFace nao estao disponiveis. "
                "Verifique se timm esta instalado e o caminho do edgeface esta correto."
            ) from exc

        edgeface_get_model = _get_edgeface_get_model(str(edgeface_dir or EDGEFACE_DIR))

        self.cv2 = cv2
        self.torch = torch
        self.functional = functional
        self.device = get_torch_device(device)
        self.precision = resolve_model_precision(precision, self.device)
        self.use_fp16 = self.precision == "fp16"

        self.model = edgeface_get_model(model_name).to(self.device)
        state_dict = torch.load(checkpoint_path, map_location=self.device)
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
        """Processa multiplas faces em um unico forward pass do EdgeFace."""
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
    "DEFAULT_EDGEFACE_CHECKPOINT",
    "DEFAULT_EDGEFACE_MODEL",
    "EDGEFACE_DIR",
    "EdgeFaceEmbedder",
]
