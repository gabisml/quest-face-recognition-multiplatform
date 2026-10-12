from __future__ import annotations

import zipfile
from pathlib import Path
from typing import Any, Optional

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_EMOTION_MODEL = PROJECT_ROOT / "models" / "emotion" / "rafdb_yolo26n_best.pt"
_EMOTION_MODEL_ZIP = PROJECT_ROOT / "models" / "emotion" / "rafdb_yolo26n_best.pt.zip"


def _resolve_model_path(model_path: str | Path) -> Path:
    """Resolve a loadable .pt path, materializing it from the .zip if needed.

    Ultralytics loads a PyTorch .pt (which is itself a zip). If only the
    *.pt.zip is present, copy it to a sibling *.pt so YOLO can open it.
    """
    path = Path(model_path)
    if path.is_file():
        return path
    zip_candidate = path.with_suffix(path.suffix + ".zip")
    if not path.exists() and zip_candidate.is_file() and zipfile.is_zipfile(zip_candidate):
        path.write_bytes(zip_candidate.read_bytes())
        return path
    if not path.exists() and _EMOTION_MODEL_ZIP.is_file():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(_EMOTION_MODEL_ZIP.read_bytes())
        return path
    raise FileNotFoundError(f"Modelo de emocao nao encontrado: {model_path}")


class EmotionClassifier:
    """YOLO classification wrapper for facial emotion (RAF-DB, 7 classes)."""

    def __init__(
        self,
        model_path: str | Path = DEFAULT_EMOTION_MODEL,
        device: Optional[str] = None,
        imgsz: int = 224,
        half: bool = False,
    ) -> None:
        from ultralytics import YOLO

        resolved = _resolve_model_path(model_path)
        self.model = YOLO(str(resolved))
        if self.model.task != "classify":
            raise ValueError(
                f"Modelo de emocao deve ser de classificacao, mas task={self.model.task}."
            )
        self.device = device
        self.imgsz = imgsz
        self.half = half
        self.names: dict[int, str] = dict(self.model.names)

    def classify(self, face_bgr: np.ndarray) -> dict[str, Any]:
        """Classify a single face crop (BGR). Returns {emotion, score, probs}."""
        results = self.classify_batch([face_bgr])
        return results[0]

    def classify_batch(self, faces_bgr: list[np.ndarray]) -> list[dict[str, Any]]:
        valid = [(i, f) for i, f in enumerate(faces_bgr) if f is not None and f.size > 0]
        outputs: list[dict[str, Any]] = [
            {"emotion": None, "score": None, "probs": {}} for _ in faces_bgr
        ]
        if not valid:
            return outputs

        crops = [f for _, f in valid]
        results = self.model.predict(
            crops,
            imgsz=self.imgsz,
            device=self.device,
            half=self.half,
            verbose=False,
        )
        for (index, _), result in zip(valid, results):
            probs = result.probs
            data = probs.data.tolist()
            outputs[index] = {
                "emotion": self.names.get(int(probs.top1), str(probs.top1)),
                "score": float(probs.top1conf),
                "probs": {self.names.get(i, str(i)): float(p) for i, p in enumerate(data)},
            }
        return outputs


__all__ = ["EmotionClassifier", "DEFAULT_EMOTION_MODEL"]
