from __future__ import annotations

from pathlib import Path
from threading import Lock
from typing import Any

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile

from utils.face_pipeline import FaceRecognizer


app = FastAPI(title="Quest Face Recognition API")

# Um único reconhecedor persistente.
# Os modelos NÃO serão recarregados a cada frame.
recognizer = FaceRecognizer(
    qdrant_url="http://localhost:6333",
    collection_name="Rostos",

    # Configuração que já validamos na Jetson
    device="cuda",

    detector_backend="yunet",
    yunet_backend="cuda",
    conf=0.5,

    embedder_backend="arcface",
    arcface_network="r100",
    arcface_compile=False,

    score_threshold=0.45,
    candidate_limit=5,
    score_margin=0.10,
    min_identity_votes=1,

    # Neste primeiro teste, queremos processar cada frame completamente
    detect_interval=1,
    temporal_cache=False,
)

# Evita duas requisições usando o mesmo pipeline/GPU simultaneamente.
recognizer_lock = Lock()


def to_json_safe(value: Any) -> Any:
    """Converte valores NumPy para tipos serializáveis em JSON."""

    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, np.generic):
        return value.item()

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, dict):
        return {
            str(key): to_json_safe(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [to_json_safe(item) for item in value]

    return value


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "device": recognizer.device,
        "detector": recognizer.detector_backend,
        "yunet_backend": recognizer.yunet_backend,
    }


@app.post("/frame")
def recognize_quest_frame(
    file: UploadFile = File(...),
) -> dict[str, Any]:

    image_bytes = file.file.read()

    if not image_bytes:
        raise HTTPException(
            status_code=400,
            detail="Arquivo de imagem vazio.",
        )

    encoded = np.frombuffer(image_bytes, dtype=np.uint8)

    frame_bgr = cv2.imdecode(
        encoded,
        cv2.IMREAD_COLOR,
    )

    if frame_bgr is None:
        raise HTTPException(
            status_code=400,
            detail="Nao foi possivel decodificar a imagem.",
        )

    with recognizer_lock:
        recognitions = recognizer.recognize_frame(frame_bgr)
        timing = dict(recognizer.last_timing)
    total_ms = float(timing.get("total_ms", 0.0))
    pipeline_fps = 1000.0 / total_ms if total_ms > 0 else 0.0

    print(
    	"[TIMING] "
    	f"detection={timing.get('detection_ms', 0.0):.2f} ms | "
   	f"alignment={timing.get('alignment_ms', 0.0):.2f} ms | "
    	f"embedding={timing.get('embedding_ms', 0.0):.2f} ms | "
    	f"search={timing.get('search_ms', 0.0):.2f} ms | "
    	f"total={total_ms:.2f} ms | "
    	f"pipeline_fps={pipeline_fps:.2f}",
    	flush=True,
	)
    return {
        "ok": True,
        "device": recognizer.device,
        "frame": {
            "width": int(frame_bgr.shape[1]),
            "height": int(frame_bgr.shape[0]),
        },
        "faces_detected": len(recognitions),
        "recognitions": to_json_safe(recognitions),
        "timing_ms": to_json_safe(timing),
    }


print(
    "[Quest API] FaceRecognizer carregado "
    f"device={recognizer.device}, "
    f"detector={recognizer.detector_backend}, "
    f"yunet_backend={recognizer.yunet_backend}"
)
