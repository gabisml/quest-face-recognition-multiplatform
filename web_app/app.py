from __future__ import annotations
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import argparse
import re
import unicodedata
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from cli.upload_face import upload_face
from utils.edgeface_inference import (
    DEFAULT_EDGEFACE_CHECKPOINT,
    DEFAULT_EDGEFACE_MODEL,
)
from utils.face_pipeline import (
    DEFAULT_ARCFACE_WEIGHT,
    DEFAULT_YUNET_MODEL,
)

STATIC_DIR = Path(__file__).resolve().parent / "static"
app = FastAPI(
    title="Face Registration API",
    version="0.1.0",
)
app.mount(
    "/static",
    StaticFiles(directory=str(STATIC_DIR)),
    name="static",
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PEOPLE_DIR = PROJECT_ROOT / "data" / "people"

ALLOWED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}


def make_entity_id(name: str) -> str:
    """
    Converte o nome para um identificador simples.

    Exemplo:
        Gabrielle Siqueira -> gabrielle_siqueira
    """

    normalized = unicodedata.normalize("NFKD", name)

    ascii_name = (
        normalized
        .encode("ascii", "ignore")
        .decode("ascii")
    )

    entity_id = re.sub(
        r"[^a-zA-Z0-9_-]+",
        "_",
        ascii_name,
    )

    entity_id = entity_id.strip("_").lower()

    if not entity_id:
        raise ValueError(
            "Nao foi possivel gerar um identificador valido."
        )

    return entity_id


def build_upload_args(
    image_path: Path,
    name: str,
    entity_id: str,
) -> argparse.Namespace:
    """
    Monta os mesmos argumentos usados pelo cli/upload_face.py.
    """

    return argparse.Namespace(
        # Imagem
        image_path=str(image_path),

        # Identidade
        point_id=None,
        entity_id=entity_id,
        name=name,

        # Qdrant
        collection="Rostos",
        qdrant_url="http://localhost:6333",
        qdrant_api_key=None,
        qdrant_timeout=15.0,

        # Embedder
        arcface_weight=str(DEFAULT_ARCFACE_WEIGHT),
        arcface_network="r100",
        embedder="arcface",

        onnx_model=None,
        tflite_model=None,
        tflite_cpu=False,

        edgeface_model=DEFAULT_EDGEFACE_MODEL,
        edgeface_checkpoint=str(
            DEFAULT_EDGEFACE_CHECKPOINT
        ),

        no_edgeface_compile=True,
        no_arcface_compile=True,

        # Detector
        detector="yunet",
        yunet_model=str(DEFAULT_YUNET_MODEL),
        yunet_nms=0.3,
        yunet_top_k=5000,
        yunet_backend="cuda",

        # Execucao
        output_dir=str(
            PROJECT_ROOT / "output" / "uploads"
        ),

        conf=0.5,
        device="cuda",

        require_alignment=False,
        fail_fast=False,

        # Argumentos do CLI que nao usamos aqui
        people_dir=None,
        entity_dir=None,
    )

@app.get("/")
def index():
    return FileResponse(
        STATIC_DIR / "index.html"
    )
@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "face-registration",
        "collection": "Rostos",
    }


@app.post("/persons")
async def create_person(
    name: str = Form(...),
    files: list[UploadFile] = File(...),
):
    name = name.strip()

    if not name:
        raise HTTPException(
            status_code=400,
            detail="O nome da pessoa e obrigatorio.",
        )

    if not files:
        raise HTTPException(
            status_code=400,
            detail="Envie pelo menos uma foto.",
        )

    try:
        entity_id = make_entity_id(name)
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    person_dir = PEOPLE_DIR / entity_id
    person_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    uploaded = []
    errors = []

    for uploaded_file in files:
        original_name = uploaded_file.filename or "image.jpg"

        suffix = Path(original_name).suffix.lower()

        if suffix not in ALLOWED_EXTENSIONS:
            errors.append(
                {
                    "file": original_name,
                    "error": "Formato de imagem nao suportado.",
                }
            )
            continue

        unique_filename = (
            f"{uuid.uuid4().hex}{suffix}"
        )

        image_path = (
            person_dir / unique_filename
        )

        image_bytes = await uploaded_file.read()

        if not image_bytes:
            errors.append(
                {
                    "file": original_name,
                    "error": "Arquivo vazio.",
                }
            )
            continue

        image_path.write_bytes(image_bytes)

        args = build_upload_args(
            image_path=image_path,
            name=name,
            entity_id=entity_id,
        )

        try:
            result = upload_face(args)

            uploaded.append(
                {
                    "original_file": original_name,
                    "stored_file": str(image_path),
                    "result": result,
                }
            )

        except Exception as exc:
            errors.append(
                {
                    "file": original_name,
                    "error": str(exc),
                }
            )

            # Se nao foi possivel cadastrar,
            # removemos apenas a foto temporariamente salva.
            image_path.unlink(
                missing_ok=True
            )

    if not uploaded:
        raise HTTPException(
            status_code=400,
            detail={
                "message": (
                    "Nenhuma foto foi cadastrada."
                ),
                "errors": errors,
            },
        )

    return {
        "ok": True,
        "name": name,
        "entity_id": entity_id,
        "collection": "Rostos",
        "images_received": len(files),
        "images_registered": len(uploaded),
        "errors_count": len(errors),
        "uploaded": uploaded,
        "errors": errors,
    }
