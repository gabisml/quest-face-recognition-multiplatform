#!/usr/bin/env bash

set -e

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

echo "=========================================="
echo " Meta Quest 3 - Facial Recognition"
echo "=========================================="

echo
echo "[1/4] Plataforma"
echo "Hostname: $(hostname)"
echo "Arquitetura: $(uname -m)"

if command -v nvidia-smi >/dev/null 2>&1; then
    echo
    echo "[GPU]"
    nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null || nvidia-smi || true
fi

echo
echo "[2/4] Verificando ambiente Python"

if [ ! -d ".venv" ]; then
    echo "ERRO: .venv não encontrada."
    echo "Execute primeiro o script de configuração da máquina."
    exit 1
fi

source .venv/bin/activate

echo "Python: $(python --version 2>&1)"

echo
echo "[3/4] Verificando Qdrant"

if ! curl -sf http://localhost:6333 >/dev/null; then
    echo "Qdrant não está ativo."
    echo "Iniciando Qdrant..."

    docker compose -f docker-compose.quest.yml up -d

    echo "Aguardando Qdrant..."

    for i in {1..30}; do
        if curl -sf http://localhost:6333 >/dev/null; then
            break
        fi

        sleep 1
    done
fi

if ! curl -sf http://localhost:6333 >/dev/null; then
    echo "ERRO: Qdrant não respondeu na porta 6333."
    exit 1
fi

echo "Qdrant OK."

echo
echo "[4/4] Verificando modelos"

if [ ! -f "models/yunet/face_detection_yunet_2023mar.onnx" ]; then
    echo "ERRO: modelo YuNet não encontrado."
    exit 1
fi

if [ ! -f "models/onnx/arcface_r100.onnx" ]; then
    echo "ERRO: ArcFace R100 não encontrado."
    echo "Execute scripts/fetch_models.sh para o hardware correspondente."
    exit 1
fi

echo "Modelos OK."

echo
echo "=========================================="
echo " Iniciando API na porta 8000"
echo "=========================================="
echo

exec python -m uvicorn quest_recognition_api:app \
    --host 0.0.0.0 \
    --port 8000 \
    --workers 1
