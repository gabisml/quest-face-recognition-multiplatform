#!/usr/bin/env bash

set -euo pipefail

SNAPSHOT="${1:-}"
QDRANT_URL="${QDRANT_URL:-http://localhost:6333}"
COLLECTION="${QDRANT_COLLECTION:-Rostos}"

if [ -z "$SNAPSHOT" ]; then
    echo "Uso:"
    echo "  $0 /caminho/arquivo.snapshot"
    exit 1
fi

if [ ! -f "$SNAPSHOT" ]; then
    echo "ERRO: snapshot não encontrado: $SNAPSHOT"
    exit 1
fi

echo "=== Restauracao Qdrant ==="
echo "Colecao: $COLLECTION"
echo "Snapshot: $SNAPSHOT"
echo "Qdrant: $QDRANT_URL"

if ! curl -sf "$QDRANT_URL" >/dev/null; then
    echo "ERRO: Qdrant não está acessível em $QDRANT_URL"
    exit 1
fi

if curl -sf "$QDRANT_URL/collections/$COLLECTION" >/dev/null; then
    if [ "${FORCE:-0}" != "1" ]; then
        echo
        echo "ERRO: a coleção '$COLLECTION' já existe."
        echo "Para substituí-la pelo snapshot, execute:"
        echo "FORCE=1 $0 \"$SNAPSHOT\""
        exit 1
    fi
fi

CHECKSUM="$(sha256sum "$SNAPSHOT" | awk '{print $1}')"

echo "SHA-256: $CHECKSUM"
echo
echo "Restaurando snapshot..."

curl -f \
    -X POST \
    "$QDRANT_URL/collections/$COLLECTION/snapshots/upload?wait=true&priority=snapshot&checksum=$CHECKSUM" \
    -F "snapshot=@$SNAPSHOT"

echo
echo
echo "=== Verificando coleção ==="

curl -fs "$QDRANT_URL/collections/$COLLECTION"

echo
echo
echo "Restauracao concluida."
