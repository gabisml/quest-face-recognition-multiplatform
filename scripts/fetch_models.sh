#!/bin/sh
# Baixa do HuggingFace os modelos grandes do alvo desta branch, com SHA-256.
#
# O HW vem do config/hw.env da branch, da env, ou do argumento:
#     bash scripts/fetch_models.sh          # numa branch hw/<id>
#     HW=rpi5 bash scripts/fetch_models.sh  # forcando o alvo
#
# POSIX sh de proposito: a Coral (Python 3.9, Mendel 5) nao tem huggingface_hub
# e nem precisa -- curl/wget + sha256sum existem em todo alvo.
set -eu
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"
[ -f config/hw.env ] && . ./config/hw.env
HW="${1:-${HW:?defina HW ou rode numa branch com config/hw.env}}"
[ -f models/MANIFEST.tsv ] || { echo "ERRO: falta models/MANIFEST.tsv"; exit 1; }

baixar() {
    # curl e wget cobrem todos os alvos; -C -/--continue retoma download parcial.
    # Prefixo _ de proposito: POSIX sh nao tem variavel local, e "dest=$2" aqui
    # sobrescreveria o $dest do loop de fora (bug real: a checagem de SHA passava
    # a olhar para $dest.part.part, que nao existe, e todo download "divergia").
    _url=$1; _destino=$2
    if command -v curl >/dev/null 2>&1; then
        curl -fL --retry 5 --retry-delay 5 -C - -o "$_destino" "$_url"
    else
        wget -c -O "$_destino" "$_url"
    fi
}

TAB="$(printf '\t')"
FALHA=0
grep -v '^#' models/MANIFEST.tsv | while IFS="$TAB" read -r dest bytes sha hws url; do
    [ -n "$dest" ] || continue
    case ",$hws," in *",$HW,"*) ;; *) continue ;; esac
    if [ -f "$dest" ] && echo "$sha  $dest" | sha256sum -c --status 2>/dev/null; then
        echo "ok      $dest"
        continue
    fi
    echo "baixando $dest ($bytes bytes)"
    mkdir -p "$(dirname "$dest")"
    baixar "$url" "$dest.part"
    if echo "$sha  $dest.part" | sha256sum -c --status 2>/dev/null; then
        mv "$dest.part" "$dest"
        echo "ok      $dest"
    else
        echo "ERRO: SHA-256 divergente em $dest -- arquivo mantido em $dest.part"
        exit 1
    fi
done
echo "modelos completos para HW=$HW"
