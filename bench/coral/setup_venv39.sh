#!/bin/bash
# Recria o venv Python 3.9 com acesso a Edge TPU na Coral Dev Board (Mendel 5 "Eagle").
#
# POR QUE NAO SE USA O pycoral
#   O caminho documentado pela Google instala pycoral, mas ele nao serve aqui e
#   nao e necessario:
#
#   - Nao serve: o wheel cp39 traz _pywrap_coral.so compilado contra GLIBC_2.29
#     (Debian 11) e esta imagem e Debian 10 com glibc 2.28 -> ImportError em
#     qualquer import de pycoral.utils.edgetpu. So o wheel cp37 casa, e ele so
#     serve pro Python 3.7 do sistema (que ja tem pycoral como dist-package).
#
#   - Nao e necessario: a unica peca util era make_interpreter, que e apenas
#     tflite.load_delegate("libedgetpu.so.1") passado a tflite.Interpreter. Os
#     seis scripts de bench desta pasta ja fazem isso direto. Os adapters
#     (pycoral.adapters.common/.detect) tratam caixas e rotulos de detectores;
#     aqui os modelos sao extratores de embedding que devolvem (1, 512), sem
#     pos-processamento que os justifique.
#
#   Ate 26/08/2026 havia um edgetpu_shim.py copiado por cima de
#   site-packages/pycoral/utils/edgetpu.py para contornar o primeiro problema.
#   Foi removido: enxerto em site-packages some em silencio na primeira
#   reinstalacao do pacote, e nada no repo importava o shim.
#
# VERSAO DO TFLITE IMPORTA
#   O libedgetpu 16.0 desta imagem (julho/2021) SEGFAULTA com tflite-runtime
#   2.13.0. 2.7.0 funciona. Nao "atualize".
#
# NUMPY FICA EM <2
#   O tflite-runtime 2.7.0 e o opencv desta placa foram compilados contra a ABI
#   do numpy 1.x.
#
# NADA DE pyarrow
#   O CPU e Cortex-A53, ARMv8.0-A: sem `atomics` (LSE), que e ARMv8.1. Os wheels
#   aarch64 do pyarrow assumem ARMv8.2 e matam o processo com SIGILL -- inclusive
#   indiretamente, porque pandas importa pyarrow em pandas/compat/pyarrow.py.
#   O motor de parquet aqui e o fastparquet.
#
# Uso:  bash setup_venv39.sh [DIR_DO_VENV]     (default: ./.venv no diretorio atual)
#
# Rode na placa. O venv NAO deve ficar dentro do repo clonado na /home interna
# (sobra ~1,4 GB); use o cartao montado em /home/mendel/data.
set -e
export PATH="$HOME/.local/bin:$PATH"
VENV="${1:-$PWD/.venv}"

uv python install 3.9
rm -rf "$VENV"
uv venv --python 3.9 "$VENV"

# tflite-runtime 2.7.0 do PyPI e manylinux2014 (GLIBC_2.17), compativel com a
# glibc 2.28 desta imagem. O delegate vem do libedgetpu.so.1 do sistema.
# psutil/pandas/fastparquet sao o que bench.probes e bench.system_monitor pedem
# para a placa gerar system_metrics.parquet como os demais alvos.
# imageio/imageio-ffmpeg sao do bench.e2e_video: ele abre video so por imageio
# (o opencv-python-headless desta placa nao tem FFmpeg), e sem isso a suite
# principal nao roda aqui -- so os scripts desta pasta.
VIRTUAL_ENV="$VENV" uv pip install \
  "numpy<2" \
  pillow \
  "tflite-runtime==2.7.0" \
  opencv-python-headless \
  psutil \
  pandas \
  fastparquet \
  imageio \
  imageio-ffmpeg

echo
echo ">>> Validando:"
"$VENV/bin/python" - <<'PY'
import numpy, tflite_runtime, psutil, pandas, imageio
import tflite_runtime.interpreter as tflite
print("  numpy         ", numpy.__version__)
print("  tflite_runtime", tflite_runtime.__version__)
print("  psutil        ", psutil.__version__)
print("  pandas        ", pandas.__version__)
print("  imageio       ", imageio.__version__)
try:
    tflite.load_delegate("libedgetpu.so.1")
    print("  delegate Edge TPU: OK")
except Exception as e:
    raise SystemExit("  delegate Edge TPU FALHOU: %s" % e)
try:
    import pyarrow  # noqa: F401
    print("  AVISO: pyarrow instalado -- ele da SIGILL neste CPU; remova-o.")
except ImportError:
    print("  pyarrow ausente: correto (parquet via fastparquet)")
import pandas as pd
pd.DataFrame({"x": [1.0, 2.0]}).to_parquet("/tmp/_setup_venv39_check.parquet")
print("  to_parquet    : OK (engine fastparquet)")
PY
echo ">>> Pronto. Ative com: source $VENV/bin/activate"
