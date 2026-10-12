"""Extrator de descritores por TensorFlow Lite, com delegacao opcional para Edge TPU.

POR QUE EXISTE

    A Coral Dev Board nao tem torch para aarch64 com glibc 2.28, e o acelerador so
    executa inteiros de 8 bits. O unico caminho de inferencia ali e TFLite int8 com o
    delegate da Edge TPU, entao o pipeline precisa de um extrator que fale essa lingua
    com a mesma interface dos demais.

PARIDADE DE PRE-PROCESSAMENTO

    A cadeia e a mesma do ArcFaceEmbedder ate a normalizacao para [-1,1]. A diferenca e
    o passo final: o modelo quantizado espera inteiros, entao o tensor e requantizado com
    o scale e o zero_point declarados pelo proprio modelo,

        x_q = clip(round(x / scale + zero_point), min, max)

    e a saida percorre o caminho inverso, (y - zp) * scale, antes da normalizacao L2.
    Usar os parametros do modelo, em vez de constantes fixas, e o que mantem o descritor
    comparavel com a referencia fp32.

LAYOUT E LOTE

    TFLite trabalha em NHWC. Modelos compilados para Edge TPU costumam ter lote fixo em
    1, entao o batch e percorrido em laco; quando o modelo aceita lote maior, um unico
    invoke e usado.

CUIDADO COM A VERSAO DO RUNTIME

    Nesta imagem da placa o libedgetpu e 16.0 e o tflite-runtime tem de ser 2.7.0; a
    2.13.0 segfalta com esse delegate. Ver bench/coral/EDGETPU_OPS.md.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

EDGETPU_LIB = "libedgetpu.so.1"


def _l2(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return np.divide(x, n, out=np.zeros_like(x), where=n != 0)


def _load_interpreter(model_path: str, use_tpu: bool, threads: int):
    """tflite_runtime na placa; no desktop, LiteRT e depois tensorflow.lite.

    A ordem importa: na Coral o tflite_runtime 2.7.0 e obrigatorio (o delegate da Edge
    TPU segfalta com runtimes mais novos). No desktop, o interpretador LiteRT aplica
    XNNPACK tambem a modelos quantizados e e uma ordem de grandeza mais rapido que o
    tf.lite embutido no TensorFlow para int8 (medido: 305 ms contra 3.560 ms por imagem
    no ArcFace R100 int8).
    """
    try:
        import tflite_runtime.interpreter as tfl
    except ImportError:  # pragma: no cover - caminho de desktop
        try:
            import ai_edge_litert.interpreter as tfl  # type: ignore
        except ImportError:
            try:
                import tensorflow.lite as tfl  # type: ignore
            except ImportError as exc:
                raise ImportError(
                    "Nem tflite_runtime, nem ai_edge_litert, nem tensorflow.lite "
                    "estao disponiveis. Na placa: tflite-runtime==2.7.0."
                ) from exc

    kwargs = {"model_path": model_path}
    if use_tpu:
        kwargs["experimental_delegates"] = [tfl.load_delegate(EDGETPU_LIB)]
    elif threads:
        kwargs["num_threads"] = int(threads)
    return tfl.Interpreter(**kwargs)


class TFLiteEmbedder:
    """Extrator TFLite int8. Interface compativel com ArcFaceEmbedder."""

    def __init__(
        self,
        model_path: str,
        device: Optional[str] = None,
        use_tpu: Optional[bool] = None,
        threads: int = 4,
        precision: str = "int8",
    ) -> None:
        try:
            import cv2
        except ImportError as exc:  # pragma: no cover - dependencia opcional
            raise ImportError("opencv-python nao esta instalado.") from exc

        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"Modelo TFLite nao encontrado em {path}.")

        # use_tpu e tri-estado de proposito. Ate 26/08/2026 um device "cpu" desligava
        # o delegate aqui dentro, mesmo quando quem chamou pediu a TPU -- e no
        # benchmark da Coral o device do DETECTOR e cpu (o YuNet roda no CPU), entao
        # a rodada rotulada int8_tpu media o extrator no CPU, com numero plausivel e
        # sem aviso. A contagem de faces nao muda nesse caso, entao a verificacao do
        # run_bench_e2e.sh tambem nao pegava.
        #
        #   None  -> decide pelo device (comportamento antigo, para chamador de app)
        #   True  -> exige o delegate; se ele nao carregar, levanta, nao cai no CPU
        #   False -> CPU, sem delegate
        if use_tpu is None:
            use_tpu = not (device or "").startswith("cpu")

        self.cv2 = cv2
        self.model_path = str(path)
        self.precision = precision
        self.use_tpu = use_tpu

        self.interpreter = _load_interpreter(self.model_path, use_tpu, threads)
        self.interpreter.allocate_tensors()

        self.inp = self.interpreter.get_input_details()[0]
        self.out = self.interpreter.get_output_details()[0]
        self.in_scale, self.in_zp = self.inp["quantization"]
        self.out_scale, self.out_zp = self.out["quantization"]
        self.in_dtype = self.inp["dtype"]
        self.batch = int(self.inp["shape"][0])
        self.size = (int(self.inp["shape"][2]), int(self.inp["shape"][1]))  # (w, h)

        self._assert_nao_degenerado()

    # ------------------------------------------------------------------ interno
    def _assert_nao_degenerado(self) -> None:
        """Uma inferencia de sonda: saida constante e extrator quebrado, nao lento.

        O tflite-runtime 2.7.0 da imagem da Coral carrega os int8 NAO compilados
        para a Edge TPU sem reclamar e devolve o zero point nas 512 posicoes --
        embedding nulo depois do L2, reconhecimento sem sentido, e a latencia
        continua com cara de normal (medido em 26/08/2026: mbf_w600k int8 no CPU da
        placa, 240 ms por face, saida constante 115). Os mesmos arquivos rodam
        certo no LiteRT do desktop, entao e o runtime da placa, nao o modelo.

        Sem esta sonda o benchmark grava precision/recall de embeddings zerados, e
        nenhuma checagem a jusante pega: a contagem de faces vem do detector e nao
        muda. Para latencia pura de um .tflite que falha aqui existe o
        bench/coral/bench_tflite_latency.py, que nao afirma nada sobre acuracia.
        """
        rng = np.random.RandomState(0)
        forma = tuple(int(v) for v in self.inp["shape"])
        if np.issubdtype(self.in_dtype, np.integer):
            info = np.iinfo(self.in_dtype)
            x = rng.randint(info.min, info.max + 1, size=forma).astype(self.in_dtype)
        else:  # modelo nao quantizado na entrada
            x = rng.uniform(-1.0, 1.0, size=forma).astype(self.in_dtype)
        self.interpreter.set_tensor(self.inp["index"], x)
        self.interpreter.invoke()
        y = self.interpreter.get_tensor(self.out["index"])
        if int(np.ptp(y)) != 0:
            return
        destino = "Edge TPU" if self.use_tpu else "CPU"
        raise RuntimeError(
            f"{Path(self.model_path).name} no {destino}: a saida e constante "
            f"({y.flat[0]}) para entrada aleatoria, ou seja, embedding nulo. "
            "O modelo carrega e roda, mas nao extrai nada -- na Coral isso e o "
            "tflite-runtime 2.7.0 com os int8 nao compilados para a Edge TPU. "
            "Use o par _edgetpu.tflite com o delegate, ou meca so latencia com "
            "bench/coral/bench_tflite_latency.py."
        )
    def _preprocess(self, aligned_face_bgr: np.ndarray) -> np.ndarray:
        face = aligned_face_bgr
        if (face.shape[1], face.shape[0]) != self.size:
            face = self.cv2.resize(face, self.size)
        face = self.cv2.cvtColor(face, self.cv2.COLOR_BGR2RGB).astype(np.float32)
        face = (face / 255.0 - 0.5) / 0.5           # [-1, 1], igual aos demais backends
        if self.in_scale:
            info = np.iinfo(self.in_dtype)
            face = np.clip(
                np.round(face / self.in_scale + self.in_zp), info.min, info.max
            )
        return face.astype(self.in_dtype)

    def _run_one(self, x: np.ndarray) -> np.ndarray:
        self.interpreter.set_tensor(self.inp["index"], x[None])
        self.interpreter.invoke()
        y = self.interpreter.get_tensor(self.out["index"])[0].astype(np.float32)
        if self.out_scale:
            y = (y - self.out_zp) * self.out_scale
        return y

    # ------------------------------------------------------------------ publico
    def embed_aligned_face(
        self,
        aligned_face_bgr: np.ndarray,
        l2_normalize: bool = True,
    ) -> np.ndarray:
        return self.embed_batch([aligned_face_bgr], l2_normalize=l2_normalize)[0]

    def embed_batch(
        self,
        aligned_faces_bgr: list[np.ndarray],
        l2_normalize: bool = True,
    ) -> list[np.ndarray]:
        if not aligned_faces_bgr:
            return []

        saidas = [self._run_one(self._preprocess(f)) for f in aligned_faces_bgr]
        saida = np.stack(saidas, axis=0)
        if l2_normalize:
            saida = _l2(saida)
        return [saida[i] for i in range(saida.shape[0])]

    def __repr__(self) -> str:  # pragma: no cover - conveniencia
        destino = "EdgeTPU" if self.use_tpu else "CPU"
        return f"TFLiteEmbedder({Path(self.model_path).name}, {destino})"
