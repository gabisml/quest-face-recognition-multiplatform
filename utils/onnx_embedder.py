"""Extrator de descritores por ONNX Runtime, com a mesma interface do ArcFaceEmbedder.

POR QUE EXISTE

    Tres necessidades convergem no mesmo lugar:

    1. O MobileFaceNet w600k do InsightFace e distribuido como ONNX. Nao existe
       checkpoint torch dele, entao nao ha como carrega-lo pelo ArcFaceEmbedder.
    2. O Raspberry Pi 5 nao comporta torch: a wheel aarch64 do PyPI declara as
       dependencias nvidia-cu13 sem filtro de arquitetura e puxa 1,5 GB de pilha CUDA
       para uma placa sem GPU; e mesmo pelo indice CPU-only sobram ~173 MB de disco
       depois dos pesos, abaixo da margem segura.
    3. Nos alvos NVIDIA o TensorRT constroi engine a partir de ONNX, entao o ONNX
       precisa existir de qualquer forma.

    Um unico backend resolve os tres.

PARIDADE COM O ArcFaceEmbedder

    O pre-processamento e identico, e isso nao e detalhe: descritor gerado com outra
    normalizacao nao e comparavel com a galeria nem com a referencia fp32. A cadeia e
    redimensionar para 112x112 se preciso, converter BGR para RGB, transpor para NCHW e
    normalizar para [-1,1] por (x/255 - 0.5)/0.5. A saida e normalizada em L2 pelo mesmo
    motivo: assim o produto interno e o cosseno.

    O layout de entrada e detectado do proprio modelo. Modelos exportados por onnx2tf
    saem em NHWC; os do InsightFace sao NCHW.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import numpy as np

# Ordem de preferencia. O primeiro disponivel no ambiente e usado, salvo escolha
# explicita do chamador.
PROVIDER_PREFERENCE = (
    "TensorrtExecutionProvider",
    "CUDAExecutionProvider",
    "CPUExecutionProvider",
)


def _l2(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return np.divide(x, n, out=np.zeros_like(x), where=n != 0)


class OnnxEmbedder:
    """Extrator ONNX. Interface compativel com ArcFaceEmbedder e EdgeFaceEmbedder."""

    def __init__(
        self,
        model_path: str,
        device: Optional[str] = None,
        precision: str = "fp32",
        providers: Optional[Sequence[str]] = None,
        intra_threads: int = 0,
        graph_optimization: str = "all",
    ) -> None:
        try:
            import cv2
            import onnxruntime as ort
        except ImportError as exc:  # pragma: no cover - dependencia opcional
            raise ImportError(
                "Dependencias do backend ONNX nao estao disponiveis. "
                "Instale onnxruntime e opencv-python no ambiente ativo."
            ) from exc

        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"Modelo ONNX nao encontrado em {path}.")

        self.cv2 = cv2
        self.model_path = str(path)
        self.precision = precision

        disponiveis = set(ort.get_available_providers())
        if providers is not None:
            escolhidos = [p for p in providers if p in disponiveis]
        elif device is not None and device.startswith("cpu"):
            escolhidos = ["CPUExecutionProvider"]
        else:
            escolhidos = [p for p in PROVIDER_PREFERENCE if p in disponiveis]
        if not escolhidos:
            escolhidos = ["CPUExecutionProvider"]

        # O nivel maximo de otimizacao do ORT tem uma fusao de LayerNorm que quebra com o
        # EdgeFace convertido para fp16 (erro em SimplifiedLayerNormFusion na inicializacao
        # da sessao). "extended" executa o mesmo grafo sem essa fusao; o resultado numerico
        # e identico, so muda velocidade.
        niveis = {
            "all": ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
            "extended": ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
            "basic": ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
            "none": ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
        }

        def abrir(nivel: str):
            opts = ort.SessionOptions()
            if intra_threads:
                opts.intra_op_num_threads = int(intra_threads)
            opts.graph_optimization_level = niveis[nivel]
            return ort.InferenceSession(self.model_path, opts, providers=escolhidos)

        # Quem chama nao tem como saber de antemao que ESTE arquivo cai na fusao
        # quebrada -- depende do grafo, nao do backend nem da precisao. Em vez de exigir
        # que o chamador acerte o nivel, tenta o pedido e desce um degrau. Sem isso,
        # `--embedder-precision fp16` no edgeface_xs aborta a rodada na criacao da sessao.
        tentativa = [graph_optimization]
        tentativa += [n for n in ("extended", "basic") if n not in tentativa]
        erros = []
        for i, nivel in enumerate(tentativa):
            try:
                self.session = abrir(nivel)
                self.graph_optimization = nivel
                if i:
                    print(f"AVISO: {path.name} nao carregou com graph_optimization="
                          f"{tentativa[i - 1]} ({erros[-1]}); usando {nivel}.")
                break
            except Exception as exc:  # noqa: BLE001 - o ORT levanta tipos variados aqui
                erros.append(str(exc).splitlines()[0][:160])
        else:
            raise RuntimeError(
                f"{path.name} nao carregou em nenhum nivel de otimizacao "
                f"({', '.join(tentativa)}). Ultimo erro: {erros[-1]}"
            )
        self.providers = self.session.get_providers()

        entrada = self.session.get_inputs()[0]
        self.input_name = entrada.name
        self.input_shape = list(entrada.shape)
        # Modelos fp16 declaram tensor(float16) e recusam float32 na entrada.
        self.input_dtype = np.float16 if "float16" in entrada.type else np.float32
        # Ultimo eixo igual a 3 indica NHWC; caso contrario NCHW.
        self.nhwc = len(self.input_shape) == 4 and self.input_shape[-1] == 3
        self.output_name = self.session.get_outputs()[0].name

    # ------------------------------------------------------------------ interno
    def _preprocess(self, aligned_face_bgr: np.ndarray) -> np.ndarray:
        face = aligned_face_bgr
        if face.shape[:2] != (112, 112):
            face = self.cv2.resize(face, (112, 112))
        face = self.cv2.cvtColor(face, self.cv2.COLOR_BGR2RGB).astype(np.float32)
        face = (face / 255.0 - 0.5) / 0.5           # [-1, 1], igual ao ArcFaceEmbedder
        if self.nhwc:
            return face
        return np.transpose(face, (2, 0, 1))

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

        lote = np.stack([self._preprocess(f) for f in aligned_faces_bgr], axis=0)
        lote = lote.astype(self.input_dtype, copy=False)
        saida = self.session.run([self.output_name], {self.input_name: lote})[0]
        saida = np.asarray(saida, dtype=np.float32)
        if saida.ndim > 2:
            saida = saida.reshape(saida.shape[0], -1)
        if l2_normalize:
            saida = _l2(saida)
        return [saida[i] for i in range(saida.shape[0])]

    def __repr__(self) -> str:  # pragma: no cover - conveniencia
        return (
            f"OnnxEmbedder({Path(self.model_path).name}, "
            f"providers={self.providers}, layout={'NHWC' if self.nhwc else 'NCHW'})"
        )
