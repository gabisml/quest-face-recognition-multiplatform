"""Busca de embedding em memoria, com o mesmo contrato do QdrantStore.

POR QUE ISSO EXISTE
    A Coral Dev Board nao roda o container do Qdrant: sao 985 MB de RAM totais, com cerca
    de 600 MB livres, e o servidor competiria com o proprio pipeline. Para uma galeria de
    algumas dezenas de rostos isso e desproporcional de qualquer forma -- 50 vetores de 512
    dimensoes em float32 sao 100 KB.

    E MEDIDO NA PLACA (ver bench/coral/embedding_search.log): forca bruta em numpy custa
    0,12 ms para 50 rostos, e ainda vence o faiss IndexFlatIP por 3,6x para uma consulta,
    porque o faiss so usa BLAS acima de 20 consultas e abaixo disso perde do GEMV do
    OpenBLAS.

CUIDADO METODOLOGICO
    O backend de busca domina a etapa `search`, e nao o hardware. Nas medicoes existentes o
    Qdrant custa ~3,0 ms no DGX Spark e ~4,3 ms no AGX Orin, contra 0,12 ms deste backend na
    Coral -- 25 a 37x, quase tudo round-trip cliente-servidor, ja que a aritmetica e a mesma
    0,1 ms nos dois casos. Comparar a etapa `search` entre plataformas exige o MESMO backend
    nas duas. Use este para a comparacao entre plataformas, e reporte o Qdrant separadamente
    como custo de uma opcao de implantacao.

PARIDADE COM O QDRANT
    A colecao do projeto usa distancia COSINE, que o Qdrant obtem normalizando os vetores
    internamente. Aqui os vetores sao normalizados em L2 na insercao e na consulta, de modo
    que o produto interno E o cosseno e os scores sao diretamente comparaveis.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence, Union

import numpy as np

# Union em vez de `|`: este alias e avaliado em tempo de execucao, e o `from
# __future__ import annotations` acima so adia anotacoes, nao atribuicoes. A
# Coral Dev Board esta presa ao Python 3.9 -- o libedgetpu 16.0 da imagem exige
# tflite-runtime 2.7.0, cujo wheel mais novo e cp39 -- e a forma com `|` so
# existe a partir do 3.10.
VectorLike = Union[Sequence[float], np.ndarray]


def _l2(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return np.divide(x, n, out=np.zeros_like(x), where=n != 0)


class NumpyVectorStore:
    """Galeria (N, dim) em memoria, busca exata por produto interno.

    Implementa o subconjunto do QdrantStore que o pipeline usa: `query` e `query_batch`
    devolvem a mesma estrutura, `{id, score, payload, vector}`, para que
    utils.identity_matching funcione sem alteracao.

    `collection_name` e aceito e ignorado: existe so para a assinatura casar com o
    QdrantStore. Um processo carrega uma galeria.
    """

    def __init__(self, dim: int = 512) -> None:
        self.dim = dim
        self.vectors = np.zeros((0, dim), dtype=np.float32)
        self.payloads: list[dict[str, Any]] = []
        self.ids: list[Any] = []

    # ------------------------------------------------------------------ cadastro
    def add(
        self,
        vectors: Iterable[VectorLike],
        payloads: Iterable[dict[str, Any]],
        ids: Optional[Iterable[Any]] = None,
    ) -> int:
        """Acrescenta vetores a galeria. Devolve o total apos a insercao."""
        v = np.asarray(list(vectors), dtype=np.float32)
        if v.ndim != 2 or v.shape[1] != self.dim:
            raise ValueError(f"esperava (N, {self.dim}), recebi {v.shape}")
        p = list(payloads)
        if len(p) != v.shape[0]:
            raise ValueError(f"{v.shape[0]} vetores mas {len(p)} payloads")
        base = len(self.ids)
        new_ids = list(ids) if ids is not None else list(range(base, base + v.shape[0]))
        if len(new_ids) != v.shape[0]:
            raise ValueError("numero de ids diferente do numero de vetores")

        self.vectors = np.ascontiguousarray(np.vstack([self.vectors, _l2(v)]))
        self.payloads.extend(p)
        self.ids.extend(new_ids)
        return len(self.ids)

    def __len__(self) -> int:
        return len(self.ids)

    # ------------------------------------------------------------------ consulta
    def query(
        self,
        collection_name: str | None = None,
        vector: VectorLike | None = None,
        limit: int = 5,
        score_threshold: Optional[float] = None,
        **_ignored: Any,
    ) -> list[dict[str, Any]]:
        if vector is None:
            return []
        return self.query_batch(
            collection_name, [vector], limit=limit, score_threshold=score_threshold
        )[0]

    def query_batch(
        self,
        collection_name: str | None = None,
        vectors: Optional[Iterable[VectorLike]] = None,
        limit: int = 5,
        score_threshold: Optional[float] = None,
        **_ignored: Any,
    ) -> list[list[dict[str, Any]]]:
        q = list(vectors or [])
        if not q or len(self.ids) == 0:
            return [[] for _ in q]

        Q = _l2(np.asarray(q, dtype=np.float32))          # (B, dim)
        sims = self.vectors @ Q.T                          # (N, B) == cosseno
        k = min(limit, sims.shape[0])

        out: list[list[dict[str, Any]]] = []
        for b in range(sims.shape[1]):
            col = sims[:, b]
            # argpartition e O(N); so o top-k precisa ser ordenado
            idx = np.argpartition(-col, k - 1)[:k] if k < col.shape[0] else np.arange(col.shape[0])
            idx = idx[np.argsort(-col[idx])]
            matches = []
            for i in idx:
                score = float(col[i])
                if score_threshold is not None and score < score_threshold:
                    continue
                matches.append({
                    "id": self.ids[i],
                    "score": score,
                    "payload": self.payloads[i],
                    "vector": None,
                })
            out.append(matches)
        return out

    # ------------------------------------------------------------------ persistencia
    def save(self, path: str | Path) -> Path:
        """Grava `<path>.npy` com os vetores e `<path>.json` com ids e payloads."""
        p = Path(path).with_suffix("")
        np.save(p.with_suffix(".npy"), self.vectors)
        p.with_suffix(".json").write_text(json.dumps(
            {"dim": self.dim, "ids": self.ids, "payloads": self.payloads},
            ensure_ascii=False, indent=2,
        ))
        return p

    @classmethod
    def load(cls, path: str | Path) -> "NumpyVectorStore":
        p = Path(path).with_suffix("")
        meta = json.loads(p.with_suffix(".json").read_text())
        store = cls(dim=int(meta["dim"]))
        store.vectors = np.ascontiguousarray(
            np.load(p.with_suffix(".npy")).astype(np.float32)
        )
        store.ids = meta["ids"]
        store.payloads = meta["payloads"]
        if store.vectors.shape[0] != len(store.ids):
            raise ValueError("galeria e metadados com tamanhos diferentes")
        return store


def build_vector_store(kind: str = "qdrant", **kwargs: Any):
    """Fabrica pelo nome, para as CLIs e o bench selecionarem por flag.

    kind='numpy'  -> NumpyVectorStore, carregado de `gallery_path` se informado
    kind='qdrant' -> QdrantStore (import tardio: a Coral nao tem o cliente instalado)
    """
    k = kind.lower().strip()
    if k == "numpy":
        gallery = kwargs.get("gallery_path")
        if gallery:
            return NumpyVectorStore.load(gallery)
        return NumpyVectorStore(dim=int(kwargs.get("dim", 512)))
    if k == "qdrant":
        from utils.qdrant import QdrantStore
        return QdrantStore(
            url=kwargs.get("qdrant_url", "http://localhost:6333"),
            api_key=kwargs.get("qdrant_api_key"),
            timeout=float(kwargs.get("qdrant_timeout", 15.0)),
        )
    raise ValueError(f"backend de busca desconhecido: {kind!r} (use 'numpy' ou 'qdrant')")
