from __future__ import annotations

from typing import Any, Iterable, Optional, Sequence, Union

try:
    from qdrant_client import QdrantClient
    from qdrant_client import models
except ImportError:  # pragma: no cover - depends on optional package
    QdrantClient = None
    models = None


VectorLike = Union[Sequence[float], Iterable[float]]
PointId = Union[int, str]


class QdrantStore:
    def __init__(
        self,
        url: str = "http://localhost:6333",
        api_key: Optional[str] = None,
        prefer_grpc: bool = False,
        timeout: float = 10.0,
    ) -> None:
        if QdrantClient is None or models is None:
            raise ImportError(
                "qdrant-client nao esta instalado. Instale com: pip install qdrant-client"
            )

        self.client = QdrantClient(
            url=url,
            api_key=api_key,
            prefer_grpc=prefer_grpc,
            timeout=timeout,
        )

    def collection_exists(self, collection_name: str) -> bool:
        if hasattr(self.client, "collection_exists"):
            return bool(self.client.collection_exists(collection_name))

        try:
            self.client.get_collection(collection_name=collection_name)
            return True
        except Exception:
            return False

    def create_collection(
        self,
        collection_name: str,
        vector_size: int,
        distance: str = "cosine",
        recreate: bool = False,
        on_disk_payload: bool = True,
    ) -> bool:
        distance_map = {
            "cosine": models.Distance.COSINE,
            "dot": models.Distance.DOT,
            "euclid": models.Distance.EUCLID,
            "manhattan": models.Distance.MANHATTAN,
        }
        distance_key = distance.lower()
        if distance_key not in distance_map:
            supported = ", ".join(sorted(distance_map))
            raise ValueError(
                f"Distancia invalida: {distance}. Use uma destas: {supported}"
            )

        vectors_config = models.VectorParams(
            size=vector_size,
            distance=distance_map[distance_key],
        )

        if recreate:
            return self.client.recreate_collection(
                collection_name=collection_name,
                vectors_config=vectors_config,
                on_disk_payload=on_disk_payload,
            )

        if self.collection_exists(collection_name):
            return False

        return self.client.create_collection(
            collection_name=collection_name,
            vectors_config=vectors_config,
            on_disk_payload=on_disk_payload,
        )

    def index_point(
        self,
        collection_name: str,
        point_id: PointId,
        vector: VectorLike,
        payload: Optional[dict[str, Any]] = None,
        wait: bool = True,
    ) -> Any:
        point = models.PointStruct(
            id=point_id,
            vector=self._to_vector(vector),
            payload=payload or {},
        )
        return self.client.upsert(
            collection_name=collection_name,
            points=[point],
            wait=wait,
        )

    def query(
        self,
        collection_name: str,
        vector: VectorLike,
        limit: int = 5,
        query_filter: Optional[Any] = None,
        with_payload: bool = True,
        with_vectors: bool = False,
        score_threshold: Optional[float] = None,
    ) -> list[dict[str, Any]]:
        response = self.client.query_points(
            collection_name=collection_name,
            query=self._to_vector(vector),
            query_filter=query_filter,
            limit=limit,
            with_payload=with_payload,
            with_vectors=with_vectors,
            score_threshold=score_threshold,
        )
        points = self._extract_points(response)

        return [
            {
                "id": point.id,
                "score": getattr(point, "score", None),
                "payload": getattr(point, "payload", None),
                "vector": getattr(point, "vector", None),
            }
            for point in points
        ]

    def query_batch(
        self,
        collection_name: str,
        vectors: list[VectorLike],
        limit: int = 5,
        query_filter: Optional[Any] = None,
        with_payload: bool = True,
        with_vectors: bool = False,
        score_threshold: Optional[float] = None,
    ) -> list[list[dict[str, Any]]]:
        """Envia múltiplas queries em uma única chamada ao Qdrant."""
        if not vectors:
            return []

        requests = [
            models.QueryRequest(
                query=self._to_vector(v),
                filter=query_filter,
                limit=limit,
                with_payload=with_payload,
                with_vector=with_vectors,
                score_threshold=score_threshold,
            )
            for v in vectors
        ]
        responses = self.client.query_batch_points(
            collection_name=collection_name,
            requests=requests,
        )
        results = []
        for response in responses:
            points = self._extract_points(response)
            results.append(
                [
                    {
                        "id": point.id,
                        "score": getattr(point, "score", None),
                        "payload": getattr(point, "payload", None),
                        "vector": getattr(point, "vector", None),
                    }
                    for point in points
                ]
            )
        return results

    @staticmethod
    def _to_vector(vector: VectorLike) -> list[float]:
        return [float(value) for value in vector]

    @staticmethod
    def _extract_points(response: Any) -> list[Any]:
        if response is None:
            return []

        points = getattr(response, "points", None)
        if points is not None:
            return list(points)

        result = getattr(response, "result", None)
        if result is not None:
            nested_points = getattr(result, "points", None)
            if nested_points is not None:
                return list(nested_points)
            if isinstance(result, list):
                return result

        if isinstance(response, list):
            return response

        return []


__all__ = ["QdrantStore"]
