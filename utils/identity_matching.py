from __future__ import annotations

from typing import Any


def resolve_identity(match: dict[str, Any]) -> dict[str, Any]:
    payload = match.get("payload")
    match_id = match.get("id")
    entity_id = None
    name = None

    if isinstance(payload, dict):
        entity_id = payload.get("entity_id")
        name = payload.get("name")

    if entity_id is None:
        entity_id = name or match_id

    return {
        "entity_id": entity_id,
        "name": name,
        "match_id": match_id,
    }


def group_matches_by_identity(
    matches: list[dict[str, Any]],
    limit: int | None = None,
) -> list[dict[str, Any]]:
    grouped: dict[Any, dict[str, Any]] = {}

    for match in matches:
        identity = resolve_identity(match)
        entity_id = identity["entity_id"]
        score = match.get("score")

        if entity_id not in grouped:
            grouped[entity_id] = {
                "entity_id": entity_id,
                "name": identity["name"],
                "score": score,
                "best_match_id": identity["match_id"],
                "match_count": 0,
                "point_ids": [],
                "matches": [],
            }

        group = grouped[entity_id]
        group["match_count"] += 1
        group["point_ids"].append(identity["match_id"])
        group["matches"].append(match)

        if group["name"] is None and identity["name"] is not None:
            group["name"] = identity["name"]

        if score is not None and (group["score"] is None or score > group["score"]):
            group["score"] = score
            group["best_match_id"] = identity["match_id"]

    grouped_matches = sorted(
        grouped.values(),
        key=lambda item: float("-inf") if item["score"] is None else float(item["score"]),
        reverse=True,
    )

    if limit is not None:
        return grouped_matches[:limit]
    return grouped_matches


def select_best_identity_match(
    matches: list[dict[str, Any]],
    *,
    limit: int | None = None,
    min_score: float | None = None,
    score_margin: float = 0.0,
    min_match_count: int = 1,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    identity_matches = group_matches_by_identity(matches, limit=limit)
    if not identity_matches:
        return None, identity_matches

    best_match = identity_matches[0]
    best_score = best_match.get("score")
    if best_score is None:
        return None, identity_matches

    if min_score is not None and best_score < min_score:
        return None, identity_matches

    if best_match.get("match_count", 0) < min_match_count:
        return None, identity_matches

    if score_margin > 0.0 and len(identity_matches) > 1:
        second_score = identity_matches[1].get("score")
        if second_score is not None and (best_score - second_score) < score_margin:
            return None, identity_matches

    return best_match, identity_matches


__all__ = [
    "group_matches_by_identity",
    "resolve_identity",
    "select_best_identity_match",
]
