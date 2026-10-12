from __future__ import annotations

from typing import Any

import cv2


def draw_recognitions(frame, recognitions: list[dict[str, Any]]) -> None:
    for item in recognitions:
        x1, y1, x2, y2 = item["bbox"]
        recognized = item.get("recognized", False)
        name = item.get("name") or "Desconhecido"
        score = item.get("score")
        color = (0, 180, 0) if recognized else (0, 165, 255)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        label = f"{name} {score:.2f}" if score is not None else name
        if not recognized and item.get("top_candidate_name") and item.get("top_candidate_score") is not None:
            label = (
                f"? {item['top_candidate_name']} "
                f"{item['top_candidate_score']:.2f} "
                f"v{item.get('top_candidate_match_count', 0)}"
            )

        label_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        text_y1 = max(0, y1 - label_size[1] - baseline - 6)
        text_y2 = text_y1 + label_size[1] + baseline + 6
        text_x2 = x1 + label_size[0] + 10
        cv2.rectangle(frame, (x1, text_y1), (text_x2, text_y2), color, -1)
        cv2.putText(
            frame,
            label,
            (x1 + 5, text_y2 - baseline - 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )


__all__ = ["draw_recognitions"]
