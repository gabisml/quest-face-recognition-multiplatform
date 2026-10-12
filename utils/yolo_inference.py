from __future__ import annotations

from typing import Any

import numpy as np
from ultralytics import YOLO


def load_yolo_model(model_path):
    return YOLO(model_path)


def detect_faces_with_model(
    model,
    image: Any,
    conf: float = 0.25,
    imgsz: int | None = None,
    max_det: int | None = None,
):
    """Run YOLO with a preloaded model and return detections sorted by confidence."""
    kwargs: dict[str, Any] = {
        "conf": conf,
        "verbose": False,
    }
    if imgsz is not None:
        kwargs["imgsz"] = imgsz
    if max_det is not None:
        kwargs["max_det"] = max_det

    if isinstance(image, np.ndarray):
        inference_image = image[:, :, ::-1]
    else:
        inference_image = image

    results = model(inference_image, **kwargs)

    boxes_xyxy = results[0].boxes.xyxy.cpu().tolist()
    confidences = results[0].boxes.conf.cpu().tolist()

    # Extract keypoints (5 facial landmarks) when available from YOLO-face models.
    kps_list: list[np.ndarray | None] = [None] * len(boxes_xyxy)
    result_kps = results[0].keypoints
    if result_kps is not None and result_kps.xy is not None:
        kps_xy = result_kps.xy.cpu().numpy()
        for idx in range(min(len(boxes_xyxy), kps_xy.shape[0])):
            pts = kps_xy[idx]  # shape (N_keypoints, 2)
            if pts.shape[0] >= 5 and np.any(pts[:5] > 0):
                kps_list[idx] = pts[:5].astype(np.float32)

    detections = []
    for idx, (box, score) in enumerate(zip(boxes_xyxy, confidences)):
        x1, y1, x2, y2 = [int(value) for value in box]
        det: dict[str, Any] = {
            "index": idx,
            "bbox": [x1, y1, x2, y2],
            "score": float(score),
            "area": max(0, x2 - x1) * max(0, y2 - y1),
        }
        if kps_list[idx] is not None:
            det["kps"] = kps_list[idx]
        detections.append(det)

    detections.sort(key=lambda item: item["score"], reverse=True)
    return detections


def detect_faces(model_path, image, conf=0.25, imgsz: int | None = None, max_det: int | None = None):
    """Run YOLO and return face detections sorted by confidence."""
    model = load_yolo_model(model_path)
    return detect_faces_with_model(model, image, conf=conf, imgsz=imgsz, max_det=max_det)


def select_primary_face(detections):
    """Select the most relevant face, favoring larger high-confidence boxes."""
    if not detections:
        raise ValueError("YOLO nao detectou nenhum rosto.")

    return max(detections, key=lambda item: (item["area"], item["score"]))


def yolo_inference(model_path, image, conf=0.25):
    """Run YOLO and return SAM3-compatible boxes and labels.

    Returns:
        tuple[list[list[list[int]]], list[list[int]]]:
            input_boxes and input_boxes_labels in SAM3 format.
    """
    detections = detect_faces(model_path, image, conf=conf)
    boxes_xyxy = [item["bbox"] for item in detections]

    input_boxes = [boxes_xyxy]
    input_boxes_labels = [[1] * len(boxes_xyxy)]

    return input_boxes, input_boxes_labels


__all__ = [
    "detect_faces",
    "detect_faces_with_model",
    "load_yolo_model",
    "select_primary_face",
    "yolo_inference",
]
