from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFont


BACKGROUND_COLOR = (248, 250, 252)
TEXT_COLOR = (15, 23, 42)
MUTED_TEXT_COLOR = (71, 85, 105)
ACCENT_COLOR = (37, 99, 235)
SUCCESS_COLOR = (22, 163, 74)
WARNING_COLOR = (234, 88, 12)
CARD_COLOR = (255, 255, 255)
BORDER_COLOR = (203, 213, 225)

try:
    RESAMPLING_LANCZOS = Image.Resampling.LANCZOS
except AttributeError:  # pragma: no cover - compatibility with older Pillow
    RESAMPLING_LANCZOS = Image.LANCZOS


def _font() -> ImageFont.ImageFont:
    return ImageFont.load_default()


def _jsonify(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {k: _jsonify(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonify(v) for v in value]
    return value


def _jsonify_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _to_rgb_array(image: Image.Image | np.ndarray) -> np.ndarray:
    if isinstance(image, Image.Image):
        return np.asarray(image.convert("RGB"))

    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("A imagem precisa ter shape HxWx3.")

    if image.dtype != np.uint8:
        return np.clip(image, 0, 255).astype(np.uint8)
    return image


def _save_rgb(path: Path, image_rgb: np.ndarray) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(_to_rgb_array(image_rgb)).save(path)
    return str(path)


def _measure_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> tuple[int, int]:
    if hasattr(draw, "textbbox"):
        bbox = draw.textbbox((0, 0), text, font=font)
        return bbox[2] - bbox[0], bbox[3] - bbox[1]

    width, height = draw.textsize(text, font=font)
    return int(width), int(height)


def _fit_image(image_rgb: np.ndarray, max_size: tuple[int, int]) -> Image.Image:
    image = Image.fromarray(_to_rgb_array(image_rgb))
    fitted = image.copy()
    fitted.thumbnail(max_size, RESAMPLING_LANCZOS)
    return fitted


def _create_canvas(width: int = 1600, height: int = 900) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    canvas = Image.new("RGB", (width, height), BACKGROUND_COLOR)
    return canvas, ImageDraw.Draw(canvas)


def _draw_header(
    draw: ImageDraw.ImageDraw,
    title: str,
    subtitle: str,
    width: int,
) -> int:
    font = _font()
    draw.text((40, 28), title, fill=TEXT_COLOR, font=font)
    draw.text((40, 54), subtitle, fill=MUTED_TEXT_COLOR, font=font)
    draw.line((40, 82, width - 40, 82), fill=BORDER_COLOR, width=2)
    return 102


def _draw_wrapped_lines(
    draw: ImageDraw.ImageDraw,
    lines: list[str],
    origin: tuple[int, int],
    max_width: int,
) -> None:
    font = _font()
    x, y = origin
    for line in lines:
        wrapped = _wrap_text(draw, line, font, max_width)
        for wrapped_line in wrapped:
            draw.text((x, y), wrapped_line, fill=TEXT_COLOR, font=font)
            _, height = _measure_text(draw, wrapped_line, font)
            y += height + 8
        y += 4


def _wrap_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.ImageFont,
    max_width: int,
) -> list[str]:
    words = text.split()
    if not words:
        return [""]

    lines: list[str] = []
    current = words[0]
    for word in words[1:]:
        candidate = f"{current} {word}"
        width, _ = _measure_text(draw, candidate, font)
        if width <= max_width:
            current = candidate
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines


def _draw_card(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int, int, int],
    title: str,
    lines: list[str],
    accent: tuple[int, int, int] = ACCENT_COLOR,
) -> None:
    x1, y1, x2, y2 = xy
    draw.rounded_rectangle(xy, radius=18, fill=CARD_COLOR, outline=BORDER_COLOR, width=2)
    draw.rounded_rectangle((x1, y1, x1 + 10, y2), radius=18, fill=accent)
    font = _font()
    draw.text((x1 + 28, y1 + 18), title, fill=TEXT_COLOR, font=font)
    _draw_wrapped_lines(draw, lines, (x1 + 28, y1 + 48), x2 - x1 - 52)


def _draw_boxes(image_rgb: np.ndarray, detections: list[dict[str, Any]], selected_bbox: Optional[list[int]]) -> np.ndarray:
    image = Image.fromarray(_to_rgb_array(image_rgb))
    draw = ImageDraw.Draw(image)
    font = _font()

    for index, detection in enumerate(detections, start=1):
        bbox = detection["bbox"]
        score = detection.get("score")
        is_selected = selected_bbox == bbox
        color = SUCCESS_COLOR if is_selected else ACCENT_COLOR
        line_width = 5 if is_selected else 3

        draw.rectangle(tuple(bbox), outline=color, width=line_width)
        label = f"face {index}"
        if score is not None:
            label += f" {score:.2f}"
        label_width, label_height = _measure_text(draw, label, font)
        x1, y1 = bbox[0], max(0, bbox[1] - label_height - 12)
        draw.rectangle((x1, y1, x1 + label_width + 12, y1 + label_height + 10), fill=color)
        draw.text((x1 + 6, y1 + 5), label, fill=(255, 255, 255), font=font)

    return np.asarray(image)


def _make_triptych(
    title: str,
    subtitle: str,
    panels: list[tuple[str, np.ndarray]],
) -> np.ndarray:
    canvas, draw = _create_canvas()
    top = _draw_header(draw, title=title, subtitle=subtitle, width=canvas.size[0])

    panel_width = 470
    panel_height = 620
    gap = 35
    x = 55
    font = _font()

    for label, image_rgb in panels:
        draw.rounded_rectangle(
            (x, top, x + panel_width, top + panel_height),
            radius=18,
            fill=CARD_COLOR,
            outline=BORDER_COLOR,
            width=2,
        )
        draw.text((x + 18, top + 16), label, fill=TEXT_COLOR, font=font)
        fitted = _fit_image(image_rgb, (panel_width - 36, panel_height - 70))
        fitted_x = x + (panel_width - fitted.size[0]) // 2
        fitted_y = top + 52 + (panel_height - 70 - fitted.size[1]) // 2
        canvas.paste(fitted, (fitted_x, fitted_y))
        x += panel_width + gap

    return np.asarray(canvas)


def _render_alignment_stage(
    original_crop_rgb: np.ndarray,
    alignment_preview_rgb: np.ndarray,
    aligned_face_rgb: np.ndarray,
    alignment_source: str,
    alignment_warning: Optional[str],
) -> np.ndarray:
    canvas, draw = _create_canvas()
    top = _draw_header(
        draw,
        title="03. Standardize Face Alignment",
        subtitle=f"Source: {alignment_source}",
        width=canvas.size[0],
    )

    panels = [
        ("Original crop", original_crop_rgb),
        ("Landmarks / alignment", alignment_preview_rgb),
        ("Aligned 112x112", aligned_face_rgb),
    ]
    panel_width = 470
    panel_height = 560
    gap = 35
    x = 55
    font = _font()

    for label, image_rgb in panels:
        draw.rounded_rectangle(
            (x, top, x + panel_width, top + panel_height),
            radius=18,
            fill=CARD_COLOR,
            outline=BORDER_COLOR,
            width=2,
        )
        draw.text((x + 18, top + 16), label, fill=TEXT_COLOR, font=font)
        fitted = _fit_image(image_rgb, (panel_width - 36, panel_height - 70))
        fitted_x = x + (panel_width - fitted.size[0]) // 2
        fitted_y = top + 52 + (panel_height - 70 - fitted.size[1]) // 2
        canvas.paste(fitted, (fitted_x, fitted_y))
        x += panel_width + gap

    lines = [
        f"alignment_source: {alignment_source}",
        f"warning: {alignment_warning or 'none'}",
    ]
    _draw_card(draw, (55, 700, 1545, 840), "Alignment details", lines)
    return np.asarray(canvas)


def _render_alignment_preview(
    crop_rgb: np.ndarray,
    alignment_info: dict[str, Any],
) -> np.ndarray:
    image = Image.fromarray(_to_rgb_array(crop_rgb))
    draw = ImageDraw.Draw(image)
    font = _font()

    bbox = alignment_info.get("bbox")
    if isinstance(bbox, list) and len(bbox) == 4:
        draw.rectangle(tuple(int(round(value)) for value in bbox), outline=ACCENT_COLOR, width=3)

    kps = alignment_info.get("kps")
    if kps is not None and len(kps):
        for index, point in enumerate(np.asarray(kps), start=1):
            x, y = int(round(float(point[0]))), int(round(float(point[1])))
            draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=SUCCESS_COLOR, outline=SUCCESS_COLOR)
            draw.text((x + 6, y - 8), str(index), fill=TEXT_COLOR, font=font)

    return np.asarray(image)


def _render_embedding_stage(embedding: list[float], network_name: str) -> np.ndarray:
    values = np.asarray(embedding, dtype=np.float32)
    total = int(values.size)
    cols = min(32, max(8, int(math.ceil(math.sqrt(total)))))
    rows = int(math.ceil(total / cols))
    cell_size = 24
    heatmap_width = cols * cell_size
    heatmap_height = rows * cell_size

    canvas, draw = _create_canvas()
    top = _draw_header(
        draw,
        title=f"04. Generate Embedding {network_name}",
        subtitle=f"Vector size: {total}",
        width=canvas.size[0],
    )

    heatmap_x = 55
    heatmap_y = top + 10
    max_abs = max(float(np.max(np.abs(values))), 1e-6)
    for index, value in enumerate(values):
        row = index // cols
        col = index % cols
        normalized = float(value) / max_abs
        if normalized >= 0:
            red = int(255 * normalized)
            green = int(120 + 80 * (1.0 - normalized))
            blue = 60
        else:
            red = 60
            green = int(120 + 80 * (1.0 + normalized))
            blue = int(255 * abs(normalized))

        x1 = heatmap_x + col * cell_size
        y1 = heatmap_y + row * cell_size
        draw.rectangle((x1, y1, x1 + cell_size, y1 + cell_size), fill=(red, green, blue))

    draw.rounded_rectangle(
        (
            heatmap_x - 10,
            heatmap_y - 10,
            heatmap_x + heatmap_width + 10,
            heatmap_y + heatmap_height + 10,
        ),
        radius=18,
        outline=BORDER_COLOR,
        width=2,
    )

    stats_lines = [
        f"min: {float(values.min()):.4f}",
        f"max: {float(values.max()):.4f}",
        f"mean: {float(values.mean()):.4f}",
        f"std: {float(values.std()):.4f}",
        "Colors: orange = positive, blue = negative",
    ]
    _draw_card(draw, (980, top + 20, 1545, top + 260), "Embedding summary", stats_lines)
    _draw_card(
        draw,
        (980, top + 290, 1545, top + 550),
        "Interpretation",
        [
            "ArcFace converts the aligned face into a numeric signature.",
            "This vector is what the system uses for similarity search in Qdrant.",
        ],
    )
    return np.asarray(canvas)


def _format_match_line(index: int, match: dict[str, Any]) -> str:
    name = match.get("name") or "Unknown"
    entity_id = match.get("entity_id") or "-"
    score = match.get("score")
    if score is None:
        return f"{index}. {name} | entity_id={entity_id}"
    return f"{index}. {name} | entity_id={entity_id} | score={float(score):.4f}"


def _render_qdrant_stage(
    identity_matches: list[dict[str, Any]],
    raw_matches: list[dict[str, Any]],
) -> np.ndarray:
    canvas, draw = _create_canvas()
    _draw_header(
        draw,
        title="05. Compare Against Qdrant",
        subtitle=f"Candidates: {len(raw_matches)} raw / {len(identity_matches)} grouped",
        width=canvas.size[0],
    )

    grouped_lines = [
        _format_match_line(index, match)
        for index, match in enumerate(identity_matches[:5], start=1)
    ] or ["No candidates returned from Qdrant."]
    raw_lines = []
    for index, match in enumerate(raw_matches[:5], start=1):
        payload = match.get("payload") if isinstance(match.get("payload"), dict) else {}
        name = payload.get("name") or "Unknown"
        entity_id = payload.get("entity_id") or "-"
        score = match.get("score")
        if score is None:
            raw_lines.append(f"{index}. point_id={match.get('id')} | {name} | {entity_id}")
        else:
            raw_lines.append(
                f"{index}. point_id={match.get('id')} | {name} | {entity_id} | score={float(score):.4f}"
            )

    _draw_card(draw, (55, 120, 770, 820), "Grouped by identity", grouped_lines)
    _draw_card(draw, (830, 120, 1545, 820), "Raw nearest neighbors", raw_lines or ["No raw matches found."])
    return np.asarray(canvas)


def _render_result_stage(
    source_image_rgb: np.ndarray,
    selected_bbox: list[int],
    recognized_name: Optional[str],
    entity_id: Optional[str],
    score: Optional[float],
    recognized: bool,
) -> np.ndarray:
    annotated = Image.fromarray(_to_rgb_array(source_image_rgb))
    draw = ImageDraw.Draw(annotated)
    font = _font()
    color = SUCCESS_COLOR if recognized else WARNING_COLOR
    label_name = recognized_name or "Unknown"
    label = label_name if score is None else f"{label_name} {float(score):.4f}"

    draw.rectangle(tuple(selected_bbox), outline=color, width=5)
    label_width, label_height = _measure_text(draw, label, font)
    x1, y1 = selected_bbox[0], max(0, selected_bbox[1] - label_height - 12)
    draw.rectangle((x1, y1, x1 + label_width + 12, y1 + label_height + 10), fill=color)
    draw.text((x1 + 6, y1 + 5), label, fill=(255, 255, 255), font=font)

    canvas, card_draw = _create_canvas()
    top = _draw_header(
        card_draw,
        title="06. Recognize Person",
        subtitle="Final decision after vector search",
        width=canvas.size[0],
    )
    fitted = _fit_image(np.asarray(annotated), (920, 680))
    canvas.paste(fitted, (55, top + 20))
    _draw_card(
        card_draw,
        (1030, top + 20, 1545, top + 320),
        "Recognition result",
        [
            f"recognized: {recognized}",
            f"name: {recognized_name or 'Unknown'}",
            f"entity_id: {entity_id or '-'}",
            f"score: {'-' if score is None else f'{float(score):.4f}'}",
        ],
        accent=SUCCESS_COLOR if recognized else WARNING_COLOR,
    )
    return np.asarray(canvas)


def save_pipeline_presentation_artifacts(
    output_dir: Path,
    source_image_rgb: Image.Image | np.ndarray,
    detections: list[dict[str, Any]],
    selected_detection: dict[str, Any],
    face_region: dict[str, Any],
    aligned_face_bgr: np.ndarray,
    alignment_info: dict[str, Any],
    alignment_source: str,
    embedding: list[float],
    raw_matches: list[dict[str, Any]],
    identity_matches: list[dict[str, Any]],
    recognized_name: Optional[str],
    entity_id: Optional[str],
    score: Optional[float],
    isolate_method: str,
    network_name: str = "ArcFace R100",
    metadata: Optional[dict[str, Any]] = None,
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)

    source_rgb = _to_rgb_array(source_image_rgb)
    selected_bbox = selected_detection["bbox"]
    original_crop_rgb = _to_rgb_array(face_region["crop_rgb"])
    segmented_crop_rgb = _to_rgb_array(face_region["segmented_crop_rgb"])
    crop_mask_rgb = np.stack(
        [face_region["crop_mask"].astype(np.uint8) * 255] * 3,
        axis=-1,
    )
    aligned_face_rgb = aligned_face_bgr[:, :, ::-1].copy()
    alignment_preview_rgb = _render_alignment_preview(original_crop_rgb, alignment_info)

    stage_paths = {
        "input": _save_rgb(output_dir / "00_input.png", source_rgb),
        "detect_faces": _save_rgb(
            output_dir / "01_detect_faces_yolo.png",
            _draw_boxes(source_rgb, detections, selected_bbox),
        ),
        "isolate_face": _save_rgb(
            output_dir / f"02_isolate_face_{isolate_method.lower()}.png",
            _make_triptych(
                title=f"02. Isolate Face {isolate_method.upper()}",
                subtitle="Face crop prepared for alignment and embedding",
                panels=[
                    ("Original crop", original_crop_rgb),
                    ("Face mask", crop_mask_rgb),
                    ("Segmented face", segmented_crop_rgb),
                ],
            ),
        ),
        "alignment": _save_rgb(
            output_dir / "03_standardize_face_alignment.png",
            _render_alignment_stage(
                original_crop_rgb=original_crop_rgb,
                alignment_preview_rgb=alignment_preview_rgb,
                aligned_face_rgb=aligned_face_rgb,
                alignment_source=alignment_source,
                alignment_warning=alignment_info.get("warning"),
            ),
        ),
        "embedding": _save_rgb(
            output_dir / "04_generate_embedding_arcface_r100.png",
            _render_embedding_stage(embedding=embedding, network_name=network_name),
        ),
        "qdrant": _save_rgb(
            output_dir / "05_compare_qdrant.png",
            _render_qdrant_stage(identity_matches=identity_matches, raw_matches=raw_matches),
        ),
        "result": _save_rgb(
            output_dir / "06_recognize_person.png",
            _render_result_stage(
                source_image_rgb=source_rgb,
                selected_bbox=selected_bbox,
                recognized_name=recognized_name,
                entity_id=entity_id,
                score=score,
                recognized=recognized_name is not None,
            ),
        ),
    }

    metadata_path = output_dir / "pipeline_metadata.json"
    metadata_payload = {
        "selected_detection": _jsonify(selected_detection),
        "alignment_source": alignment_source,
        "alignment_warning": alignment_info.get("warning"),
        "recognized_name": recognized_name,
        "entity_id": entity_id,
        "score": score,
        "artifacts": stage_paths,
    }
    if metadata:
        metadata_payload.update(metadata)

    with metadata_path.open("w", encoding="utf-8") as file:
        json.dump(metadata_payload, file, ensure_ascii=True, indent=2, default=_jsonify_default)

    stage_paths["metadata"] = str(metadata_path)
    return stage_paths


__all__ = ["save_pipeline_presentation_artifacts"]
