"""Export evenly spaced, transparent PNG frames from a restored animation."""

from __future__ import annotations

import json
from pathlib import Path
import zipfile

import cv2
import numpy as np

from workflow import align_frame, crop_to_master, load_master_frame, read_json


FRAME_COUNT = 20


def remove_flat_background(frame: np.ndarray, expected_bgr: np.ndarray | None = None) -> tuple[np.ndarray, dict]:
    """Key a nearly uniform edge color, retaining enclosed dark artwork."""
    height, width = frame.shape[:2]
    strip = max(2, min(height, width) // 80)
    border = np.concatenate((frame[:strip].reshape(-1, 3),
                             frame[-strip:].reshape(-1, 3),
                             frame[:, :strip].reshape(-1, 3),
                             frame[:, -strip:].reshape(-1, 3)))
    seed = np.asarray(expected_bgr, dtype=np.float32) if expected_bgr is not None else np.median(border, axis=0)
    seed_distance = np.max(np.abs(border.astype(np.float32) - seed), axis=1)
    background_samples = seed_distance <= 24
    if float(np.mean(background_samples)) < (0.1 if expected_bgr is not None else 0.5):
        raise ValueError("画面边缘背景不是单色；当前去背景节点无法可靠处理，请使用透明母版或配置 AI 抠图")
    color = np.median(border[background_samples], axis=0)
    border_distance = np.max(np.abs(border.astype(np.float32) - color), axis=1)
    background_samples = border_distance <= 24
    background_fraction = float(np.mean(background_samples))
    # The subject may touch one edge. Measure the background cluster itself,
    # rather than treating foreground pixels along that edge as background noise.
    spread = float(np.percentile(border_distance[background_samples], 90))

    distance = np.max(np.abs(frame.astype(np.float32) - color), axis=2)
    low = max(5.0, spread + 3.0)
    high = low + 22.0
    alpha = np.clip((distance - low) * 255.0 / (high - low), 0, 255).astype(np.uint8)

    # Dark details enclosed by the subject must stay opaque. Only transparent
    # areas connected to the canvas edge belong to the background.
    background = (alpha < 220).astype(np.uint8)
    count, labels = cv2.connectedComponents(background, connectivity=4)
    edge_labels = np.unique(np.concatenate((labels[0], labels[-1], labels[:, 0], labels[:, -1])))
    exterior = np.isin(labels, edge_labels)
    alpha[(background != 0) & ~exterior] = 255
    alpha = cv2.medianBlur(alpha, 3)

    rgb = frame[:, :, ::-1].astype(np.float32)
    fraction = alpha.astype(np.float32) / 255.0
    visible = fraction > 0.02
    foreground = np.zeros_like(rgb)
    background_rgb = color[::-1]
    foreground[visible] = np.clip(
        (rgb[visible] - (1.0 - fraction[visible, None]) * background_rgb)
        / fraction[visible, None], 0, 255)
    rgba = np.dstack((foreground.astype(np.uint8), alpha))
    return rgba, {"method": "flat_edge_color", "background_rgb": [int(v) for v in color[::-1]],
                  "border_spread": round(spread, 2),
                  "border_background_fraction": round(background_fraction, 3)}


def export_frame_assets(task_folder: Path, count: int = FRAME_COUNT) -> dict:
    source = task_folder / "generated.mp4"
    if not source.is_file():
        raise FileNotFoundError(source)
    analysis = read_json(task_folder / "analysis.json")
    master_frame = load_master_frame(analysis)
    expected_bgr = np.asarray(analysis["transform"]["background_rgb"][::-1], dtype=np.float32)
    quality = read_json(task_folder / "quality_report.json")
    alignment = quality["first_frame_alignment"]
    scale = float(alignment.get("scale", 1.0)) if alignment["applied"] else 1.0
    dx, dy = alignment.get("offset", [0.0, 0.0]) if alignment["applied"] else [0.0, 0.0]
    matrix = np.array([[scale, 0.0, dx], [0.0, scale, dy]], dtype=np.float32)
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise RuntimeError(f"无法读取恢复后的视频：{source}")
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if total < count:
        capture.release()
        raise ValueError(f"视频只有 {total} 帧，无法抽取 {count} 个不同帧")
    fps = capture.get(cv2.CAP_PROP_FPS) or 24.0
    indices = np.rint(np.linspace(0, total - 1, count)).astype(int).tolist()
    frame_dir = task_folder / "transparent_frames"
    frame_dir.mkdir(exist_ok=True)
    records = []
    background_info = None
    try:
        for sequence, index in enumerate(indices, 1):
            capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError(f"无法读取第 {index} 帧")
            restored = align_frame(crop_to_master(frame, analysis, master_frame), matrix, analysis)
            rgba, info = remove_flat_background(restored, expected_bgr)
            background_info = info
            name = f"frame_{sequence:02d}.png"
            if not cv2.imwrite(str(frame_dir / name), rgba[:, :, [2, 1, 0, 3]]):
                raise RuntimeError(f"无法写入 {name}")
            records.append({"file": name, "source_frame": index, "time_seconds": round(index / fps, 4)})
    finally:
        capture.release()

    master_width, master_height = analysis["master_size"]
    manifest = {"source": source.name, "frame_count": count, "width": master_width,
                "height": master_height, "fps": fps, "background": background_info,
                "composition": {"operation": analysis["transform"]["operation"],
                                "source_rect": analysis["transform"].get("source_rect", [0, 0, master_width, master_height]),
                                "quality_review_required": quality["review_required"]},
                "frames": records}
    manifest_path = frame_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    archive = task_folder / "transparent_frames_20.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as output:
        for record in records:
            output.write(frame_dir / record["file"], record["file"])
        output.write(manifest_path, manifest_path.name)
    return {"archive": str(archive), "frame_count": count, "size": [master_width, master_height],
            "background": background_info}
