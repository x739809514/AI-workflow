#!/usr/bin/env python3
"""Master image -> selected video model -> video normalization workflow."""

from __future__ import annotations

import argparse
import base64
from datetime import datetime, timezone
from dataclasses import replace
import json
import os
import shutil
import sys
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps
from video_providers import MODELS, RUNWAY_RATIOS, create_task, model_for, query_task


def size_arg(value: str) -> tuple[int, int]:
    try:
        width, height = (int(part) for part in value.lower().split("x"))
        if width < 2 or height < 2:
            raise ValueError
        return width, height
    except ValueError as exc:
        raise argparse.ArgumentTypeError("尺寸格式应为 宽x高，例如 720x1280") from exc


def bbox_arg(value: str) -> list[int]:
    try:
        box = [int(part) for part in value.split(",")]
        if len(box) != 4 or min(box[2:]) <= 0:
            raise ValueError
        return box
    except ValueError as exc:
        raise argparse.ArgumentTypeError("角色框格式应为 x,y,宽,高") from exc


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: dict) -> None:
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                     prefix=f".{path.name}.", suffix=".tmp", delete=False) as handle:
        temp_path = Path(handle.name)
        handle.write(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    os.replace(temp_path, path)


def detect_subject(image: Image.Image, manual: list[int] | None) -> dict:
    width, height = image.size
    if manual:
        x, y, w, h = manual
        if x < 0 or y < 0 or x + w > width or y + h > height:
            raise ValueError("--bbox 必须位于母版图范围内")
        too_broad = (w * h > width * height * 0.85 or
                     (w > width * 0.95 and h > height * 0.95))
        sparse_top = False
        if too_broad:
            rgba = np.asarray(image.convert("RGBA"))
            alpha = rgba[:, :, 3]
            if np.any(alpha < 250) and np.any(alpha > 10):
                foreground = alpha > 10
            else:
                rgb = rgba[:, :, :3].astype(np.float32)
                edge = np.concatenate([rgb[0], rgb[-1], rgb[:, 0], rgb[:, -1]])
                background = np.median(edge, axis=0)
                foreground = np.linalg.norm(rgb - background, axis=2) > 24
            half = max(1, height // 2)
            sparse_top = (float(np.mean(foreground[:half])) < 0.05 and
                          float(np.mean(foreground[half:])) > 0.2)
        return {"bbox": manual, "method": "manual", "confidence": 0.3 if sparse_top else 1.0,
                "needs_review": sparse_top,
                "warning": "上半部主要是细线；请只框住角色及会移动的道具" if sparse_top else ""}

    rgba = np.asarray(image.convert("RGBA"))
    alpha = rgba[:, :, 3]
    if np.any(alpha < 250) and np.any(alpha > 10):
        mask = alpha > 10
        method = "alpha"
        confidence = 0.95
    else:
        rgb = rgba[:, :, :3].astype(np.float32)
        edge = np.concatenate([rgb[0], rgb[-1], rgb[:, 0], rgb[:, -1]])
        background = np.median(edge, axis=0)
        variation = float(np.median(np.linalg.norm(edge - background, axis=1)))
        diff = np.linalg.norm(rgb - background, axis=2)
        mask = diff > max(24.0, variation * 2.5)
        method = "border_color"
        confidence = 0.75 if variation < 12 else 0.35

    binary = (mask.astype(np.uint8) * 255)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    candidates = [i for i in range(1, count) if stats[i, cv2.CC_STAT_AREA] >= width * height * 0.004]
    if not candidates:
        return {"bbox": [0, 0, width, height], "method": "full_frame_fallback", "confidence": 0.0, "needs_review": True}
    # Keep major connected regions; a hand or tool may be detached from the torso.
    largest = max(stats[i, cv2.CC_STAT_AREA] for i in candidates)
    selected = [i for i in candidates if stats[i, cv2.CC_STAT_AREA] >= largest * 0.08]
    pixels = np.isin(labels, selected)
    ys, xs = np.where(pixels)
    x0, x1, y0, y1 = int(xs.min()), int(xs.max() + 1), int(ys.min()), int(ys.max() + 1)
    box = [x0, y0, x1 - x0, y1 - y0]
    touches_edge = x0 == 0 or y0 == 0 or x1 == width or y1 == height
    area_fraction = (x1 - x0) * (y1 - y0) / (width * height)
    if touches_edge or area_fraction > 0.9 or area_fraction < 0.03:
        confidence = min(confidence, 0.3)
    return {"bbox": box, "method": method, "confidence": confidence, "needs_review": confidence < 0.8}


def focus_rect(image_size: tuple[int, int], box: list[int], target: tuple[int, int]) -> list[int] | None:
    """Choose a safe, aspect-matched view around the subject with generous motion room."""
    src_w, src_h = image_size
    x, y, width, height = box
    ratio = target[0] / target[1]
    view_w = max(width * 1.4, height * 1.4 * ratio)
    view_h = view_w / ratio
    if view_w > src_w or view_h > src_h:
        return None
    view_w = min(src_w, max(2, round(view_w)))
    view_h = min(src_h, max(2, round(view_w / ratio)))
    left = max(0, min(src_w - view_w, round(x + width / 2 - view_w / 2)))
    top = max(0, min(src_h - view_h, round(y + height / 2 - view_h / 2)))
    return [left, top, view_w, view_h]


def make_api_image(image: Image.Image, target: tuple[int, int], box: list[int],
                   mode: str = "auto") -> tuple[Image.Image, dict]:
    src_w, src_h = image.size
    dst_w, dst_h = target
    if mode not in {"auto", "full_frame", "focus"}:
        raise ValueError("composition_mode 须为 auto、full_frame 或 focus")
    source_rect = [0, 0, src_w, src_h]
    candidate = focus_rect(image.size, box, target) if mode != "full_frame" else None
    full_scale = min(dst_w / src_w, dst_h / src_h)
    if candidate is not None:
        focused_scale = min(dst_w / candidate[2], dst_h / candidate[3])
        if mode == "focus" or focused_scale >= full_scale * 1.15:
            source_rect = candidate
    source_x, source_y, source_w, source_h = source_rect
    scale = min(dst_w / source_w, dst_h / source_h)
    resized_w = max(1, round(source_w * scale))
    resized_h = max(1, round(source_h * scale))
    left = (dst_w - resized_w) // 2
    top = (dst_h - resized_h) // 2
    rgb = image.convert("RGB")
    sample = np.asarray(rgb)
    border = np.concatenate([sample[0], sample[-1], sample[:, 0], sample[:, -1]])
    fill = tuple(int(v) for v in np.median(border, axis=0))
    canvas = Image.new("RGB", target, fill)
    crop = rgb.crop((source_x, source_y, source_x + source_w, source_y + source_h))
    canvas.paste(crop.resize((resized_w, resized_h), Image.Resampling.LANCZOS), (left, top))
    transform = {
        "scale_x": resized_w / source_w,
        "scale_y": resized_h / source_h,
        "source_rect": source_rect,
        "content_rect": [left, top, resized_w, resized_h],
        "padding": [left, top, dst_w - left - resized_w, dst_h - top - resized_h],
        "operation": "focus_crop_and_resize" if source_rect != [0, 0, src_w, src_h] else "resize_and_pad",
        "requested_mode": mode,
        "scale_gain": round(scale / full_scale, 3),
        "background_rgb": list(fill),
    }
    return canvas, transform


def preview_image(image: Image.Image, box: list[int], path: Path,
                  source_rect: list[int] | None = None) -> None:
    from PIL import ImageDraw

    preview = image.convert("RGB").copy()
    draw = ImageDraw.Draw(preview)
    x, y, w, h = box
    draw.rectangle((x, y, x + w, y + h), outline="#ff4030", width=max(2, image.width // 250))
    if source_rect and source_rect != [0, 0, image.width, image.height]:
        sx, sy, sw, sh = source_rect
        draw.rectangle((sx, sy, sx + sw, sy + sh), outline="#31a9ff", width=max(2, image.width // 250))
    preview.save(path)


def prepare(args: argparse.Namespace) -> None:
    image_path = Path(args.image).expanduser().resolve()
    if not image_path.is_file():
        raise FileNotFoundError(image_path)
    name = args.name or image_path.stem
    folder = Path(args.workdir).expanduser().resolve() / name
    folder.mkdir(parents=True, exist_ok=True)
    with Image.open(image_path) as raw:
        image = ImageOps.exif_transpose(raw).copy()
    subject = detect_subject(image, args.bbox)
    api_image, transform = make_api_image(image, args.api_size, subject["bbox"],
                                          getattr(args, "composition_mode", "auto"))
    api_image.save(folder / "api_input.png")
    preview_image(image, subject["bbox"], folder / "preview.png", transform["source_rect"])
    analysis = {
        "version": 2,
        "master_path": str(image_path),
        "master_size": list(image.size),
        "api_size": list(args.api_size),
        "output_size": list(args.output_size),
        "subject": subject,
        "transform": transform,
        "target_subject_height_fraction": args.target_height,
        "target_subject_bottom_fraction": args.anchor_y,
        "api_input_path": str(folder / "api_input.png"),
    }
    write_json(folder / "analysis.json", analysis)
    print(json.dumps({"folder": str(folder), "subject": subject, "api_padding": transform["padding"]}, ensure_ascii=False, indent=2))


def prompt_for(action: str, analysis: dict) -> str:
    box = analysis["subject"]["bbox"]
    sx, sy, sw, sh = analysis["transform"].get("source_rect", [0, 0, *analysis["master_size"]])
    px, py, pw, ph = analysis["transform"]["content_rect"]
    api_w, api_h = analysis["api_size"]
    center_x = (px + (box[0] + box[2] / 2 - sx) * pw / sw) / api_w
    bottom_y = (py + (box[1] + box[3] - sy) * ph / sh) / api_h
    return (
        "Use the input image as the first frame and composition reference. "
        f"Action: {action.strip()}\n"
        "Keep a static locked camera. No zoom, pan, tilt, reframing, camera shake, or perspective change. "
        "Keep the character at the same screen-space scale and the background layout unchanged. "
        f"In the input frame the main subject center is near {center_x:.3f} of frame width "
        f"and its bottom is near {bottom_y:.3f} of frame height; maintain this framing. "
        "Animate only the requested action. Keep fixed furniture and background elements stationary."
    )


def submit(args: argparse.Namespace) -> None:
    master_folder = Path(args.folder).expanduser().resolve()
    analysis = read_json(master_folder / "analysis.json")
    spec = model_for(getattr(args, "model", None))
    api_w, api_h = analysis["api_size"]
    requested_ratio = getattr(args, "ratio", None)
    if spec.provider == "runway":
        ratio = requested_ratio or spec.ratio
        if ratio not in RUNWAY_RATIOS:
            raise ValueError(f"{spec.label} 不支持输出比例 {ratio}")
        ratio_w, ratio_h = size_arg(ratio.replace(":", "x"))
        if abs(api_w / api_h - ratio_w / ratio_h) > 0.005:
            raise ValueError("--ratio 与 prepare 的 --api-size 比例不一致")
        spec = replace(spec, ratio=ratio)
    elif (api_w, api_h) != spec.image_size:
        raise ValueError(f"{spec.label} 的预处理图片应为 {spec.image_size[0]}×{spec.image_size[1]}；请重新 prepare")
    elif requested_ratio and requested_ratio != "adaptive":
        raise ValueError(f"{spec.label} 的图生视频输出比例由首帧决定；请使用 adaptive")
    duration = getattr(args, "duration", None) or spec.duration
    if spec.provider == "runway" and not 2 <= duration <= 10:
        raise ValueError("Runway Gen-4.5 的时长须为 2 到 10 秒")
    if spec.provider == "minimax" and not 4 <= duration <= 15:
        raise ValueError("MiniMax H3 的时长须为 4 到 15 秒")
    spec = replace(spec, duration=duration)
    if analysis["subject"]["needs_review"] and not (args.dry_run or args.allow_unreviewed):
        raise RuntimeError("母版角色框需要复核；请用 prepare --bbox 修正，或明确指定 --allow-unreviewed")
    key = getattr(args, "api_key", None) or os.environ.get(spec.key_name)
    if not args.dry_run and not key:
        raise RuntimeError(f"缺少 {spec.key_name} 环境变量")
    source_image = master_folder / "api_input.png"
    job_name = args.name or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    if not job_name or job_name in {".", ".."} or any(c in job_name for c in "/\\"):
        raise ValueError("--name 只能是单层目录名")
    folder = master_folder / "jobs" / job_name
    if folder.exists():
        raise RuntimeError(f"动作任务目录已存在: {folder}，请换一个 --name")
    folder.mkdir(parents=True)
    shutil.copy2(master_folder / "analysis.json", folder / "analysis.json")
    shutil.copy2(master_folder / "api_input.png", folder / "api_input.png")
    prompt = args.prompt or prompt_for(args.action, analysis)
    encoded = "data:image/png;base64," + base64.b64encode(source_image.read_bytes()).decode("ascii")
    api_base = getattr(args, "api_base", None) or os.environ.get("MINIMAX_API_BASE")
    task, summary = create_task(spec, prompt, encoded, key or "", api_base, dry_run=args.dry_run)
    # Keep the image and credentials out of the review file.
    write_json(folder / "request_summary.json", {**summary, "prompt": prompt,
                                                   "image": str(source_image)})
    if args.dry_run:
        print(f"Dry run: {folder / 'request_summary.json'}; API request not sent")
        return
    write_json(folder / "task.json", task)
    print(f"已提交 {spec.label} 任务 {task['id']}。运行 poll 查询并下载结果。")


def poll(args: argparse.Namespace) -> None:
    folder = Path(args.folder).expanduser().resolve()
    task = read_json(folder / "task.json")
    spec = model_for(task.get("model_choice", "runway_gen45"))
    key = getattr(args, "api_key", None) or os.environ.get(spec.key_name)
    if not key:
        raise RuntimeError(f"缺少 {spec.key_name} 环境变量")
    status, url, failure = query_task(task, key)
    task.update({"status": status, "failure": failure})
    write_json(folder / "task.json", task)
    on_status = getattr(args, "on_status", None)
    if callable(on_status):
        on_status(status)
    if status == "SUCCEEDED":
        if not url:
            raise RuntimeError(f"{spec.label} 任务成功，但没有返回视频 URL")
        output = folder / "generated.mp4"
        if not output.exists():
            urllib.request.urlretrieve(url, output)
        print(f"下载完成: {output}")
    elif status in {"FAILED", "CANCELED"}:
        raise RuntimeError(f"{spec.label} 任务 {status}: {failure or ''}")
    else:
        print(f"任务状态: {status}。稍后再次运行 poll。")


def load_master_frame(analysis: dict) -> np.ndarray:
    with Image.open(analysis["master_path"]) as image:
        master = ImageOps.exif_transpose(image).convert("RGB")
        if master.size != tuple(analysis["master_size"]):
            raise RuntimeError("母版尺寸与预处理记录不一致")
        return cv2.cvtColor(np.asarray(master), cv2.COLOR_RGB2BGR)


def crop_to_master(frame: np.ndarray, analysis: dict,
                   master_frame: np.ndarray | None = None) -> np.ndarray:
    api_w, api_h = analysis["api_size"]
    left, top, content_w, content_h = analysis["transform"]["content_rect"]
    frame_h, frame_w = frame.shape[:2]
    x0 = max(0, round(left / api_w * frame_w))
    y0 = max(0, round(top / api_h * frame_h))
    x1 = min(frame_w, round((left + content_w) / api_w * frame_w))
    y1 = min(frame_h, round((top + content_h) / api_h * frame_h))
    if x1 <= x0 or y1 <= y0:
        raise RuntimeError("视频画面与预处理记录不兼容")
    master_w, master_h = analysis["master_size"]
    sx, sy, sw, sh = analysis["transform"].get("source_rect", [0, 0, master_w, master_h])
    if not (0 <= sx < master_w and 0 <= sy < master_h and
            0 < sw <= master_w - sx and 0 < sh <= master_h - sy):
        raise RuntimeError("裁切区域超出母版画布")
    patch = cv2.resize(frame[y0:y1, x0:x1], (sw, sh), interpolation=cv2.INTER_CUBIC)
    if [sx, sy, sw, sh] == [0, 0, master_w, master_h]:
        return patch
    output = (master_frame if master_frame is not None else load_master_frame(analysis)).copy()
    # Blend at the crop boundary so a small model background-color shift does not
    # leave a rectangular seam on the restored master canvas.
    feather = min(24, max(4, min(sw, sh) // 40))
    xx = np.minimum(np.arange(sw), np.arange(sw)[::-1]).astype(np.float32)
    yy = np.minimum(np.arange(sh), np.arange(sh)[::-1]).astype(np.float32)
    opacity = np.minimum(1.0, np.minimum(yy[:, None], xx[None, :]) / feather)[:, :, None]
    base = output[sy:sy + sh, sx:sx + sw]
    output[sy:sy + sh, sx:sx + sw] = np.clip(base * (1.0 - opacity) + patch * opacity, 0, 255).astype(np.uint8)
    return output


def alignment_for_first_frame(first_frame: np.ndarray, analysis: dict) -> tuple[np.ndarray, dict]:
    observed = detect_subject(Image.fromarray(cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)), None)
    reference = analysis["subject"]["bbox"]
    observed_box = observed["bbox"]
    identity = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    if analysis["transform"].get("operation") == "focus_crop_and_resize":
        return identity, {"applied": True, "scale": 1.0, "offset": [0.0, 0.0],
                          "reason": "已按聚焦裁切记录映射回母版；不移动母版静态区域", "detected": observed}
    if observed["confidence"] < 0.6 or reference[3] == 0:
        return identity, {"applied": False, "reason": "首帧角色检测置信度不足", "detected": observed}
    scale = reference[3] / observed_box[3]
    dx = reference[0] + reference[2] / 2 - (observed_box[0] + observed_box[2] / 2) * scale
    dy = reference[1] + reference[3] - (observed_box[1] + observed_box[3]) * scale
    width, height = analysis["master_size"]
    if not (0.8 <= scale <= 1.25 and abs(dx) <= width * 0.15 and abs(dy) <= height * 0.15):
        return identity, {"applied": False, "reason": "偏差过大，需人工检查", "detected": observed,
                          "proposed_scale": round(scale, 4), "proposed_offset": [round(dx, 2), round(dy, 2)]}
    matrix = np.array([[scale, 0.0, dx], [0.0, scale, dy]], dtype=np.float32)
    return matrix, {"applied": True, "scale": round(scale, 4), "offset": [round(dx, 2), round(dy, 2)],
                    "detected": observed}


def align_frame(frame: np.ndarray, matrix: np.ndarray, analysis: dict) -> np.ndarray:
    width, height = analysis["master_size"]
    bg = tuple(analysis["transform"]["background_rgb"][::-1])
    return cv2.warpAffine(frame, matrix, (width, height), flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=bg)


def project_frame(frame: np.ndarray, analysis: dict) -> np.ndarray:
    out_w, out_h = analysis["output_size"]
    x, y, w, h = analysis["subject"]["bbox"]
    target_h = analysis["target_subject_height_fraction"] * out_h
    scale = target_h / h
    scaled_w = max(1, round(frame.shape[1] * scale))
    scaled_h = max(1, round(frame.shape[0] * scale))
    scaled = cv2.resize(frame, (scaled_w, scaled_h), interpolation=cv2.INTER_CUBIC)
    target_center_x = out_w / 2
    target_bottom_y = analysis["target_subject_bottom_fraction"] * out_h
    offset_x = round(target_center_x - (x + w / 2) * scale)
    offset_y = round(target_bottom_y - (y + h) * scale)
    # Use the same canvas color on every frame to avoid flashing at the edges.
    bg = np.array(analysis["transform"]["background_rgb"][::-1], dtype=np.uint8)
    output = np.empty((out_h, out_w, 3), dtype=np.uint8)
    output[:] = bg
    dest_x0, dest_y0 = max(0, offset_x), max(0, offset_y)
    dest_x1, dest_y1 = min(out_w, offset_x + scaled_w), min(out_h, offset_y + scaled_h)
    if dest_x1 > dest_x0 and dest_y1 > dest_y0:
        src_x0, src_y0 = dest_x0 - offset_x, dest_y0 - offset_y
        output[dest_y0:dest_y1, dest_x0:dest_x1] = scaled[src_y0:src_y0 + dest_y1 - dest_y0,
                                                          src_x0:src_x0 + dest_x1 - dest_x0]
    return output


def video_writer(path: Path, fps: float, size: tuple[int, int]) -> cv2.VideoWriter:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    if not writer.isOpened():
        raise RuntimeError(f"无法创建视频: {path}")
    return writer


def normalize(args: argparse.Namespace) -> None:
    folder = Path(args.folder).expanduser().resolve()
    analysis = read_json(folder / "analysis.json")
    source = Path(args.video).expanduser().resolve() if args.video else folder / "generated.mp4"
    if not source.is_file():
        raise FileNotFoundError(source)
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise RuntimeError(f"无法读取视频: {source}")
    fps = capture.get(cv2.CAP_PROP_FPS) or 24.0
    master_size = tuple(analysis["master_size"])
    output_size = tuple(analysis["output_size"])
    ok, first_raw = capture.read()
    if not ok:
        capture.release()
        raise RuntimeError("输入视频没有可读取的帧")
    master_frame = load_master_frame(analysis)
    first_unaligned = crop_to_master(first_raw, analysis, master_frame)
    matrix, alignment = alignment_for_first_frame(first_unaligned, analysis)
    restored = video_writer(folder / "restored.mp4", fps, master_size)
    normalized = video_writer(folder / "normalized.mp4", fps, output_size)
    frames = 0
    first_frame = None
    pending = first_raw
    try:
        while True:
            if pending is not None:
                frame = pending
                pending = None
            else:
                ok, frame = capture.read()
                if not ok:
                    break
            restored_frame = align_frame(crop_to_master(frame, analysis, master_frame), matrix, analysis)
            if first_frame is None:
                first_frame = restored_frame.copy()
            restored.write(restored_frame)
            normalized.write(project_frame(restored_frame, analysis))
            frames += 1
    finally:
        capture.release()
        restored.release()
        normalized.release()
    if frames == 0 or first_frame is None:
        raise RuntimeError("输入视频没有可读取的帧")
    cv2.imwrite(str(folder / "restored_first_frame.png"), first_frame)
    difference = float(np.mean(cv2.absdiff(master_frame, first_frame)))
    detected = alignment["detected"]
    reference_box = analysis["subject"]["bbox"]
    bx, by, bw, bh = reference_box
    subject_difference = float(np.mean(cv2.absdiff(
        master_frame[by:by + bh, bx:bx + bw], first_frame[by:by + bh, bx:bx + bw])))
    observed_box = detected["bbox"]
    if detected["confidence"] >= 0.6:
        ref_x = reference_box[0] + reference_box[2] / 2
        ref_y = reference_box[1] + reference_box[3] / 2
        obs_x = observed_box[0] + observed_box[2] / 2
        obs_y = observed_box[1] + observed_box[3] / 2
        displacement = [round((obs_x - ref_x) / master_size[0], 4), round((obs_y - ref_y) / master_size[1], 4)]
        height_difference = round(observed_box[3] / reference_box[3] - 1, 4)
    else:
        displacement = None
        height_difference = None
    report = {
        "source": str(source), "frames": frames, "fps": fps,
        "restored": str(folder / "restored.mp4"), "normalized": str(folder / "normalized.mp4"),
        "first_frame_alignment": alignment,
        "first_frame_mean_absolute_difference": difference,
        "first_frame_subject_difference": subject_difference,
        "generated_region": analysis["transform"].get("source_rect", [0, 0, *master_size]),
        "detected_first_frame_subject": detected,
        "first_frame_center_displacement_fraction": displacement,
        "first_frame_height_difference_fraction": height_difference,
        "review_required": analysis["subject"]["needs_review"] or detected["needs_review"]
                           or (not alignment["applied"])
                           or subject_difference > 25
                           or (displacement is not None and max(abs(v) for v in displacement) > 0.05)
                           or (height_difference is not None and abs(height_difference) > 0.1),
        "note": "自动检测与首帧像素差只能提示漂移；聚焦区域外保留原母版，动作超出区域时须重新选择完整画面；请检查 restored_first_frame.png。",
    }
    write_json(folder / "quality_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def frames(args: argparse.Namespace) -> None:
    from frame_assets import export_frame_assets

    result = export_frame_assets(Path(args.folder).expanduser().resolve())
    print(json.dumps(result, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare", help="分析母版并适配 API 输入")
    prep.add_argument("image")
    prep.add_argument("--name")
    prep.add_argument("--workdir", default="work")
    prep.add_argument("--api-size", type=size_arg, default=(720, 1280))
    prep.add_argument("--output-size", type=size_arg, default=(1080, 1920))
    prep.add_argument("--bbox", type=bbox_arg)
    prep.add_argument("--composition-mode", choices=["auto", "full_frame", "focus"], default="auto")
    prep.add_argument("--target-height", type=float, default=0.70)
    prep.add_argument("--anchor-y", type=float, default=0.85)
    prep.set_defaults(func=prepare)
    submit_cmd = sub.add_parser("submit", help="提交所选视频模型的异步任务")
    submit_cmd.add_argument("folder")
    submit_cmd.add_argument("--action", required=True)
    submit_cmd.add_argument("--prompt", help="由外部工作流生成的完整提示词")
    submit_cmd.add_argument("--name", help="动作任务名，默认使用 UTC 时间")
    submit_cmd.add_argument("--model", choices=list(MODELS), default="runway_gen45")
    submit_cmd.add_argument("--ratio", help="可选；需与所选模型的预处理画布一致")
    submit_cmd.add_argument("--duration", type=int, help="可选；默认 5 秒")
    submit_cmd.add_argument("--api-base", help="MiniMax API 地址：国际版或中国版")
    submit_cmd.add_argument("--dry-run", action="store_true")
    submit_cmd.add_argument("--allow-unreviewed", action="store_true")
    submit_cmd.set_defaults(func=submit)
    poll_cmd = sub.add_parser("poll", help="查询任务并下载视频")
    poll_cmd.add_argument("folder")
    poll_cmd.set_defaults(func=poll)
    norm = sub.add_parser("normalize", help="恢复母版坐标并输出统一画布")
    norm.add_argument("folder")
    norm.add_argument("--video")
    norm.set_defaults(func=normalize)
    frame_cmd = sub.add_parser("frames", help="均匀抽取 20 帧、去除单色背景并打包透明 PNG")
    frame_cmd.add_argument("folder", help="已完成 normalize 的动作任务目录")
    frame_cmd.set_defaults(func=frames)
    args = parser.parse_args()
    if args.command == "prepare" and not (0 < args.target_height <= 1 and 0 < args.anchor_y <= 1):
        parser.error("--target-height 和 --anchor-y 必须位于 0 到 1 之间")
    try:
        args.func(args)
    except (ValueError, FileNotFoundError, RuntimeError, urllib.error.URLError) as exc:
        print(f"错误: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
