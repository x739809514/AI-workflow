#!/usr/bin/env python3
"""HTTP companion for the Dify animation workflow."""

from __future__ import annotations

import argparse
import base64
import cgi
from datetime import datetime, timezone
import hashlib
import html
import json
import os
import re
import secrets
import time
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock
from urllib.parse import parse_qs, urlsplit

from cryptography.fernet import Fernet, InvalidToken

from frame_assets import export_frame_assets
from video_providers import minimax_base, model_for
from workflow import bbox_arg, normalize, poll, prepare, read_json, submit, write_json


env_path = Path(__file__).with_name(".env")
if env_path.is_file():
    for line in env_path.read_text(encoding="utf-8").splitlines():
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key, value)


ROOT = Path(os.environ.get("WORKDIR", "work")).resolve()
TOKEN = os.environ.get("ANIMATION_SERVICE_TOKEN", "")
PUBLIC_BASE = os.environ.get("PUBLIC_BASE", "http://localhost:8765").rstrip("/")
POOL = ThreadPoolExecutor(max_workers=2)
ACTIVE_JOBS: set[str] = set()
ACTIVE_LOCK = Lock()
ID_RE = re.compile(r"[0-9a-f]{24}")


def job_folder(job_id: str) -> Path:
    if not ID_RE.fullmatch(job_id):
        raise ValueError("invalid job id")
    return ROOT / job_id


def status_path(job_id: str) -> Path:
    return job_folder(job_id) / "status.json"


def update_status(job_id: str, stage: str, message: str = "", **extra: object) -> None:
    path = status_path(job_id)
    old = read_json(path) if path.exists() else {}
    now = datetime.now(timezone.utc).isoformat()
    if not old.get("created_at"):
        old["created_at"] = now
    if old.get("stage") != stage:
        old["stage_started_at"] = now
    old.update({"job_id": job_id, "stage": stage, "message": message, **extra})
    write_json(path, old)


def credential_cipher() -> Fernet:
    key = hashlib.pbkdf2_hmac("sha256", TOKEN.encode("utf-8"),
                              b"animation-normalizer-credentials-v1", 200_000, dklen=32)
    return Fernet(base64.urlsafe_b64encode(key))


def save_credential(job_id: str, api_key: str, api_base: str | None) -> None:
    path = job_folder(job_id) / "credential.bin"
    payload = json.dumps({"api_key": api_key, "api_base": api_base}).encode("utf-8")
    path.write_bytes(credential_cipher().encrypt(payload))
    path.chmod(0o600)


def load_credential(job_id: str) -> tuple[str, str | None]:
    path = job_folder(job_id) / "credential.bin"
    data = json.loads(credential_cipher().decrypt(path.read_bytes()))
    return data["api_key"], data.get("api_base")


def queue_job(job_id: str, callback, *args) -> bool:
    with ACTIVE_LOCK:
        if job_id in ACTIVE_JOBS:
            return False
        ACTIVE_JOBS.add(job_id)

    def run() -> None:
        try:
            callback(job_id, *args)
        finally:
            with ACTIVE_LOCK:
                ACTIVE_JOBS.discard(job_id)

    POOL.submit(run)
    return True


def recover_pending_jobs() -> None:
    for path in ROOT.glob("*/status.json"):
        job_id = path.parent.name
        if not ID_RE.fullmatch(job_id):
            continue
        try:
            state = read_json(path)
            if state.get("stage") not in {"queued", "submitting", "generating", "downloading",
                                          "normalizing", "extracting_frames"}:
                continue
            task_path = path.parent / "jobs" / "animation" / "task.json"
            if not task_path.is_file():
                update_status(job_id, "interrupted", "服务重启时任务尚未保存模型 ID；请重新提交动作")
                continue
            if not state.get("generation_started_at"):
                state["generation_started_at"] = datetime.fromtimestamp(task_path.stat().st_mtime, timezone.utc).isoformat()
                write_json(path, state)
            try:
                api_key, api_base = load_credential(job_id)
            except (FileNotFoundError, ValueError, InvalidToken, json.JSONDecodeError):
                update_status(job_id, "interrupted", "服务重启中断轮询；需恢复 API 凭据继续查询现有任务")
                continue
            update_status(job_id, "generating", "服务重启后继续查询现有模型任务")
            queue_job(job_id, finish_existing, api_key, api_base)
        except (OSError, KeyError, json.JSONDecodeError):
            continue


def progress_info(job_id: str, state: dict) -> dict:
    task_path = job_folder(job_id) / "jobs" / "animation" / "task.json"
    model_status = state.get("provider_status")
    last_checked = state.get("last_checked_at")
    if task_path.is_file():
        task = read_json(task_path)
        model_status = task.get("status", model_status)
        if not last_checked:
            last_checked = datetime.fromtimestamp(task_path.stat().st_mtime, timezone.utc).isoformat()
    started = state.get("generation_started_at") or state.get("stage_started_at") or state.get("created_at")
    elapsed = max(0, int(datetime.now(timezone.utc).timestamp() - datetime.fromisoformat(started).timestamp())) if started else 0
    poll_age = max(0, int(datetime.now(timezone.utc).timestamp() - datetime.fromisoformat(last_checked).timestamp())) if last_checked else None
    return {"stage": state["stage"], "model_status": model_status,
            "elapsed_seconds": elapsed, "last_checked_at": last_checked,
            "seconds_since_last_check": poll_age, "poll_count": state.get("poll_count", 0),
            "interrupted": state["stage"] == "interrupted" or
                           (state["stage"] == "generating" and poll_age is not None and poll_age > 90)}


def execute(job_id: str, action: str, prompt: str, model_choice: str,
            api_key: str, api_base: str | None) -> None:
    folder = job_folder(job_id)
    task_folder = folder / "jobs" / "animation"
    try:
        spec = model_for(model_choice)
        update_status(job_id, "submitting", f"正在提交 {spec.label} 任务")
        submit(argparse.Namespace(folder=str(folder), action=action, prompt=prompt or None,
                                  name="animation", ratio=None, duration=None, model=model_choice,
                                  dry_run=False, allow_unreviewed=True, api_key=api_key,
                                  api_base=api_base))
        task = read_json(task_folder / "task.json")
        update_status(job_id, "generating", "视频生成中", task_id=task["id"])
        state = read_json(status_path(job_id))
        state["generation_started_at"] = datetime.now(timezone.utc).isoformat()
        write_json(status_path(job_id), state)
    except Exception as exc:
        failed_at = read_json(status_path(job_id)).get("stage", "unknown")
        update_status(job_id, "failed", str(exc), failed_at=failed_at)
        return
    finish_existing(job_id, api_key, api_base)


def finish_existing(job_id: str, api_key: str, api_base: str | None) -> None:
    task_folder = job_folder(job_id) / "jobs" / "animation"
    try:
        state = read_json(status_path(job_id))
        started = state.get("generation_started_at")
        if not started:
            started = datetime.fromtimestamp(task_folder.stat().st_mtime, timezone.utc).isoformat()
            update_status(job_id, "generating", "已恢复模型任务轮询", generation_started_at=started)
        deadline = datetime.fromisoformat(started).timestamp() + 45 * 60
        checks = int(state.get("poll_count") or 0)
        while True:
            checks += 1

            def record_provider_status(provider_status: str) -> None:
                stage = "downloading" if provider_status == "SUCCEEDED" else "generating"
                message = "模型已完成，正在下载视频" if stage == "downloading" else "视频生成中"
                update_status(job_id, stage, message, provider_status=provider_status,
                              last_checked_at=datetime.now(timezone.utc).isoformat(), poll_count=checks)

            poll(argparse.Namespace(folder=str(task_folder), api_key=api_key,
                                   on_status=record_provider_status))
            task = read_json(task_folder / "task.json")
            if task["status"] == "SUCCEEDED":
                break
            if time.time() >= deadline:
                raise RuntimeError("视频生成超过 45 分钟")
            time.sleep(15)
        update_status(job_id, "normalizing", "正在恢复画布并校准角色")
        normalize(argparse.Namespace(folder=str(task_folder), video=None))
        report = read_json(task_folder / "quality_report.json")
        update_status(job_id, "extracting_frames", "正在抽取 20 帧并去除背景")
        try:
            assets = export_frame_assets(task_folder)
            update_status(job_id, "completed", "视频及 20 帧透明 PNG 图片包已完成",
                          review_required=report["review_required"], quality_report=report,
                          frame_assets=assets)
        except Exception as exc:
            update_status(job_id, "completed", "视频已完成，但图片包生成失败",
                          review_required=True, quality_report=report,
                          frame_assets_error=str(exc))
    except Exception as exc:
        failed_at = read_json(status_path(job_id)).get("stage", "unknown")
        update_status(job_id, "failed", str(exc), failed_at=failed_at)


class Handler(BaseHTTPRequestHandler):
    def reply(self, code: int, data: dict | str | bytes, content_type: str = "application/json") -> None:
        if isinstance(data, dict):
            raw = json.dumps(data, ensure_ascii=False).encode("utf-8")
        elif isinstance(data, str):
            raw = data.encode("utf-8")
        else:
            raw = data
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(raw)

    def authorized(self) -> bool:
        return bool(TOKEN) and secrets.compare_digest(self.headers.get("Authorization", ""), f"Bearer {TOKEN}")

    def form(self) -> cgi.FieldStorage:
        size = int(self.headers.get("Content-Length", "0"))
        if size < 1 or size > 31_000_000:
            raise ValueError("request size must be 1 byte to 31 MB")
        if not self.headers.get("Content-Type", "").startswith("multipart/form-data"):
            raise ValueError("expected multipart/form-data")
        return cgi.FieldStorage(fp=self.rfile, headers=self.headers,
                                environ={"REQUEST_METHOD": "POST", "CONTENT_TYPE": self.headers["Content-Type"],
                                         "CONTENT_LENGTH": str(size)}, keep_blank_values=True)

    def do_POST(self) -> None:
        if not self.authorized():
            return self.reply(401, {"error": "unauthorized"})
        try:
            if self.path == "/prepare":
                fields = self.form()
                if "master" not in fields or not fields["master"].file:
                    raise ValueError("master image is required")
                raw = fields["master"].file.read(30_000_001)
                if not raw or len(raw) > 30_000_000:
                    raise ValueError("master image must be under 30 MB")
                job_id = secrets.token_hex(12)
                folder = job_folder(job_id)
                folder.mkdir(parents=True)
                input_path = folder / "master_upload"
                input_path.write_bytes(raw)
                box_text = fields.getfirst("bbox", "").strip()
                box = bbox_arg(box_text) if box_text else None
                composition_mode = fields.getfirst("composition_mode", "auto").strip() or "auto"
                prepare(argparse.Namespace(image=str(input_path), name=job_id, workdir=str(ROOT),
                                           api_size=(720, 1280), output_size=(1080, 1920), bbox=box,
                                           target_height=0.70, anchor_y=0.85,
                                           composition_mode=composition_mode))
                view_token = secrets.token_urlsafe(24)
                update_status(job_id, "prepared", "母版解析和输入图适配已完成", view_token=view_token)
                analysis = read_json(folder / "analysis.json")
                return self.reply(200, {"job_id": job_id, "analysis": analysis,
                                        "needs_review": analysis["subject"]["needs_review"],
                                        "preview_url": f"{PUBLIC_BASE}/view/{job_id}?t={view_token}"})
            if self.path == "/generate":
                fields = self.form()
                job_id = fields.getfirst("job_id", "")
                action = fields.getfirst("action", "").strip()
                prompt = fields.getfirst("prompt", "").strip()
                model_choice = fields.getfirst("model", "runway_gen45").strip() or "runway_gen45"
                spec = model_for(model_choice)
                api_key = fields.getfirst("runway_api_key" if spec.provider == "runway"
                                          else "minimax_api_key", "").strip()
                api_base = fields.getfirst("minimax_api_base", "").strip() or None
                if not action:
                    raise ValueError("action is required")
                if not api_key:
                    raise ValueError(f"在 Dify 环境变量中设置 {spec.key_name}")
                if spec.provider == "minimax":
                    api_base = minimax_base(api_base)
                state = read_json(status_path(job_id))
                if state["stage"] != "prepared":
                    raise ValueError("job is not in prepared state")
                if read_json(job_folder(job_id) / "analysis.json")["subject"]["needs_review"]:
                    raise ValueError("母版角色框需要复核；请用 bbox 重新提交")
                save_credential(job_id, api_key, api_base)
                update_status(job_id, "queued", f"{spec.label} 已加入生成队列",
                              model_choice=model_choice)
                queue_job(job_id, execute, action, prompt, model_choice, api_key, api_base)
                return self.reply(202, {"job_id": job_id, "status": "queued", "model": spec.label,
                                        "preview_url": f"{PUBLIC_BASE}/view/{job_id}?t={state['view_token']}",
                                        "frames_url": f"{PUBLIC_BASE}/frames/{job_id}?t={state['view_token']}"})
            if self.path == "/resume":
                fields = self.form()
                job_id = fields.getfirst("job_id", "").strip()
                state = read_json(status_path(job_id))
                if state["stage"] not in {"generating", "downloading", "interrupted"}:
                    raise ValueError("任务不在可恢复状态")
                task = read_json(job_folder(job_id) / "jobs" / "animation" / "task.json")
                if not state.get("generation_started_at"):
                    state["generation_started_at"] = datetime.fromtimestamp(
                        (job_folder(job_id) / "jobs" / "animation" / "task.json").stat().st_mtime,
                        timezone.utc).isoformat()
                    write_json(status_path(job_id), state)
                spec = model_for(task["model_choice"])
                api_key = fields.getfirst("runway_api_key" if spec.provider == "runway"
                                          else "minimax_api_key", "").strip()
                if not api_key:
                    raise ValueError(f"缺少 {spec.key_name}")
                api_base = fields.getfirst("minimax_api_base", "").strip() or task.get("api_base")
                save_credential(job_id, api_key, api_base)
                queued = queue_job(job_id, finish_existing, api_key, api_base)
                if queued:
                    update_status(job_id, "generating", "已恢复现有模型任务的轮询")
                return self.reply(202, {"job_id": job_id, "resumed": queued,
                                        "model_status": task["status"]})
            self.reply(404, {"error": "not found"})
        except (ValueError, OSError, KeyError, argparse.ArgumentTypeError) as exc:
            self.reply(400, {"error": str(exc)})

    def do_GET(self) -> None:
        route = urlsplit(self.path)
        if route.path == "/health":
            return self.reply(200, {"status": "ok"})
        match = re.fullmatch(r"/(view|status|download|frame|preview|frames)/([0-9a-f]{24})", route.path)
        if not match:
            return self.reply(404, {"error": "not found"})
        kind, job_id = match.groups()
        try:
            state = read_json(status_path(job_id))
        except (ValueError, FileNotFoundError):
            return self.reply(404, {"error": "not found"})
        supplied = parse_qs(route.query).get("t", [""])[0]
        if not secrets.compare_digest(supplied, state["view_token"]):
            return self.reply(403, {"error": "forbidden"})
        if kind == "status":
            return self.reply(200, {"job_id": job_id, "message": state.get("message", ""),
                                    **progress_info(job_id, state)})
        if kind == "download":
            video = job_folder(job_id) / "jobs" / "animation" / "normalized.mp4"
            if not video.is_file():
                return self.reply(404, {"error": "video not ready"})
            return self.reply(200, video.read_bytes(), "video/mp4")
        if kind == "frames":
            archive = job_folder(job_id) / "jobs" / "animation" / "transparent_frames_20.zip"
            if not archive.is_file():
                return self.reply(404, {"error": "frame package not ready"})
            data = archive.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "application/zip")
            self.send_header("Content-Disposition", f'attachment; filename="animation-{job_id}-20-frames.zip"')
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            return self.wfile.write(data)
        if kind == "frame":
            frame = job_folder(job_id) / "jobs" / "animation" / "restored_first_frame.png"
            if not frame.is_file():
                return self.reply(404, {"error": "frame not ready"})
            return self.reply(200, frame.read_bytes(), "image/png")
        if kind == "preview":
            preview = job_folder(job_id) / "preview.png"
            if not preview.is_file():
                return self.reply(404, {"error": "preview not ready"})
            return self.reply(200, preview.read_bytes(), "image/png")
        token = state["view_token"]
        links = ""
        details = ""
        analysis_path = job_folder(job_id) / "analysis.json"
        if analysis_path.is_file():
            analysis = read_json(analysis_path)
            width, height = analysis["master_size"]
            bbox = analysis["subject"]["bbox"]
            bbox_text = ",".join(str(v) for v in bbox)
            review_text = ("红框是自动识别的角色范围。如果框住了背景或道具，"
                           "请回到 Dify，在 bbox 输入框填写 x,y,宽,高 后重新运行。"
                           if analysis["subject"]["needs_review"] else
                           "红框是本次使用的角色范围。")
            transform = analysis["transform"]
            focus_text = (f"蓝框是送给视频模型的聚焦区域，放大倍数约 {transform['scale_gain']:.2f}；"
                          "蓝框外会保留母版原像素。请检查动作是否可能超出蓝框。"
                          if transform["operation"] == "focus_crop_and_resize" else
                          "本次保留完整母版画面。")
            details = (
                f"<h2>母版解析</h2><p>原图尺寸：{width} × {height} px；"
                f"角色框：<code>{bbox_text}</code>；识别方式："
                f"{html.escape(analysis['subject']['method'])}；置信度："
                f"{analysis['subject']['confidence']:.2f}</p>"
                f"<p>{review_text} {focus_text}</p>"
                f'<img src="/preview/{job_id}?t={token}" alt="母版角色框预览" '
                'style="display:block;max-width:100%;max-height:75vh;background:#222;border:1px solid #aaa">'
            )
        if state["stage"] == "completed":
            links = (f'<p><a href="/download/{job_id}?t={token}">下载统一画布视频</a></p>'
                     f'<p><a href="/frame/{job_id}?t={token}">查看恢复后的首帧</a></p>')
            if (job_folder(job_id) / "jobs" / "animation" / "transparent_frames_20.zip").is_file():
                links += f'<p><a href="/frames/{job_id}?t={token}">下载 20 帧透明 PNG 图片包</a></p>'
            if state.get("frame_assets_error"):
                links += f'<p>图片包生成失败：{html.escape(state["frame_assets_error"])}</p>'
        if state["stage"] == "failed" and state.get("failed_at"):
            links += f"<p>失败阶段：{html.escape(state['failed_at'])}</p>"
        stage_text = "待复核角色框" if state["stage"] == "prepared" and analysis_path.is_file() and analysis["subject"]["needs_review"] else state["stage"]
        progress = progress_info(job_id, state)
        elapsed = progress["elapsed_seconds"]
        model_status = html.escape(str(progress["model_status"] or "等待查询"))
        progress_html = ""
        if state["stage"] in {"generating", "downloading", "interrupted"}:
            progress_html = (f"<h2>生成进度</h2><p>模型任务状态：<strong>{model_status}</strong>；"
                             f"已等待：{elapsed // 60} 分 {elapsed % 60} 秒；"
                             f"查询次数：{progress['poll_count']}</p>")
            if progress["last_checked_at"]:
                last_checked = datetime.fromisoformat(progress["last_checked_at"]).astimezone().strftime("%H:%M:%S")
                progress_html += f"<p>最近查询：{last_checked}；距今 {progress['seconds_since_last_check']} 秒</p>"
            if state["stage"] == "generating" and not progress["interrupted"]:
                progress_html += "<progress aria-label='模型正在生成'></progress><p>模型接口只提供任务状态，没有准确的生成百分比。</p>"
            if progress["interrupted"]:
                progress_html += "<p style='color:#a33'>轮询已中断，正在等待恢复；现有模型任务不会重复提交。</p>"
        elif state["stage"] in {"queued", "submitting", "normalizing", "extracting_frames"}:
            progress_html = f"<h2>处理进度</h2><p>当前阶段：{html.escape(state['stage'])}；已用时：{elapsed // 60} 分 {elapsed % 60} 秒</p>"
        quality = state.get("quality_report")
        quality_html = (f"<pre>{html.escape(json.dumps(quality, ensure_ascii=False, indent=2))}</pre>"
                        if quality else "")
        content = ("<!doctype html><meta charset='utf-8'><meta http-equiv='refresh' content='15'>"
                   "<title>动画任务</title><main style='max-width:800px;margin:48px auto;font:18px sans-serif'>"
                   f"<h1>动画任务 {job_id}</h1><p>状态：{html.escape(stage_text)}</p>"
                   f"<p>{html.escape(state.get('message', ''))}</p>{progress_html}{details}{links}"
                   f"{quality_html}"
                   "</main>")
        self.reply(200, content, "text/html; charset=utf-8")


if __name__ == "__main__":
    if not TOKEN or TOKEN.startswith("replace_"):
        raise SystemExit("Set ANIMATION_SERVICE_TOKEN before starting the service")
    ROOT.mkdir(parents=True, exist_ok=True)
    recover_pending_jobs()
    ThreadingHTTPServer(("0.0.0.0", 8765), Handler).serve_forever()
