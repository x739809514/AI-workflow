"""Provider-specific image-to-video API contracts and model choices."""

from __future__ import annotations

from dataclasses import dataclass
import json
import urllib.error
import urllib.request


RUNWAY_BASE = "https://api.dev.runwayml.com/v1"
RUNWAY_VERSION = "2024-11-06"
RUNWAY_RATIOS = {"1280:720", "720:1280", "1584:672", "1104:832",
                 "832:1104", "672:1584", "960:960"}
MINIMAX_BASES = {"https://api.minimax.io", "https://api.minimax.cn"}


@dataclass(frozen=True)
class ModelSpec:
    choice: str
    label: str
    provider: str
    model_id: str
    duration: int
    image_size: tuple[int, int]
    ratio: str
    resolution: str | None = None

    @property
    def key_name(self) -> str:
        return "RUNWAY_API_KEY" if self.provider == "runway" else "MINIMAX_API_KEY"


MODELS = {
    model.choice: model for model in (
        ModelSpec("runway_gen45", "Runway Gen-4.5", "runway", "gen4.5", 5,
                  (720, 1280), "720:1280"),
        ModelSpec("minimax_h3_768p", "MiniMax H3 · 768P", "minimax", "MiniMax-H3", 5,
                  (720, 1280), "adaptive", "768P"),
        ModelSpec("minimax_h3_2k", "MiniMax H3 · 2K", "minimax", "MiniMax-H3", 5,
                  (720, 1280), "adaptive", "2K"),
    )
}


def model_for(choice: str | None) -> ModelSpec:
    choice = choice or "runway_gen45"
    if choice not in MODELS:
        raise ValueError(f"不支持的视频模型：{choice}；请选择 {', '.join(MODELS)}")
    return MODELS[choice]


def minimax_base(value: str | None) -> str:
    base = (value or "https://api.minimax.io").rstrip("/")
    if base not in MINIMAX_BASES:
        raise ValueError("MINIMAX_API_BASE 只支持 https://api.minimax.io 或 https://api.minimax.cn")
    return base


def api_request(method: str, url: str, key: str, provider: str,
                payload: dict | None = None) -> dict:
    body = json.dumps(payload).encode("utf-8") if payload is not None else None
    request = urllib.request.Request(url, data=body, method=method)
    request.add_header("Authorization", f"Bearer {key}")
    request.add_header("Content-Type", "application/json")
    if provider == "runway":
        request.add_header("X-Runway-Version", RUNWAY_VERSION)
    try:
        with urllib.request.urlopen(request, timeout=45) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:1200]
        name = "Runway" if provider == "runway" else "MiniMax"
        if exc.code == 401:
            raise RuntimeError(f"{name} API 鉴权失败 (401)：请检查对应的 API Key。{detail}") from exc
        if exc.code == 402:
            raise RuntimeError(f"{name} API 余额不足或未开通按量计费 (402)：{detail}") from exc
        if exc.code == 429:
            raise RuntimeError(f"{name} API 请求过于频繁 (429)：{detail}") from exc
        raise RuntimeError(f"{name} API HTTP {exc.code}: {detail}") from exc


def create_task(spec: ModelSpec, prompt: str, image_data_uri: str, key: str,
                api_base: str | None = None, dry_run: bool = False) -> tuple[dict, dict]:
    if spec.provider == "runway":
        if len(image_data_uri) > 5_000_000:
            raise ValueError("适配后的首帧超过 Runway 的 5 MB data URI 限制")
        payload = {"model": spec.model_id, "promptImage": image_data_uri,
                   "promptText": prompt, "ratio": spec.ratio, "duration": spec.duration}
        url = RUNWAY_BASE + "/image_to_video"
    else:
        base = minimax_base(api_base)
        payload = {"model": spec.model_id,
                   "content": [{"type": "text", "text": prompt},
                               {"type": "image_url", "image_url": {"url": image_data_uri},
                                "role": "first_frame"}],
                   "resolution": spec.resolution, "duration": spec.duration,
                   "ratio": "adaptive"}
        if len(json.dumps(payload).encode("utf-8")) > 64_000_000:
            raise ValueError("MiniMax 请求超过 64 MB 上限")
        url = base + "/v2/video_generation"
    summary = {"model_choice": spec.choice, "model": spec.model_id,
               "provider": spec.provider, "ratio": spec.ratio,
               "resolution": spec.resolution, "duration": spec.duration,
               "request_bytes": len(json.dumps(payload).encode("utf-8"))}
    if dry_run:
        return {}, summary
    result = api_request("POST", url, key, spec.provider, payload)
    task_id = result.get("id") if spec.provider == "runway" else result.get("task_id")
    status = result.get("status", "PENDING") if spec.provider == "runway" else "queued"
    if not task_id:
        raise RuntimeError(f"{spec.label} API 未返回任务 ID：{result}")
    task = {"id": str(task_id), "status": status, "model_choice": spec.choice,
            "model": spec.model_id, "provider": spec.provider}
    if spec.provider == "minimax":
        task["api_base"] = minimax_base(api_base)
        task["resolution"] = spec.resolution
    return task, summary


def query_task(task: dict, key: str) -> tuple[str, str | None, str | None]:
    provider = task.get("provider", "runway")
    if provider == "runway":
        result = api_request("GET", f"{RUNWAY_BASE}/tasks/{task['id']}", key, provider)
        status = result.get("status", "UNKNOWN")
        if status == "SUCCEEDED":
            urls = result.get("output") or []
            return status, urls[0] if urls else None, None
        if status in {"FAILED", "CANCELED"}:
            return status, None, f"{result.get('failureCode') or ''} {result.get('failure') or ''}"
        return status, None, None

    base = minimax_base(task.get("api_base"))
    result = api_request("GET", f"{base}/v2/query/video_generation/{task['id']}", key, provider)
    if result.get("error"):
        raise RuntimeError(f"MiniMax 查询任务失败：{result['error']}")
    item = result.get("task") or {}
    if not item:
        raise RuntimeError(f"MiniMax 查询任务未返回 task：{result}")
    status = str(item.get("status", "unknown")).lower()
    if status == "succeeded":
        if item.get("ratio") not in (None, "9:16"):
            raise RuntimeError(f"MiniMax 输出比例为 {item['ratio']}，与预处理的 9:16 画布不匹配")
        return "SUCCEEDED", (item.get("content") or {}).get("url"), None
    if status in {"failed", "cancelled"}:
        return status.upper(), None, str(item.get("failure_reason") or item.get("error") or result.get("error") or "")
    return status.upper(), None, None
