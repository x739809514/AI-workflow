#!/usr/bin/env python3
"""Build a Dify 1.17 workflow DSL for the animation service."""

from pathlib import Path
import yaml


AUTH = "Authorization:Bearer {{#env.ANIMATION_SERVICE_TOKEN#}}"
SERVICE = "http://host.docker.internal:8765"


def node(node_id, title, node_type, x, y, data, height=110):
    return {
        "id": node_id, "type": "custom", "width": 244, "height": height,
        "position": {"x": x, "y": y}, "positionAbsolute": {"x": x, "y": y},
        "sourcePosition": "right", "targetPosition": "left", "selected": False,
        "data": {"title": title, "type": node_type, "desc": "", "selected": False, **data},
    }


def edge(source, target, source_type, target_type, handle="source"):
    return {"id": f"{source}-{handle}-{target}", "source": source, "target": target,
            "sourceHandle": handle, "targetHandle": "target", "type": "custom", "zIndex": 0,
            "data": {"isInIteration": False, "isInLoop": False,
                     "sourceType": source_type, "targetType": target_type}}


def form_field(key, value, field_id):
    return {"id": field_id, "key": key, "type": "text", "value": value}


def http_node(url, fields):
    return {"method": "post", "url": SERVICE + url, "authorization": {"type": "no-auth", "config": None},
            "headers": AUTH, "params": "", "body": {"type": "form-data", "data": fields},
            "timeout": {"max_connect_timeout": 10, "max_read_timeout": 90, "max_write_timeout": 90},
            "retry_config": {"retry_enabled": False, "max_retries": 0, "retry_interval": 100}}


def code_node(code, variables, outputs):
    return {"code_language": "python3", "code": code, "variables": variables,
            "outputs": {key: {"type": kind, "children": None} for key, kind in outputs.items()}}


def var(name, node_id, key):
    return {"variable": name, "value_selector": [node_id, key]}


parse_prepared = '''def main(body: str, status_code: int) -> dict:
    import json
    if status_code != 200:
        try:
            detail = json.loads(body).get("error", "母版解析失败")
        except (ValueError, TypeError, AttributeError):
            detail = "动画处理服务不可用；请检查 animation-normalizer 容器和 8765 端口"
        raise ValueError(f"母版解析请求失败 (HTTP {status_code}): {detail}")
    try:
        data = json.loads(body)
    except (ValueError, TypeError):
        raise ValueError("母版解析服务返回了非 JSON 响应，请检查处理服务日志")
    analysis = data["analysis"]
    return {
        "job_id": data["job_id"],
        "analysis_json": json.dumps(analysis, ensure_ascii=False),
        "review_flag": "review" if data["needs_review"] else "ready",
        "review_message": "请打开预览页检查红色角色框和蓝色聚焦区域；角色框不准时填写 bbox=x,y,宽,高 后重新运行。",
        "preview_url": data["preview_url"],
    }
'''

make_prompt = '''def main(analysis_json: str, action: str) -> dict:
    import json
    analysis = json.loads(analysis_json)
    x, y, w, h = analysis["subject"]["bbox"]
    width, height = analysis["api_size"]
    sx, sy, sw, sh = analysis["transform"]["source_rect"]
    px, py, pw, ph = analysis["transform"]["content_rect"]
    center = (px + (x + w / 2 - sx) * pw / sw) / width
    bottom = (py + (y + h - sy) * ph / sh) / height
    prompt = (
        "Use the input image as the exact first-frame composition reference. "
        f"Action: {action}. "
        "Static locked camera. No zoom, pan, tilt, reframing or perspective change. "
        "Maintain the character's screen-space size and position; keep furniture and background stationary. "
        f"Subject center near {center:.3f} of frame width, bottom near {bottom:.3f} of frame height. "
        "Animate only the requested action."
    )
    return {"prompt": prompt}
'''

parse_submitted = '''def main(body: str, status_code: int) -> dict:
    import json
    if status_code != 202:
        try:
            detail = json.loads(body).get("error", "视频任务提交失败")
        except (ValueError, TypeError, AttributeError):
            detail = "动画处理服务不可用；请检查 animation-normalizer 容器和 8765 端口"
        raise ValueError(f"视频任务提交失败 (HTTP {status_code}): {detail}")
    try:
        data = json.loads(body)
    except (ValueError, TypeError):
        raise ValueError("视频任务服务返回了非 JSON 响应，请检查处理服务日志")
    return {"job_id": data["job_id"], "model": data["model"],
            "preview_url": data["preview_url"],
            "frames_url": data["frames_url"],
            "message": "任务已提交。打开进度页，完成后下载视频和 20 帧透明 PNG 图片包。"}
'''

frame_package_link = '''def main(frames_url: str) -> dict:
    return {"frames_url": frames_url,
            "message": "图片包将在生成、恢复画布、抽取 20 帧及去背景完成后可下载。"}
'''


nodes = [
    node("start", "上传母版与动作", "start", 30, 240, {"variables": [
        {"variable": "master", "label": "母版图片", "type": "file", "required": True,
         "allowed_file_types": ["image"], "allowed_file_upload_methods": ["local_file", "remote_url"],
         "allowed_file_extensions": [".png", ".jpg", ".jpeg", ".webp"]},
        {"variable": "action", "label": "动作描述", "type": "paragraph", "required": True,
         "max_length": 2000, "options": []},
        {"variable": "model", "label": "视频模型", "type": "select", "required": True,
         "default": "runway_gen45", "options": ["runway_gen45", "minimax_h3_768p", "minimax_h3_2k"]},
        {"variable": "bbox", "label": "角色框（可选：x,y,宽,高）", "type": "text-input",
         "required": False, "default": "", "max_length": 80, "options": []},
        {"variable": "composition_mode", "label": "构图适配", "type": "select", "required": True,
         "default": "auto", "options": ["auto", "full_frame", "focus"]},
    ]}, 230),
    node("prepare", "解析母版并适配画布", "http-request", 334, 240,
         http_node("/prepare", [
             {"id": "master-file", "key": "master", "type": "file", "file": ["start", "master"], "value": ""},
             form_field("bbox", "{{#start.bbox#}}", "bbox-field"),
             form_field("composition_mode", "{{#start.composition_mode#}}", "composition-mode-field"),
         ])),
    node("parse", "读取角色框与补边记录", "code", 638, 240,
         code_node(parse_prepared, [var("body", "prepare", "body"),
                                    var("status_code", "prepare", "status_code")],
                   {"job_id": "string", "analysis_json": "string", "review_flag": "string",
                    "review_message": "string",
                    "preview_url": "string"})),
    node("review", "母版是否需要复核", "if-else", 942, 240, {"cases": [{
        "case_id": "review", "id": "review", "logical_operator": "and", "conditions": [{
            "id": "review-condition", "variable_selector": ["parse", "review_flag"],
            "comparison_operator": "is", "value": "review", "varType": "string"}]}]}),
    node("review_end", "复核角色框", "end", 1246, 60, {"outputs": [
        {"variable": "review_job_id", "value_selector": ["parse", "job_id"], "value_type": "string"},
        {"variable": "review_preview_url", "value_selector": ["parse", "preview_url"], "value_type": "string"},
        {"variable": "review_message", "value_selector": ["parse", "review_message"], "value_type": "string"},
    ]}),
    node("prompt", "固定构图提示词", "code", 1246, 300,
         code_node(make_prompt, [var("analysis_json", "parse", "analysis_json"),
                                 var("action", "start", "action")], {"prompt": "string"})),
    node("generate", "按所选模型生成并启动校准", "http-request", 1550, 300,
         http_node("/generate", [
             form_field("job_id", "{{#parse.job_id#}}", "job-id-field"),
             form_field("action", "{{#start.action#}}", "action-field"),
             form_field("prompt", "{{#prompt.prompt#}}", "prompt-field"),
             form_field("model", "{{#start.model#}}", "model-field"),
             form_field("runway_api_key", "{{#env.RUNWAY_API_KEY#}}", "runway-key-field"),
             form_field("minimax_api_key", "{{#env.MINIMAX_API_KEY#}}", "minimax-key-field"),
             form_field("minimax_api_base", "{{#env.MINIMAX_API_BASE#}}", "minimax-base-field"),
         ])),
    node("result", "读取任务地址", "code", 1854, 300,
         code_node(parse_submitted, [var("body", "generate", "body"),
                                     var("status_code", "generate", "status_code")],
                   {"job_id": "string", "model": "string", "preview_url": "string", "frames_url": "string",
                    "message": "string"})),
    node("frame_link", "20 帧去背景图片包地址", "code", 2158, 300,
         code_node(frame_package_link, [var("frames_url", "result", "frames_url")],
                   {"frames_url": "string", "message": "string"})),
    node("end", "生成任务与结果入口", "end", 2462, 300, {"outputs": [
        {"variable": "job_id", "value_selector": ["result", "job_id"], "value_type": "string"},
        {"variable": "model", "value_selector": ["result", "model"], "value_type": "string"},
        {"variable": "preview_url", "value_selector": ["result", "preview_url"], "value_type": "string"},
        {"variable": "frames_url", "value_selector": ["frame_link", "frames_url"], "value_type": "string"},
        {"variable": "message", "value_selector": ["result", "message"], "value_type": "string"},
    ]}),
]

edges = [
    edge("start", "prepare", "start", "http-request"),
    edge("prepare", "parse", "http-request", "code"),
    edge("parse", "review", "code", "if-else"),
    edge("review", "review_end", "if-else", "end", "review"),
    edge("review", "prompt", "if-else", "code", "false"),
    edge("prompt", "generate", "code", "http-request"),
    edge("generate", "result", "http-request", "code"),
    edge("result", "frame_link", "code", "code"),
    edge("frame_link", "end", "code", "end"),
]

dsl = {
    "app": {"description": "母版解析、可选 Runway Gen-4.5 或 MiniMax H3、画布恢复与角色校准",
            "icon": "🎞️", "icon_background": "#E4F2FF", "mode": "workflow",
            "name": "角色动画统一生成与校准", "use_icon_as_answer_icon": False},
    "dependencies": [], "kind": "app", "version": "0.3.1",
    "workflow": {"conversation_variables": [], "environment_variables": [{
        "id": "animation-service-token", "name": "ANIMATION_SERVICE_TOKEN", "value": "",
        "value_type": "secret", "description": "动画处理服务的访问令牌"}, {
        "id": "runway-api-key", "name": "RUNWAY_API_KEY", "value": "",
        "value_type": "secret", "description": "Runway Dev API 密钥"}, {
        "id": "minimax-api-key", "name": "MINIMAX_API_KEY", "value": "",
        "value_type": "secret", "description": "MiniMax 按量计费 API 密钥"}, {
        "id": "minimax-api-base", "name": "MINIMAX_API_BASE", "value": "https://api.minimax.io",
        "value_type": "string", "description": "国际版 https://api.minimax.io；中国版 https://api.minimax.cn"}],
        "features": {"file_upload": {"enabled": True, "allowed_file_types": ["image"],
                                      "allowed_file_upload_methods": ["local_file", "remote_url"],
                                      "image": {"enabled": True, "number_limits": 1,
                                                "transfer_methods": ["local_file", "remote_url"]}},
                     "opening_statement": "", "retriever_resource": {"enabled": False},
                     "sensitive_word_avoidance": {"enabled": False},
                     "speech_to_text": {"enabled": False}, "suggested_questions": [],
                     "suggested_questions_after_answer": {"enabled": False},
                     "text_to_speech": {"enabled": False}},
        "graph": {"edges": edges, "nodes": nodes,
                  "viewport": {"x": 0, "y": 0, "zoom": 0.7}}},
}

Path("dify_workflow.yml").write_text(yaml.safe_dump(dsl, allow_unicode=True, sort_keys=False), encoding="utf-8")
print("dify_workflow.yml")
