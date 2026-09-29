# 角色动画统一生成与校准

将不同尺寸的角色母版送入所选视频模型，再将生成视频映射回母版坐标并输出统一画布。Dify 工作流负责模型选择、上传、复核和提交；本地处理服务负责图片适配、异步查询、视频校准，以及抽取 20 帧透明 PNG 并打包。

## 在 Dify 使用

导入 [dify_workflow.yml](./dify_workflow.yml)。在起始节点的「视频模型」选择 `runway_gen45`、`minimax_h3_768p` 或 `minimax_h3_2k`；「构图适配」通常选 `auto`。随后工作流解析母版、复核角色框、固定构图、调用所选模型并返回进度页和图片包地址。抽帧和去背景在本地服务中异步执行，须在进度页显示完成后下载。

在 Dify Editor 的环境变量中配置：

- `ANIMATION_SERVICE_TOKEN`：填本地 `.env` 中同名的服务令牌。
- `RUNWAY_API_KEY`：填 Runway Dev 的 API 密钥，设为密钥类型。
- `MINIMAX_API_KEY`：填 MiniMax 按量计费 API 密钥，设为密钥类型。仅在选择 H3 时需要。
- `MINIMAX_API_BASE`：国际版用 `https://api.minimax.io`；中国版用 `https://api.minimax.cn`。默认国际版。

本地服务首次运行：

```bash
cp .env.example .env
# 在 .env 中填写 ANIMATION_SERVICE_TOKEN
python3 service.py
```

Dify 容器通过 `host.docker.internal:8765` 访问处理服务；进度页由本机 `localhost:8765` 提供。也可以使用 `docker compose up -d --build`，此时须把 DSL 的服务地址改为 `http://animation-normalizer:8765`。模型密钥在 Dify 的密钥变量中配置，提交时传给本地服务，并用服务令牌派生的密钥加密保存在任务目录，供本地服务重启后继续查询已提交的模型任务；明文密钥不写入状态或日志。

上传母版并填写动作描述。角色自动识别置信度不足时，工作流返回预览页；在预览图检查红色角色框，然后在 `bbox` 输入框填写 `x,y,宽,高` 并重新运行。如果上半部只是吊线，不要把整张图当作角色框；当前木偶母版的参考值是 `105,810,335,480`。如果全身立绘几乎铺满画面，可以填写接近整张图的角色框，例如 238×658 立绘可填 `2,0,235,658`。蓝框是实际送给模型的聚焦区域：若手臂或道具的动作可能超出蓝框，改选 `full_frame`；希望即使放大收益较小仍聚焦时，可选 `focus`。提交后打开进度页；完成后可下载统一画布视频、查看恢复后的首帧，并下载含 20 张透明 PNG 与 `manifest.json` 的 ZIP 图片包。当前最终图片产物是 ZIP，不生成单张 Sprite Sheet。

需要复核母版时，Dify 输出 `review_job_id`、`review_preview_url` 和 `review_message`；正常提交生成任务时，输出 `job_id`、`model`、`preview_url`、`frames_url` 和 `message`。

进度页每 15 秒刷新一次，显示模型的 `PENDING` / `RUNNING` / `SUCCEEDED` 状态、已等待时间、查询次数和最近查询时间。Runway 任务接口不返回可靠的生成百分比，因此模型生成阶段使用不定进度条。进度数据也可通过任务页同一令牌的 `/status/<job_id>?t=<token>` 地址读取。本地服务重启后会自动恢复有加密凭据的任务轮询，不重新提交视频生成请求。

两个模型都使用适配后的 720×1280 PNG 首帧，默认生成 5 秒竖屏视频。Runway Gen-4.5 使用 `720:1280`；MiniMax H3 在图生视频模式使用 `adaptive`，可选择 768P 或 2K。服务按模型调用对应的任务查询接口，完成后立即下载视频，再统一恢复画布和处理图片包。Runway 和 MiniMax 分别计费；H3 需开通按量计费 API。

## 命令行

```bash
python3 -m pip install -r requirements.txt
python3 workflow.py prepare master.png --name chef --api-size 720x1280 --output-size 1080x1920
python3 workflow.py prepare master.png --name chef_focus --bbox 105,810,335,480 --composition-mode auto
python3 workflow.py submit work/chef --name wave --action "角色抬起右手，缓慢挥手，然后回到起始姿势" --dry-run
python3 workflow.py submit work/chef --name wave_h3 --model minimax_h3_768p --action "角色抬起右手，缓慢挥手，然后回到起始姿势" --dry-run
```

检查 `work/chef/analysis.json`、`work/chef/api_input.png` 和 `work/chef/preview.png`。角色框不准确时，重新执行 `prepare` 并指定 `--bbox x,y,width,height`，坐标基于原图。

```bash
export RUNWAY_API_KEY='你的 Runway Dev API 密钥'
python3 workflow.py submit work/chef --name wave_live --action "角色抬起右手，缓慢挥手，然后回到起始姿势"
python3 workflow.py poll work/chef/jobs/wave_live
python3 workflow.py normalize work/chef/jobs/wave_live
python3 workflow.py frames work/chef/jobs/wave_live
```

使用 MiniMax H3 的命令行方式：设置 `MINIMAX_API_KEY`，然后在 `submit` 中添加 `--model minimax_h3_768p` 或 `--model minimax_h3_2k`。中国版另设 `MINIMAX_API_BASE=https://api.minimax.cn`。

`poll` 可重复运行。没有 API Key 时，可以使用 `--dry-run` 检查请求摘要；也可以用 `python3 workflow.py normalize work/chef/jobs/wave --video existing.mp4` 处理已有视频。

## 画布与限制

- `prepare` 有 `auto`、`focus` 和 `full_frame` 三种构图模式。`auto` 在可把主体放大至少 15% 时聚焦，否则保留全图；`focus` 尽量选取包含角色框及四周活动余量的区域。`analysis.json` 记录原图尺寸、角色框、裁切坐标和完整变换。预览图用红框表示角色、蓝框表示模型输入区域。
- 聚焦输入会裁去模型画面以外的部分；恢复时只将生成区域贴回母版，外部保留原图，并在边界做轻微过渡。模型无法生成聚焦区域外的新动作；长线条或活动道具穿过蓝框边界时，预览后应改选 `full_frame` 或调整角色框。
- `restored.mp4` 恢复母版画面；MP4 编码器可能将奇数宽度裁成偶数。透明 PNG 直接从原始生成视频和变换记录恢复，保持母版的准确像素尺寸。`normalized.mp4` 使用统一项目画布，默认 1080×1920。
- 图片包直接从原始生成视频均匀选取含首尾帧的 20 个时间点，映射回母版尺寸，从画面边缘采样单色背景并生成透明通道。`manifest.json` 记录帧时间、构图模式和质量复核标记。背景边缘复杂时会明确报告图片包失败，视频仍可下载；这种素材需要另接 AI 抠图服务。
- 复杂背景、多人或遮挡时应手工填写角色框。视频模型仍可能重绘背景或改变角色比例；质量报告和首帧预览用于复核。
- 模型输出链接可能过期，因此服务在任务完成后立即保存视频。图片须符合所选模型的输入限制；当前适配图为 720×1280 PNG。

Runway 官方文档：[图生视频入门](https://docs.dev.runwayml.com/guides/using-the-api/)、[输入限制](https://docs.dev.runwayml.com/assets/inputs/)、[任务查询](https://docs.dev.runwayml.com/api-details/sdks/)。
MiniMax 官方文档：[创建 H3 视频任务](https://platform.minimax.io/docs/api-reference/video-generation-v2-create)、[查询任务](https://platform.minimax.io/docs/api-reference/video-generation-v2-query)。
