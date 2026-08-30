# kira-ai-plugin-vllm-omni-tts

**vLLM-Omni 语音合成** — 让 KiraAI 用自部署的 vLLM-Omni TTS 服务发语音消息。

## 功能

基于 vLLM-Omni 的 OpenAI 兼容语音端点（`POST /v1/audio/speech`），以消息标签模式集成（与内置 `<record>` 同机制）：LLM 在回复里直接写 `<vllm_tts>要说的文本</vllm_tts>`，决策、合成、发送在一个 LLM 轮次内完成，无函数调用开销；合成失败自动降级为文本发送，消息不丢。

- **流式 / 非流式**：非流式直接取完整 WAV；流式拉取裸 PCM（int16 mono）本地封装 WAV，首包延迟更低
- **采样率可选**：不同 TTS 模型输出采样率不同（Qwen3-TTS 24000Hz、VoxCPM2 48000Hz），按部署模型配置即可
- **内联情绪指令（可选）**：部分模型支持（如 VoxCPM2），开启后 LLM 才能看到 `emotion` 属性说明，可按句指定情绪/语气/节奏；LLM 未传时对文本不做任何情绪控制
- **声音克隆（可选）**：配置参考音频与对应文本（`ref_audio` + `ref_text`）即以克隆音色合成；资产缺失自动回退预设音色
- **语音格式自动修复**：语音与 @/回复混排时拆成单条发送（QQ 显示兼容），自动摊平标签内嵌套写法；对内置 `<record>` 产生的语音同样生效

## 使用

对话中对 KiraAI 说「用语音跟我说……」「读一下这段话」即可。LLM 输出示例：

```
<msg><vllm_tts>今天天气真不错，出门走走吧。</vllm_tts></msg>
```

开启内联情绪指令后（仅支持的模型）：

```
<msg><vllm_tts emotion="Excited and fast-paced">快告诉我考得怎么样！</vllm_tts></msg>
```

## 配置

前提：已用 vLLM-Omni 部署 TTS 模型（`vllm serve <model> --omni`），KiraAI 可访问其服务地址。

| 字段 | 说明 | 默认值 |
|------|------|--------|
| `api_base_url` | vLLM-Omni 服务地址（OpenAI 兼容根，含 `/v1`），如 `http://127.0.0.1:8000/v1`；也可填不带版本段的地址 | 空（留空则语音功能不可用） |
| `model` | TTS 模型名（`--served-model-name`），如 `Qwen3-TTS` / `voxcpm2` | `Qwen3-TTS` |
| `api_key` | 服务端开启鉴权时的 Bearer 密钥，本地部署通常留空 | 空 |
| `default_voice` | 非克隆模式的音色名（voice 字段） | `default` |
| `enable_inline_emotion` | 启用内联情绪指令（仅支持的模型，如 VoxCPM2） | `false` |
| `enable_clone` | 启用声音克隆 | `false` |
| `ref_audio_path` | 克隆参考音频路径（3-10 秒干净人声；相对路径基于 KiraAI data/ 目录解析） | 空 |
| `ref_text` | 参考音频的逐字文本，与音频严格对应可提升克隆效果 | 空 |
| `sample_rate` | 流式 PCM 封 WAV 的采样率：Qwen3-TTS 填 24000，VoxCPM2 填 48000；非流式不受影响 | `24000` |
| `use_stream` | 流式合成（首包延迟低）/ 非流式（WAV 直出） | `false` |
| `speed` | 语速倍率（0.25-4.0） | `1.0` |
| `language` | 随请求传给服务端的 language 字段；留空则不传 | 空 |
| `timeout_seconds` | 单次合成超时秒数（服务冷启动首次请求较慢，建议不低于 30） | `60` |
| `max_text_length` | 单次合成最大文本长度，超出截断 | `800` |
| `auto_fix_voice_format` | 语音消息格式自动修复开关 | `true` |
| `temp_keep` | 临时语音文件保留个数 | `5` |

## 安装

1. KiraAI WebUI → 插件页 → 从 GitHub 安装，填入本仓库地址
2. 也可以下载本仓库 zip，通过 WebUI 上传安装
3. 安装完成后在插件配置页填入 `api_base_url`（按部署的模型确认 `model` 与 `sample_rate`）

## 依赖

无额外依赖。HTTP 客户端复用 KiraAI 本体自带的 `httpx`（本体依赖已含 `httpx>=0.28.1`）。

## 本地测试

```
python tests/test_vllm_omni_tts.py
```

自包含验证脚本（无需 pytest）：复刻 KiraAI 插件加载路径装载 main.py，覆盖标签注册面、配置容错、克隆预加载、情绪组装规则、格式修复钩子与 client 的 payload 构造、认证、流式 PCM 封装、错误契约。`tests/live_check.py` 为部署环境回环验证脚本（对真实 vLLM-Omni 服务做合成验证）。

## 信息

- **插件 ID**: `kira-ai-plugin-vllm-omni-tts`
- **版本**: 1.0.0
- **作者**: skyzhishui
