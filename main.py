"""vLLM-Omni 语音合成插件（KiraAI 版）。

经 vLLM-Omni 的 OpenAI 兼容端点 /v1/audio/speech 合成语音，以标签模式
暴露给 LLM（与内置 <record> 同机制）：LLM 在回复里直接写
<vllm_tts>文本</vllm_tts>，本插件在 XML 解析阶段合成并返回 Record 随
消息链以语音发出；合成失败降级为文本发送，内容不丢。

特性与配置对应：
- 采样率：vLLM-Omni 上不同 TTS 模型输出采样率不同（Qwen3-TTS 24000Hz、
  VoxCPM2 48000Hz），sample_rate 仅用于流式 PCM 封装 WAV。
- 内联情绪指令：仅部分模型支持（如 VoxCPM2）。enable_inline_emotion 开启后
  标签说明才向 LLM 暴露 emotion 属性；仅当 LLM 显式传入 emotion 时才以
  (指令)文本 形式拼接到文本前，未传则原文本零注入。
- 声音克隆：enable_clone 开启后使用 ref_audio_path + ref_text（预加载为
  base64 data URL）替代预设音色；资产缺失回退预设音色。
- 语音格式修复：语音与 @/回复混排时拆条发送（QQ 显示兼容），并摊平标签内
  嵌套写法，对内置 <record> 产生的语音同样生效。

api_base_url 未配置时标签不注册，不影响主程序。
"""

from __future__ import annotations

import os
import re
import time
from typing import Any, Optional

import httpx

from core.plugin import BasePlugin, logger, on
from core.chat import MessageChain
from core.chat.message_elements import At, Record, Reply, Text
from core.provider import LLMResponse
from core.tag import BaseTag
from core.utils.path_utils import get_data_path

from .client import TtsRequestError, VllmOmniTtsClient, ref_audio_data_url

TAG_NAME = "vllm_tts"

# 文本已以内联指令 (…) / （…） 开头时视为自带指令，不再包裹
_INLINE_PREFIX_RE = re.compile(r"^\s*[（(][^）)]+[）)]")

# 采样率合理范围，超出按默认处理（防配置手误生成无效 WAV 头）
_MIN_SAMPLE_RATE, _MAX_SAMPLE_RATE = 8000, 192000


def _as_int(value: Any, default: int) -> int:
    """配置项安全转 int；None/空串/非法值落回默认（单字段容错，不拖垮整插件）。"""
    try:
        return int(value) if value not in (None, "") else default
    except (TypeError, ValueError):
        return default


def _as_float(value: Any, default: float) -> float:
    try:
        return float(value) if value not in (None, "") else default
    except (TypeError, ValueError):
        return default


def _as_bool(value: Any, default: bool) -> bool:
    if value is None or value == "":
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "on")
    return bool(value)


def clean_voice_text(value: str) -> str:
    """清洗标签内容：剥掉残留的 <text> 等 XML 标签，只保留纯文本。

    常见错误输出：<vllm_tts><text>要说的话</text></vllm_tts>。
    llm_response 阶段的摊平是主防线，这里是第二道兜底。
    """
    if not value:
        return ""
    return re.sub(r"</?[a-zA-Z][^>]*>", "", value).strip()


class _VllmTtsTag(BaseTag):
    """<vllm_tts> 标签：描述由插件按配置动态生成（emotion 属性按开关暴露）。"""

    name = TAG_NAME
    parent = "msg"

    def __init__(self, plugin: "VllmOmniTtsPlugin"):
        super().__init__(ctx=plugin.ctx)
        self._plugin = plugin
        self.description = plugin.build_tag_description()

    async def handle(self, value: str, emotion: Optional[str] = None, **kwargs):
        return await self._plugin.handle_tts_tag(value, emotion)


class VllmOmniTtsPlugin(BasePlugin):
    """vLLM-Omni 语音合成插件。"""

    def __init__(self, ctx, cfg: dict):
        super().__init__(ctx, cfg)
        self._client: Optional[VllmOmniTtsClient] = None
        self._use_stream = False
        self._enable_inline_emotion = False
        self._auto_fix = True
        self._max_text_length = 800
        self._temp_keep = 5
        self._temp_dir = ""
        self._clone_ready = False
        self._ref_audio_data_url: Optional[str] = None
        self._ref_text = ""

    async def initialize(self):
        """按插件配置构造客户端（支持热重载重入；未配置则功能不可用）。"""
        cfg = self.plugin_cfg or {}
        base_url = str(cfg.get("api_base_url", "") or "").strip()
        model = str(cfg.get("model", "") or "").strip() or "Qwen3-TTS"
        voice = str(cfg.get("default_voice", "") or "").strip() or "default"
        speed = _as_float(cfg.get("speed"), 1.0)
        language = str(cfg.get("language", "") or "").strip()
        timeout = _as_float(cfg.get("timeout_seconds"), 60.0)
        sample_rate = _as_int(cfg.get("sample_rate"), 24000)
        if not (_MIN_SAMPLE_RATE <= sample_rate <= _MAX_SAMPLE_RATE):
            logger.warning(
                "vLLM-Omni TTS: sample_rate=%s 超出合理范围，回退默认 24000", sample_rate
            )
            sample_rate = 24000
        self._use_stream = _as_bool(cfg.get("use_stream"), False)
        self._enable_inline_emotion = _as_bool(cfg.get("enable_inline_emotion"), False)
        self._auto_fix = _as_bool(cfg.get("auto_fix_voice_format"), True)
        self._max_text_length = max(1, _as_int(cfg.get("max_text_length"), 800))
        self._temp_keep = max(1, _as_int(cfg.get("temp_keep"), 5))
        self._temp_dir = os.path.join(str(get_data_path()), "temp", "vllm_omni_tts")
        os.makedirs(self._temp_dir, exist_ok=True)

        if not base_url:
            logger.info("vLLM-Omni TTS: 未配置 api_base_url，语音合成功能不可用")
            self._client = None
            return

        self._client = VllmOmniTtsClient(
            base_url,
            model,
            api_key=str(cfg.get("api_key", "") or ""),
            voice=voice,
            speed=speed,
            language=language,
            timeout_seconds=timeout,
            sample_rate=sample_rate,
        )
        self._prepare_clone(cfg)
        logger.info(
            "vLLM-Omni TTS: 已就绪（model=%s，%s 模式，sample_rate=%d，流式=%s，情绪指令=%s）",
            model,
            "克隆" if self._clone_ready else "预设音色",
            sample_rate,
            self._use_stream,
            self._enable_inline_emotion,
        )

    def _prepare_clone(self, cfg: dict) -> None:
        """克隆模式预加载参考音频；未配置完整或读取失败一律回退预设音色。"""
        self._clone_ready = False
        self._ref_audio_data_url = None
        self._ref_text = ""
        if not _as_bool(cfg.get("enable_clone"), False):
            return
        ref_path = str(cfg.get("ref_audio_path", "") or "").strip()
        ref_text = str(cfg.get("ref_text", "") or "").strip()
        if not ref_path or not ref_text:
            logger.warning(
                "vLLM-Omni TTS: enable_clone=true 但 ref_audio_path/ref_text 未配置完整，回退预设音色"
            )
            return
        try:
            self._ref_audio_data_url = ref_audio_data_url(ref_path, get_data_path())
            self._ref_text = ref_text
            self._clone_ready = True
            logger.info("vLLM-Omni TTS: 克隆模式参考音频已加载: %s", ref_path)
        except Exception as exc:
            logger.warning("vLLM-Omni TTS: 参考音频读取失败（%s），回退预设音色", exc)

    async def terminate(self):
        self._client = None
        self._clone_ready = False
        self._cleanup_temp_files(keep=0)

    # ------------------------------------------------------------------
    #  标签注册与描述
    # ------------------------------------------------------------------

    def build_tag_description(self) -> str:
        """生成标签说明：emotion 属性文档仅在开关开启时暴露给 LLM。"""
        desc = (
            f"<{TAG_NAME}>voice_text</{TAG_NAME}> # 用 vLLM-Omni TTS 发送语音消息，"
            "voice_text 直接写要朗读的文本，不要在标签内嵌套 <text> 等其他标签；"
            "保留正常标点符号帮助控制停顿和语调，避免 @、#、markdown 格式、emoji 表情等符号；"
            "用户要求你发语音、朗读、或用语音回复时使用（普通文本回复不要用）；"
            "同一条消息中最多使用一次"
        )
        if self._enable_inline_emotion:
            desc += (
                "；可用 emotion 属性为本句指定情绪/语气/节奏，用英文自然语言描述，"
                f'如 <{TAG_NAME} emotion="Excited and fast-paced">文本</{TAG_NAME}>；'
                "emotion 仅描述情绪与说话风格，不要描述音色特征（性别/年龄/音质），"
                "不传则对文本不做任何情绪控制"
            )
        return desc

    @on.llm_request()
    async def inject_tts_tag(self, event, req, tag_set, *args, **kwargs):
        """按需注册 <vllm_tts> 标签（与内置标签同模式）。

        未配置服务、平台不支持语音（supported_elements 无 record）时不注册，
        LLM 不会看到该标签。
        """
        if self._client is None:
            return
        supported_elements = getattr(event, "supported_elements", None) or []
        if "record" not in supported_elements:
            return
        tag_set.register(_VllmTtsTag(self))

    # ------------------------------------------------------------------
    #  标签处理与合成
    # ------------------------------------------------------------------

    def _compose_input(self, text: str, emotion: Optional[str]) -> str:
        """内联情绪组装：仅 LLM 显式传 emotion 且开关开启时拼 (指令)文本 前缀。

        未传 emotion 时原文本零注入；文本已自带 (…) 前缀时不重复包裹。
        """
        if not self._enable_inline_emotion:
            return text
        instruction = (emotion or "").strip().strip("()（）").strip()
        if not instruction or _INLINE_PREFIX_RE.match(text):
            return text
        return f"({instruction}){text}"

    async def handle_tts_tag(self, value: str, emotion: Optional[str] = None) -> list:
        """<vllm_tts> 处理器：合成并返回 [Record]；任何失败降级 [Text]，内容不丢。"""
        text = clean_voice_text(value)
        if not text:
            return []
        if self._client is None or not self._temp_dir:
            logger.warning("vLLM-Omni TTS: 未配置或不可用，语音内容降级为文本发送")
            return [Text(text)]

        original = text
        if len(text) > self._max_text_length:
            logger.warning(
                "vLLM-Omni TTS: 文本过长(%d字)，截断至%d字", len(text), self._max_text_length
            )
            text = text[: self._max_text_length]

        input_text = self._compose_input(text, emotion)
        ref_audio = self._ref_audio_data_url if self._clone_ready else None
        ref_text = self._ref_text if self._clone_ready else None

        try:
            logger.info(
                "vLLM-Omni TTS: 开始合成（%s 模式，文本长度=%d）",
                "克隆" if self._clone_ready else "预设音色",
                len(input_text),
            )
            wav_bytes = await self._client.synthesize(
                input_text,
                stream=self._use_stream,
                ref_audio=ref_audio,
                ref_text=ref_text,
            )
            filename = f"vllm_tts_{int(time.time() * 1000)}.wav"
            file_path = os.path.join(self._temp_dir, filename)
            with open(file_path, "wb") as f:
                f.write(wav_bytes)
            self._cleanup_temp_files(keep=self._temp_keep)
            logger.info("vLLM-Omni TTS: 合成完成，文件大小=%d bytes", len(wav_bytes))
            return [Record(record=file_path, name=filename)]
        except TtsRequestError as exc:
            logger.error("vLLM-Omni TTS: 合成失败: %s", exc)
        except Exception:
            logger.exception("vLLM-Omni TTS: 合成异常")

        return [Text(original)]

    # ------------------------------------------------------------------
    #  临时文件管理
    # ------------------------------------------------------------------

    def _cleanup_temp_files(self, keep: int = 5) -> None:
        """只保留最近 keep 个 wav（keep=0 全清）。"""
        try:
            if not self._temp_dir or not os.path.isdir(self._temp_dir):
                return
            files = []
            for name in os.listdir(self._temp_dir):
                if name.endswith(".wav"):
                    full = os.path.join(self._temp_dir, name)
                    try:
                        files.append((os.path.getmtime(full), full))
                    except OSError:
                        pass
            files.sort()
            to_remove = files if keep <= 0 else files[:-keep]
            for _, full in to_remove:
                try:
                    os.remove(full)
                except OSError:
                    pass
        except Exception as exc:
            logger.warning("vLLM-Omni TTS: 清理临时文件失败: %s", exc)

    # ------------------------------------------------------------------
    #  语音消息格式自动修复（auto_fix_voice_format）
    # ------------------------------------------------------------------

    _TTS_BLOCK_RE = re.compile(rf"<{TAG_NAME}(\s[^>]*)?>(.*?)</{TAG_NAME}>", re.DOTALL)

    @on.llm_response()
    async def flatten_nested_tts_tags(self, event, resp: LLMResponse, *_):
        """XML 解析前的兜底：摊平 <vllm_tts><text>…</text></vllm_tts> 嵌套写法。

        框架解析器只取标签的直接文本（child.text），标签内再嵌套子标签时
        内容会在解析阶段静默丢失（处理器收到空串，语音发不出去）。
        """
        if getattr(resp, "tool_calls", None):
            return
        if not self._auto_fix:
            return
        text = getattr(resp, "text_response", "") or ""
        if f"<{TAG_NAME}" not in text:
            return
        new_text = self._TTS_BLOCK_RE.sub(self._flatten_block, text)
        if new_text != text:
            resp.text_response = new_text
            logger.info("vLLM-Omni TTS: 已自动摊平 <%s> 内的嵌套标签", TAG_NAME)

    @staticmethod
    def _flatten_block(m) -> str:
        attrs = m.group(1) or ""
        inner = re.sub(r"</?[a-zA-Z][^>]*>", "", m.group(2))
        return f"<{TAG_NAME}{attrs}>{inner}</{TAG_NAME}>"

    @on.after_xml_parse()
    async def fix_voice_format(self, event, actions, *_):
        """发送前兜底：把混在消息里的语音拆成单条干净消息（不带 @ 和回复）。

        QQ 上语音与 @/回复等内容混在同一条消息里会无法正常显示，这里在
        after_xml_parse 阶段改写待发送的 actions：每个 Record 各自单独成链，
        @/回复迁移到首个有实际内容的文字链。对内置 <record> 产生的语音同样生效。
        """
        if not self._auto_fix:
            return
        try:
            new_actions = []
            changed = False
            for action in actions:
                if not isinstance(action, MessageChain):
                    new_actions.append(action)
                    continue
                if not any(isinstance(e, Record) for e in action.message_list):
                    new_actions.append(action)
                    continue
                changed = True
                new_actions.extend(self._split_voice_chain(action))
            if changed:
                actions[:] = new_actions
                logger.info("vLLM-Omni TTS: 已自动修复语音消息格式（语音单条发送，不带@/回复）")
        except Exception:
            logger.exception("vLLM-Omni TTS: 语音格式修复异常")

    @staticmethod
    def _split_voice_chain(chain: MessageChain) -> list:
        """把含 Record 的消息链按顺序拆分：每个 Record 单独成链，其余内容保持原顺序成链。

        仅含 @/回复的碎片链会并入首个有实际内容的链（回复保持最前）；
        若整条消息只有语音，则 @/回复直接丢弃，保证语音消息绝对干净。
        """
        runs = []  # 按原始顺序的元素分组：语音单独一组，其余连续成组
        run = []
        for e in chain.message_list:
            if isinstance(e, Record):
                if run:
                    runs.append(run)
                    run = []
                runs.append([e])
            else:
                run.append(e)
        if run:
            runs.append(run)

        def has_real_content(elems) -> bool:
            return any(not isinstance(x, (At, Reply)) for x in elems)

        real_runs = [
            r for r in runs
            if not (len(r) == 1 and isinstance(r[0], Record)) and has_real_content(r)
        ]
        stray = [
            e for r in runs
            if not (len(r) == 1 and isinstance(r[0], Record)) and not has_real_content(r)
            for e in r
        ]

        if real_runs and stray:
            # 回复需保持在消息最前，@ 其次，其余按原相对顺序
            stray.sort(key=lambda x: 0 if isinstance(x, Reply) else 1)
            real_runs[0] = stray + real_runs[0]
        # 没有文字内容时 stray 直接丢弃

        # 按原顺序重组：文字链和语音链的先后关系保持不变
        ordered = []
        for r in runs:
            if len(r) == 1 and isinstance(r[0], Record):
                ordered.append(r)
            elif has_real_content(r):
                # 首个内容链可能已并入 stray，用 real_runs 里对应版本
                ordered.append(real_runs.pop(0) if real_runs else r)
        # real_runs 若有剩余（理论上不会），追加到末尾
        ordered.extend(real_runs)

        return [MessageChain(list(r)) for r in ordered]
