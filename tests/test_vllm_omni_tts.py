"""kira-ai-plugin-vllm-omni-tts 自包含验证脚本。

不依赖 pytest：python test_vllm_omni_tts.py 直接跑。
- 用 stub 的 core.* 装载 main.py（复刻 KiraAI 插件加载路径：
  plugins.<文件夹名> 包 + spec_from_file_location，验证相对导入可用）；
- 合成层用 FakeTtsClient 注入 canned 结果/异常；
- HTTP 层用 httpx.MockTransport 走真实请求/编码/解析路径。
"""

from __future__ import annotations

import asyncio
import importlib.util
import io
import logging
import os
import sys
import tempfile
import types
import wave
from pathlib import Path
from typing import Any

import httpx

PLUGIN_DIR = Path(__file__).resolve().parent.parent
TMP_ROOT = Path(tempfile.mkdtemp(prefix="vllm_tts_test_"))


# ---------------------------------------------------------------- 装载设施


def _install_core_stub() -> None:
    """stub core.*（BasePlugin/on/MessageChain/元素/BaseTag/get_data_path）。"""
    if "core.plugin" in sys.modules:
        return

    core = types.ModuleType("core")

    core_plugin = types.ModuleType("core.plugin")

    class BasePlugin:
        def __init__(self, ctx, cfg):
            self.ctx = ctx
            self.plugin_cfg = cfg or {}

    def _hook(event_type):
        def deco(*_args, **_kwargs):
            def wrap(func):
                func._kira_hook = event_type
                return func

            return wrap

        return deco

    core_plugin.BasePlugin = BasePlugin
    core_plugin.logger = logging.getLogger("kira.vllmtts.test")
    core_plugin.on = types.SimpleNamespace(
        llm_request=_hook("on_llm_request"),
        llm_response=_hook("on_llm_response"),
        after_xml_parse=_hook("after_xml_parse"),
    )
    core.plugin = core_plugin
    sys.modules["core"] = core
    sys.modules["core.plugin"] = core_plugin

    core_chat = types.ModuleType("core.chat")

    class MessageChain:
        def __init__(self, message_list):
            self.message_list = list(message_list)

    core_chat.MessageChain = MessageChain
    sys.modules["core.chat"] = core_chat

    core_elements = types.ModuleType("core.chat.message_elements")

    class Text:
        def __init__(self, text=""):
            self.text = text

    class Record:
        def __init__(self, record=None, mime=None, name=None, transcript=None):
            self.record = record
            self.mime = mime
            self.name = name
            self.transcript = transcript

    class At:
        def __init__(self, user_id=""):
            self.user_id = user_id

    class Reply:
        def __init__(self, message_id=""):
            self.message_id = message_id

    core_elements.Text = Text
    core_elements.Record = Record
    core_elements.At = At
    core_elements.Reply = Reply
    sys.modules["core.chat.message_elements"] = core_elements

    core_provider = types.ModuleType("core.provider")

    class LLMResponse:
        def __init__(self, text_response="", tool_calls=None):
            self.text_response = text_response
            self.tool_calls = tool_calls

    core_provider.LLMResponse = LLMResponse
    sys.modules["core.provider"] = core_provider

    core_tag = types.ModuleType("core.tag")

    class BaseTag:
        name = None
        description = None
        parent = "msg"

        def __init__(self, ctx=None, **kwargs):
            self.ctx = ctx

        async def handle(self, value, **kwargs):
            raise NotImplementedError

    core_tag.BaseTag = BaseTag
    sys.modules["core.tag"] = core_tag

    core_utils = types.ModuleType("core.utils")
    core_path_utils = types.ModuleType("core.utils.path_utils")
    core_path_utils.get_data_path = lambda: TMP_ROOT
    sys.modules["core.utils"] = core_utils
    sys.modules["core.utils.path_utils"] = core_path_utils


def _load_plugin_module():
    """按 KiraAI 加载器的方式（plugins.<entry> 包 + spec exec）装载 main.py。"""
    _install_core_stub()
    pkg = types.ModuleType("plugins")
    pkg.__path__ = [str(PLUGIN_DIR.parent)]
    sys.modules["plugins"] = pkg
    sub = types.ModuleType("plugins.kira-ai-plugin-vllm-omni-tts")
    sub.__path__ = [str(PLUGIN_DIR)]
    sys.modules["plugins.kira-ai-plugin-vllm-omni-tts"] = sub
    spec = importlib.util.spec_from_file_location(
        "plugins.kira-ai-plugin-vllm-omni-tts.main", PLUGIN_DIR / "main.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["plugins.kira-ai-plugin-vllm-omni-tts.main"] = mod
    spec.loader.exec_module(mod)
    return mod


class FakeTtsClient:
    """记录调用并返回预置结果/异常的假 VllmOmniTtsClient。"""

    def __init__(self, result: Any = b"RIFFxxxxWAVE", error: Any = None):
        self.result = result
        self.error = error
        self.calls: list[dict] = []

    async def synthesize(self, text, *, stream=False, ref_audio=None, ref_text=None):
        self.calls.append({
            "text": text,
            "stream": stream,
            "ref_audio": ref_audio,
            "ref_text": ref_text,
        })
        if self.error is not None:
            raise self.error
        return self.result


class _TagSetStub:
    def __init__(self):
        self.registered = []

    def register(self, *tags):
        self.registered.extend(tags)


def _make_plugin(cfg: dict, mod=None):
    mod = mod or _load_plugin_module()
    return mod.VllmOmniTtsPlugin(None, cfg)


async def _ready_plugin(cfg: dict, mod=None):
    """带有效 api_base_url 且完成 initialize 的插件实例。"""
    inst = _make_plugin({"api_base_url": "http://127.0.0.1:8000/v1", **cfg}, mod)
    await inst.initialize()
    return inst


def _wav_bytes(rate: int = 24000, frames: bytes = b"\x00\x00\x00\x00") -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(frames)
    return buf.getvalue()


# ---------------------------------------------------------------- 注册面


async def test_tag_class_attributes():
    mod = _load_plugin_module()
    tag_cls = mod._VllmTtsTag
    assert tag_cls.name == "vllm_tts", tag_cls.name
    assert tag_cls.parent == "msg", tag_cls.parent


async def test_hook_decorators_marked():
    mod = _load_plugin_module()
    cls = mod.VllmOmniTtsPlugin
    assert cls.inject_tts_tag._kira_hook == "on_llm_request"
    assert cls.flatten_nested_tts_tags._kira_hook == "on_llm_response"
    assert cls.fix_voice_format._kira_hook == "after_xml_parse"


# ---------------------------------------------------------------- 配置与初始化


async def test_defaults_applied():
    inst = await _ready_plugin({})
    assert inst._client is not None
    assert inst._client._model == "Qwen3-TTS"
    assert inst._client._voice == "default"
    assert inst._client._sample_rate == 24000
    assert inst._use_stream is False
    assert inst._enable_inline_emotion is False
    assert inst._auto_fix is True
    assert inst._max_text_length == 800
    assert inst._clone_ready is False


async def test_base_url_empty_client_none():
    inst = _make_plugin({})
    await inst.initialize()
    assert inst._client is None


async def test_garbage_config_falls_back():
    inst = await _ready_plugin({
        "sample_rate": "abc",
        "speed": object(),
        "timeout_seconds": None,
        "max_text_length": [],
        "use_stream": "yes",
        "enable_inline_emotion": "false",
        "auto_fix_voice_format": 0,
    })
    assert inst._client._sample_rate == 24000  # 非法值 → 默认
    assert inst._client._speed == 1.0
    assert inst._client._timeout_seconds == 60.0
    assert inst._max_text_length == 800
    assert inst._use_stream is True        # "yes" → True
    assert inst._enable_inline_emotion is False
    assert inst._auto_fix is False         # 0 → False


async def test_sample_rate_out_of_range_falls_back():
    inst = await _ready_plugin({"sample_rate": 0})
    assert inst._client._sample_rate == 24000
    inst2 = await _ready_plugin({"sample_rate": 999999})
    assert inst2._client._sample_rate == 24000
    inst3 = await _ready_plugin({"sample_rate": 48000})
    assert inst3._client._sample_rate == 48000


async def test_client_url_built_from_base():
    inst = await _ready_plugin({"api_base_url": "http://127.0.0.1:8000"})
    assert inst._client._url == "http://127.0.0.1:8000/v1/audio/speech"


# ---------------------------------------------------------------- 克隆配置


async def test_clone_ready_with_valid_file():
    ref = TMP_ROOT / "ref_clone.wav"
    ref.write_bytes(_wav_bytes())
    inst = await _ready_plugin({
        "enable_clone": True,
        "ref_audio_path": str(ref),
        "ref_text": "参考音频文本",
    })
    assert inst._clone_ready is True
    assert inst._ref_audio_data_url.startswith("data:audio/wav;base64,")
    assert inst._ref_text == "参考音频文本"


async def test_clone_relative_path_resolved_against_data_root():
    ref = TMP_ROOT / "rel_ref.wav"
    ref.write_bytes(_wav_bytes())
    inst = await _ready_plugin({
        "enable_clone": True,
        "ref_audio_path": "rel_ref.wav",
        "ref_text": "文本",
    })
    assert inst._clone_ready is True


async def test_clone_missing_ref_text_falls_back():
    ref = TMP_ROOT / "ref_notext.wav"
    ref.write_bytes(_wav_bytes())
    inst = await _ready_plugin({"enable_clone": True, "ref_audio_path": str(ref)})
    assert inst._clone_ready is False


async def test_clone_bad_path_falls_back():
    inst = await _ready_plugin({
        "enable_clone": True,
        "ref_audio_path": str(TMP_ROOT / "no_such_file.wav"),
        "ref_text": "文本",
    })
    assert inst._clone_ready is False


async def test_clone_disabled_no_preload():
    ref = TMP_ROOT / "ref_off.wav"
    ref.write_bytes(_wav_bytes())
    inst = await _ready_plugin({"enable_clone": False, "ref_audio_path": str(ref), "ref_text": "文本"})
    assert inst._clone_ready is False
    assert inst._ref_audio_data_url is None


# ---------------------------------------------------------------- 标签描述与注册


async def test_description_without_emotion():
    inst = await _ready_plugin({})
    desc = inst.build_tag_description()
    assert "vllm_tts" in desc
    assert "emotion" not in desc


async def test_description_with_emotion():
    inst = await _ready_plugin({"enable_inline_emotion": True})
    desc = inst.build_tag_description()
    assert 'emotion="Excited and fast-paced"' in desc
    assert "不要描述音色特征" in desc


async def test_inject_registers_tag_when_ready():
    """服务已配置且 supported_elements 含 record 时注册语音标签。"""
    mod = _load_plugin_module()
    inst = await _ready_plugin({}, mod)
    ts = _TagSetStub()
    event = types.SimpleNamespace(supported_elements=["text", "record"])
    await inst.inject_tts_tag(event, None, ts)
    assert len(ts.registered) == 1
    tag = ts.registered[0]
    assert isinstance(tag, mod._VllmTtsTag)
    assert tag.name == "vllm_tts"
    assert tag.parent == "msg"
    assert "<vllm_tts>" in tag.description


async def test_inject_skips_when_not_configured():
    """未配置服务时，即使 supported_elements 含 record 也不注册标签。"""
    inst = _make_plugin({})
    await inst.initialize()
    ts = _TagSetStub()
    event = types.SimpleNamespace(supported_elements=["text", "record"])
    await inst.inject_tts_tag(event, None, ts)
    assert ts.registered == []


async def test_inject_skips_when_platform_no_record():
    """supported_elements 不含 record 时不注册语音标签。"""
    inst = await _ready_plugin({})
    ts = _TagSetStub()
    event = types.SimpleNamespace(supported_elements=["text", "img"])
    await inst.inject_tts_tag(event, None, ts)
    assert ts.registered == []


# ---------------------------------------------------------------- 标签处理与合成


async def test_handle_success_returns_record():
    inst = await _ready_plugin({})
    inst._client = FakeTtsClient(result=_wav_bytes())
    res = await inst.handle_tts_tag("你好世界")
    assert len(res) == 1
    rec = res[0]
    assert rec.name.endswith(".wav")
    assert os.path.isfile(rec.record)
    assert rec.record.startswith(str(Path(inst._temp_dir)))


async def test_handle_failure_returns_text():
    mod = _load_plugin_module()
    inst = await _ready_plugin({}, mod)
    inst._client = FakeTtsClient(error=mod.TtsRequestError("HTTP 500: boom"))
    res = await inst.handle_tts_tag("你好世界")
    assert len(res) == 1
    assert isinstance(res[0], mod.Text)
    assert res[0].text == "你好世界"


async def test_handle_unexpected_exception_returns_text():
    mod = _load_plugin_module()
    inst = await _ready_plugin({}, mod)
    inst._client = FakeTtsClient(error=ValueError("bad thing"))
    res = await inst.handle_tts_tag("你好世界")
    assert len(res) == 1
    assert isinstance(res[0], mod.Text)
    assert res[0].text == "你好世界"


async def test_handle_empty_value_returns_empty_list():
    inst = await _ready_plugin({})
    assert await inst.handle_tts_tag("") == []
    assert await inst.handle_tts_tag("   ") == []


async def test_handle_not_configured_returns_text():
    mod = _load_plugin_module()
    inst = _make_plugin({}, mod)
    await inst.initialize()
    res = await inst.handle_tts_tag("你好")
    assert len(res) == 1
    assert isinstance(res[0], mod.Text)
    assert res[0].text == "你好"


async def test_handle_strips_residual_tags():
    inst = await _ready_plugin({})
    fake = FakeTtsClient()
    inst._client = fake
    await inst.handle_tts_tag("<text>你好呀</text>")
    assert fake.calls[0]["text"] == "你好呀"


async def test_handle_truncates_long_text():
    mod = _load_plugin_module()
    inst = await _ready_plugin({"max_text_length": 5}, mod)
    fake = FakeTtsClient(error=mod.TtsRequestError("x"))
    inst._client = fake
    long_text = "一二三四五六七八九十"
    res = await inst.handle_tts_tag(long_text)
    assert fake.calls[0]["text"] == "一二三四五"  # 合成输入被截断至 5 字
    assert isinstance(res[0], mod.Text)
    assert res[0].text == long_text  # 降级文本保留原文，内容不丢


async def test_emotion_composed_when_enabled_and_passed():
    inst = await _ready_plugin({"enable_inline_emotion": True})
    fake = FakeTtsClient()
    inst._client = fake
    await inst.handle_tts_tag("今天真开心", emotion="Excited and fast-paced")
    assert fake.calls[0]["text"] == "(Excited and fast-paced)今天真开心"


async def test_emotion_not_composed_when_disabled():
    inst = await _ready_plugin({})
    fake = FakeTtsClient()
    inst._client = fake
    await inst.handle_tts_tag("今天真开心", emotion="Excited")
    assert fake.calls[0]["text"] == "今天真开心"  # 开关关闭时属性被忽略


async def test_emotion_not_composed_when_not_passed():
    """LLM 未传 emotion → 原文本零注入（无 default_emotion 兜底）。"""
    inst = await _ready_plugin({"enable_inline_emotion": True})
    fake = FakeTtsClient()
    inst._client = fake
    await inst.handle_tts_tag("今天真开心", emotion=None)
    assert fake.calls[0]["text"] == "今天真开心"
    await inst.handle_tts_tag("今天真开心", emotion="   ")
    assert fake.calls[1]["text"] == "今天真开心"


async def test_emotion_wrapped_parens_stripped():
    inst = await _ready_plugin({"enable_inline_emotion": True})
    fake = FakeTtsClient()
    inst._client = fake
    await inst.handle_tts_tag("文本", emotion="（Calm and warm）")
    assert fake.calls[0]["text"] == "(Calm and warm)文本"


async def test_text_with_own_paren_prefix_not_double_wrapped():
    inst = await _ready_plugin({"enable_inline_emotion": True})
    fake = FakeTtsClient()
    inst._client = fake
    await inst.handle_tts_tag("(笑)哈哈哈", emotion="Excited")
    assert fake.calls[0]["text"] == "(笑)哈哈哈"


async def test_clone_mode_passes_ref_args():
    ref = TMP_ROOT / "ref_used.wav"
    ref.write_bytes(_wav_bytes())
    inst = await _ready_plugin({
        "enable_clone": True,
        "ref_audio_path": str(ref),
        "ref_text": "参考文本",
    })
    fake = FakeTtsClient()
    inst._client = fake
    await inst.handle_tts_tag("你好")
    assert fake.calls[0]["ref_audio"] == inst._ref_audio_data_url
    assert fake.calls[0]["ref_text"] == "参考文本"


async def test_voice_mode_no_ref_args():
    inst = await _ready_plugin({})
    fake = FakeTtsClient()
    inst._client = fake
    await inst.handle_tts_tag("你好")
    assert fake.calls[0]["ref_audio"] is None
    assert fake.calls[0]["ref_text"] is None


async def test_stream_flag_forwarded():
    inst = await _ready_plugin({"use_stream": True})
    fake = FakeTtsClient()
    inst._client = fake
    await inst.handle_tts_tag("你好")
    assert fake.calls[0]["stream"] is True


# ---------------------------------------------------------------- 临时文件管理


async def test_temp_keep_cleanup():
    inst = await _ready_plugin({"temp_keep": 2})
    inst._client = FakeTtsClient()
    for _ in range(4):
        await inst.handle_tts_tag("你好")
    wavs = [n for n in os.listdir(inst._temp_dir) if n.endswith(".wav")]
    assert len(wavs) == 2, wavs


async def test_terminate_cleans_temp():
    inst = await _ready_plugin({})
    inst._client = FakeTtsClient()
    await inst.handle_tts_tag("你好")
    assert os.listdir(inst._temp_dir)
    await inst.terminate()
    assert inst._client is None
    assert not os.path.exists(inst._temp_dir) or not os.listdir(inst._temp_dir)


# ---------------------------------------------------------------- 格式修复钩子


async def test_flatten_nested_tags():
    mod = _load_plugin_module()
    inst = await _ready_plugin({}, mod)
    resp = mod.LLMResponse(
        text_response='<msg><vllm_tts><text>要说的话</text></vllm_tts></msg>'
    )
    await inst.flatten_nested_tts_tags(None, resp)
    assert "<text>" not in resp.text_response
    assert "<vllm_tts>要说的话</vllm_tts>" in resp.text_response


async def test_flatten_keeps_attrs_and_other_tags():
    mod = _load_plugin_module()
    inst = await _ready_plugin({"enable_inline_emotion": True}, mod)
    raw = '<msg><text>前置</text><vllm_tts emotion="Calm">内<b>x</b>容</vllm_tts></msg>'
    resp = mod.LLMResponse(text_response=raw)
    await inst.flatten_nested_tts_tags(None, resp)
    assert '<text>前置</text>' in resp.text_response
    # 剥掉嵌套标签本身、保留其内文本（mimo-tts 同款语义：防标签名被念出）
    assert '<vllm_tts emotion="Calm">内x容</vllm_tts>' in resp.text_response


async def test_flatten_skips_when_disabled():
    mod = _load_plugin_module()
    inst = await _ready_plugin({"auto_fix_voice_format": False}, mod)
    raw = "<msg><vllm_tts><text>x</text></vllm_tts></msg>"
    resp = mod.LLMResponse(text_response=raw)
    await inst.flatten_nested_tts_tags(None, resp)
    assert resp.text_response == raw


async def test_flatten_skips_when_tool_calls():
    mod = _load_plugin_module()
    inst = await _ready_plugin({}, mod)
    raw = "<msg><vllm_tts><text>x</text></vllm_tts></msg>"
    resp = mod.LLMResponse(text_response=raw, tool_calls=[{"id": "t"}])
    await inst.flatten_nested_tts_tags(None, resp)
    assert resp.text_response == raw


async def test_fix_voice_format_isolates_record_and_merges_stray_at():
    mod = _load_plugin_module()
    Text, Record, At = mod.Text, mod.Record, mod.At
    inst = await _ready_plugin({}, mod)
    chain = mod.MessageChain([At("u1"), Record(record="/tmp/a.wav"), Text("正文")])
    actions = [chain]
    await inst.fix_voice_format(None, actions)
    assert len(actions) == 2
    a, b = actions
    assert isinstance(a.message_list[0], Record) and len(a.message_list) == 1
    # @/回复并入首个有实际内容的文字链
    assert [type(e) for e in b.message_list] == [At, Text]


async def test_fix_voice_format_drops_stray_when_only_voice():
    mod = _load_plugin_module()
    Record, At, Reply = mod.Record, mod.At, mod.Reply
    inst = await _ready_plugin({}, mod)
    chain = mod.MessageChain([Reply("m1"), At("u1"), Record(record="/tmp/a.wav")])
    actions = [chain]
    await inst.fix_voice_format(None, actions)
    assert len(actions) == 1
    assert isinstance(actions[0].message_list[0], Record)
    assert len(actions[0].message_list) == 1  # 纯语音时 @/回复直接丢弃


async def test_fix_voice_format_reply_goes_first():
    """碎片 Reply 并入文字链时保持在最前（mimo-tts 同款语义）。"""
    mod = _load_plugin_module()
    Text, Record, Reply = mod.Text, mod.Record, mod.Reply
    inst = await _ready_plugin({}, mod)
    chain = mod.MessageChain([Reply("m1"), Record(record="/tmp/a.wav"), Text("后说")])
    actions = [chain]
    await inst.fix_voice_format(None, actions)
    merged = [a for a in actions if not (len(a.message_list) == 1 and isinstance(a.message_list[0], Record))]
    assert len(merged) == 1
    first = merged[0].message_list
    assert isinstance(first[0], Reply)  # 回复保持在最前
    assert first[0] is chain.message_list[0]
    assert isinstance(first[1], Text) and first[1].text == "后说"


async def test_fix_voice_format_mixed_run_keeps_original_order():
    """同段混排（文字+回复）的 run 保持原相对顺序，不强行前移回复。"""
    mod = _load_plugin_module()
    Text, Record, Reply = mod.Text, mod.Record, mod.Reply
    inst = await _ready_plugin({}, mod)
    chain = mod.MessageChain([Text("先说"), Reply("m1"), Record(record="/tmp/a.wav"), Text("后说")])
    actions = [chain]
    await inst.fix_voice_format(None, actions)
    merged = [a for a in actions if not (len(a.message_list) == 1 and isinstance(a.message_list[0], Record))]
    assert len(merged) == 2
    first = merged[0].message_list
    assert [type(e) for e in first] == [Text, Reply]  # 原序保持
    assert first[0] is chain.message_list[0]


async def test_fix_voice_format_disabled_no_change():
    mod = _load_plugin_module()
    Text, Record = mod.Text, mod.Record
    inst = await _ready_plugin({"auto_fix_voice_format": False}, mod)
    chain = mod.MessageChain([Text("a"), Record(record="/tmp/a.wav")])
    actions = [chain]
    await inst.fix_voice_format(None, actions)
    assert actions == [chain]


async def test_fix_voice_format_keeps_non_chain_actions():
    mod = _load_plugin_module()
    Text, Record = mod.Text, mod.Record
    inst = await _ready_plugin({}, mod)
    marker = object()
    chain = mod.MessageChain([Text("a"), Record(record="/tmp/a.wav")])
    actions = [marker, chain]
    await inst.fix_voice_format(None, actions)
    assert actions[0] is marker
    assert len(actions) == 3  # marker + 拆分后的两条


async def test_fix_voice_format_no_record_untouched():
    mod = _load_plugin_module()
    Text = mod.Text
    inst = await _ready_plugin({}, mod)
    chain = mod.MessageChain([Text("纯文本"), mod.At("u")])
    actions = [chain]
    await inst.fix_voice_format(None, actions)
    assert actions == [chain]


# ---------------------------------------------------------------- client 层（httpx.MockTransport）


def test_speech_url_variants():
    from importlib import util as _util
    spec = _util.spec_from_file_location(
        "vllm_tts_client_under_test", PLUGIN_DIR / "client.py"
    )
    cl = _util.module_from_spec(spec)
    spec.loader.exec_module(cl)
    assert cl.speech_url("http://h:8000/v1") == "http://h:8000/v1/audio/speech"
    assert cl.speech_url("http://h:8000/v1/") == "http://h:8000/v1/audio/speech"
    assert cl.speech_url("http://h:8000") == "http://h:8000/v1/audio/speech"
    assert cl.speech_url("http://h:8000/") == "http://h:8000/v1/audio/speech"


def _client_mod():
    from importlib import util as _util
    spec = _util.spec_from_file_location(
        "vllm_tts_client_under_test", PLUGIN_DIR / "client.py"
    )
    cl = _util.module_from_spec(spec)
    spec.loader.exec_module(cl)
    return cl


def test_build_payload_preset():
    cl = _client_mod()
    p = cl.build_payload(model="Qwen3-TTS", text="hi", voice="v1", speed=1.5)
    assert p == {
        "model": "Qwen3-TTS", "input": "hi", "voice": "v1",
        "response_format": "wav", "stream": False, "speed": 1.5,
    }
    assert "language" not in p and "ref_audio" not in p


def test_build_payload_clone_replaces_voice():
    cl = _client_mod()
    p = cl.build_payload(
        model="m", text="hi", voice="v1", speed=1.0,
        ref_audio="data:audio/wav;base64,AAA", ref_text="参考",
    )
    assert "voice" not in p
    assert p["ref_audio"].startswith("data:audio/wav")
    assert p["ref_text"] == "参考"


def test_build_payload_language_and_stream():
    cl = _client_mod()
    p = cl.build_payload(model="m", text="hi", voice="v", speed=1.0,
                         language="Chinese", stream=True, response_format="pcm")
    assert p["language"] == "Chinese"
    assert p["stream"] is True and p["response_format"] == "pcm"


def test_pcm_to_wav_header_fields():
    cl = _client_mod()
    pcm = b"\x01\x02" * 100
    for rate in (24000, 48000):
        wav = cl.pcm_to_wav(pcm, rate)
        w = wave.open(io.BytesIO(wav), "rb")
        assert w.getframerate() == rate
        assert w.getnchannels() == 1
        assert w.getsampwidth() == 2
        assert w.getnframes() == 100
        w.close()


async def test_client_batch_success_and_auth_header():
    cl = _client_mod()
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["auth"] = request.headers.get("Authorization")
        seen["payload"] = request.read()
        return httpx.Response(200, content=_wav_bytes())

    transport = httpx.MockTransport(handler)
    client = cl.VllmOmniTtsClient(
        "http://h:8000/v1", "Qwen3-TTS", api_key="sk-test", transport=transport
    )
    out = await client.synthesize("你好")
    assert out == _wav_bytes()
    assert seen["url"] == "http://h:8000/v1/audio/speech"
    assert seen["auth"] == "Bearer sk-test"
    body = __import__("json").loads(seen["payload"])
    assert body["model"] == "Qwen3-TTS" and body["input"] == "你好"
    assert body["response_format"] == "wav" and body["stream"] is False


async def test_client_no_auth_when_key_empty_or_empty_string():
    cl = _client_mod()
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.headers.get("Authorization"))
        return httpx.Response(200, content=b"x")

    client = cl.VllmOmniTtsClient(
        "http://h:8000/v1", "m", api_key="", transport=httpx.MockTransport(handler)
    )
    await client.synthesize("a")
    client2 = cl.VllmOmniTtsClient(
        "http://h:8000/v1", "m", api_key="EMPTY", transport=httpx.MockTransport(handler)
    )
    await client2.synthesize("a")
    assert seen == [None, None]


async def test_client_batch_http_error():
    cl = _client_mod()

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="server boom")

    client = cl.VllmOmniTtsClient(
        "http://h:8000/v1", "m", transport=httpx.MockTransport(handler)
    )
    try:
        await client.synthesize("a")
        raise AssertionError("should raise")
    except cl.TtsRequestError as e:
        assert "HTTP 500" in str(e) and "server boom" in str(e)


async def test_client_timeout_error():
    cl = _client_mod()

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectTimeout("timed out", request=request)

    client = cl.VllmOmniTtsClient(
        "http://h:8000/v1", "m", timeout_seconds=15,
        transport=httpx.MockTransport(handler),
    )
    try:
        await client.synthesize("a")
        raise AssertionError("should raise")
    except cl.TtsRequestError as e:
        assert "timeout" in str(e) and "15" in str(e)


async def test_client_stream_collects_pcm_and_wraps_wav():
    cl = _client_mod()
    pcm = b"\x01\x02" * 200 + b"\xff"  # 末尾 1 字节未对齐，应丢弃

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=pcm)

    client = cl.VllmOmniTtsClient(
        "http://h:8000/v1", "voxcpm2", sample_rate=48000, transport=httpx.MockTransport(handler)
    )
    out = await client.synthesize("你好", stream=True)
    w = wave.open(io.BytesIO(out), "rb")
    assert w.getframerate() == 48000
    assert w.getnframes() == 200  # 未对齐的尾字节被丢弃
    w.close()


async def test_client_stream_http_error():
    cl = _client_mod()

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(422, text='{"detail":"bad"}')

    client = cl.VllmOmniTtsClient(
        "http://h:8000/v1", "m", transport=httpx.MockTransport(handler)
    )
    try:
        await client.synthesize("a", stream=True)
        raise AssertionError("should raise")
    except cl.TtsRequestError as e:
        assert "HTTP 422" in str(e)


def test_ref_audio_data_url_relative_resolution():
    cl = _client_mod()
    ref = TMP_ROOT / "rel_client_ref.wav"
    ref.write_bytes(_wav_bytes())
    url = cl.ref_audio_data_url("rel_client_ref.wav", TMP_ROOT)
    assert url.startswith("data:audio/wav;base64,")
    url_abs = cl.ref_audio_data_url(str(ref))
    assert url_abs == url


# ---------------------------------------------------------------- 运行器


async def _main() -> int:
    tests = [
        (name, fn) for name, fn in sorted(globals().items())
        if name.startswith("test_") and callable(fn)
    ]
    failures = []
    for name, fn in tests:
        try:
            ret = fn()
            if asyncio.iscoroutine(ret):
                await ret
            print(f"[PASS] {name}")
        except Exception as exc:
            failures.append((name, exc))
            print(f"[FAIL] {name}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - len(failures)}/{len(tests)} passed")
    if failures:
        print("failures:")
        for name, exc in failures:
            print(f"  - {name}: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(_main()))
