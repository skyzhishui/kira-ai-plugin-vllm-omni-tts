"""vLLM-Omni TTS HTTP 客户端。

调用 vLLM-Omni 的 OpenAI 兼容语音合成端点 POST /v1/audio/speech：
- 预设音色模式（voice）
- 声音克隆模式（ref_audio 为 base64 data URL + ref_text）
- 非流式：response_format=wav 直接取完整 WAV
- 流式：response_format=pcm 收集裸 PCM（int16 mono）后按配置采样率封装 WAV

本模块不依赖 KiraAI 本体（仅 httpx + 标准库），便于独立测试与 live_check 复用。
"""

from __future__ import annotations

import base64
import io
import mimetypes
import wave
from pathlib import Path
from typing import Optional

import httpx

# vLLM-Omni 上不同 TTS 模型输出采样率不同（如 Qwen3-TTS 24000Hz、VoxCPM2 48000Hz），
# 仅影响流式 PCM 封装 WAV 时的头字段；非流式 WAV 自带头。
DEFAULT_SAMPLE_RATE = 24000

_PCM_CHANNELS = 1
_PCM_SAMPLE_WIDTH = 2  # int16


class TtsRequestError(Exception):
    """合成请求失败（超时/连接/HTTP 错误），消息可直接用于日志与降级提示。"""


def speech_url(base_url: str) -> str:
    """由 base_url 推导语音合成端点。

    约定 base_url 指到 OpenAI 兼容根（以 /v1 结尾或不含版本段均可）：
    - http://host:port/v1 -> http://host:port/v1/audio/speech
    - http://host:port    -> http://host:port/v1/audio/speech
    """
    base = (base_url or "").strip().rstrip("/")
    if base.endswith("/v1"):
        return f"{base}/audio/speech"
    return f"{base}/v1/audio/speech"


def build_payload(
    *,
    model: str,
    text: str,
    voice: str,
    speed: float,
    language: str = "",
    stream: bool = False,
    response_format: str = "wav",
    ref_audio: Optional[str] = None,
    ref_text: Optional[str] = None,
) -> dict:
    """构造 /v1/audio/speech 请求体。

    ref_audio 与 ref_text 同时非空时为克隆模式：移除 voice 字段，
    改用参考音频（base64 data URL）+ 参考文本。language 留空则不传该字段
    （不同模型接受的字段名有差异，留空最稳妥）。
    """
    payload: dict = {
        "model": model,
        "input": text,
        "voice": voice,
        "response_format": response_format,
        "stream": stream,
        "speed": speed,
    }
    if ref_audio and ref_text:
        payload.pop("voice")
        payload["ref_audio"] = ref_audio
        payload["ref_text"] = ref_text
    if language:
        payload["language"] = language
    return payload


def pcm_to_wav(
    pcm: bytes,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    channels: int = _PCM_CHANNELS,
    sample_width: int = _PCM_SAMPLE_WIDTH,
) -> bytes:
    """把裸 PCM（int16 mono）封装为 WAV（RIFF/WAVE）字节。"""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(sample_width)
        w.setframerate(sample_rate)
        w.writeframes(pcm)
    return buf.getvalue()


def ref_audio_data_url(path: str, data_root: Optional[Path] = None) -> str:
    """读取参考音频文件并编码为 base64 data URL。

    绝对路径原样使用；相对路径基于 data_root（KiraAI data 目录）解析。
    文件不存在或读取失败抛 OSError，由调用方决定降级方式。
    """
    audio_path = Path(path)
    if not audio_path.is_absolute() and data_root is not None:
        audio_path = data_root / audio_path
    audio_bytes = audio_path.read_bytes()
    mime = mimetypes.guess_type(str(audio_path))[0] or "audio/wav"
    b64 = base64.b64encode(audio_bytes).decode("ascii")
    return f"data:{mime};base64,{b64}"


class VllmOmniTtsClient:
    """vLLM-Omni /v1/audio/speech 客户端（每次请求独立连接，无需长连接池）。"""

    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        api_key: str = "",
        voice: str = "default",
        speed: float = 1.0,
        language: str = "",
        timeout_seconds: float = 60.0,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        transport: Optional[httpx.AsyncBaseTransport] = None,
    ):
        self._url = speech_url(base_url)
        self._model = model
        self._api_key = (api_key or "").strip()
        self._voice = voice or "default"
        self._speed = speed
        self._language = language or ""
        self._timeout_seconds = float(timeout_seconds)
        self._timeout = httpx.Timeout(self._timeout_seconds, connect=10.0)
        self._sample_rate = int(sample_rate)
        self._transport = transport

    def _headers(self) -> dict:
        headers = {"Content-Type": "application/json"}
        if self._api_key and self._api_key.upper() != "EMPTY":
            headers["Authorization"] = f"Bearer {self._api_key}"
        return headers

    async def synthesize(
        self,
        text: str,
        *,
        stream: bool = False,
        ref_audio: Optional[str] = None,
        ref_text: Optional[str] = None,
    ) -> bytes:
        """合成语音，返回 WAV 字节；失败抛 TtsRequestError。"""
        if stream:
            return await self._synthesize_stream(text, ref_audio=ref_audio, ref_text=ref_text)
        return await self._synthesize_batch(text, ref_audio=ref_audio, ref_text=ref_text)

    async def _synthesize_batch(self, text: str, *, ref_audio, ref_text) -> bytes:
        payload = build_payload(
            model=self._model,
            text=text,
            voice=self._voice,
            speed=self._speed,
            language=self._language,
            stream=False,
            response_format="wav",
            ref_audio=ref_audio,
            ref_text=ref_text,
        )
        try:
            async with httpx.AsyncClient(timeout=self._timeout, transport=self._transport) as client:
                resp = await client.post(self._url, json=payload, headers=self._headers())
        except httpx.TimeoutException:
            raise TtsRequestError(f"timeout ({self._timeout_seconds:.0f}s)")
        except httpx.HTTPError as exc:
            raise TtsRequestError(f"connection: {exc}")

        if resp.status_code >= 400:
            detail = resp.text[:200] if resp.text else "无错误详情"
            raise TtsRequestError(f"HTTP {resp.status_code}: {detail}")
        audio = resp.content
        if not audio:
            raise TtsRequestError("服务端返回空音频数据")
        return audio

    async def _synthesize_stream(self, text: str, *, ref_audio, ref_text) -> bytes:
        """流式拉取裸 PCM（int16 mono）并按配置采样率封装 WAV。

        末尾不足 int16 对齐的残余字节丢弃（极少发生）。
        """
        payload = build_payload(
            model=self._model,
            text=text,
            voice=self._voice,
            speed=self._speed,
            language=self._language,
            stream=True,
            response_format="pcm",
            ref_audio=ref_audio,
            ref_text=ref_text,
        )
        pcm = bytearray()
        try:
            async with httpx.AsyncClient(timeout=self._timeout, transport=self._transport) as client:
                async with client.stream("POST", self._url, json=payload, headers=self._headers()) as resp:
                    if resp.status_code >= 400:
                        body = (await resp.aread()).decode("utf-8", "replace")[:200]
                        raise TtsRequestError(f"HTTP {resp.status_code}: {body}")
                    async for chunk in resp.aiter_bytes():
                        if chunk:
                            pcm.extend(chunk)
        except httpx.TimeoutException:
            raise TtsRequestError(f"timeout ({self._timeout_seconds:.0f}s)")
        except httpx.HTTPError as exc:
            raise TtsRequestError(f"connection: {exc}")

        if len(pcm) % _PCM_SAMPLE_WIDTH:
            del pcm[len(pcm) - (len(pcm) % _PCM_SAMPLE_WIDTH):]
        if not pcm:
            raise TtsRequestError("服务端返回空 PCM 数据")
        return pcm_to_wav(bytes(pcm), self._sample_rate)
