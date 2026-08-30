"""部署环境回环验证：对真实 vLLM-Omni 服务做合成验证。

用法（在部署机上）：
    python tests/live_check.py
可选环境变量：
    VLLM_TTS_PLUGIN_CONFIG  插件配置 JSON 路径
                            （默认 /data/KiraAI/data/config/plugins/kira-ai-plugin-vllm-omni-tts.json）

检查项：
1. 配置加载与端点推导
2. 非流式合成（response_format=wav 直取）
3. 流式合成（PCM → 按配置采样率封装 WAV）
4. 内联情绪（enable_inline_emotion=true 时带 (指令) 前缀合成）
5. 声音克隆（enable_clone=true 时带 ref_audio/ref_text 合成）
"""

from __future__ import annotations

import asyncio
import importlib.util
import io
import json
import os
import sys
import tempfile
import time
import wave
from pathlib import Path

PLUGIN_DIR = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = "/data/KiraAI/data/config/plugins/kira-ai-plugin-vllm-omni-tts.json"


def _load_client_module():
    spec = importlib.util.spec_from_file_location("vllm_tts_client", PLUGIN_DIR / "client.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load_config() -> dict:
    path = os.environ.get("VLLM_TTS_PLUGIN_CONFIG", DEFAULT_CONFIG)
    with open(path, "rb") as f:
        return json.loads(f.read().decode("utf-8"))


def _wav_info(data: bytes):
    w = wave.open(io.BytesIO(data), "rb")
    info = (w.getframerate(), w.getnchannels(), w.getsampwidth(), w.getnframes())
    w.close()
    return info


async def main() -> int:
    cl = _load_client_module()
    results: list[tuple[str, bool, str]] = []

    # 1. 配置
    try:
        cfg = _load_config()
        base_url = str(cfg.get("api_base_url", "") or "").strip()
        assert base_url, "api_base_url 未配置"
        model = str(cfg.get("model", "") or "Qwen3-TTS")
        sample_rate = int(cfg.get("sample_rate") or 24000)
        results.append(("配置加载", True, f"model={model} sample_rate={sample_rate}"))
    except Exception as exc:
        results.append(("配置加载", False, str(exc)))
        _report(results)
        return 1

    client = cl.VllmOmniTtsClient(
        base_url,
        model,
        api_key=str(cfg.get("api_key", "") or ""),
        voice=str(cfg.get("default_voice", "") or "default"),
        speed=float(cfg.get("speed") or 1.0),
        language=str(cfg.get("language", "") or ""),
        timeout_seconds=float(cfg.get("timeout_seconds") or 60),
        sample_rate=sample_rate,
    )

    # 2. 非流式
    try:
        wav = await client.synthesize("你好，这是语音合成回环测试。", stream=False)
        rate, ch, sw, nframes = _wav_info(wav)
        assert nframes > 0, "非流式合成无音频帧"
        results.append(("非流式合成", True, f"WAV rate={rate} ch={ch} sw={sw} frames={nframes}"))
    except Exception as exc:
        results.append(("非流式合成", False, str(exc)))

    # 3. 流式
    try:
        wav = await client.synthesize("流式合成测试。", stream=True)
        rate, _, _, nframes = _wav_info(wav)
        assert nframes > 0, "流式合成无音频帧"
        assert rate == sample_rate, f"封装采样率 {rate} != 配置 {sample_rate}"
        results.append(("流式合成", True, f"WAV rate={rate} frames={nframes}"))
    except Exception as exc:
        results.append(("流式合成", False, str(exc)))

    # 4. 内联情绪（开启时才测）
    if str(cfg.get("enable_inline_emotion", "")).lower() in ("1", "true", "yes", "on"):
        try:
            wav = await client.synthesize("(Excited and fast-paced)情绪指令测试。", stream=False)
            assert _wav_info(wav)[3] > 0
            results.append(("内联情绪合成", True, "服务端接受 (指令)文本 前缀"))
        except Exception as exc:
            results.append(("内联情绪合成", False, str(exc)))
    else:
        results.append(("内联情绪合成", True, "未启用，跳过"))

    # 5. 声音克隆（开启时才测）
    if str(cfg.get("enable_clone", "")).lower() in ("1", "true", "yes", "on"):
        try:
            ref_path = str(cfg.get("ref_audio_path", "") or "")
            assert ref_path, "ref_audio_path 未配置"
            # 相对路径基于 KiraAI data/ 目录解析（配置位于 data/config/plugins/ 下）
            data_root = Path(DEFAULT_CONFIG).parent.parent.parent
            url = cl.ref_audio_data_url(ref_path, data_root)
            wav = await client.synthesize(
                "克隆模式回环测试。",
                stream=False,
                ref_audio=url,
                ref_text=str(cfg.get("ref_text", "") or ""),
            )
            assert _wav_info(wav)[3] > 0
            results.append(("声音克隆合成", True, f"ref={ref_path}"))
        except Exception as exc:
            results.append(("声音克隆合成", False, str(exc)))
    else:
        results.append(("声音克隆合成", True, "未启用，跳过"))

    ok = _report(results)
    return 0 if ok else 1


def _report(results) -> bool:
    ok = all(r[1] for r in results)
    for name, passed, detail in results:
        print(f"[{'PASS' if passed else 'FAIL'}] {name}: {detail}")
    print(f"\n{sum(1 for r in results if r[1])}/{len(results)} checks passed")
    return ok


if __name__ == "__main__":
    t0 = time.time()
    code = asyncio.run(main())
    print(f"(耗时 {time.time() - t0:.1f}s)")
    sys.exit(code)
