"""
搜救機器人 — 語音對講模組
============================
操作員透過網頁與受困者通話。

方向 1（操作員→受困者）：網頁輸入文字 → TTS → USB 喇叭播放
方向 2（受困者→操作員）：USB 麥克風 → 音訊串流 → 瀏覽器播放

不需要 HTTPS，不需要瀏覽器麥克風權限。
"""

import io
import time
import struct
import threading
import logging
import subprocess
import sys
import os
import tempfile

logger = logging.getLogger("rescue.intercom")

try:
    import numpy as np
    NP_OK = True
except ImportError:
    NP_OK = False

# 預設訊息（操作員快速選擇）
PRESET_MESSAGES = [
    "有人聽到嗎？我們是搜救隊",
    "請不要移動，救援即將到達",
    "請發出聲音讓我們定位你的位置",
    "你受傷了嗎？請回答",
    "請保持冷靜，我們正在靠近",
]


class Intercom:
    """語音對講（文字轉語音 + 麥克風串流）"""

    # 對講音訊處理參數
    INTERCOM_GAIN = 5.0         # 音訊增益倍率（提高受困者說話可聽度）
    INTERCOM_NOISE_GATE = 0.003  # 低於此 RMS 門檻視為背景噪音，壓制為靜音

    def __init__(self, audio_reader=None):
        self._audio_reader = audio_reader
        self._active = False
        self._speaking = False
        self._lock = threading.Lock()
        # 串流連續性：追蹤上次送出的環形緩衝位置，避免相鄰 fetch 重疊或漏段
        self._listen_last_pos = -1
        logger.info("對講模組初始化完成")

    @property
    def is_active(self) -> bool:
        return self._active

    @property
    def is_speaking(self) -> bool:
        return self._speaking

    def start(self):
        self._active = True
        self._listen_last_pos = -1   # 重置位置追蹤
        logger.info("對講模式已開啟")

    def stop(self):
        self._active = False
        self._listen_last_pos = -1
        logger.info("對講模式已關閉")

    def fetch_audio_chunk(self) -> bytes:
        """抓自上次 fetch 以來的新音訊，回傳 WAV bytes（對重疊與空隙無縫）。"""
        if not self._audio_reader or not self._audio_reader.is_available or not NP_OK:
            return b''
        try:
            seg, new_pos = self._audio_reader.get_audio_since(
                self._listen_last_pos, max_duration_sec=0.8
            )
            self._listen_last_pos = new_pos
            if seg is None or len(seg) == 0:
                return b''

            # 1. 噪音閘門：RMS 低於閾值的片段壓為 0（抑制 hiss）
            rms = float(np.sqrt(np.mean(seg * seg) + 1e-12))
            if rms < self.INTERCOM_NOISE_GATE:
                seg = seg * 0.3   # 不完全歸零，保留低強度底噪避免不自然靜音
            # 2. 增益 + soft clipping（避免突波削波產生爆音）
            amplified = seg * self.INTERCOM_GAIN
            amplified = np.tanh(amplified)   # soft saturation，保留動態範圍
            pcm_16 = (amplified * 32767).astype(np.int16)
            return self._pcm_to_wav(pcm_16.tobytes(), self._audio_reader._sample_rate)
        except Exception as e:
            logger.debug(f"fetch_audio_chunk 失敗: {e}")
            return b''

    # ── 方向 1：操作員 → 受困者（TTS）──

    def speak(self, text: str):
        """將文字轉語音播放到 USB 喇叭（非阻塞）"""
        if not text or self._speaking:
            return
        threading.Thread(target=self._do_speak, args=(text,), daemon=True).start()

    def _do_speak(self, text: str):
        """TTS 播放實作 — 統一走 tts_utils（aplay + 麥克風序列化）"""
        self._speaking = True
        try:
            from tts_utils import speak as tts_speak
            tts_speak(text)
            logger.info(f"[對講] 播放: {text[:20]}...")
        except Exception as e:
            logger.error(f"[對講] 播放錯誤: {e}")
        finally:
            self._speaking = False

    # ── 方向 2：受困者 → 操作員（麥克風串流）──

    def generate_audio_stream(self):
        """
        串流 RPi 麥克風到瀏覽器。
        產生連續的 WAV 片段，瀏覽器用 <audio> 播放。
        """
        if not NP_OK or not self._audio_reader or not self._audio_reader.is_available:
            return

        while self._active:
            try:
                buf = self._audio_reader.get_audio_buffer(0.5)
                if buf is None or len(buf) == 0:
                    time.sleep(0.1)
                    continue

                pcm_16 = (buf * 32767).astype(np.int16)
                wav = self._pcm_to_wav(pcm_16.tobytes(), 48000)
                yield wav
                time.sleep(0.3)

            except Exception as e:
                logger.debug(f"對講串流錯誤: {e}")
                time.sleep(0.5)

    @staticmethod
    def get_presets() -> list:
        """取得預設訊息清單"""
        return PRESET_MESSAGES

    @staticmethod
    def _pcm_to_wav(pcm: bytes, sr: int, ch: int = 1, sw: int = 2) -> bytes:
        """PCM → WAV"""
        sz = len(pcm)
        buf = io.BytesIO()
        buf.write(b'RIFF')
        buf.write(struct.pack('<I', 36 + sz))
        buf.write(b'WAVE')
        buf.write(b'fmt ')
        buf.write(struct.pack('<IHHIIHH', 16, 1, ch, sr, sr * ch * sw, ch * sw, sw * 8))
        buf.write(b'data')
        buf.write(struct.pack('<I', sz))
        buf.write(pcm)
        return buf.getvalue()
