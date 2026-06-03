"""
搜救機器人 — 音訊 AI 分析模組
==============================
VAD（語音活動偵測）+ 呼救聲分類 + 敲擊聲偵測

技術方案：
  - VAD：基於能量 + 過零率的輕量級偵測（無需額外模型）
  - 呼救聲：音訊能量 + 頻率特徵分析（頻譜中心 > 閾值 = 人聲/喊叫）
  - 敲擊聲：短時能量突變偵測（衝擊性音訊特徵）
"""

import logging
from collections import deque
import numpy as np
from dataclasses import dataclass

import config

logger = logging.getLogger("rescue.audio_detector")


@dataclass
class AudioResult:
    """音訊分析結果"""
    has_voice: bool = False           # 是否偵測到語音活動
    help_score: float = 0.0          # 呼救聲置信度 (0~1)
    knock_detected: bool = False      # 是否偵測到敲擊聲
    knock_score: float = 0.0         # 敲擊聲置信度 (0~1)
    rms_level: float = 0.0           # 當前音量 RMS
    dominant_freq: float = 0.0       # 主頻率 (Hz)


class AudioDetector:
    """音訊事件偵測器（規則式，輕量級）"""

    def __init__(self, sample_rate: int = None):
        self._sample_rate = sample_rate or config.MIC_SAMPLE_RATE
        self._vad_threshold = config.VAD_THRESHOLD
        self._help_threshold = config.HELP_THRESHOLD
        self._knock_threshold = config.KNOCK_THRESHOLD

        # 歷史能量（用於突變偵測，deque 自動淘汰舊值，O(1) 追加）
        self._max_history = 30
        self._energy_history = deque(maxlen=self._max_history)

        # 呼救聲連續偵測計數
        self._voice_consecutive = 0
        self._voice_confirm_frames = 3

        logger.info("✅ 音訊偵測器初始化完成（規則式引擎）")

    def detect(self, audio_chunk: np.ndarray) -> AudioResult:
        """
        分析單一音訊片段。

        參數:
            audio_chunk: float32 numpy array，取樣率對應 config.MIC_SAMPLE_RATE

        回傳:
            AudioResult 包含 VAD、呼救、敲擊偵測結果
        """
        result = AudioResult()

        if audio_chunk is None or len(audio_chunk) == 0:
            return result

        # 基礎特徵
        rms = float(np.sqrt(np.mean(audio_chunk ** 2)))
        result.rms_level = rms

        # --- VAD：語音活動偵測（含呻吟/低頻人聲）---
        zcr = self._zero_crossing_rate(audio_chunk)
        vad_score = 0.0
        if rms > 0.008:  # 降低音量門檻，捕捉微弱呻吟
            # 人聲過零率範圍放寬（0.01~0.25），涵蓋低頻呻吟和高頻喊叫
            if 0.01 < zcr < 0.25:
                vad_score = min(rms * 12, 1.0)
            # 額外：即使過零率不在範圍，高音量也算語音
            elif rms > 0.04:
                vad_score = min(rms * 8, 1.0)
            result.has_voice = bool(vad_score > self._vad_threshold)

        # --- 呼救聲分類 ---
        if result.has_voice:
            freq_features = self._spectral_features(audio_chunk)
            result.dominant_freq = freq_features['dominant_freq']

            # 呼救聲特徵：人聲頻率範圍 (100~4000Hz，含呻吟) + 頻譜質心 + 音量
            help_score = 0.0
            if 100 < freq_features['dominant_freq'] < 4000:
                help_score += 0.4  # 在人聲範圍（含低頻呻吟）
            if freq_features['spectral_centroid'] > 500:
                help_score += 0.3  # 中高頻重心（呼喊/呻吟特徵）
            if rms > 0.02:
                help_score += 0.3  # 音量（降低門檻）
            result.help_score = min(help_score, 1.0)

            # 連續幀確認
            if result.help_score >= self._help_threshold:
                self._voice_consecutive += 1
                if self._voice_consecutive < self._voice_confirm_frames:
                    result.help_score *= 0.5  # 降低單次偵測的信心
            else:
                self._voice_consecutive = 0
        else:
            self._voice_consecutive = 0

        # --- 敲擊聲偵測 ---
        knock_score = self._detect_knock(rms)
        result.knock_score = knock_score
        result.knock_detected = bool(knock_score > self._knock_threshold)

        return result

    def detect_buffer(self, audio_buffer: np.ndarray,
                       window_size: int = None) -> AudioResult:
        """
        分析較長的音訊緩衝（逐窗分析，取最高分）。
        用於主動詢問後的靜聽分析。
        """
        window = window_size or config.MIC_CHUNK_SIZE
        best = AudioResult()

        for i in range(0, len(audio_buffer) - window, window // 2):
            chunk = audio_buffer[i:i + window]
            r = self.detect(chunk)
            
            # 使用聯集保留任何一次含有講話的紀錄
            best.has_voice = bool(best.has_voice or r.has_voice)
            best.knock_detected = bool(best.knock_detected or r.knock_detected)
            
            if r.help_score > best.help_score:
                best.help_score = float(r.help_score)
                best.dominant_freq = float(r.dominant_freq)
            if r.knock_score > best.knock_score:
                best.knock_score = float(r.knock_score)
            best.rms_level = float(max(best.rms_level, r.rms_level))

        return best

    # ------------------------------------------------------------------
    # 特徵提取
    # ------------------------------------------------------------------

    @staticmethod
    def _zero_crossing_rate(audio: np.ndarray) -> float:
        """計算過零率"""
        if len(audio) < 2:
            return 0.0
        signs = np.sign(audio)
        crossings = np.sum(np.abs(np.diff(signs)) > 0)
        return crossings / len(audio)

    def _spectral_features(self, audio: np.ndarray) -> dict:
        """提取頻譜特徵"""
        n = len(audio)
        # FFT
        fft_vals = np.abs(np.fft.rfft(audio * np.hanning(n)))
        freqs = np.fft.rfftfreq(n, d=1.0 / self._sample_rate)

        # 主頻率
        if len(fft_vals) > 1:
            dominant_idx = np.argmax(fft_vals[1:]) + 1  # 跳過 DC
            dominant_freq = float(freqs[dominant_idx])
        else:
            dominant_freq = 0.0

        # 頻譜質心
        total_power = np.sum(fft_vals)
        if total_power > 0:
            spectral_centroid = float(np.sum(freqs * fft_vals) / total_power)
        else:
            spectral_centroid = 0.0

        return {
            'dominant_freq': dominant_freq,
            'spectral_centroid': spectral_centroid,
        }

    def _detect_knock(self, current_rms: float) -> float:
        """
        偵測敲擊聲（能量突變）。
        敲擊特徵：短暫的高能量脈衝，前後能量較低。
        """
        self._energy_history.append(current_rms)

        if len(self._energy_history) < 5:
            return 0.0

        # 計算背景能量（排除最後 3 個值的中位數）
        hist_list = list(self._energy_history)
        bg_energy = np.median(hist_list[:-3]) if len(hist_list) > 3 else 0.01
        bg_energy = max(bg_energy, 0.001)  # 防止除零

        # 能量比
        ratio = current_rms / bg_energy

        # 敲擊判定：能量突升 > 4 倍
        if ratio > 4.0 and current_rms > 0.015:
            score = min((ratio - 4.0) / 10.0, 1.0)
            return score

        return 0.0
