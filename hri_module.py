"""
搜救機器人 — 主動語音互動模組 (HRI)
=====================================
Human-Robot Interaction：主動詢問 -> 錄音等待 -> 分析回應 -> 更新分數

所有 TTS 和等待操作均為非阻塞（使用 Popen + 短超時），
避免卡住其他執行緒。
"""

import time
import logging
import threading
from dataclasses import dataclass

import config
import numpy as np

try:
    import speech_recognition as sr
    STT_AVAILABLE = True
except ImportError:
    STT_AVAILABLE = False

from tts_utils import speak as tts_speak

logger = logging.getLogger("rescue.hri")


@dataclass
class InquiryResult:
    """主動詢問結果"""
    voice_detected: bool = False
    help_score: float = 0.0
    knock_detected: bool = False
    motion_response_score: float = 0.0
    completed: bool = False
    critical_help_requested: bool = False
    recognized_text: str = ""


class HRIModule:
    """主動式生命跡象確認互動"""

    PROMPTS = {
        "inquiry_1": "前方 是否 有 人員 受困, 若 聽得到, 請 回答 或 發出 聲音",
        "inquiry_2": "若 可 移動, 請 揮手 或 敲擊 周圍 物體",
        "confirm":   "系統 偵測到 疑似 受困者, 正在 標記 位置",
    }

    def __init__(self, speaker, audio_reader, audio_detector):
        self._speaker = speaker
        self._audio_reader = audio_reader
        self._audio_detector = audio_detector
        self._is_running = False
        self._cancel_requested = False
        self._lock = threading.Lock()
        logger.info("HRI 主動互動模組初始化完成")

    @property
    def is_running(self) -> bool:
        with self._lock:
            return self._is_running

    def cancel(self):
        """外部請求中斷 HRI 流程（例如揮手偵測觸發跳過詢問）"""
        self._cancel_requested = True
        logger.info("[HRI] 收到中斷請求")

    def run_inquiry_sequence(self, listen_duration: float = 3.0) -> InquiryResult:
        """
        執行完整的主動詢問流程（非阻塞式 TTS）。
        支援 cancel() 中斷（每個步驟前檢查）。
        """
        with self._lock:
            if self._is_running:
                logger.warning("HRI 正在進行中，跳過")
                return InquiryResult()
            self._is_running = True
            self._cancel_requested = False

        result = InquiryResult()

        try:
            if self._cancel_requested:
                logger.info("[HRI] 啟動前被中斷")
                return result

            # === 第一輪：語音詢問 ===
            logger.info("[HRI] 第一輪：播放語音詢問...")
            self._speak(self.PROMPTS["inquiry_1"])

            if self._cancel_requested:
                logger.info("[HRI] 第一輪語音後被中斷")
                return result

            # 靜聽並分析
            logger.info(f"[HRI] 靜聽中... ({listen_duration}秒)")
            audio_result_1 = self._listen_and_analyze(listen_duration)

            if audio_result_1:
                result.voice_detected = audio_result_1[0].has_voice
                result.help_score = audio_result_1[0].help_score
                result.knock_detected = audio_result_1[0].knock_detected
                result.critical_help_requested = result.critical_help_requested or audio_result_1[2]
                if audio_result_1[1]:
                    result.recognized_text += " " + audio_result_1[1]

                if audio_result_1[0].has_voice:
                    logger.info(f"[HRI] 偵測到語音回應 (help: {audio_result_1[0].help_score:.2f})")
                if audio_result_1[0].knock_detected:
                    logger.info("[HRI] 偵測到敲擊回應")

            if self._cancel_requested:
                logger.info("[HRI] 第一輪後被中斷")
                return result

            # === 第二輪：要求動作 ===
            logger.info("[HRI] 第二輪：請求揮手/敲擊...")
            self._speak(self.PROMPTS["inquiry_2"])

            if self._cancel_requested:
                logger.info("[HRI] 第二輪語音後被中斷")
                return result

            logger.info(f"[HRI] 第二輪靜聽... ({listen_duration}秒)")
            audio_result_2 = self._listen_and_analyze(listen_duration)

            if audio_result_2:
                result.voice_detected = result.voice_detected or audio_result_2[0].has_voice
                result.help_score = max(result.help_score, audio_result_2[0].help_score)
                result.knock_detected = result.knock_detected or audio_result_2[0].knock_detected
                result.critical_help_requested = result.critical_help_requested or audio_result_2[2]
                if audio_result_2[1]:
                    result.recognized_text += " " + audio_result_2[1]

            # === 計算互動回應分數 ===
            motion_score = 0.0
            if result.voice_detected:
                motion_score += 0.4
            if result.help_score > 0.5:
                motion_score += 0.3
            if result.knock_detected:
                motion_score += 0.3
            result.motion_response_score = min(motion_score, 1.0)

            # 搜救場景：有任何正面回應（語音/敲擊/聲音）→ 視為需要救助
            # 不能只依賴 STT 精確關鍵字，否則會漏報
            if result.motion_response_score > 0.3 and not result.critical_help_requested:
                result.critical_help_requested = True
                logger.warning(
                    f"[HRI] 🚨 偵測到正面回應（回應分={result.motion_response_score:.2f}），"
                    f"標記為需要救助！"
                )

            if self._cancel_requested:
                logger.info("[HRI] 第二輪後被中斷")
                return result

            # === 第三輪：確認通報 ===
            if result.motion_response_score > 0.3:
                logger.info("[HRI] 播放確認訊息...")
                self._speak(self.PROMPTS["confirm"])

            result.completed = True
            logger.info(
                f"[HRI] 完成 | 語音:{result.voice_detected} "
                f"呼救:{result.help_score:.2f} 敲擊:{result.knock_detected} "
                f"回應分:{result.motion_response_score:.2f}"
            )

        except Exception as e:
            logger.error(f"[HRI] 互動錯誤: {e}")
            result.completed = True
        finally:
            with self._lock:
                self._is_running = False

        return result

    def _speak(self, text: str):
        """使用共用 TTS 工具播放語音（已移除警報音效備援）。"""
        tts_speak(text)

    def _listen_and_analyze(self, duration_sec: float):
        """靜聽指定時長並分析音訊（支援 cancel 快速中斷）"""
        if not self._audio_reader or not self._audio_reader.is_available:
            # 可中斷的 sleep（每 0.1s 檢查一次 cancel）
            for _ in range(int(duration_sec / 0.1)):
                if self._cancel_requested:
                    return None
                time.sleep(0.1)
            return None

        # 可中斷的聽音等待
        for _ in range(int(duration_sec / 0.1)):
            if self._cancel_requested:
                return None
            time.sleep(0.1)

        if self._cancel_requested:
            return None

        audio_buffer = self._audio_reader.get_audio_buffer(duration_sec)
        if audio_buffer is None or len(audio_buffer) == 0:
            return None

        # 1. 規則分析
        audio_result = self._audio_detector.detect_buffer(audio_buffer)
        
        # 2. 語音辨識 (STT) 流程
        recognized_text = ""
        critical_help = False
        if STT_AVAILABLE:  # 拔除 audio_result.has_voice 的前置阻擋，無條件送交 Google STT
            try:
                # 轉 int16 bytes
                audio_16bit = (audio_buffer * 32767).astype(np.int16)
                audio_data = sr.AudioData(
                    audio_16bit.tobytes(),
                    sample_rate=config.MIC_SAMPLE_RATE,
                    sample_width=2
                )
                recognizer = sr.Recognizer()
                text = recognizer.recognize_google(audio_data, language="zh-TW")
                recognized_text = text.lower()
                
                # STT 既然有聽到字，代表一定有聲音，反向寫回標記
                audio_result.has_voice = True
                
                logger.info(f"[HRI-STT] 聽到回應: '{recognized_text}'")
                
                # 關鍵字判斷
                keywords = ["救", "幫", "需要", "help", "sos", "please", "痛", "受傷",
                            "救命", "幫忙", "有人嗎", "來人", "快來", "危險",
                            "hurt", "emergency", "danger", "嗚", "啊", "哎"]
                if any(kw in recognized_text for kw in keywords):
                    critical_help = True
                    audio_result.help_score = 1.0  # 強制拉高求救分數
                    logger.warning("[HRI-STT] 🚨 偵測到明確求救關鍵字！")
                elif recognized_text:
                    audio_result.help_score = max(audio_result.help_score, 0.4)
                    
                # 英文備援
                if not recognized_text:
                    text_en = recognizer.recognize_google(audio_data, language="en-US")
                    if text_en:
                        recognized_text = text_en.lower()
                        audio_result.has_voice = True
                        logger.info(f"[HRI-STT] 聽到英文回應: '{recognized_text}'")
                        if any(kw in recognized_text for kw in keywords):
                            critical_help = True
                            audio_result.help_score = 1.0
                            logger.warning("[HRI-STT] 🚨 偵測到明確英文求救關鍵字！")
                        else:
                            audio_result.help_score = max(audio_result.help_score, 0.4)
            except sr.UnknownValueError:
                pass
            except sr.RequestError as e:
                logger.error(f"[HRI-STT] API 請求失敗 (請檢查網路): {e}")
            except Exception as e:
                logger.error(f"[HRI-STT] 辨識異常: {e}")

        return (audio_result, recognized_text, critical_help)
