"""
搜救機器人 — 警報系統模組
Telegram 即時通報（截圖 + 事件訊息）+ USB 喇叭語音警報
"""

import os
import time
import threading
import subprocess
import logging
import cv2
import numpy as np

logger = logging.getLogger("carbot.alert")

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    logger.warning("requests 未安裝，Telegram 通知不可用")
    REQUESTS_AVAILABLE = False

import config
from location_service import LocationService

from tts_utils import speak_emergency


class AlertSystem:
    """多管道警報系統"""

    def __init__(self):
        logger.info("初始化警報系統...")
        self._last_alert_time = 0
        self._last_rescue_time = 0
        self._alert_lock = threading.Lock()
        self._alert_count = 0
        self._broadcasting = False      # 警報正在播放中
        self._cancel_requested = False  # 中斷請求旗標
        logger.info("警報系統初始化完成")

    @property
    def is_broadcasting(self):
        """警報是否正在播放（供 audio_loop 防自我觸發用）"""
        return self._broadcasting

    @property
    def last_alert_time(self):
        """最後一次警報觸發時間"""
        return max(self._last_alert_time, self._last_rescue_time)

    @property
    def alert_count(self):
        return self._alert_count

    def is_cooldown(self):
        """檢查是否在冷卻期間"""
        return (time.time() - self._last_alert_time) < config.ALERT_COOLDOWN

    # 前次警報線程 alive 超過此秒數視為 hang，強制重置允許新警報觸發
    # 略大於 ALERT_TOTAL_BUDGET_SEC (30s) + pause/resume 保守值
    _STALE_ALERT_THREAD_SEC = 45.0

    def trigger_alert(self, frame=None, message=None, critical=False,
                      vital_status: str = None, consciousness: str = None):
        """
        觸發警報（含冷卻與防重疊機制 + hang 強制解鎖）

        B5: vital_status / consciousness 用於 TTS 動態文字決定;
        若提供,失意識/微弱等狀態會改變播報內容與緊急程度。
        """
        prev = getattr(self, "_active_alert_thread", None)
        if prev and prev.is_alive():
            # 僵持保護：若前次警報 alive 超過 _STALE_ALERT_THREAD_SEC 視為 hang
            age = time.time() - getattr(self, "_active_alert_started_at", 0.0)
            if age > self._STALE_ALERT_THREAD_SEC:
                logger.warning(
                    f"前次警報線程 alive {age:.0f}s (> {self._STALE_ALERT_THREAD_SEC:.0f}s) "
                    f"視為 hang，強制重置 _broadcasting 並允許新警報"
                )
                self._broadcasting = False
                self._cancel_requested = False
                # 嘗試喚醒上一次 cancel（若線程 polling _cancel_requested 會自動退出）
                try:
                    self._cancel_requested = True
                except Exception:
                    pass
            else:
                logger.debug("目前已有警報廣播正在進行中，略過本次觸發確保系統穩定")
                return False

        with self._alert_lock:
            now = time.time()
            if critical:
                # 特急救援模式，獨立冷卻時間
                if (now - self._last_rescue_time) < 10.0:
                    return False
                self._last_rescue_time = now
            else:
                # 一般警報冷卻
                if self.is_cooldown():
                    logger.debug("警報冷卻中，略過本次觸發")
                    return False
                self._last_alert_time = now
                
            self._alert_count += 1

        alert_msg = message or "【搜救系統】偵測到疑似受困者！"
        if critical:
            logger.warning(f"[CRITICAL] 特急救援觸發: {alert_msg}")
        else:
            logger.warning(f"警報觸發: {alert_msg}")

        # 非同步執行警報動作
        t = threading.Thread(
            target=self._execute_alert,
            args=(frame, alert_msg, critical, vital_status, consciousness),
            daemon=True
        )
        t.start()
        self._active_alert_thread = t
        self._active_alert_started_at = time.time()   # 供 stale 偵測用

        return True

    def cancel_alert(self):
        """中斷正在播放的警報（倒地者恢復正常時呼叫）"""
        self._cancel_requested = True
        logger.info("警報中斷請求已發送")

    def _execute_alert(self, frame, message, critical=False,
                       vital_status: str = None, consciousness: str = None):
        """執行所有警報動作"""
        self._broadcasting = True
        self._cancel_requested = False
        try:
            # 1. 獨立非同步發送 Telegram 通知
            threading.Thread(
                target=self._send_telegram,
                args=(message, frame, critical),
                daemon=True
            ).start()

            # 2. 播放警報語音（純 TTS，可中斷）
            self._play_alert_sequence(critical, vital_status, consciousness)
        finally:
            self._broadcasting = False
            self._cancel_requested = False

    # 警報流程硬性 timeout：避免 aplay/ALSA/gTTS 偶發 hang 拖死整個系統
    ALERT_TOTAL_BUDGET_SEC = 30.0   # 警報語音總預算(秒)，超時立刻恢復 USB 釋放相機
    MAX_CONSECUTIVE_FAILURES = 2    # 連續失敗 N 次後放棄剩餘 loop

    def _play_alert_sequence(self, critical, vital_status: str = None,
                             consciousness: str = None):
        """
        播放警報語音（純 TTS）。輪數由 config.ALERT_TTS_ROUNDS 控制(預設 2);
        每輪須暫停相機讓出 USB,輪數越少相機停擺越短、YOLO 中斷越少。
        critical 旗標供 _tts_fallback_no_pause 選不同文字;
        B5: vital_status / consciousness 用於動態決定文字內容(失意識→最緊急播報)。
        """
        loop_count = int(getattr(config, 'ALERT_TTS_ROUNDS', 2))

        # ═══ 整段警報只暫停一次 USB 裝置 ═══
        try:
            from tts_utils import _pause_usb_devices, _resume_usb_devices
            _pause_usb_devices()
        except Exception:
            pass

        start_ts = time.time()
        consecutive_fails = 0
        try:
            for i in range(loop_count):
                if self._cancel_requested:
                    logger.info(f"警報已中斷（第 {i+1}/{loop_count} 次前）")
                    return

                # 總預算 watchdog：防 aplay/espeak 等偶發 hang 連環拖延
                elapsed = time.time() - start_ts
                if elapsed >= self.ALERT_TOTAL_BUDGET_SEC:
                    logger.warning(
                        f"警報超過 {self.ALERT_TOTAL_BUDGET_SEC:.0f}s 總預算 "
                        f"(elapsed={elapsed:.1f}s)，提前結束剩餘 {loop_count - i} 輪"
                    )
                    return

                # 連續失敗保護：2 次 aplay 失敗代表 ALSA/USB 狀態異常，不再硬撐
                if consecutive_fails >= self.MAX_CONSECUTIVE_FAILURES:
                    logger.warning(
                        f"警報 TTS 連續失敗 {consecutive_fails} 次，跳過剩餘 {loop_count - i} 輪"
                    )
                    return

                logger.info(f"[警報] 語音 ({i+1}/{loop_count})")
                ok = self._tts_fallback_no_pause(
                    critical=critical,
                    vital_status=vital_status,
                    consciousness=consciousness,
                )
                if ok:
                    consecutive_fails = 0
                else:
                    consecutive_fails += 1

                # 輪間間隔:給 USB 喇叭時間關閉並釋放裝置,避免下一輪「Device busy」
                # aplay timeout 已加長至 15s(輪 1 較少被殺),故間隔可縮短提升警報節奏;
                # 失敗時稍等久一點讓 USB 有 recovery 時間。
                if i < loop_count - 1:
                    wait_sec = 2.5 if ok else 4.0
                    ticks = int(wait_sec / 0.2)
                    for _ in range(ticks):
                        if self._cancel_requested:
                            logger.info("警報已中斷(等待間隔中)")
                            return
                        time.sleep(0.2)
        finally:
            # ═══ 警報結束，統一恢復 USB 裝置（即使異常也要執行）═══
            total_elapsed = time.time() - start_ts
            logger.info(f"[警報] 流程結束（耗時 {total_elapsed:.1f}s，失敗次數 {consecutive_fails}）")
            try:
                from tts_utils import _resume_usb_devices
                _resume_usb_devices()
            except Exception as e:
                logger.warning(f"USB 裝置恢復失敗: {e}")

    def _tts_fallback(self, critical=False):
        """使用共用 TTS 工具播放語音"""
        if critical:
            text_zh = "緊急警報！這裡有傷患！這裡有傷患！請盡速前往救援！"
            text_en = "Emergency! Victim located here! Please rescue immediately!"
        else:
            text_zh = "前方偵測到異常，系統正在確認中。"
            text_en = "Anomaly detected, system is verifying."
        speak_emergency(text_zh, text_en)

    def _tts_fallback_no_pause(self, critical=False,
                                vital_status: str = None,
                                consciousness: str = None) -> bool:
        """同步播放快取 WAV。外層 _play_alert_sequence 已負責 pause/resume USB，
        這裡直接呼叫底層函數避免雙重 pause。回傳 True=成功、False=失敗。

        B5: 若提供 vital_status / consciousness,從 VitalSignsAggregator.tts_text 取得
        動態中英文播報(失意識/微弱/正常 各有不同緊急程度的文字),否則用預設文字。

        注意:不再 fallback 到 espeak-ng — 它的中文合成聲音含糊不清會嚇到聽者,
        寧可這輪靜音由連續失敗保護放棄,也不要播放糟糕音質。
        """
        from tts_utils import _generate_wav_cache, _play_wav

        # 動態文字決定
        text = None
        if critical and (vital_status or consciousness):
            try:
                from vital_signs import VitalSignsAggregator
                text_zh, _text_en = VitalSignsAggregator.tts_text(
                    vital_status or "", consciousness or "UNKNOWN"
                )
                text = text_zh
            except Exception:
                text = None
        if text is None:
            # 預設文字(舊行為相容)
            text = "緊急警報！有人受傷！請立即前往救援！" if critical else "前方偵測到異常"

        try:
            wav = _generate_wav_cache(text)
            if wav and _play_wav(wav):
                return True
            # gTTS+aplay 失敗 → 不 fallback espeak,直接放棄這輪
            return False
        except Exception as e:
            logger.debug(f"TTS 失敗: {e}")
            return False

    def _send_telegram(self, message, frame=None, critical=False):
        """發送 Telegram 通知（含截圖）"""
        if not REQUESTS_AVAILABLE:
            logger.warning("requests 未安裝，無法發送 Telegram")
            return

        token = config.TELEGRAM_BOT_TOKEN
        chat_id = config.TELEGRAM_CHAT_ID
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

        loc_str = ""
        if critical:
            loc_str = "\n" + LocationService.get_location()

        full_message = (
            f"{message}\n"
            f"時間: {timestamp}\n"
            f"累計警報: 第 {self._alert_count} 次"
            f"{loc_str}"
        )

        try:
            if frame is not None:
                # 發送截圖
                _, buffer = cv2.imencode('.jpg', frame,
                                         [cv2.IMWRITE_JPEG_QUALITY, 85])
                url = f"https://api.telegram.org/bot{token}/sendPhoto"
                files = {"photo": ("alert.jpg", buffer.tobytes(), "image/jpeg")}
                data = {"chat_id": chat_id, "caption": full_message}
                resp = requests.post(url, files=files, data=data, timeout=10)
                resp.raise_for_status()
                logger.info("Telegram 截圖通知已發送")
            else:
                # 僅發送文字
                url = f"https://api.telegram.org/bot{token}/sendMessage"
                data = {"chat_id": chat_id, "text": full_message}
                resp = requests.post(url, data=data, timeout=10)
                resp.raise_for_status()
                logger.info("Telegram 文字通知已發送")

        except Exception as e:
            err_msg = str(e)
            if hasattr(e, 'response') and e.response is not None:
                err_msg += f" | 回應: {e.response.text}"
            logger.error(f"Telegram 發送失敗: {err_msg}")

    def cleanup(self):
        """清理"""
        logger.info("警報系統已關閉")
