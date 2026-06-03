"""
搜救機器人 — TTS 工具（極簡版）
================================
啟動時：gTTS 下載 MP3 → ffmpeg 轉 WAV 48kHz 立體聲 → 存檔
播放時：只用 aplay 播 WAV（最簡單可靠的方式）
"""

import os
import sys
import time
import hashlib
import shutil
import subprocess
import logging
import threading

logger = logging.getLogger("rescue.tts")

_tts_lock = threading.RLock()

# TTS 快取目錄
_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tts_cache")
try:
    os.makedirs(_CACHE_DIR, exist_ok=True)
    # 清除舊的 mp3 檔案（改用 wav）
    for _old in os.listdir(_CACHE_DIR):
        if _old.endswith(".mp3"):
            try:
                os.remove(os.path.join(_CACHE_DIR, _old))
            except OSError:
                pass
except Exception:
    pass

# 依賴檢查
GTTS_AVAILABLE = False
try:
    from gtts import gTTS
    GTTS_AVAILABLE = True
except ImportError:
    pass

_HAS_FFMPEG = shutil.which("ffmpeg") is not None
_HAS_APLAY = shutil.which("aplay") is not None

logger.info(f"TTS: gTTS={GTTS_AVAILABLE}, ffmpeg={_HAS_FFMPEG}, aplay={_HAS_APLAY}")


# ── 外部依賴引用 ──
_camera_ref = None
_ros_bridge_ref = None
_audio_reader_ref = None

def set_camera(cam):
    global _camera_ref
    _camera_ref = cam

def set_ros_bridge(bridge):
    global _ros_bridge_ref
    _ros_bridge_ref = bridge

def set_audio_reader(reader):
    global _audio_reader_ref
    _audio_reader_ref = reader


_usb_pause_depth = 0
_usb_pause_lock = threading.Lock()


def _safe_call(label, fn, timeout=1.0):
    """在 daemon thread 內呼叫 fn，最多等 timeout 秒，超時就放棄。
    確保任何單一裝置 hang 都不會吊死整個警報流程。"""
    def _wrap():
        try:
            fn()
        except Exception as e:
            logger.debug(f"{label} 失敗: {e}")
    t = threading.Thread(target=_wrap, daemon=True)
    t.start()
    t.join(timeout=timeout)
    if t.is_alive():
        logger.warning(f"{label} 逾時（>{timeout}s），跳過")


def _pause_usb_devices():
    """TTS 播放前釋放 USB 頻寬：硬暫停相機 + 暫停麥克風 + 暫停 map poll。
    每個操作都用 _safe_call 包住，避免任何一個 ALSA/V4L2 hang 吊死警報流程。"""
    global _usb_pause_depth
    with _usb_pause_lock:
        _usb_pause_depth += 1
        if _usb_pause_depth > 1:
            return
    if _camera_ref:
        _safe_call("camera.pause", _camera_ref.pause, timeout=2.0)
    if _audio_reader_ref:
        _safe_call("audio.pause", _audio_reader_ref.pause, timeout=1.0)
    if _ros_bridge_ref:
        _safe_call("map.pause", _ros_bridge_ref.pause_map_poll, timeout=0.5)


def _resume_usb_devices():
    """TTS 播放後恢復所有 USB 裝置。
    所有操作都用 _safe_call 防止 ALSA/V4L2 hang 吊死流程。
    Camera reopen 在背景並行，不阻塞呼叫端。"""
    global _usb_pause_depth
    with _usb_pause_lock:
        _usb_pause_depth = max(0, _usb_pause_depth - 1)
        if _usb_pause_depth > 0:
            logger.info("外層仍在暫停中，跳過恢復")
            return

    # Camera reopen 直接 fire-and-forget 在背景 thread 內，不等
    if _camera_ref:
        threading.Thread(
            target=lambda: _camera_ref.resume(),
            daemon=True
        ).start()

    if _audio_reader_ref:
        _safe_call("audio.resume", _audio_reader_ref.resume, timeout=1.0)
        logger.info("麥克風恢復指令已送出")

    if _ros_bridge_ref:
        _safe_call("map.resume", _ros_bridge_ref.resume_map_poll, timeout=0.5)


# 向後相容
_pause_before_audio = _pause_usb_devices
_resume_after_audio = _resume_usb_devices


# ── 喇叭裝置偵測 ──
_SPEAKER_DEVICE = None


def _find_speaker_device():
    """找 C-Media USB 喇叭的 ALSA 裝置"""
    global _SPEAKER_DEVICE
    if _SPEAKER_DEVICE is not None:
        return _SPEAKER_DEVICE if _SPEAKER_DEVICE else None
    try:
        result = subprocess.run(
            ["aplay", "-l"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            timeout=5
        )
        output = result.stdout.decode("utf-8", errors="ignore")
        for line in output.split("\n"):
            if "card" not in line.lower():
                continue
            line_lower = line.lower()
            if "usb" in line_lower or "c-media" in line_lower:
                try:
                    card_num = line.split("card")[1].strip().split(":")[0].strip()
                    _SPEAKER_DEVICE = f"plughw:{card_num},0"
                    logger.info(f"偵測到 USB 喇叭: {_SPEAKER_DEVICE}")
                    return _SPEAKER_DEVICE
                except (IndexError, ValueError):
                    pass
    except Exception as e:
        logger.debug(f"偵測 USB 喇叭失敗: {e}")
    _SPEAKER_DEVICE = ""
    return None


def _get_wav_path(text: str, lang: str = "zh-TW") -> str:
    """以文字+語言雜湊為 WAV 檔名"""
    key = hashlib.md5(f"{lang}::{text}".encode("utf-8")).hexdigest()
    return os.path.join(_CACHE_DIR, f"{key}.wav")


def _generate_wav_cache(text: str, lang: str = "zh-TW") -> str:
    """
    取得 WAV 路徑。
    - 命中 → 直接回傳
    - 未命中 → gTTS 下載 MP3 → ffmpeg 轉 WAV → 回傳
    """
    wav_path = _get_wav_path(text, lang)
    if os.path.exists(wav_path) and os.path.getsize(wav_path) > 1000:
        return wav_path

    if not GTTS_AVAILABLE or not _HAS_FFMPEG:
        logger.warning("gTTS 或 ffmpeg 不可用，無法生成 TTS")
        return None

    # 下載 MP3
    mp3_path = wav_path + ".tmp.mp3"
    try:
        tts = gTTS(text=text, lang=lang)
        tts.save(mp3_path)
    except Exception as e:
        logger.warning(f"gTTS 下載失敗 '{text[:15]}': {e}")
        return None

    # ffmpeg 轉 WAV 48kHz 立體聲 16-bit（USB audio 原生格式）
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", mp3_path,
             "-ar", "48000", "-ac", "2", "-f", "wav", wav_path],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            timeout=15, check=True
        )
    except Exception as e:
        logger.warning(f"ffmpeg 轉檔失敗 '{text[:15]}': {e}")
        return None
    finally:
        try:
            os.remove(mp3_path)
        except OSError:
            pass

    if os.path.exists(wav_path) and os.path.getsize(wav_path) > 1000:
        logger.debug(f"TTS WAV 生成完成: {text[:20]}")
        return wav_path
    return None


def _play_wav(wav_path: str) -> bool:
    """用 aplay 播 WAV（最簡單的音訊播放方式）"""
    if not wav_path or not os.path.exists(wav_path):
        return False
    if not _HAS_APLAY:
        return False

    device = _find_speaker_device()
    cmd = ["aplay", "-q"]
    if device:
        cmd.extend(["-D", device])
    cmd.append(wav_path)

    start = time.time()
    try:
        # 典型 TTS 3-7s（B5 生命跡象文字較長);15s timeout 容納長文字 + USB 開啟延遲,
        # 失敗則由外層連續失敗保護放棄,避免單輪卡死整段警報
        r = subprocess.run(
            cmd,
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
            timeout=15
        )
        dur = time.time() - start
        if r.returncode == 0:
            logger.info(f"aplay WAV 完成 ({dur:.1f}s)")
            return True
        else:
            err = r.stderr.decode('utf-8', errors='ignore').strip()
            logger.warning(f"aplay 失敗 (rc={r.returncode}, {dur:.1f}s): {err}")
            # 備援：用預設裝置重試（同樣 timeout 較寬鬆）
            if device:
                logger.info("嘗試用預設裝置重播...")
                r2 = subprocess.run(
                    ["aplay", "-q", wav_path],
                    stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                    timeout=12
                )
                if r2.returncode == 0:
                    logger.info(f"aplay 預設裝置播放成功")
                    return True
            return False
    except subprocess.TimeoutExpired:
        logger.warning("aplay 逾時（已殺掉 subprocess，繼續流程）")
        return False
    except Exception as e:
        logger.warning(f"aplay 異常: {e}")
        return False


def _play_offline(text: str) -> bool:
    """espeak-ng 離線備用（縮短 timeout 避免卡住警報）"""
    if not shutil.which("espeak-ng"):
        return False
    try:
        subprocess.run(
            ["espeak-ng", "-vcmn", "-s", "140", text],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            timeout=10
        )
        return True
    except Exception:
        return False


def preload_common_phrases(phrases: list):
    """啟動時預載：gTTS → WAV，存入快取"""
    for text in phrases:
        try:
            _generate_wav_cache(text)
        except Exception as e:
            logger.debug(f"預載失敗 '{text[:15]}': {e}")


def speak(text: str, fallback_alert_fn=None):
    """播放語音（自動暫停 USB 攝影機/麥克風避免頻寬衝突）"""
    with _tts_lock:
        wav = _generate_wav_cache(text)
        _pause_usb_devices()
        try:
            if wav and _play_wav(wav):
                return
            if _play_offline(text):
                return
        finally:
            _resume_usb_devices()
    if fallback_alert_fn:
        fallback_alert_fn()


def speak_emergency(text_zh: str, text_en: str, fallback_alert_fn=None):
    """播放緊急語音（自動暫停 USB 攝影機/麥克風避免頻寬衝突）"""
    with _tts_lock:
        wav_zh = _generate_wav_cache(text_zh)
        wav_en = _generate_wav_cache(text_en, lang="en")
        _pause_usb_devices()
        try:
            if wav_zh and _play_wav(wav_zh):
                return
            if wav_en and _play_wav(wav_en):
                return
            if _play_offline(text_zh):
                return
        finally:
            _resume_usb_devices()
    if fallback_alert_fn:
        fallback_alert_fn()
