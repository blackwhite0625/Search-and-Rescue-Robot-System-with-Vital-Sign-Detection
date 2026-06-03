"""
搜救機器人主程式 (V2)
====================
Flask 伺服器入口，整合所有硬體與 AI 模組。
7 階段任務狀態機 + 多模態融合 + 音訊偵測 + 主動語音互動
"""

import os
import sys
import subprocess
import json

# ─── 在載入 cv2 前先抑制 V4L2 噪音 ───
# OpenCV 的 V4L2 後端會直接往 stderr 寫錯誤訊息，無法從 Python 攔截。
# 把 stderr (fd 2) 重新導向到 /dev/null，但保留 Python logging 的輸出（走 stdout）。
_devnull_fd = os.open(os.devnull, os.O_WRONLY)
os.dup2(_devnull_fd, 2)
os.close(_devnull_fd)

os.environ.setdefault("OPENCV_LOG_LEVEL", "SILENT")
os.environ.setdefault("OPENCV_VIDEOIO_DEBUG", "0")

import threading
import time
import logging
from concurrent.futures import ThreadPoolExecutor

# 強制 Python logging 走 stdout（而非預設的 stderr），避免被剛剛的 fd 2 重導影響
logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
    force=True,
)
from flask import Flask, render_template, request, jsonify, Response, send_from_directory
import cv2

import config
from camera import Camera
from motor import MotorController
from servo import ServoController
from detector import RescueDetector
from notifier import AlertSystem
from speaker import Speaker
from audio_reader import AudioReader
from audio_detector import AudioDetector, AudioResult
from fusion import VictimFusion, FusionInput
from event_logger import EventLogger
from hri_module import HRIModule
from mission_controller import MissionController
from victim_memory import VictimMemory

# 新功能模組
try:
    from heat_map import HeatMap
    HEAT_MAP_AVAILABLE = True
except ImportError:
    HEAT_MAP_AVAILABLE = False

try:
    from intercom import Intercom
    INTERCOM_AVAILABLE = True
except ImportError:
    INTERCOM_AVAILABLE = False

try:
    from ros_bridge import RosBridge
    ROS_BRIDGE_AVAILABLE = True
except ImportError:
    ROS_BRIDGE_AVAILABLE = False

# ============================================================
# 日誌設定
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("rescue")

# ============================================================
# Flask
# ============================================================
app = Flask(__name__)

# 背景工作佇列（確保 Flask 請求立即返回）
_executor = ThreadPoolExecutor(max_workers=3)

def _bg(fn):
    """將耗時操作推入背景執行緒池"""
    _executor.submit(fn)

# ============================================================
# 初始化所有模組
# ============================================================
logger.info("=" * 60)
logger.info("  搜救機器人系統 V2 啟動中...")
logger.info("=" * 60)

# 硬體
camera = Camera()
motor = MotorController()
servo = ServoController()
speaker = Speaker()

# AI
detector = RescueDetector()
audio_reader = AudioReader()
audio_detector = AudioDetector()

# 融合與決策
fusion = VictimFusion()
event_logger = EventLogger()
victim_memory = VictimMemory()
hri = HRIModule(speaker, audio_reader, audio_detector)
mission = MissionController(motor, servo, speaker, fusion, event_logger, hri)

# 通知
alert_manager = AlertSystem()

# 熱區記憶地圖
heat_map = HeatMap() if (HEAT_MAP_AVAILABLE and config.HEAT_MAP_ENABLED) else None

# 雙向語音對講
intercom = Intercom(audio_reader) if INTERCOM_AVAILABLE else None

# ROS 2 橋接（LiDAR + SLAM + /cmd_vel）
ros_bridge = None
if ROS_BRIDGE_AVAILABLE and config.ROS_BRIDGE_ENABLED:
    ros_bridge = RosBridge()
    if ros_bridge.is_available():
        motor.set_ros_bridge(ros_bridge)
        from tts_utils import set_ros_bridge as _set_tts_bridge
        _set_tts_bridge(ros_bridge)
        logger.info("ROS 2 Bridge 已連接，/cmd_vel 同步發布中")
    else:
        logger.warning("ROS 2 Bridge 初始化但不可用，降級為純超聲波模式")
        ros_bridge = None

# 讓 TTS 播放時能暫停麥克風，避免 ALSA 衝突
from tts_utils import set_camera as _set_tts_camera, set_audio_reader as _set_tts_audio, preload_common_phrases
_set_tts_camera(camera)
_set_tts_audio(audio_reader)

# 預載常用 HRI 語句到 TTS 快取（避免第一次播放的下載延遲）
# B5: 同時預載所有「生命跡象感知警報」的中文文字,避免首次警報邊下載邊播放導致 aplay 超時
try:
    _preload_list = [
        hri.PROMPTS["inquiry_1"],
        hri.PROMPTS["inquiry_2"],
        hri.PROMPTS["confirm"],
        "緊急警報！有人受傷！請立即前往救援！",
        "前方偵測到異常",
    ]
    # 自動補上 vital_signs.tts_text() 全部 5 種狀態的中文文字
    try:
        from vital_signs import VitalSignsAggregator as _VSA
        for _st, _cons in [("失去意識", "UNCONSCIOUS"), ("無反應", "UNKNOWN"),
                           ("微弱", "DROWSY"), ("正常", "AWAKE"), ("未知", "UNKNOWN")]:
            _zh, _ = _VSA.tts_text(_st, _cons)
            if _zh and _zh not in _preload_list:
                _preload_list.append(_zh)
    except Exception:
        pass
    preload_common_phrases(_preload_list)
    logger.info(f"TTS 常用語句已預載 ({len(_preload_list)} 條)")
except Exception as e:
    logger.debug(f"TTS 預載失敗: {e}")

# SLAM 驅動掃描巡邏實例（在 search_loop 中完成初始化，供 /control /status 讀取）
scan_patrol_instance = None

# ============================================================
# 全域狀態
# ============================================================
state_lock = threading.Lock()
app_state = {
    "mode": "manual",
    "mission_stage": "STANDBY",
    "person_count": 0,
    "fallen_count": 0,
    "pose_anomaly_score": 0.0,
    "victim_score": 0.0,
    "risk_level": "LOW",
    "distance_cm": -1,
    "audio_event": {
        "has_voice": False,
        "help_score": 0.0,
        "knock_detected": False,
        "rms_level": 0.0,
    },
    "fusion_components": {},
    "search_mode": config.SEARCH_MODE,
    "alert_count": 0,
    "event_count": 0,
    "fps": 0,
    "detect_ms": 0,
    "last_detection_ts": 0.0,
    "ai_inference_paused": False,
    "detection_stale": False,
    "brightness": 0, # Added brightness state
    "night_vision": config.CLAHE_MODE,      # "auto" / "on" / "off"（預設 off）
    "brightness_avg": 128,                  # 幀平均亮度
    "unique_person_count": 0,               # 歷史不重複人數
    "unreported_count": 0,                  # 未通報可見人數
    "heat_map_coverage": 0.0,               # 熱區覆蓋率 %
    # B5: 整合性生命跡象初始值
    "vital_score": -1.0,
    "vital_status": "未知",
    "vital_confidence": 0.0,
    "victim_vital_score": 0.0,
    "vital_components": {},
    "fallen_persons": [],
    "reported_victim_count": victim_memory.count,
    "reported_victims": [],
    "near_reported_victim": None,
    "departing_victim": False,
    "last_reported_victim_id": None,
    "manual_last_cmd_ts": 0.0,              # 最近一次手動移動/停止命令時間
    "manual_motion_active": False,          # 手動命令仍在推動車體
    "manual_last_seq": -1,                  # 前端手動控制序號，避免舊封包覆蓋停止命令
    "manual_session": None,                 # 前端頁面 session，重新整理後允許序號重來
    "objects": [],                          # 偵測到的災區物件
    "ground_obstacle_level": "CLEAR",       # 視覺低障礙等級 CLEAR/WARN/BRAKE
    "ground_obstacles": [],                 # 視覺低障礙清單
    "ai_loaded": False,
    "mic_ok": audio_reader.is_available,
    "camera_ok": camera.is_opened(),
    "gpio_ok": config.GPIO_AVAILABLE,
    "patrol_pan": 0,
    "logs": [],
    "events": [],
}

_latest_frame = None
_frame_lock = threading.Lock()
MAX_LOGS = 50


def add_log(level, msg):
    entry = {"time": time.strftime("%H:%M:%S"), "level": level, "msg": msg}
    with state_lock:
        app_state["logs"].insert(0, entry)
        if len(app_state["logs"]) > MAX_LOGS:
            app_state["logs"] = app_state["logs"][:MAX_LOGS]


def get_state():
    with state_lock:
        s = dict(app_state)
        s["logs"] = list(app_state["logs"])
        s["events"] = list(app_state["events"])
        s["audio_event"] = dict(app_state["audio_event"])
        s["fusion_components"] = dict(app_state.get("fusion_components", {}))
        s["reported_victims"] = list(app_state.get("reported_victims", []))
        near = app_state.get("near_reported_victim")
        s["near_reported_victim"] = dict(near) if isinstance(near, dict) else near
        return s


def _clear_live_detection_state(paused=False, stale=False):
    """清掉即時 AI 偵測欄位，避免畫面已更新但 UI 沿用上一輪倒地結果。"""
    with state_lock:
        app_state["person_count"] = 0
        app_state["fallen_count"] = 0
        app_state["fallen_persons"] = []
        app_state["pose_anomaly_score"] = 0.0
        app_state["victim_score"] = 0.0
        app_state["risk_level"] = "LOW"
        app_state["fusion_components"] = {}
        app_state["unreported_count"] = 0
        app_state["objects"] = []
        app_state["ground_obstacle_level"] = "CLEAR"
        app_state["ground_obstacles"] = []
        app_state["eye_state"] = "UNKNOWN"
        app_state["heart_rate_bpm"] = -1.0
        app_state["rppg_confidence"] = 0.0
        app_state["rppg_signal_quality"] = "UNKNOWN"
        app_state["respiration_rate"] = -1.0
        app_state["resp_confidence"] = 0.0
        app_state["rr_buffer_ratio"] = 0.0
        app_state["hr_buffer_ratio"] = 0.0
        app_state["blink_rate_per_min"] = -1.0
        app_state["consciousness_state"] = "UNKNOWN"
        app_state["blink_warmup_ratio"] = 0.0
        app_state["micro_motion_score"] = 0.0
        app_state["vital_score"] = -1.0
        app_state["vital_status"] = "未知"
        app_state["vital_confidence"] = 0.0
        app_state["victim_vital_score"] = 0.0
        app_state["vital_components"] = {}
        app_state["ai_inference_paused"] = bool(paused)
        app_state["detection_stale"] = bool(stale)
        app_state["detect_ms"] = 0


def _current_slam_pose():
    try:
        if ros_bridge and ros_bridge.is_available():
            return ros_bridge.get_slam_pose()
    except Exception:
        pass
    return None


def _visible_track_ids():
    try:
        if detector.tracker:
            return [t.get("id") for t in detector.tracker.get_tracks_info()
                    if t.get("id") is not None]
    except Exception:
        pass
    return []


def _current_victim_already_reported(result=None):
    """以 tracker/victim memory 判斷目前畫面中的人是否已完成通報。"""
    try:
        tracks = getattr(result, "tracks", None) if result is not None else None
        if tracks and any(bool(t.get("reported")) for t in tracks if isinstance(t, dict)):
            return True
    except Exception:
        pass
    try:
        return victim_memory.is_track_reported(_visible_track_ids())
    except Exception:
        return False


def _format_vital_block_from_values(vital_status, vital_score, vital_confidence,
                                    hr_bpm, hr_conf, rr_bpm, rr_conf,
                                    blink_rate, consciousness, micro_motion,
                                    eye_state):
    try:
        from vital_signs import VitalSignsAggregator as _VSA
        return _VSA.format_telegram_full(
            vital_status=vital_status or "未知",
            vital_score=float(vital_score if vital_score is not None else -1.0),
            hr_bpm=float(hr_bpm if hr_bpm is not None else -1.0),
            hr_conf=float(hr_conf if hr_conf is not None else 0.0),
            rr_bpm=float(rr_bpm if rr_bpm is not None else -1.0),
            rr_conf=float(rr_conf if rr_conf is not None else 0.0),
            blink_rate=float(blink_rate if blink_rate is not None else -1.0),
            consciousness=consciousness or "UNKNOWN",
            micro_motion=float(micro_motion if micro_motion is not None else 0.0),
            eye_state=eye_state or "UNKNOWN",
            vital_confidence=float(vital_confidence if vital_confidence is not None else 0.0),
        )
    except Exception as e:
        logger.debug(f"vital block 生成失敗: {e}")
        return ""


def _format_vital_block_from_result(result):
    return _format_vital_block_from_values(
        vital_status=getattr(result, "vital_status", "未知"),
        vital_score=getattr(result, "vital_score", -1.0),
        vital_confidence=getattr(result, "vital_confidence", 0.0),
        hr_bpm=getattr(result, "heart_rate_bpm", -1.0),
        hr_conf=getattr(result, "rppg_confidence", 0.0),
        rr_bpm=getattr(result, "respiration_rate", -1.0),
        rr_conf=getattr(result, "resp_confidence", 0.0),
        blink_rate=getattr(result, "blink_rate_per_min", -1.0),
        consciousness=getattr(result, "consciousness_state", "UNKNOWN"),
        micro_motion=getattr(result, "micro_motion_score", 0.0),
        eye_state=getattr(result, "eye_state", "UNKNOWN"),
    )


def _format_vital_block_from_state():
    s = get_state()
    return _format_vital_block_from_values(
        vital_status=s.get("vital_status", "未知"),
        vital_score=s.get("vital_score", -1.0),
        vital_confidence=s.get("vital_confidence", 0.0),
        hr_bpm=s.get("heart_rate_bpm", -1.0),
        hr_conf=s.get("rppg_confidence", 0.0),
        rr_bpm=s.get("respiration_rate", -1.0),
        rr_conf=s.get("resp_confidence", 0.0),
        blink_rate=s.get("blink_rate_per_min", -1.0),
        consciousness=s.get("consciousness_state", "UNKNOWN"),
        micro_motion=s.get("micro_motion_score", 0.0),
        eye_state=s.get("eye_state", "UNKNOWN"),
    )


def _refresh_victim_memory_state(pose=None):
    now = time.time()
    if pose is None and now - getattr(_refresh_victim_memory_state, "_last_ts", 0.0) < 1.0:
        with state_lock:
            near = app_state.get("near_reported_victim")
            return dict(near) if isinstance(near, dict) else near
    _refresh_victim_memory_state._last_ts = now
    pose = pose if pose is not None else _current_slam_pose()
    near, dist = victim_memory.nearest(pose)
    near_info = None
    if near:
        near_info = {
            "victim_id": near.victim_id,
            "distance_m": round(dist, 2) if dist is not None else None,
            "status": near.status,
            "report_count": near.report_count,
        }
    with state_lock:
        app_state["reported_victim_count"] = victim_memory.count
        app_state["reported_victims"] = victim_memory.to_status()
        app_state["near_reported_victim"] = near_info
    return near_info


def _remember_current_victim(source_stage, victim_score=None, risk_level=None,
                             result=None, event=""):
    pose = _current_slam_pose()
    tracks = _visible_track_ids()
    if result is not None:
        vital_status = getattr(result, "vital_status", None)
        vital_conf = getattr(result, "vital_confidence", None)
        consciousness = getattr(result, "consciousness_state", None)
    else:
        s = get_state()
        vital_status = s.get("vital_status")
        vital_conf = s.get("vital_confidence")
        consciousness = s.get("consciousness_state")
        victim_score = s.get("victim_score") if victim_score is None else victim_score
        risk_level = s.get("risk_level") if risk_level is None else risk_level
    rec = victim_memory.remember(
        pose=pose,
        track_ids=tracks,
        source_stage=source_stage,
        victim_score=victim_score or 0.0,
        risk_level=risk_level or "LOW",
        vital_status=vital_status or "未知",
        vital_confidence=vital_conf or 0.0,
        consciousness_state=consciousness or "UNKNOWN",
        event=event,
    )
    with state_lock:
        app_state["last_reported_victim_id"] = rec.victim_id
    _refresh_victim_memory_state(pose)
    return rec


def _victim_visual_guard():
    """Return visual stop/slow hints from fallen-person boxes."""
    try:
        latest = detector.latest_result
        boxes = getattr(latest, "fallen_persons", None) or []
        if not boxes and getattr(latest, "fallen_count", 0) > 0:
            boxes = getattr(latest, "persons", None) or []
    except Exception:
        boxes = []

    if not boxes:
        return {"stop": False, "slow": False, "reason": "", "y_ratio": 0.0, "area": 0.0}

    w = float(getattr(config, "CAMERA_WIDTH", 640) or 640)
    h = float(getattr(config, "CAMERA_HEIGHT", 480) or 480)
    best_y = 0.0
    best_area = 0.0
    for b in boxes:
        try:
            y2 = float(b.get("y2", 0.0)) / h
            area = max(0.0, float(b.get("x2", 0.0)) - float(b.get("x1", 0.0)))
            area *= max(0.0, float(b.get("y2", 0.0)) - float(b.get("y1", 0.0)))
            area /= max(1.0, w * h)
            best_y = max(best_y, y2)
            best_area = max(best_area, area)
        except Exception:
            continue

    stop = (
        best_y >= float(getattr(config, "VICTIM_VISUAL_STOP_Y_RATIO", 0.72))
        or best_area >= float(getattr(config, "VICTIM_VISUAL_STOP_AREA", 0.18))
    )
    slow = (
        best_y >= float(getattr(config, "VICTIM_VISUAL_SLOW_Y_RATIO", 0.60))
        or best_area >= float(getattr(config, "VICTIM_VISUAL_SLOW_AREA", 0.10))
    )
    reason = f"visual y={best_y:.2f} area={best_area:.2f}"
    return {"stop": stop, "slow": slow, "reason": reason, "y_ratio": best_y, "area": best_area}


# ============================================================
# 背景執行緒：AI 偵測
# ============================================================
def detection_loop():
    global _latest_frame
    logger.info("AI 偵測迴圈啟動")
    add_log("info", "AI 偵測迴圈啟動")

    frame_count = 0
    fps_timer = time.time()
    _report_alert_sent = False   # 防止 REPORT 期間重複觸發警報
    _report_memory_marked_id = None
    _fallen_alert_sent = False   # 倒地即時警報
    _fallen_first_seen = 0.0    # 首次偵測到倒地的時間（持續 3 秒才發警報）
    FALLEN_ALERT_DELAY = 3.0    # 倒地持續秒數門檻
    _last_known_victim_log = 0.0

    # CLAHE（僅在使用者手動開啟時才建立）
    _clahe = None
    _cached_apply_clahe = False
    _cached_avg_br = 128
    _hm_cov = 0.0

    while True:
        try:
            frame = camera.get_frame()
            if frame is None:
                _clear_live_detection_state(stale=True)
                time.sleep(0.3)
                continue

            # 一次性讀取所有需要的狀態（減少鎖競爭）
            with state_lock:
                audio_ev = dict(app_state["audio_event"])
                mode = app_state["mode"]
                distance = app_state["distance_cm"]
                brightness = app_state["brightness"]
                night_vision = app_state.get("night_vision", "auto")

            # ── REPORT / STANDBY 期間跳過 AI 推論，釋放 CPU 給警報 TTS ──
            _cur_mission = mission.current_stage
            if _cur_mission in ("REPORT", "STANDBY") and mode == "auto":
                camera.set_display_frame(frame)
                with state_lock:
                    app_state["mission_stage"] = _cur_mission
                if _cur_mission == "REPORT" and not _report_alert_sent:
                    s = get_state()
                    with _frame_lock:
                        alert_frame = _latest_frame if _latest_frame is not None else frame
                    vital_block = _format_vital_block_from_state()
                    if _report_memory_marked_id is None:
                        rec = _remember_current_victim(
                            "REPORT",
                            s.get("victim_score", 0.0),
                            s.get("risk_level", "LOW"),
                            result=None,
                            event="final_report_retry",
                        )
                        _report_memory_marked_id = rec.victim_id
                    else:
                        rec = None
                    victim_id = _report_memory_marked_id or (rec.victim_id if rec else None)
                    _report_msg = (f"✅ 搜救系統通報：完成搜救流程\n"
                                   f"VictimScore: {s.get('victim_score', 0.0):.2f} "
                                   f"({s.get('risk_level', 'LOW')})")
                    if vital_block:
                        _report_msg += "\n" + vital_block
                    if victim_id:
                        _report_msg += f"\n已標記傷患 #{victim_id}"
                    if alert_manager.trigger_alert(
                        frame=alert_frame,
                        message=_report_msg,
                        critical=True,
                        vital_status=s.get("vital_status"),
                        consciousness=s.get("consciousness_state"),
                    ):
                        _report_alert_sent = True
                        if detector.tracker:
                            detector.tracker.mark_all_visible_reported()
                        add_log("danger", f"REPORT 補充警報已發送，傷患 #{victim_id}")

                # YOLO 暫停時仍要推進 REPORT 冷卻，否則會永遠卡在 REPORT。
                stage_after_tick = mission.tick()
                if stage_after_tick == "STANDBY":
                    _report_alert_sent = False
                    _report_memory_marked_id = None

                # 這裡顯示的是最新 raw 畫面，不是最新 AI 推論；清掉 live 偵測避免 UI 殘留上一位倒地者。
                try:
                    detector.clear_transient_state(frame)
                except Exception:
                    pass
                _clear_live_detection_state(paused=True)
                with state_lock:
                    app_state["mission_stage"] = stage_after_tick
                time.sleep(0.033)   # ~30fps 原始畫面直通
                continue

            # ── ONNX 效能優化：隔幀推論，中間幀用最新 raw frame 重畫短期 overlay ──
            detect_every = max(1, int(getattr(config, "AI_DETECT_EVERY_N_FRAMES", 3)))
            if detect_every > 1 and frame_count % detect_every != 1:
                # 不再直接吐 raw frame，否則框線會「有一幀、消失兩幀」看起來像 YOLO 斷續。
                # 這裡只重畫短時間內的上一輪 YOLO 結果；過期就回 raw，避免舊框殘留。
                overlay = None
                try:
                    overlay = detector.draw_cached_overlay(
                        frame,
                        max_age_sec=float(getattr(config, "AI_DISPLAY_OVERLAY_TTL_SEC", 0.45)),
                    )
                except Exception:
                    overlay = None
                camera.set_display_frame(overlay if overlay is not None else frame)
                frame_count += 1
                time.sleep(0.01)
                continue

            # ── 低光增強（僅在使用者開啟時執行，預設完全跳過）──
            if night_vision != "off":
                if _clahe is None:
                    _clahe = cv2.createCLAHE(
                        clipLimit=config.CLAHE_CLIP_LIMIT,
                        tileGridSize=config.CLAHE_TILE_SIZE)
                if frame_count % 30 == 0:
                    _small = cv2.resize(frame, (80, 60))
                    _cached_avg_br = int(cv2.cvtColor(_small, cv2.COLOR_BGR2GRAY).mean())
                    _cached_apply_clahe = (night_vision == "on") or \
                        (night_vision == "auto" and _cached_avg_br < config.CLAHE_AUTO_THRESHOLD)
                if _cached_apply_clahe:
                    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
                    lab[:, :, 0] = _clahe.apply(lab[:, :, 0])
                    frame = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

            # 套用亮度調整
            if brightness != 0:
                frame = cv2.convertScaleAbs(frame, alpha=1.0, beta=brightness)

            frame_count += 1

            t0 = time.time()
            result = detector.detect(frame)
            detect_ms = int((time.time() - t0) * 1000)

            with _frame_lock:
                _latest_frame = result.annotated_frame
            camera.set_display_frame(result.annotated_frame)
            elapsed = time.time() - fps_timer
            if elapsed >= 1.0:
                fps = round(frame_count / elapsed, 1)
                frame_count = 0
                fps_timer = time.time()
            else:
                fps = app_state.get("fps", 0)

            # 融合計算（包裹在獨立 try 中，不影響影像串流）
            score = 0.0
            risk = "LOW"
            components = {}
            suppress_known_victim = False
            try:
                near_known_victim = _refresh_victim_memory_state()
                already_reported_track = _current_victim_already_reported(result)
                suppress_known_victim = (
                    mode == "auto"
                    and (near_known_victim is not None or already_reported_track)
                    and getattr(result, "fallen_count", 0) > 0
                    and mission.current_stage in ("SEARCH", "ANOMALY", "STANDBY")
                )
                if suppress_known_victim:
                    if time.time() - _last_known_victim_log > 5.0:
                        if near_known_victim:
                            msg = f"已通報傷患 #{near_known_victim['victim_id']} 在附近，略過重複鎖定"
                        else:
                            msg = "目前 track 已通報，略過重複鎖定"
                        add_log("info", msg)
                        _last_known_victim_log = time.time()
                distance_for_mission = distance
                if getattr(result, "fallen_count", 0) > 0:
                    vg = _victim_visual_guard()
                    stop_cm = float(getattr(config, "VICTIM_APPROACH_STOP_CM", 55))
                    if vg.get("stop") and (distance_for_mission < 0 or distance_for_mission > stop_cm):
                        distance_for_mission = stop_cm
                if mode == "auto":
                    with _frame_lock:
                        current_frame = _latest_frame
                    fusion_result = mission.update(
                        person_count=0 if suppress_known_victim else result.person_count,
                        pose_anomaly_score=0.0 if suppress_known_victim else result.pose_anomaly_score,
                        audio_help_score=audio_ev.get("help_score", 0),
                        audio_knock=audio_ev.get("knock_detected", False),
                        distance_cm=distance_for_mission,
                        frame=current_frame,
                        # B5: 傳入整合性生命跡象(取代舊「僅看心率」邏輯)
                        # vital_status / consciousness 用於 CONFIRM 階段失意識立即強制通報
                        victim_vital_score=0.0 if suppress_known_victim else getattr(result, 'victim_vital_score', -1.0),
                        heart_rate_bpm=result.heart_rate_bpm,
                        rppg_confidence=result.rppg_confidence,
                        vital_status=getattr(result, 'vital_status', None),
                        consciousness_state=getattr(result, 'consciousness_state', None),
                    )
                    score = fusion_result.victim_score
                    risk = fusion_result.risk_level
                    components = fusion_result.components
                else:
                    from fusion import FusionInput as _FI
                    _inp = _FI(
                        person_detected=(result.person_count > 0),
                        person_count=result.person_count,
                        pose_anomaly_score=result.pose_anomaly_score,
                        audio_help_score=audio_ev.get("help_score", 0),
                        audio_knock=audio_ev.get("knock_detected", False),
                        distance_cm=distance,
                        heart_rate_bpm=result.heart_rate_bpm,
                        rppg_confidence=result.rppg_confidence,
                        victim_vital_score=getattr(result, 'victim_vital_score', -1.0),
                    )
                    _fr = fusion.compute(_inp)
                    score = _fr.victim_score
                    risk = _fr.risk_level
                    components = _fr.components
            except Exception as e:
                logger.debug(f"融合計算錯誤: {e}")

            # 在鎖外預先計算（降頻避免鎖競爭）
            if heat_map and frame_count % 30 == 0:
                _hm_cov = heat_map.get_coverage_percent()

            # 更新全域狀態（快進快出）
            with state_lock:
                app_state["person_count"] = result.person_count
                app_state["fallen_count"] = result.fallen_count
                app_state["fallen_persons"] = getattr(result, "fallen_persons", []) or []
                app_state["pose_anomaly_score"] = result.pose_anomaly_score
                app_state["victim_score"] = score
                app_state["risk_level"] = risk
                app_state["fusion_components"] = components
                app_state["mission_stage"] = mission.current_stage
                app_state["alert_count"] = alert_manager.alert_count
                app_state["event_count"] = event_logger.event_count
                app_state["fps"] = fps
                app_state["detect_ms"] = detect_ms
                app_state["last_detection_ts"] = time.time()
                app_state["ai_inference_paused"] = False
                app_state["detection_stale"] = False
                app_state["brightness_avg"] = _cached_avg_br
                # 新功能狀態
                app_state["unique_person_count"] = result.unique_person_count
                app_state["unreported_count"] = result.unreported_count
                app_state["objects"] = result.objects
                # 視覺低障礙 (補 LiDAR 車頂盲區)
                app_state["ground_obstacle_level"] = getattr(result, "ground_obstacle_level", "CLEAR")
                app_state["ground_obstacles"] = getattr(result, "ground_obstacles", [])
                app_state["eye_state"] = result.eye_state
                app_state["heart_rate_bpm"] = result.heart_rate_bpm
                app_state["rppg_confidence"] = result.rppg_confidence
                app_state["rppg_signal_quality"] = result.rppg_signal_quality
                # 新：B1 呼吸率、B2 微動、B4 眨眼率 / 意識
                app_state["respiration_rate"] = getattr(result, "respiration_rate", -1.0)
                app_state["resp_confidence"] = getattr(result, "resp_confidence", 0.0)
                app_state["rr_buffer_ratio"] = getattr(result, "rr_buffer_ratio", 0.0)
                app_state["hr_buffer_ratio"] = getattr(result, "hr_buffer_ratio", 0.0)
                app_state["blink_rate_per_min"] = getattr(result, "blink_rate_per_min", -1.0)
                app_state["consciousness_state"] = getattr(result, "consciousness_state", "UNKNOWN")
                app_state["blink_warmup_ratio"] = getattr(result, "blink_warmup_ratio", 0.0)
                app_state["micro_motion_score"] = getattr(result, "micro_motion_score", 0.0)
                # B5: 整合性生命跡象
                app_state["vital_score"] = getattr(result, "vital_score", -1.0)
                app_state["vital_status"] = getattr(result, "vital_status", "未知")
                app_state["vital_confidence"] = getattr(result, "vital_confidence", 0.0)
                app_state["victim_vital_score"] = getattr(result, "victim_vital_score", 0.0)
                app_state["vital_components"] = getattr(result, "vital_components", {}) or {}
                app_state["heat_map_coverage"] = _hm_cov

            # 每幀同步 SLAM 位姿到 heat_map（取代 dead reckoning，消除累積漂移）
            if heat_map and ros_bridge:
                try:
                    slam_pose = ros_bridge.get_slam_pose()
                    if slam_pose:
                        heat_map.update_from_slam_pose(*slam_pose)
                except Exception:
                    pass
            if frame_count % 15 == 0:
                _refresh_victim_memory_state()

            # 事件列表更新（降頻：每 10 幀一次，減少鎖競爭）
            if frame_count % 10 == 0 or frame_count == 0:
                with state_lock:
                    app_state["events"] = event_logger.get_events()[:10]

            # ── 倒地警報（持續 3 秒觸發，交給 alert_manager 的背景執行緒處理）──
            # B5: 若同時偵測到失意識,延遲縮短至 1 秒(極度危險不能等)
            try:
                _fallen_stage = mission.current_stage
                _fallen_alert_allowed = (
                    _fallen_stage in ("SEARCH", "ANOMALY", "LOCK_ON")
                    and not _current_victim_already_reported(result)
                    and not alert_manager.is_broadcasting
                )
                if result.fallen_count > 0 and not suppress_known_victim and _fallen_alert_allowed:
                    now_t = time.time()
                    if _fallen_first_seen == 0.0:
                        _fallen_first_seen = now_t

                    fallen_duration = now_t - _fallen_first_seen
                    # B5: 失意識時加速觸發
                    _is_unconscious = (
                        getattr(result, 'vital_status', None) == "失去意識"
                        or getattr(result, 'consciousness_state', None) == "UNCONSCIOUS"
                    )
                    _alert_delay = 1.0 if _is_unconscious else FALLEN_ALERT_DELAY

                    if fallen_duration >= _alert_delay and not _fallen_alert_sent:
                        with _frame_lock:
                            alert_frame = _latest_frame

                        # B5: 完整生命跡象摘要(供 Telegram 與事件紀錄使用)
                        vital_block = _format_vital_block_from_result(result)

                        msg = (f"🚨 搜救系統偵測到人員倒地！(持續 {fallen_duration:.0f} 秒)\n"
                               f"人數: {result.person_count}  倒地: {result.fallen_count}\n"
                               f"VictimScore: {score:.2f} ({risk})")
                        if vital_block:
                            msg += "\n" + vital_block

                        # 用 alert_manager.trigger_alert 處理整段警報（背景執行緒 + 批次 pause/resume）
                        # 將 vital_status / consciousness 一併傳入,讓 TTS 動態決定播報文字
                        if alert_manager.trigger_alert(
                            frame=alert_frame,
                            message=msg,
                            critical=True,
                            vital_status=getattr(result, 'vital_status', None),
                            consciousness=getattr(result, 'consciousness_state', None),
                        ):
                            _fallen_alert_sent = True
                            add_log("danger", f"倒地持續 {fallen_duration:.0f}s，已發送警報 (Score:{score:.2f})")
                            event_logger.log_event(
                                frame=alert_frame, victim_score=score, risk_level=risk,
                                mission_stage=mission.current_stage,
                                person_count=result.person_count,
                                fallen_count=result.fallen_count,
                                components=components)
                            with state_lock:
                                app_state["events"] = event_logger.get_events()[:10]
                                app_state["event_count"] = event_logger.event_count
                            if detector.tracker:
                                detector.tracker.mark_all_visible_reported()
                            if heat_map:
                                try:
                                    heat_map.mark_person("rescued")
                                except Exception:
                                    pass
                else:
                    # 只有「真的沒有倒地」才中斷正在播放的倒地警報。
                    # 問答/REPORT/已通報 track 期間只是抑制重複警報，不代表人員恢復正常。
                    if result.fallen_count <= 0 and _fallen_alert_sent and alert_manager.is_broadcasting:
                        alert_manager.cancel_alert()
                        add_log("info", "人員恢復正常，警報已中斷")
                    _fallen_first_seen = 0.0
                    if result.fallen_count <= 0 or _fallen_stage in ("INQUIRY", "CONFIRM", "REPORT", "STANDBY"):
                        _fallen_alert_sent = False
            except Exception as e:
                logger.error(f"倒地警報錯誤: {e}")

            # Telegram 通知（REPORT 階段補充通報，與上方倒地即時通報互補）
            try:
                _cur_stage = mission.current_stage
                if _cur_stage == "REPORT" and not _report_alert_sent:
                    with _frame_lock:
                        alert_frame = _latest_frame
                    # B5: REPORT 通報也帶生命跡象摘要
                    vital_block = _format_vital_block_from_result(result)
                    _report_msg = (f"✅ 搜救系統通報：完成搜救流程\n"
                                   f"VictimScore: {score:.2f} ({risk})")
                    if vital_block:
                        _report_msg += "\n" + vital_block
                    if _report_memory_marked_id is None:
                        rec = _remember_current_victim(
                            "REPORT", score, risk, result=result, event="final_report"
                        )
                        _report_memory_marked_id = rec.victim_id
                    if _report_memory_marked_id:
                        _report_msg += f"\n已標記傷患 #{_report_memory_marked_id}"
                    if alert_manager.trigger_alert(
                        frame=alert_frame,
                        message=_report_msg,
                        critical=True,
                        vital_status=getattr(result, 'vital_status', None),
                        consciousness=getattr(result, 'consciousness_state', None),
                    ):
                        _report_alert_sent = True
                        if detector.tracker:
                            detector.tracker.mark_all_visible_reported()
                elif _cur_stage != "REPORT":
                    _report_alert_sent = False
                    _report_memory_marked_id = None
            except Exception:
                pass

            time.sleep(0.003)

        except Exception as e:
            logger.error(f"偵測迴圈錯誤: {e}")
            # 極短等待後繼續，避免畫面長時間凍結
            try:
                f = camera.get_frame()
                if f is not None:
                    camera.set_display_frame(f)
                    try:
                        detector.clear_transient_state(f)
                    except Exception:
                        pass
            except Exception:
                pass
            _clear_live_detection_state(stale=True)
            time.sleep(0.1)


# ============================================================
# 背景執行緒：音訊偵測
# ============================================================
def audio_loop():
    logger.info("音訊偵測迴圈啟動")
    add_log("info", "音訊偵測迴圈啟動")

    import time
    try:
        import speech_recognition as sr
        import numpy as np
        STT_AVAILABLE = True
    except ImportError:
        logger.warning("speech_recognition 套件未找到，停用全局 STT")
        STT_AVAILABLE = False

    last_stt_time = 0.0  # 限流計時器（取代脆弱的函數屬性）

    while True:
        try:
            if not audio_reader.is_available:
                time.sleep(2)
                continue

            # 防自我觸發：警報播放中或剛結束時暫停收音
            # （避免麥克風聽到自己的 TTS/警報聲 → STT 辨識出「救援」→ 無限迴圈崩潰）
            _audio_stage = mission.current_stage
            if _audio_stage in ("REPORT", "STANDBY"):
                time.sleep(2)
                continue
            # 警報系統正在播放（即使已切離 REPORT 階段）
            if alert_manager.is_broadcasting:
                time.sleep(2)
                continue
            # REPORT 結束後冷卻 15 秒，等警報音完全播完+餘音消散
            if hasattr(alert_manager, 'last_alert_time') and \
               time.time() - alert_manager.last_alert_time < 15:
                time.sleep(1)
                continue

            audio_buffer = audio_reader.get_audio_buffer(config.MIC_BUFFER_SEC)
            audio_result = audio_detector.detect_buffer(
                audio_buffer,
                window_size=config.MIC_CHUNK_SIZE
            )

            critical_help_detected = False

            # 全局收音防護網：只要測到語音，就送交 STT 找求救關鍵字 (限流每 3 秒一次避免 API 撐爆)
            if STT_AVAILABLE and audio_result.has_voice:
                current_time = time.time()
                if current_time - last_stt_time > 3.0:
                    last_stt_time = current_time
                    try:
                        audio_16bit = (audio_buffer * 32767).astype(np.int16)
                        audio_data = sr.AudioData(audio_16bit.tobytes(), sample_rate=config.MIC_SAMPLE_RATE, sample_width=2)
                        
                        recognizer = sr.Recognizer()
                        text = recognizer.recognize_google(audio_data, language="zh-TW")
                        recognized_text = text.lower()
                        logger.info(f"[全局收音] 背後監聽文字: '{recognized_text}'")
                        
                        keywords = ["救", "幫", "需要", "help", "sos", "please", "痛", "受傷",
                                    "救命", "幫忙", "有人嗎", "來人", "快來", "危險",
                                    "hurt", "emergency", "danger", "嗚", "啊", "哎"]
                        if any(kw in recognized_text for kw in keywords):
                            logger.warning("🚨 [全局收音] 偵測到明確求救關鍵字，強行介入通報系統！")
                            critical_help_detected = True
                    except sr.UnknownValueError:
                        pass
                    except Exception as e:
                        logger.error(f"STT 錯誤: {e}") # Added error logging for STT

            # 將結果更新到 global 事件狀態
            with state_lock:
                app_state["audio_event"] = {
                    "has_voice": bool(audio_result.has_voice),
                    "help_score": float(audio_result.help_score),
                    "knock_detected": bool(audio_result.knock_detected),
                    "rms_level": float(audio_result.rms_level)
                }
                
                # 如果背景全局聽到了救命，記錄 Panic Mode，並進行 360 度搜尋
                if critical_help_detected:
                    mission.panic_mode_until = time.time() + 60.0  # 持續警戒 60 秒
                    if app_state["mode"] != "auto":
                        app_state["mode"] = "auto"
                    
                    if mission.current_stage in ["STANDBY", "SEARCH"]:
                        mission.transition_to("SEARCH")
                        add_log("warn", "🚨 [全局求救] 啟動 360 度全景尋人視角與強制鎖定！")

            # Original logic for logging help/knock detection (kept for consistency)
            if audio_result.help_score > config.HELP_THRESHOLD:
                add_log("warn", f"🗣️ 偵測到呼救聲 (score: {audio_result.help_score:.2f})")
            if audio_result.knock_detected:
                add_log("warn", "🔨 偵測到敲擊聲！")

            time.sleep(0.5) # Changed from 0.05 to 0.5 as per new code

        except Exception as e:
            logger.error(f"音訊迴圈錯誤: {e}")
            time.sleep(1)


# ============================================================
# 背景執行緒：手動控制失聯保護
# ============================================================
def manual_safety_loop():
    """手動蘑菇頭心跳逾時即停，避免停止封包遺失後持續前進。"""
    add_log("info", "手動控制 watchdog 啟動")
    while True:
        try:
            with state_lock:
                mode = app_state.get("mode")
                active = bool(app_state.get("manual_motion_active", False))
                last_ts = float(app_state.get("manual_last_cmd_ts", 0.0) or 0.0)

            if (
                mode == "manual"
                and active
                and time.time() - last_ts > config.MANUAL_CONTROL_TIMEOUT_SEC
            ):
                motor.stop()
                with state_lock:
                    app_state["manual_motion_active"] = False
                add_log("warn", "手動控制逾時，已自動停車")

            time.sleep(0.05)
        except Exception as e:
            logger.error(f"手動控制 watchdog 錯誤: {e}")
            time.sleep(0.2)


# ============================================================
# 背景執行緒：LiDAR 安全煞車
# ============================================================
def lidar_safety_loop():
    """
    LiDAR 360° 全車身即時安全煞車（20 Hz）。
    自動巡邏中任何可能移動的階段（SEARCH / ANOMALY / LOCK_ON），
    任何方向碰撞距離 < safety margin 立刻停車 + 後退 + 轉彎。
    內建 watchdog：LiDAR 資料超過 2 秒未更新 → 記錄警告。
    """
    _escape_dir = 1  # 交替左右轉避免卡死
    _watchdog_warned = False
    LOOP_PERIOD = 0.05  # 50ms = 20Hz
    # scanning_lock watchdog：避免 scan_patrol 卡住鎖導致防撞永久被跳過
    _lock_first_seen = 0.0
    SCAN_LOCK_MAX_SEC = 5.0

    while True:
        try:
            if not ros_bridge or not ros_bridge.is_available():
                with state_lock:
                    app_state["distance_cm"] = -1
                time.sleep(0.5)
                continue

            # LiDAR watchdog
            if not ros_bridge.is_lidar_alive(config.ROS_LIDAR_WATCHDOG_SEC):
                if not _watchdog_warned:
                    add_log("warn", "⚠️ LiDAR 資料超時")
                    _watchdog_warned = True
                with state_lock:
                    app_state["distance_cm"] = -1
                time.sleep(0.5)
                continue
            _watchdog_warned = False

            # 取得前方最近距離（給 UI、給其他模組沿用）
            front_dist_cm = ros_bridge.get_front_distance_cm(config.ROS_LIDAR_FRONT_ARC_DEG)
            with state_lock:
                app_state["distance_cm"] = front_dist_cm
                cur_mode = app_state["mode"]

            # 360° 全方位最近障礙物（UI 顯示）
            lidar_obs = ros_bridge.get_lidar_obstacles(config.ROS_LIDAR_OBSTACLE_THRESHOLD_M)
            if lidar_obs:
                closest = min(lidar_obs, key=lambda o: o[1])
                with state_lock:
                    app_state["lidar_closest_cm"] = round(closest[1] * 100, 1)
                    app_state["lidar_closest_angle"] = round(closest[0], 1)

            # 全車身防撞：把車當成非對稱矩形 footprint
            body = ros_bridge.check_body_collision()
            with state_lock:
                app_state["body_clear_cm"] = body["min_clear_cm"]
                app_state["body_min_angle"] = body["min_angle_deg"]

            # 掃描旋轉期間跳過安全煞車（避免搶馬達控制權）
            # 但加上 watchdog：若 lock 被持有 > SCAN_LOCK_MAX_SEC，強制清除並繼續防撞
            from scan_patrol import scanning_lock
            if scanning_lock.is_set():
                if _lock_first_seen == 0.0:
                    _lock_first_seen = time.time()
                elif time.time() - _lock_first_seen > SCAN_LOCK_MAX_SEC:
                    add_log("warn",
                            f"⚠️ scanning_lock 持有 >{SCAN_LOCK_MAX_SEC:.0f}s，"
                            "強制清除以恢復防撞")
                    scanning_lock.clear()
                    _lock_first_seen = 0.0
                else:
                    time.sleep(LOOP_PERIOD)
                    continue
            else:
                _lock_first_seen = 0.0

            cur_stage = mission.current_stage
            moving_stages = ("SEARCH", "ANOMALY", "LOCK_ON")

            # 防撞觸發：全車身任一方向碰撞 OR 前方傳統閾值
            brake_cm = config.ROS_LIDAR_BRAKE_THRESHOLD_M * 100
            front_too_close = (0 < front_dist_cm <= brake_cm)
            body_collision = body["collision"]

            if (cur_mode == "auto"
                    and cur_stage in moving_stages
                    and (front_too_close or body_collision)):
                motor.stop()
                if body_collision:
                    add_log("warn",
                            f"⚠️ 車身防撞 {body['min_clear_cm']:.0f}cm @ {body['min_angle_deg']:.0f}° → 後退+轉彎")
                else:
                    add_log("warn", f"⚠️ LiDAR 安全煞車 {front_dist_cm:.0f}cm → 後退+轉彎")

                # 後退（先確認車尾安全）
                if body["rear_clear_cm"] > 8 or body["rear_clear_cm"] < 0:
                    motor.move(0, -config.PATROL_REVERSE_SPEED, 0)
                    time.sleep(0.4)
                    motor.stop()
                # 轉向：用兩側淨空判斷
                left_clear = body["left_clear_cm"]
                right_clear = body["right_clear_cm"]
                if left_clear > 0 and right_clear > 0:
                    _escape_dir = 1 if right_clear > left_clear else -1
                else:
                    ranges = ros_bridge.get_lidar_ranges()
                    if ranges:
                        left_free = sum(1 for a, d in ranges.items() if -135 <= a <= -45 and d > 0.5)
                        right_free = sum(1 for a, d in ranges.items() if 45 <= a <= 135 and d > 0.5)
                        if left_free != right_free:
                            _escape_dir = 1 if right_free > left_free else -1
                motor.move(0, 0, config.PATROL_TURN_SPEED * _escape_dir)
                time.sleep(0.4)
                motor.stop()
                _escape_dir *= -1

            time.sleep(LOOP_PERIOD)
        except Exception:
            time.sleep(1)


# ============================================================
# 背景執行緒：搜索巡檢 (V4 — D 靜止 + E 智慧巡邏)
# ============================================================
def search_loop():
    global scan_patrol_instance
    add_log("info", "搜索巡檢執行緒啟動")
    from scan_patrol import ScanPatrol

    _expected_sm = [None]  # 用 list 讓閉包可修改

    def _check_active():
        with state_lock:
            m = app_state["mode"]
            sm = app_state.get("search_mode", "E")
        # 模式或搜索策略改變 → 立即中斷當前週期
        if _expected_sm[0] is not None and sm != _expected_sm[0]:
            return False
        # 允許 SEARCH 和 ANOMALY 階段繼續巡邏（ANOMALY 只是觀察確認，不需要停車）
        return m == "auto" and mission.current_stage in ("SEARCH", "ANOMALY")

    def _get_search_mode():
        with state_lock:
            return app_state.get("search_mode", "E")

    def _get_distance():
        """取得前方最近距離 (cm)，優先從 LiDAR 即時讀取。"""
        if ros_bridge and ros_bridge.is_available():
            d = ros_bridge.get_front_distance_cm(config.ROS_LIDAR_FRONT_ARC_DEG)
            if d > 0:
                return d
        with state_lock:
            return app_state["distance_cm"]

    def _get_victim_score():
        with state_lock:
            return app_state.get("victim_score", 0.0)

    def _update_pan(pan):
        with state_lock:
            app_state["patrol_pan"] = pan

    def _ground_escape_turn_dir():
        """視覺低障礙煞車時決定轉向:用 LiDAR 兩側淨空挑較空的一側。
        回傳 +1(右) / -1(左)。LiDAR 不可用時預設右轉。"""
        try:
            if ros_bridge and ros_bridge.is_available():
                ranges = ros_bridge.get_lidar_ranges()
                if ranges:
                    # 右 = 正角度區(45~135), 左 = 負角度區(-135~-45),取較空一側
                    right_free = sum(1 for a, d in ranges.items() if 45 <= a <= 135 and d > 0.6)
                    left_free = sum(1 for a, d in ranges.items() if -135 <= a <= -45 and d > 0.6)
                    if right_free != left_free:
                        return 1 if right_free > left_free else -1
        except Exception:
            pass
        return 1

    _unstuck_dir = [1]   # 交替轉向,避免又卡回同一處

    def _force_unstuck():
        """高階脫困:被小物卡死時的強力掙脫(長後退 + 大角度轉向,交替方向)。
        比 scan_patrol 內部 escape 更激進,專門對付 LiDAR/視覺都沒抓到的卡夾。"""
        rev_pwm = float(getattr(config, 'PATROL_UNSTUCK_REVERSE_PWM', 0.45))
        rev_sec = float(getattr(config, 'PATROL_UNSTUCK_REVERSE_SEC', 1.0))
        turn_sec = float(getattr(config, 'PATROL_UNSTUCK_TURN_SEC', 1.2))
        motor.stop()
        # 1. 強力後退掙脫
        motor.move(0, -rev_pwm, 0)
        time.sleep(rev_sec)
        motor.stop()
        # 2. 大角度轉向(朝 LiDAR 較空側,失敗則交替)
        d = _ground_escape_turn_dir()
        if d == 0:
            d = _unstuck_dir[0]
        motor.move(0, 0, config.PATROL_TURN_PWM * d)
        time.sleep(turn_sec)
        motor.stop()
        _unstuck_dir[0] = -d   # 下次反向
        # 3. 清掉 scan_patrol 當前 bucket 記憶,避免殘留近距讀值又立刻判定阻塞
        try:
            if scan_patrol_instance:
                scan_patrol_instance._bucket_memory.clear()
        except Exception:
            pass

    def _depart_reported_victim():
        """Leave the reported victim before resuming patrol."""
        with state_lock:
            app_state["departing_victim"] = True
        try:
            victim_id = app_state.get("last_reported_victim_id")
            add_log("info", f"傷患 #{victim_id or '-'} 已標記，倒車離場後恢復巡邏")
            motor.stop()
            time.sleep(0.2)

            rear_clear = -1
            left_clear = -1
            right_clear = -1
            try:
                if ros_bridge and ros_bridge.is_available():
                    body = ros_bridge.check_body_collision()
                    rear_clear = body.get("rear_clear_cm", -1)
                    left_clear = body.get("left_clear_cm", -1)
                    right_clear = body.get("right_clear_cm", -1)
            except Exception:
                pass

            rear_min = float(getattr(config, "VICTIM_DEPART_REAR_MIN_CM", 12))
            if rear_clear < 0 or rear_clear >= rear_min:
                motor.move(0, -float(getattr(config, "VICTIM_DEPART_REVERSE_PWM", 0.30)), 0)
                time.sleep(float(getattr(config, "VICTIM_DEPART_REVERSE_SEC", 1.4)))
                motor.stop()
            else:
                add_log("warn", f"後方餘裕 {rear_clear:.0f}cm 不足，略過倒車離場")

            turn_dir = 1
            if left_clear > 0 and right_clear > 0:
                turn_dir = 1 if right_clear >= left_clear else -1
            else:
                turn_dir = _ground_escape_turn_dir()
            motor.move(0, 0, float(getattr(config, "VICTIM_DEPART_TURN_PWM", 0.32)) * turn_dir)
            time.sleep(float(getattr(config, "VICTIM_DEPART_TURN_SEC", 0.9)))
            motor.stop()

            if victim_id:
                try:
                    victim_memory.mark_departed(int(victim_id))
                except Exception:
                    pass
            _refresh_victim_memory_state()
        finally:
            motor.stop()
            with state_lock:
                app_state["departing_victim"] = False

    # ── SLAM 驅動反應式巡邏實例（V12：模式 D / E / F 統一，巡邏時不動舵機）──
    scan = ScanPatrol(
        motor=motor, servo=servo,
        get_distance_fn=_get_distance,
        get_victim_score_fn=_get_victim_score,
        check_active_fn=_check_active,
        add_log_fn=add_log,
        update_pan_fn=_update_pan,
        heat_map=heat_map,
        ros_bridge=ros_bridge,
    )
    scan_patrol_instance = scan  # 公開給 /control /status 讀取

    _was_auto = False
    _prev_mode = None
    _was_reporting = False     # 是否在本次巡邏週期內已進入過 REPORT（供 REPORT→STANDBY 自動恢復用）
    _last_known_avoid_ts = 0.0

    # 脫困 watchdog 狀態
    _wd_last_move_ts = time.time()   # 上次「確實移動」的時間
    _wd_last_pose = None             # 上次 SLAM pose (x, y, yaw)

    while True:
        try:
            with state_lock:
                mode = app_state["mode"]

            if mode != "auto":
                if _was_auto:
                    motor.stop()
                    try:
                        servo.sweep_end()
                    except Exception:
                        pass
                    # 離開 auto → 關閉視覺低障礙高頻偵測(回到每 10 幀省 CPU)
                    try:
                        detector.obstacle_scan_active = False
                    except Exception:
                        pass
                    _was_auto = False
                    _was_reporting = False   # 切回手動 → 清空 REPORT 記憶
                time.sleep(0.5)
                continue
            if not _was_auto:
                # 進入自動巡邏：把舵機固定在預設位，之後巡邏迴圈完全不動舵機
                try:
                    servo.set_angle(config.SERVO_DEFAULT_PAN, config.SERVO_DEFAULT_TILT)
                    _update_pan(int(config.SERVO_DEFAULT_PAN))
                except Exception:
                    pass
                # 進入 auto → 啟用視覺低障礙高頻偵測(補 LiDAR 車頂盲區)
                try:
                    detector.obstacle_scan_active = True
                except Exception:
                    pass
                _wd_last_move_ts = time.time()   # 重置脫困計時,避免剛進 auto 誤觸
                _wd_last_pose = None
                _was_auto = True
            _was_auto = True

            current_stage = mission.current_stage
            if current_stage == "REPORT":
                current_stage = mission.tick()

            # ── 高階脫困 watchdog:卡死 + 前方無人 → 強制脫困 ──
            # 追蹤 SLAM 位移;只在「SEARCH/ANOMALY 巡邏 + 無人 + 有 pose 卻長時間沒動」時觸發。
            _wd_moved = False
            _wd_have_pose = False
            if ros_bridge and ros_bridge.is_available():
                _wp = ros_bridge.get_slam_pose()
                if _wp:
                    _wd_have_pose = True
                    if _wd_last_pose is not None:
                        _dx = _wp[0] - _wd_last_pose[0]
                        _dy = _wp[1] - _wd_last_pose[1]
                        _dyaw = _wp[2] - _wd_last_pose[2]
                        while _dyaw > 3.14159265:
                            _dyaw -= 6.28318530
                        while _dyaw < -3.14159265:
                            _dyaw += 6.28318530
                        _eps = float(getattr(config, 'PATROL_STUCK_WATCHDOG_POSE_EPS_M', 0.05))
                        _yeps = float(getattr(config, 'PATROL_STUCK_WATCHDOG_YAW_EPS', 0.10))
                        if (_dx * _dx + _dy * _dy) >= (_eps * _eps) or abs(_dyaw) >= _yeps:
                            _wd_moved = True
                    _wd_last_pose = _wp
            with state_lock:
                _wd_person = app_state.get("person_count", 0)
            # 重置條件:有移動 / 有人(合法停車) / 非巡邏階段 / 無 pose(LiDAR 死交給 scan watchdog)
            if (_wd_moved or _wd_person > 0
                    or current_stage not in ("SEARCH", "ANOMALY")
                    or not _wd_have_pose):
                _wd_last_move_ts = time.time()
            elif (time.time() - _wd_last_move_ts) > config.PATROL_STUCK_WATCHDOG_SEC:
                _immobile = time.time() - _wd_last_move_ts
                add_log("warn", f"⚠️ 卡死 {_immobile:.0f}s 且前方無人 → 強制脫困")
                _force_unstuck()
                _wd_last_move_ts = time.time()
                _wd_last_pose = None
                continue

            # ── LOCK_ON：漸進靠近目標 ──
            if current_stage == "LOCK_ON":
                with state_lock:
                    pan = app_state.get("patrol_pan", 0)

                if abs(pan) > 15:
                    turn_speed = config.PATROL_TURN_SPEED if pan > 0 else -config.PATROL_TURN_SPEED
                    motor.move(0, 0, turn_speed)
                    new_pan = pan - (15 if pan > 0 else -15)
                    if abs(new_pan) < 15:
                        new_pan = 0
                    servo.set_angle(new_pan, config.SERVO_DEFAULT_TILT)
                    _update_pan(new_pan)
                    time.sleep(0.2)
                    continue

                dist_now = _get_distance()
                visual_guard = _victim_visual_guard()
                stop_cm = float(getattr(config, "VICTIM_APPROACH_STOP_CM", 55))
                reverse_cm = float(getattr(config, "VICTIM_APPROACH_REVERSE_CM", 38))
                slow_cm = float(getattr(config, "VICTIM_APPROACH_SLOW_CM", 110))

                if 0 < dist_now < reverse_cm:
                    motor.move(0, -config.PATROL_REVERSE_SPEED * 0.55, 0)
                elif (0 < dist_now <= stop_cm) or visual_guard.get("stop"):
                    motor.stop()
                elif (0 < dist_now <= slow_cm) or visual_guard.get("slow"):
                    motor.move(0, config.SMART_PATROL_SPEED * 0.22, 0)
                else:
                    motor.move(0, config.SMART_PATROL_SPEED * 0.45, 0)

                time.sleep(0.2)
                continue

            # ── 非搜索階段 → 停車 ──
            if current_stage not in ("SEARCH", "ANOMALY"):
                if current_stage in ("INQUIRY", "CONFIRM", "REPORT", "STANDBY"):
                    motor.stop()

                # 追蹤：是否曾進入 REPORT；用於 STANDBY 自動恢復巡邏判定
                if current_stage == "REPORT":
                    _was_reporting = True

                # ── REPORT → STANDBY 轉換：警報週期結束 → 自動恢復巡邏 ──
                # 條件：自動模式 + 剛剛才從 REPORT 出來 + 警報 TTS 已播完
                if (current_stage == "STANDBY" and _was_reporting
                        and not alert_manager.is_broadcasting):
                    add_log("info", "通報完成，開始離場動作")
                    _was_reporting = False
                    _depart_reported_victim()
                    add_log("info", "離場完成，自動恢復搜索巡邏")
                    _bg(lambda: mission.start_mission())

                time.sleep(0.5)
                continue

            # ── SEARCH / ANOMALY 階段 ──
            sm = _get_search_mode()

            # 模式切換時停車
            if sm != _prev_mode:
                motor.stop()
                _prev_mode = sm

            _expected_sm[0] = sm  # 記錄當前模式，切換時 _check_active 會立即回 False

            # ── 安全優先：偵測到倒地人員立即停車 ──
            # 融合分數可能還沒升到 ANOMALY/LOCK_ON，但此時機器人仍在前進會輾壓受困者。
            # 只要偵測到倒地個案就硬停，由後續階段（LOCK_ON → INQUIRY → CONFIRM）接管。
            with state_lock:
                fallen_now = app_state.get("fallen_count", 0)
                near_known = app_state.get("near_reported_victim")
            if fallen_now > 0:
                if near_known or _current_victim_already_reported(detector.latest_result):
                    now_avoid = time.time()
                    if now_avoid - _last_known_avoid_ts > 3.0:
                        if near_known:
                            add_log("info", f"靠近已通報傷患 #{near_known.get('victim_id')}，保持距離並繞開")
                        else:
                            add_log("info", "靠近已通報 track，保持距離並繞開")
                        _last_known_avoid_ts = now_avoid
                    motor.stop()
                    motor.move(0, -float(getattr(config, "VICTIM_DEPART_REVERSE_PWM", 0.30)), 0)
                    time.sleep(0.45)
                    motor.stop()
                    motor.move(0, 0, float(getattr(config, "VICTIM_DEPART_TURN_PWM", 0.32)) * _ground_escape_turn_dir())
                    time.sleep(0.45)
                    motor.stop()
                    continue
                motor.stop()
                time.sleep(0.15)
                continue

            # ── 視覺低障礙避障：補 LiDAR 車頂盲區（椅腳/桌腳/行李等）──
            # LiDAR 掃描平面看不到低物,改用攝影機 YOLO 偵測。
            # BRAKE = 非常近 → 停車 + 後退 + 轉向;WARN 由 scan_patrol 內部減速(交給 LiDAR gap)。
            try:
                g_level = detector.latest_result.ground_obstacle_level
            except Exception:
                g_level = "CLEAR"
            if g_level == "BRAKE":
                motor.stop()
                # 朝 LiDAR 較空的一側轉(若可用),否則固定右轉
                turn_dir = _ground_escape_turn_dir()
                add_log("warn", f"👁️ 視覺偵測前方低障礙(LiDAR 盲區)→ 停車後退轉向({'右' if turn_dir > 0 else '左'})")
                motor.move(0, -config.PATROL_REVERSE_SPEED, 0)
                time.sleep(0.45)
                motor.stop()
                motor.move(0, 0, config.PATROL_TURN_SPEED * turn_dir)
                time.sleep(0.6)
                motor.stop()
                time.sleep(0.1)
                continue

            # 統一使用 SLAM 反應式控制器（scan_patrol V13）；巡邏時不動舵機
            scan.run_cycle()

        except Exception as e:
            logger.warning(f"搜索迴圈錯誤: {e}")
            motor.stop()
            time.sleep(1)


# ============================================================
# Flask 路由
# ============================================================
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/demo")
def demo_page():
    return render_template("demo.html")


@app.route("/video_feed")
def video_feed():
    return Response(
        camera.generate_mjpeg(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/events/<path:filename>")
def serve_event_image(filename):
    return send_from_directory(config.EVENT_DIR, filename)


@app.route("/control", methods=["POST"])
def control():
    """所有操作均為非阻塞（耗時工作推入背景）"""
    data = request.get_json(silent=True)
    if data is None and request.data:
        try:
            data = json.loads(request.data.decode("utf-8"))
        except Exception:
            data = None
    if not isinstance(data, dict):
        return jsonify({"status": "error"}), 400

    action = data.get("action", "")

    try:
        if action == "move":
            def _manual_axis(value):
                v = max(-1.0, min(1.0, float(value)))
                return 0.0 if abs(v) < config.MANUAL_COMMAND_DEADZONE else v

            vx = _manual_axis(data.get("vx", 0))
            vy = _manual_axis(data.get("vy", 0))
            omega = _manual_axis(data.get("omega", 0))
            is_zero = vx == 0.0 and vy == 0.0 and omega == 0.0

            try:
                seq = int(data["seq"]) if "seq" in data else -1
            except (TypeError, ValueError):
                seq = -1

            manual_session = str(data.get("session", ""))[:80] or None
            now = time.time()
            need_force_manual = False
            stale = False
            with state_lock:
                if manual_session and manual_session != app_state.get("manual_session"):
                    app_state["manual_session"] = manual_session
                    app_state["manual_last_seq"] = -1
                last_seq = int(app_state.get("manual_last_seq", -1) or -1)
                if seq >= 0 and seq < last_seq and not is_zero:
                    stale = True
                else:
                    if seq >= 0 and seq > last_seq:
                        app_state["manual_last_seq"] = seq
                    app_state["manual_last_cmd_ts"] = now
                    app_state["manual_motion_active"] = not is_zero
                    if app_state.get("mode") != "manual":
                        app_state["mode"] = "manual"
                        need_force_manual = True

            if stale:
                return jsonify({"status": "stale", "last_seq": last_seq})

            if need_force_manual:
                mission.force_manual()

            # 零值立即煞停（繞過 motor 內部 ramp，避免搖桿放開後車輛殘餘滑行）
            if is_zero:
                motor.stop()
            else:
                motor.move(vx, vy, omega, accel_rate=config.MANUAL_ACCEL_RATE)
        elif action == "servo":
            # ── 網頁雲台控制已暫時停用 ──
            add_log("info", "雲台控制已暫時停用")
        elif action == "mode_auto":
            with state_lock:
                app_state["mode"] = "auto"
                app_state["manual_motion_active"] = False
            motor.stop()
            _bg(lambda: mission.start_mission())
            add_log("info", "切換至自動搜索模式")
        elif action == "mode_manual":
            with state_lock:
                app_state["mode"] = "manual"
                app_state["manual_motion_active"] = False
                app_state["manual_last_seq"] = -1
                app_state["manual_session"] = None
            mission.force_manual()
            motor.stop()
            try:
                servo.sweep_end()   # 立即停止掃描中的 servo
            except Exception:
                pass
            add_log("info", "切換至手動模式")
        elif action == "stop":
            motor.stop()
            try:
                servo.sweep_end()
            except Exception:
                pass
            with state_lock:
                app_state["mode"] = "manual"
                app_state["manual_motion_active"] = False
                app_state["manual_last_seq"] = -1
                app_state["manual_session"] = None
            mission.force_manual()
            add_log("warn", "緊急停止")
        elif action == "home":
            # 雲台歸位到 config 預設角度（手動調整硬體後的還原入口）
            servo.home()
            add_log("info", f"雲台歸位 (pan={config.SERVO_DEFAULT_PAN}, tilt={config.SERVO_DEFAULT_TILT})")
        elif action == "search_mode":
            # D (靜止掃描) / E (智慧巡邏) 已淘汰；統一使用 F (SLAM 反應式巡邏)
            with state_lock:
                app_state["search_mode"] = "F"
            add_log("info", "搜索模式: SLAM 掃描巡邏")
        elif action == "set_brightness":
            val = int(data.get("value", 0))
            with state_lock:
                app_state["brightness"] = val
            add_log("info", f"設定亮度偏移: {val}")
        elif action == "toggle_night_vision":
            with state_lock:
                cur = app_state.get("night_vision", "auto")
                nxt = {"auto": "on", "on": "off", "off": "auto"}.get(cur, "auto")
                app_state["night_vision"] = nxt
            mode_name = {"auto": "自動", "on": "強制開啟", "off": "關閉"}
            add_log("info", f"夜視模式: {mode_name.get(nxt, nxt)}")
        elif action == "reset_heat_map":
            if heat_map:
                heat_map.reset()
            if ros_bridge:
                ros_bridge.reset_pose_history()
            if scan_patrol_instance:
                scan_patrol_instance.reset_memory()
            victim_memory.clear()
            _refresh_victim_memory_state()
            add_log("info", "熱區地圖、SLAM 軌跡、巡邏記憶已重置")
        elif action == "refresh_slam_map":
            ok = ros_bridge.refresh_slam_map() if ros_bridge else False
            add_log("info", "SLAM 地圖已刷新" if ok else "SLAM 刷新失敗")
        elif action == "restart_slam":
            # 完全重建地圖：重啟 rescue-lidar.service（清空 slam_toolbox 內部 pose graph + occupancy grid）。
            # 需要 /etc/sudoers.d/ 裡有這行讓帳號免密執行：
            #   blackwhite0625 ALL=(ALL) NOPASSWD: /bin/systemctl restart rescue-lidar.service
            def _restart_slam_bg():
                try:
                    motor.stop()
                    add_log("warn", "⏳ SLAM 服務重啟中…請等候約 15 秒")
                    # 同時清本地記憶，避免停用舊資料
                    if heat_map:
                        heat_map.reset()
                    if ros_bridge:
                        ros_bridge.reset_pose_history()
                    if scan_patrol_instance:
                        scan_patrol_instance.reset_memory()
                    victim_memory.clear()
                    _refresh_victim_memory_state()
                    result = subprocess.run(
                        ["sudo", "-n", "/bin/systemctl", "restart", "rescue-lidar.service"],
                        capture_output=True, text=True, timeout=20,
                    )
                    if result.returncode == 0:
                        add_log("info", "✅ SLAM 服務已重啟，地圖已清空 — 等待 LiDAR 重新連線…")
                    else:
                        err = (result.stderr or result.stdout or "").strip()[:200]
                        add_log("danger", f"❌ SLAM 重啟失敗：{err}")
                        add_log("warn", "請確認 sudoers 已設定免密：blackwhite0625 NOPASSWD /bin/systemctl restart rescue-lidar.service")
                except subprocess.TimeoutExpired:
                    add_log("danger", "❌ SLAM 重啟逾時（>20s）")
                except FileNotFoundError:
                    add_log("danger", "❌ 找不到 systemctl（僅 Linux systemd 系統支援）")
                except Exception as e:
                    add_log("danger", f"❌ SLAM 重啟錯誤：{e}")
            _bg(_restart_slam_bg)
        elif action == "toggle_intercom":
            if intercom:
                if intercom.is_active:
                    intercom.stop()
                    add_log("info", "對講模式已關閉")
                else:
                    intercom.start()
                    add_log("info", "對講模式已開啟")
        elif action == "intercom_speak":
            text = data.get("text", "")
            if intercom and text:
                intercom.speak(text)
                add_log("info", f"[對講] {text[:30]}")

        # --- 手動測試（全部背景執行） ---
        elif action == "set_stage":
            stage = data.get("stage", "STANDBY")
            if stage in mission.STAGES:
                mission.transition_to(stage)
                with state_lock:
                    app_state["mission_stage"] = stage
                add_log("info", f"[手動] 狀態: {stage}")
        elif action == "test_inquiry":
            with state_lock:
                app_state["mode"] = "auto"
                mission.transition_to("INQUIRY")
            mission._start_inquiry_async(None)
            add_log("info", "[手動] 強制進入 INQUIRY 測試真實收音與通報流程")
        elif action == "test_report":
            add_log("info", "[手動] 事件回報中...")
            def _do_report():
                with state_lock:
                    sc = app_state["victim_score"]
                    rk = app_state["risk_level"]
                    pc = app_state["person_count"]
                    fc = app_state["fallen_count"]
                    cp = dict(app_state.get("fusion_components", {}))
                with _frame_lock:
                    report_frame = _latest_frame
                event_logger.log_event(
                    frame=report_frame, victim_score=sc, risk_level=rk,
                    mission_stage=mission.current_stage,
                    person_count=pc, fallen_count=fc, components=cp)
                with state_lock:
                    app_state["events"] = event_logger.get_events()[:10]
                    app_state["event_count"] = event_logger.event_count
                add_log("warn", f"[手動] 回報完成 | Score: {sc:.2f}")
            _bg(_do_report)
        elif action == "simulate_audio":
            t = data.get("event_type", "voice")
            with state_lock:
                if t == "voice":
                    app_state["audio_event"] = {"has_voice": True, "help_score": 0.8, "knock_detected": False, "rms_level": 0.1}
                elif t == "knock":
                    app_state["audio_event"] = {"has_voice": False, "help_score": 0.0, "knock_detected": True, "rms_level": 0.15}
                elif t == "clear":
                    app_state["audio_event"] = {"has_voice": False, "help_score": 0.0, "knock_detected": False, "rms_level": 0.0}
            add_log("info", f"[模擬] 音訊: {t}")
        elif action == "trigger_critical_help":
            # 手動模擬收到關鍵字求救，並強制推入 CONFIRM 階段觸發事件
            from hri_module import InquiryResult
            mission._inquiry_result = InquiryResult(completed=True, critical_help_requested=True, recognized_text="救命 (Demo)")
            with state_lock:
                app_state["mode"] = "auto"
                mission.transition_to("CONFIRM")
            add_log("warn", "[模擬] 收到明確求救語音！強制進入確認並發報")
        elif action == "simulate_score":
            sc = float(data.get("score", 0))
            with state_lock:
                app_state["victim_score"] = sc
                app_state["risk_level"] = "HIGH" if sc >= 0.6 else ("SUSPECT" if sc >= 0.3 else "LOW")
                app_state["fusion_components"] = {
                    "person": min(sc * 1.5, 1.0), "pose": min(sc * 1.2, 1.0),
                    "audio": min(sc * 0.8, 1.0), "motion": min(sc * 0.5, 1.0), "distance": 0.5}
            add_log("info", f"[模擬] Score = {sc:.2f}")
    except Exception as e:
        logger.error(f"control 錯誤: {e}")

    return jsonify({"status": "ok"})


# ── 對講 API ──
@app.route("/intercom/presets")
def intercom_presets():
    """取得預設對講訊息"""
    if not intercom:
        return jsonify([])
    from intercom import PRESET_MESSAGES
    return jsonify(PRESET_MESSAGES)


@app.route("/intercom/listen")
def intercom_listen():
    """取得自上次呼叫以來的新麥克風音訊（WAV 格式）。
    v2: 使用位置追蹤保證相鄰 fetch 無重疊亦無空隙，並內建增益 + 噪音閘門。"""
    if not intercom or not intercom.is_active:
        return b'', 204
    if not audio_reader.is_available:
        return b'', 204
    try:
        wav = intercom.fetch_audio_chunk()
        if not wav:
            return b'', 204
        return Response(wav, mimetype="audio/wav")
    except Exception:
        return b'', 204


@app.route("/status")
def status():
    s = get_state()
    map_requested = request.args.get("map", "0") == "1"
    return jsonify({
        "mission_stage": s["mission_stage"],
        "mode": s["mode"],
        "person_count": s["person_count"],
        "fallen_count": s["fallen_count"],
        "fallen_persons": s.get("fallen_persons", []),
        "pose_anomaly_score": s["pose_anomaly_score"],
        "victim_score": s["victim_score"],
        "risk_level": s["risk_level"],
        "distance_cm": s["distance_cm"],
        "audio_event": s["audio_event"],
        "fusion_components": s["fusion_components"],
        "search_mode": s["search_mode"],
        "alert_count": s["alert_count"],
        "event_count": s["event_count"],
        "fps": s["fps"],
        "detect_ms": s["detect_ms"],
        "ai_inference_paused": s.get("ai_inference_paused", False),
        "detection_stale": s.get("detection_stale", False),
        "last_detection_age_sec": round(time.time() - s.get("last_detection_ts", 0.0), 2)
                                  if s.get("last_detection_ts", 0.0) else None,
        "ai_loaded": s["ai_loaded"],
        "ai_backend": detector.backend_name,
        "mic_ok": s["mic_ok"],
        "camera_ok": s["camera_ok"],
        "gpio_ok": s["gpio_ok"],
        "patrol_pan": s["patrol_pan"],
        "eye_state": s.get("eye_state", "UNKNOWN"),
        "heart_rate_bpm": s.get("heart_rate_bpm", -1.0),
        "rppg_confidence": s.get("rppg_confidence", 0.0),
        "rppg_signal_quality": s.get("rppg_signal_quality", "UNKNOWN"),
        "respiration_rate": s.get("respiration_rate", -1.0),
        "resp_confidence": s.get("resp_confidence", 0.0),
        "rr_buffer_ratio": s.get("rr_buffer_ratio", 0.0),
        "hr_buffer_ratio": s.get("hr_buffer_ratio", 0.0),
        "blink_warmup_ratio": s.get("blink_warmup_ratio", 0.0),
        "blink_rate_per_min": s.get("blink_rate_per_min", -1.0),
        "consciousness_state": s.get("consciousness_state", "UNKNOWN"),
        "micro_motion_score": s.get("micro_motion_score", 0.0),
        # B5: 整合性生命跡象 (心率+呼吸+眨眼+意識+微動 → 綜合分數 + 中文狀態)
        "vital_score": s.get("vital_score", -1.0),
        "vital_status": s.get("vital_status", "未知"),
        "vital_confidence": s.get("vital_confidence", 0.0),
        "victim_vital_score": s.get("victim_vital_score", 0.0),
        "vital_components": s.get("vital_components", {}),
        "reported_victim_count": s.get("reported_victim_count", 0),
        "reported_victims": s.get("reported_victims", []),
        "near_reported_victim": s.get("near_reported_victim"),
        "departing_victim": s.get("departing_victim", False),
        "night_vision": s.get("night_vision", "auto"),
        "brightness_avg": s.get("brightness_avg", 128),
        "unique_person_count": s.get("unique_person_count", 0),
        "unreported_count": s.get("unreported_count", 0),
        "heat_map_coverage": s.get("heat_map_coverage", 0.0),
        "heat_map": heat_map.get_grid_data() if (map_requested and heat_map) else None,
        "objects": s.get("objects", []),
        "ground_obstacle_level": s.get("ground_obstacle_level", "CLEAR"),
        "ground_obstacles": s.get("ground_obstacles", []),
        "tracks": detector.tracker.get_tracks_info() if detector.tracker else [],
        "intercom_active": intercom.is_active if intercom else False,
        # ROS 2 SLAM 資料
        "lidar_available": ros_bridge.is_available() if ros_bridge else False,
        "slam_map": ros_bridge.get_slam_map() if (map_requested and ros_bridge) else None,
        "slam_pose": ros_bridge.get_slam_pose() if (map_requested and ros_bridge) else None,
        "pose_history": ros_bridge.get_pose_history() if (map_requested and ros_bridge) else [],
        "lidar_points": ros_bridge.get_lidar_world_points() if (map_requested and ros_bridge) else [],
        "frontiers": ros_bridge.get_frontier_points(max_points=60) if (map_requested and ros_bridge) else [],
        "lidar_closest_cm": s.get("lidar_closest_cm", -1),
        "lidar_closest_angle": s.get("lidar_closest_angle", 0),
        "patrol_memory": scan_patrol_instance.get_memory_stats() if scan_patrol_instance else None,
        "logs": s["logs"][:20],
        "events": s["events"][:10],
    })


# ============================================================
# 啟動
# ============================================================
if __name__ == "__main__":
    with state_lock:
        app_state["ai_loaded"] = detector.is_loaded

    threading.Thread(target=detection_loop, daemon=True).start()
    threading.Thread(target=audio_loop, daemon=True).start()
    threading.Thread(target=manual_safety_loop, daemon=True).start()
    threading.Thread(target=lidar_safety_loop, daemon=True).start()
    threading.Thread(target=search_loop, daemon=True).start()

    logger.info("所有模組初始化完成")
    add_log("info", "搜救機器人系統 V2 啟動")

    print()
    print("=" * 52)
    print("  搜救機器人系統 V2 已啟動！")
    print(f"  控制台: http://0.0.0.0:{config.FLASK_PORT}")
    print(f"  Demo:   http://0.0.0.0:{config.FLASK_PORT}/demo")
    print("=" * 52)
    print()

    try:
        app.run(
            host=config.FLASK_HOST,
            port=config.FLASK_PORT,
            debug=False,
            threaded=True,
        )
    except KeyboardInterrupt:
        logger.info("收到中斷訊號...")
    finally:
        motor.cleanup()
        servo.cleanup()
        camera.cleanup()
        audio_reader.cleanup()
        if ros_bridge:
            ros_bridge.shutdown()
        logger.info("系統安全關閉")
