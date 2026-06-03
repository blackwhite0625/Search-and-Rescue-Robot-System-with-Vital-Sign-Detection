"""
搜救機器人 — 事件紀錄模組
==========================
事件截圖 + VictimScore + 時間戳紀錄
"""

import os
import time
import logging
import threading
from dataclasses import dataclass, field
from typing import List

logger = logging.getLogger("rescue.event_logger")

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

import config


@dataclass
class EventRecord:
    """單一事件紀錄"""
    event_id: int = 0
    timestamp: str = ""
    unix_time: float = 0.0
    victim_score: float = 0.0
    risk_level: str = "LOW"
    mission_stage: str = ""
    person_count: int = 0
    fallen_count: int = 0
    audio_event: str = ""
    screenshot_path: str = ""
    components: dict = field(default_factory=dict)


class EventLogger:
    """事件截圖 + 分數紀錄管理器"""

    MAX_SCREENSHOTS = 100  # 磁碟最多保留的截圖數量

    def __init__(self, event_dir: str = None):
        self._event_dir = event_dir or config.EVENT_DIR
        os.makedirs(self._event_dir, exist_ok=True)
        self._events: List[EventRecord] = []
        self._lock = threading.Lock()
        self._event_counter = 0
        self._max_events = 50
        self._cleanup_old_screenshots()
        logger.info(f"✅ 事件紀錄器初始化（目錄: {self._event_dir}）")

    def _cleanup_old_screenshots(self):
        """清理過多的舊截圖，防止磁碟爆滿"""
        try:
            files = sorted(
                [f for f in os.listdir(self._event_dir) if f.endswith('.jpg')],
                key=lambda f: os.path.getmtime(os.path.join(self._event_dir, f))
            )
            if len(files) > self.MAX_SCREENSHOTS:
                to_remove = files[:len(files) - self.MAX_SCREENSHOTS]
                for f in to_remove:
                    try:
                        os.remove(os.path.join(self._event_dir, f))
                    except OSError:
                        pass
                logger.info(f"🗑️ 已清理 {len(to_remove)} 張舊截圖")
        except Exception as e:
            logger.debug(f"截圖清理失敗: {e}")

    def log_event(self, frame=None, victim_score: float = 0.0,
                   risk_level: str = "LOW", mission_stage: str = "",
                   person_count: int = 0, fallen_count: int = 0,
                   audio_event: str = "", components: dict = None) -> EventRecord:
        """
        記錄一個事件：截圖 + 分數 + 時間戳
        """
        with self._lock:
            self._event_counter += 1
            event_id = self._event_counter

        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        unix_time = time.time()

        # 截圖
        screenshot_path = ""
        if frame is not None and CV2_AVAILABLE:
            filename = f"event_{event_id:04d}_{int(unix_time)}.jpg"
            filepath = os.path.join(self._event_dir, filename)
            try:
                cv2.imwrite(filepath, frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                screenshot_path = filepath
                logger.info(f"📸 事件截圖已儲存: {filename}")
            except Exception as e:
                logger.error(f"截圖儲存失敗: {e}")

        record = EventRecord(
            event_id=event_id,
            timestamp=timestamp,
            unix_time=unix_time,
            victim_score=round(victim_score, 3),
            risk_level=risk_level,
            mission_stage=mission_stage,
            person_count=person_count,
            fallen_count=fallen_count,
            audio_event=audio_event,
            screenshot_path=screenshot_path,
            components=components or {},
        )

        with self._lock:
            self._events.insert(0, record)
            if len(self._events) > self._max_events:
                self._events = self._events[:self._max_events]

        # 每 10 次事件清理一次舊截圖
        if event_id % 10 == 0:
            self._cleanup_old_screenshots()

        logger.warning(
            f"📋 事件 #{event_id} | Score: {victim_score:.2f} ({risk_level}) | "
            f"Stage: {mission_stage} | 人數: {person_count} 倒地: {fallen_count}"
        )
        return record

    def get_events(self) -> List[dict]:
        """取得所有事件紀錄（適合 JSON 序列化）"""
        with self._lock:
            return [
                {
                    "event_id": e.event_id,
                    "timestamp": e.timestamp,
                    "victim_score": e.victim_score,
                    "risk_level": e.risk_level,
                    "mission_stage": e.mission_stage,
                    "person_count": e.person_count,
                    "fallen_count": e.fallen_count,
                    "audio_event": e.audio_event,
                    "has_screenshot": bool(e.screenshot_path),
                    "components": e.components,
                }
                for e in self._events
            ]

    def get_latest_screenshot_path(self) -> str:
        """取得最新事件截圖路徑"""
        with self._lock:
            if self._events and self._events[0].screenshot_path:
                return self._events[0].screenshot_path
        return ""

    @property
    def event_count(self) -> int:
        with self._lock:
            return len(self._events)
