"""
搜救機器人 — 多人追蹤模組
===========================
IoU-based 簡易多目標追蹤，為每個偵測到的人員分配持久 ID。
避免對同一受困者重複回報。
"""

import time
import logging
from dataclasses import dataclass
from typing import List

import numpy as np

import config

logger = logging.getLogger("rescue.tracker")


@dataclass
class Track:
    """單一追蹤目標"""
    track_id: int
    bbox: dict                    # {x1, y1, x2, y2, conf}
    first_seen: float = 0.0
    last_seen: float = 0.0
    frames_since_seen: int = 0
    hit_count: int = 1            # 總共被偵測到的幀數
    reported: bool = False        # 是否已通報過

    @property
    def age_sec(self) -> float:
        return self.last_seen - self.first_seen


def _compute_iou(box_a: dict, box_b: dict) -> float:
    """計算兩個 bbox 的 IoU"""
    x1 = max(box_a["x1"], box_b["x1"])
    y1 = max(box_a["y1"], box_b["y1"])
    x2 = min(box_a["x2"], box_b["x2"])
    y2 = min(box_a["y2"], box_b["y2"])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    if inter == 0:
        return 0.0

    area_a = (box_a["x2"] - box_a["x1"]) * (box_a["y2"] - box_a["y1"])
    area_b = (box_b["x2"] - box_b["x1"]) * (box_b["y2"] - box_b["y1"])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


class PersonTracker:
    """
    IoU 匹配的多人追蹤器。
    每幀呼叫 update() 傳入偵測結果，回傳活躍 Track 列表。
    """

    def __init__(self):
        self._next_id = 1
        self._tracks: List[Track] = []
        self._max_lost = config.TRACKER_MAX_LOST_FRAMES
        self._min_iou = config.TRACKER_MIN_IOU
        logger.info(f"✅ 多人追蹤器初始化 | IoU≥{self._min_iou} | 遺失上限={self._max_lost}幀")

    def update(self, persons: List[dict]) -> List[Track]:
        """
        用新的偵測結果更新追蹤。
        persons: [{x1, y1, x2, y2, conf}, ...]
        回傳: 所有活躍的 Track 列表
        """
        now = time.time()

        if not persons:
            for t in self._tracks:
                t.frames_since_seen += 1
            self._cleanup()
            return self._tracks

        if not self._tracks:
            for p in persons:
                self._create_track(p, now)
            return self._tracks

        # 計算 IoU 矩陣
        n_tracks = len(self._tracks)
        n_dets = len(persons)
        iou_matrix = np.zeros((n_tracks, n_dets), dtype=np.float32)

        for i, track in enumerate(self._tracks):
            for j, det in enumerate(persons):
                iou_matrix[i, j] = _compute_iou(track.bbox, det)

        # 貪心匹配
        matched_tracks = set()
        matched_dets = set()

        # 複製一份用於標記已處理
        work_matrix = iou_matrix.copy()

        while True:
            if work_matrix.size == 0:
                break
            max_val = work_matrix.max()
            if max_val < self._min_iou:
                break
            max_idx = np.unravel_index(np.argmax(work_matrix), work_matrix.shape)
            ti, di = int(max_idx[0]), int(max_idx[1])

            if ti not in matched_tracks and di not in matched_dets:
                self._tracks[ti].bbox = persons[di]
                self._tracks[ti].last_seen = now
                self._tracks[ti].frames_since_seen = 0
                self._tracks[ti].hit_count += 1
                matched_tracks.add(ti)
                matched_dets.add(di)

            work_matrix[ti, :] = -1
            work_matrix[:, di] = -1

        # 未匹配的追蹤 → 遺失計數 +1
        for i in range(n_tracks):
            if i not in matched_tracks:
                self._tracks[i].frames_since_seen += 1

        # 未匹配的偵測 → 新追蹤
        for j in range(n_dets):
            if j not in matched_dets:
                self._create_track(persons[j], now)

        self._cleanup()
        return self._tracks

    def mark_all_visible_reported(self):
        """標記所有目前可見的追蹤為已通報"""
        for t in self._tracks:
            if t.frames_since_seen == 0:
                t.reported = True
                logger.info(f"📋 Track #{t.track_id} 標記為已通報")

    def get_unreported_count(self) -> int:
        """回傳未通報且目前可見的人數"""
        return sum(1 for t in self._tracks
                   if t.frames_since_seen == 0 and not t.reported)

    @property
    def total_unique_persons(self) -> int:
        """歷史上偵測到的不重複人數"""
        return self._next_id - 1

    @property
    def active_tracks(self) -> List[Track]:
        """目前可見的追蹤"""
        return [t for t in self._tracks if t.frames_since_seen == 0]

    def get_tracks_info(self) -> list:
        """回傳可序列化的追蹤資訊（供 API）"""
        return [
            {
                "id": t.track_id,
                "bbox": t.bbox,
                "age_sec": round(t.age_sec, 1),
                "hit_count": t.hit_count,
                "reported": t.reported,
            }
            for t in self._tracks
            if t.frames_since_seen <= 2
        ]

    def reset(self):
        """重置追蹤器"""
        self._tracks.clear()
        self._next_id = 1

    def _create_track(self, person: dict, now: float):
        track = Track(
            track_id=self._next_id,
            bbox=person,
            first_seen=now,
            last_seen=now,
        )
        self._tracks.append(track)
        self._next_id += 1
        logger.debug(f"🆕 新追蹤 Track #{track.track_id}")

    def _cleanup(self):
        """移除過期追蹤"""
        before = len(self._tracks)
        self._tracks = [t for t in self._tracks
                        if t.frames_since_seen <= self._max_lost]
        removed = before - len(self._tracks)
        if removed > 0:
            logger.debug(f"🗑️ 移除 {removed} 個過期追蹤")
