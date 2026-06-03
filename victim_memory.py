"""
Reported victim memory.

Stores rescue/report records in map coordinates when SLAM pose is available.
The goal is to avoid treating the same reported victim as a new target after
the robot leaves and later patrols through the same area again.
"""

import json
import math
import os
import threading
import time
from dataclasses import asdict, dataclass, field
from typing import Iterable, Optional

import config


@dataclass
class VictimRecord:
    victim_id: int
    created_at: float
    updated_at: float
    x: Optional[float] = None
    y: Optional[float] = None
    yaw: Optional[float] = None
    report_count: int = 0
    status: str = "reported"
    source_stage: str = ""
    victim_score: float = 0.0
    risk_level: str = "LOW"
    vital_status: str = "未知"
    vital_confidence: float = 0.0
    consciousness_state: str = "UNKNOWN"
    track_ids: list = field(default_factory=list)
    last_event: str = ""

    @property
    def has_pose(self) -> bool:
        return self.x is not None and self.y is not None


class VictimMemory:
    def __init__(self, file_path=None, merge_radius_m=None):
        self.file_path = file_path or getattr(config, "VICTIM_MEMORY_FILE", None)
        self.merge_radius_m = float(
            merge_radius_m
            if merge_radius_m is not None
            else getattr(config, "VICTIM_MEMORY_MERGE_RADIUS_M", 0.85)
        )
        self._lock = threading.Lock()
        self._records = []
        self._next_id = 1
        self._load()

    def remember(self, pose=None, track_ids=None, source_stage="", victim_score=0.0,
                 risk_level="LOW", vital_status="未知", vital_confidence=0.0,
                 consciousness_state="UNKNOWN", event="") -> VictimRecord:
        now = time.time()
        x, y, yaw = self._pose_parts(pose)
        tids = self._normalize_track_ids(track_ids)
        with self._lock:
            rec = self._find_merge_target_locked(x, y, tids)
            if rec is None:
                rec = VictimRecord(
                    victim_id=self._next_id,
                    created_at=now,
                    updated_at=now,
                    x=x,
                    y=y,
                    yaw=yaw,
                )
                self._records.append(rec)
                self._next_id += 1
            else:
                rec.updated_at = now
                if x is not None and y is not None:
                    rec.x = x
                    rec.y = y
                    rec.yaw = yaw

            rec.report_count += 1
            rec.status = "reported"
            rec.source_stage = source_stage or rec.source_stage
            rec.victim_score = round(float(victim_score or 0.0), 3)
            rec.risk_level = risk_level or rec.risk_level
            rec.vital_status = vital_status or rec.vital_status
            rec.vital_confidence = round(float(vital_confidence or 0.0), 3)
            rec.consciousness_state = consciousness_state or rec.consciousness_state
            rec.last_event = event or rec.last_event
            rec.track_ids = sorted(set(rec.track_ids).union(tids))
            self._save_locked()
            return rec

    def mark_departed(self, victim_id: int):
        with self._lock:
            rec = self._get_locked(victim_id)
            if rec:
                rec.status = "departed"
                rec.updated_at = time.time()
                self._save_locked()

    def nearest(self, pose=None, radius_m=None):
        x, y, _ = self._pose_parts(pose)
        if x is None or y is None:
            return None, None
        limit = float(radius_m if radius_m is not None
                      else getattr(config, "VICTIM_MEMORY_NEAR_RADIUS_M", 1.10))
        with self._lock:
            best = None
            best_dist = None
            for rec in self._records:
                if not rec.has_pose:
                    continue
                dist = math.hypot(float(rec.x) - x, float(rec.y) - y)
                if dist <= limit and (best_dist is None or dist < best_dist):
                    best = rec
                    best_dist = dist
            return best, best_dist

    def is_track_reported(self, track_ids: Iterable[int]) -> bool:
        tids = set(self._normalize_track_ids(track_ids))
        if not tids:
            return False
        with self._lock:
            for rec in self._records:
                if tids.intersection(rec.track_ids):
                    return True
        return False

    def to_status(self, max_records=20) -> list:
        now = time.time()
        with self._lock:
            records = sorted(self._records, key=lambda r: r.updated_at, reverse=True)
            out = []
            for rec in records[:max_records]:
                item = asdict(rec)
                item["age_sec"] = round(now - rec.created_at, 1)
                item["updated_ago_sec"] = round(now - rec.updated_at, 1)
                if rec.has_pose:
                    item["x"] = round(float(rec.x), 3)
                    item["y"] = round(float(rec.y), 3)
                    item["yaw"] = round(float(rec.yaw or 0.0), 3)
                out.append(item)
            return out

    @property
    def count(self) -> int:
        with self._lock:
            return len(self._records)

    def clear(self):
        with self._lock:
            self._records.clear()
            self._next_id = 1
            self._save_locked()

    def _find_merge_target_locked(self, x, y, track_ids):
        tids = set(track_ids or [])
        if x is not None and y is not None:
            best = None
            best_dist = None
            for rec in self._records:
                if not rec.has_pose:
                    continue
                dist = math.hypot(float(rec.x) - x, float(rec.y) - y)
                if dist <= self.merge_radius_m and (best_dist is None or dist < best_dist):
                    best = rec
                    best_dist = dist
            if best:
                return best
        if tids:
            for rec in self._records:
                if tids.intersection(rec.track_ids):
                    return rec
        if not tids and x is None and y is None and self._records:
            recent_sec = float(getattr(config, "VICTIM_MEMORY_NO_POSE_MERGE_SEC", 90))
            newest = max(self._records, key=lambda r: r.updated_at)
            if time.time() - newest.updated_at <= recent_sec:
                return newest
        return None

    def _get_locked(self, victim_id):
        for rec in self._records:
            if rec.victim_id == victim_id:
                return rec
        return None

    def _load(self):
        if not self.file_path or not os.path.exists(self.file_path):
            return
        try:
            with open(self.file_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            records = payload.get("records", []) if isinstance(payload, dict) else []
            self._records = [VictimRecord(**r) for r in records if isinstance(r, dict)]
            if self._records:
                self._next_id = max(r.victim_id for r in self._records) + 1
        except Exception:
            self._records = []
            self._next_id = 1

    def _save_locked(self):
        if not self.file_path:
            return
        try:
            os.makedirs(os.path.dirname(self.file_path), exist_ok=True)
            payload = {
                "version": 1,
                "updated_at": time.time(),
                "records": [asdict(r) for r in self._records],
            }
            tmp_path = self.file_path + ".tmp"
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            os.replace(tmp_path, self.file_path)
        except Exception:
            pass

    @staticmethod
    def _pose_parts(pose):
        if not pose or len(pose) < 2:
            return None, None, None
        try:
            x = float(pose[0])
            y = float(pose[1])
            yaw = float(pose[2]) if len(pose) >= 3 else None
            return x, y, yaw
        except Exception:
            return None, None, None

    @staticmethod
    def _normalize_track_ids(track_ids):
        result = []
        for tid in track_ids or []:
            try:
                result.append(int(tid))
            except Exception:
                continue
        return result
