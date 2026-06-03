"""
搜救機器人 — SLAM 反應式巡邏 (V13 Gap-Following)
================================================
V12 的 heading-P 控制在牆角會算出 turn≈0 且 forward=0，送出 (0,0,0) 原地踏步。
V13 改為 gap-following：每 cycle 掃 360° LiDAR，切成 36 個 10° bucket，
評分 = clearance − angle_diff_to_goal − 後向懲罰，直接朝最佳 bucket 前進。

流程：
  1. 讀 360° LiDAR → 36 buckets
  2. 從 SLAM frontier 取得「探索 bearing」（世界座標，相對當前 yaw 轉成 local）
  3. 找最佳 bucket（最寬闊 + 最接近 bearing + 偏好前方）
  4. 若最佳 bucket 對準但前方仍阻塞 → 觸發 escape（不送 0,0,0）
  5. 對準程度分級：
       < 8°  → 全速前進
       < 45° → 半速前進 + 轉向
       ≥ 45° → 停車原地轉
  6. Idle 偵測：連續 15 cycle forward=0 → escape
  7. Escape：倒車 0.5s + SLAM yaw 閉環轉 160° + 強制前進 1s
             + 該區域 5×5 cell 標記為 visited（脹大），下次不會再選回

巡邏完全不動舵機；刪除 backtrack 返回功能。
"""

import time
import math
import logging
import threading
import collections
import config

logger = logging.getLogger("rescue.scan_patrol")

# 保留模組層級 event 供向下相容（V13 巡邏不使用）
scanning_lock = threading.Event()


class ScanPatrol:
    # ════════════════════════════════════════════════════
    # 控制與記憶（速度/距離閾值改由 config.py 提供，於 __init__ 載入）
    # ════════════════════════════════════════════════════
    BUCKET_DEG        = 10      # LiDAR 分箱解析度
    ALIGNED_DEG       = 8       # 航向誤差 < 此值即可全速前進
    TURN_ONLY_DEG     = 45      # 誤差 > 此值則停車原地轉
    HEADING_P_GAIN    = 2.0
    CYCLE_SLEEP_SEC   = 0.08    # ~12 Hz
    GOAL_REFRESH_SEC  = 2.5
    GOAL_REACH_M      = 0.5
    IMMOBILE_LIMIT    = 15      # 僅作觀察值，不再直接觸發 escape
    STUCK_WIN_SEC     = 3.0
    STUCK_MOVE_MIN_M  = 0.08

    def __init__(self, motor, servo, get_distance_fn,
                 get_victim_score_fn, check_active_fn,
                 add_log_fn, update_pan_fn, heat_map=None,
                 ros_bridge=None):
        self.motor = motor
        self.servo = servo
        self._dist = get_distance_fn
        self._score = get_victim_score_fn
        self._ok = check_active_fn
        self._log = add_log_fn
        self._pan_fn = update_pan_fn
        self._hm = heat_map
        self._rb = ros_bridge

        self.VISITED_CELL_M   = float(getattr(config, 'PATROL_VISITED_CELL_M', 0.25))
        self.UNREACH_CELL_M   = float(getattr(config, 'PATROL_UNREACH_CELL_M', 0.30))
        self.VISITED_SATURATE = int(getattr(config, 'PATROL_VISITED_SATURATE', 2))

        self._lidar_front_cm = float(getattr(config, 'ROS_LIDAR_TO_FRONT_CM', 30.0))
        self._lidar_side_cm  = float(getattr(config, 'ROS_LIDAR_TO_SIDE_CM', 6.0))
        self._self_reflect_cutoff_cm = self._lidar_front_cm + 3.0

        # Body-clearance → raw LiDAR 距離（預先換算，避免每 cycle 重算）
        self.CRUISE_FAST_PWM   = float(getattr(config, 'PATROL_CRUISE_FAST_PWM', 0.50))
        self.CRUISE_NORMAL_PWM = float(getattr(config, 'PATROL_CRUISE_NORMAL_PWM', 0.38))
        self.MIN_FWD_PWM       = float(getattr(config, 'PATROL_MIN_FWD_PWM', 0.22))
        self.TURN_PWM          = float(getattr(config, 'PATROL_TURN_PWM', 0.36))
        self.MIN_TURN_PWM      = float(getattr(config, 'PATROL_MIN_TURN_PWM', 0.28))
        self.SIDE_CAUTION_MULT = float(getattr(config, 'PATROL_SIDE_CAUTION_MULT', 0.6))
        self.BUCKET_HYSTERESIS = float(getattr(config, 'PATROL_BUCKET_HYSTERESIS', 1.15))
        self.BUCKET_MEM_TTL    = float(getattr(config, 'PATROL_BUCKET_MEMORY_TTL_SEC', 0.8))
        self.BUCKET_MEM_THRESH = float(getattr(config, 'PATROL_BUCKET_MEMORY_THRESH_CM', 100))
        self.TIGHT_BUCKET_N    = int(getattr(config, 'PATROL_TIGHT_BUCKET_COUNT', 4))
        self.TIGHT_BODY_CM     = float(getattr(config, 'PATROL_TIGHT_BODY_CM', 40))
        self.TIGHT_SPACE_MULT  = float(getattr(config, 'PATROL_TIGHT_SPACE_MULT', 0.45))
        self.SLAM_PAUSE_EVERY  = int(getattr(config, 'PATROL_SLAM_PAUSE_EVERY_CYCLES', 40))
        self.SLAM_PAUSE_SEC    = float(getattr(config, 'PATROL_SLAM_PAUSE_SEC', 0.4))

        front_min = float(getattr(config, 'PATROL_FRONT_MIN_BODY_CM', 10))
        front_brake = float(getattr(config, 'PATROL_FRONT_BRAKE_BODY_CM', 40))
        front_cruise = float(getattr(config, 'PATROL_FRONT_CRUISE_BODY_CM', 80))
        front_deadend = float(getattr(config, 'PATROL_FRONT_DEAD_END_BODY_CM', 5))
        min_gap = float(getattr(config, 'PATROL_MIN_GAP_BODY_CM', 8))
        self._FRONT_CRITICAL_RAW = self._lidar_front_cm + front_min
        self._FRONT_BRAKE_RAW    = self._lidar_front_cm + front_brake
        self._FRONT_CRUISE_RAW   = self._lidar_front_cm + front_cruise
        self._DEAD_END_RAW       = self._lidar_front_cm + front_deadend
        self._MIN_GAP_RAW        = self._lidar_front_cm + min_gap
        self._SIDE_MIN_BODY_CM     = float(getattr(config, 'PATROL_SIDE_MIN_BODY_CM', 8))
        self._SIDE_CAUTION_BODY_CM = float(getattr(config, 'PATROL_SIDE_CAUTION_BODY_CM', 20))

        # Escape 參數
        self._ESC_REVERSE_SEC = float(getattr(config, 'PATROL_ESCAPE_REVERSE_SEC', 0.3))
        self._ESC_REVERSE_PWM = float(getattr(config, 'PATROL_ESCAPE_REVERSE_PWM', 0.32))
        self._ESC_ROT_TIMEOUT = float(getattr(config, 'PATROL_ESCAPE_ROTATE_TIMEOUT', 1.8))
        self._ESC_VIS_RADIUS  = int(getattr(config, 'PATROL_ESCAPE_VISITED_RADIUS', 1))
        self._ESC_VIS_TTL_SEC = float(getattr(config, 'PATROL_ESCAPE_VISITED_TTL_SEC', 30.0))
        # 級聯脫困參數
        self._ESC_CASCADE_WIN   = float(getattr(config, 'PATROL_ESCAPE_CASCADE_WINDOW_SEC', 15.0))
        self._ESC_CASCADE_LIMIT = int(getattr(config, 'PATROL_ESCAPE_CASCADE_LIMIT', 3))
        self._ESC_CASCADE_REV   = float(getattr(config, 'PATROL_ESCAPE_CASCADE_REV_SEC', 1.0))
        self._ESC_CASCADE_ROT   = float(getattr(config, 'PATROL_ESCAPE_CASCADE_ROT_TIMEOUT', 3.2))
        self._ESC_CLEAR_RAW     = float(getattr(config, 'PATROL_ESCAPE_CLEAR_RAW_CM', 120))
        self._ESC_CLOSE_BODY    = float(getattr(config, 'PATROL_ESCAPE_CLOSE_BODY_CM', 15))
        self._ESC_DEEP_REV      = float(getattr(config, 'PATROL_ESCAPE_DEEP_REV_SEC', 0.7))

        # 訪問記憶參數
        self._RECENT_VISIT_SEC = float(getattr(config, 'PATROL_RECENT_VISIT_SEC', 60.0))
        self._RECENT_VISIT_PEN = float(getattr(config, 'PATROL_RECENT_VISIT_PENALTY', 2.5))
        self._VISITED_CAP = self.VISITED_SATURATE * 4   # 單 cell 訪問計數上限（防爆衝）

        self._cycle = 0
        self._visited_cells: dict = {}
        self._last_visit_time: dict = {}    # cell → 最後一次訪問 epoch（時間加權懲罰用）
        self._escape_visited: dict = {}     # cell → expiry epoch
        self._bucket_memory: dict = {}      # bucket_deg → (min_raw_cm, expiry_epoch) ← 低障礙記憶
        self._unreachable: set = set()
        self._current_target = None
        self._goal_set_time = 0.0
        self._no_frontier_cycles = 0
        self._immobile_count = 0
        self._prev_best_bucket = None       # Hysteresis state
        self._cycles_since_slam_pause = 0   # SLAM 掃描暫停計數
        self._escape_timestamps = collections.deque(maxlen=20)  # 逃脫時間戳（級聯偵測）
        self._in_tight_space = False                            # 當 cycle 是否處於擁擠空間

        # Stuck 時間窗
        self._pose_trace = collections.deque(maxlen=80)

    # ════════════════════════════════════════════════════
    # 公開 API（向下相容）
    # ════════════════════════════════════════════════════
    def reset_memory(self):
        self._visited_cells.clear()
        self._last_visit_time.clear()
        self._escape_visited.clear()
        self._bucket_memory.clear()
        self._unreachable.clear()
        self._current_target = None
        self._no_frontier_cycles = 0
        self._immobile_count = 0
        self._pose_trace.clear()
        self._prev_best_bucket = None
        logger.info("[F] 巡邏記憶已清空")

    def set_profile(self, profile: str):
        _ = profile

    def get_memory_stats(self) -> dict:
        return {
            "visited_cells": len(self._visited_cells),
            "escape_visited": len(self._escape_visited),
            "unreachable": len(self._unreachable),
            "current_target": self._current_target,
            "no_frontier_cycles": self._no_frontier_cycles,
            "immobile_count": self._immobile_count,
        }

    # ════════════════════════════════════════════════════
    # 馬達封裝：強制 diff-drive
    # ════════════════════════════════════════════════════
    def _drive_cmd(self, forward_pwm: float, turn_pwm: float):
        f = max(-self.CRUISE_FAST_PWM, min(self.CRUISE_FAST_PWM, float(forward_pwm)))
        t = max(-self.TURN_PWM,        min(self.TURN_PWM,        float(turn_pwm)))
        self.motor.move(0.0, f, t)

    # ════════════════════════════════════════════════════
    # 擁擠空間偵測（椅下/桌下/家具群 → 強制慢速）
    # ════════════════════════════════════════════════════
    def _is_tight_space(self, buckets: dict) -> bool:
        """前方 ±90° 內若有 >= TIGHT_BUCKET_N 個 bucket 的 body 餘裕 < TIGHT_BODY_CM，
        視為擁擠（家具群/椅下），此時強制降速。"""
        front_body_threshold_raw = self._lidar_front_cm + self.TIGHT_BODY_CM
        close_count = 0
        for b, raw in buckets.items():
            if -90 <= b <= 90 and raw < front_body_threshold_raw:
                close_count += 1
                if close_count >= self.TIGHT_BUCKET_N:
                    return True
        return False

    # ════════════════════════════════════════════════════
    # 側邊餘裕檢查（body clearance，cm）
    # ════════════════════════════════════════════════════
    def _check_side_clearance(self, ranges: dict) -> tuple:
        """回傳 (min_body_cm, status)。
        status: 'OK' / 'CAUTION' / 'BLOCK_L' / 'BLOCK_R'
        """
        left_vals  = [ranges[a] * 100.0 for a in ranges if 60 <= a <= 120 and ranges[a] > 0]
        right_vals = [ranges[a] * 100.0 for a in ranges if -120 <= a <= -60 and ranges[a] > 0]
        left_min  = min(left_vals)  if left_vals  else 9999.0
        right_min = min(right_vals) if right_vals else 9999.0
        left_body  = left_min  - self._lidar_side_cm
        right_body = right_min - self._lidar_side_cm
        min_body = min(left_body, right_body)
        if min_body < self._SIDE_MIN_BODY_CM:
            return min_body, ('BLOCK_L' if left_body < right_body else 'BLOCK_R')
        if min_body < self._SIDE_CAUTION_BODY_CM:
            return min_body, 'CAUTION'
        return min_body, 'OK'

    # ════════════════════════════════════════════════════
    # 自適應三檔前進 PWM（由 body clearance 推算）
    # ════════════════════════════════════════════════════
    def _compute_forward_pwm(self, front_raw_cm: float,
                             abs_err_deg: float,
                             side_status: str) -> float:
        # 1) 前方距離分段（線性內插）
        if front_raw_cm < self._FRONT_CRITICAL_RAW:
            return 0.0
        if front_raw_cm >= self._FRONT_CRUISE_RAW:
            base = self.CRUISE_FAST_PWM
        elif front_raw_cm >= self._FRONT_BRAKE_RAW:
            span = max(1.0, self._FRONT_CRUISE_RAW - self._FRONT_BRAKE_RAW)
            t = (front_raw_cm - self._FRONT_BRAKE_RAW) / span
            base = self.CRUISE_NORMAL_PWM + t * (self.CRUISE_FAST_PWM - self.CRUISE_NORMAL_PWM)
        else:
            span = max(1.0, self._FRONT_BRAKE_RAW - self._FRONT_CRITICAL_RAW)
            t = (front_raw_cm - self._FRONT_CRITICAL_RAW) / span
            base = self.MIN_FWD_PWM + t * (self.CRUISE_NORMAL_PWM - self.MIN_FWD_PWM)

        # 2) 航向誤差衰減
        if abs_err_deg < self.ALIGNED_DEG:
            mult = 1.0
        elif abs_err_deg < self.TURN_ONLY_DEG:
            mult = 0.5
        else:
            return 0.0

        # 3) 側邊狀態
        if side_status == 'CAUTION':
            mult *= self.SIDE_CAUTION_MULT
        elif side_status in ('BLOCK_L', 'BLOCK_R'):
            return 0.0

        # 4) 擁擠空間：整體大幅降速（家具群/椅下強迫慢行，避免輾上椅腳/桌腳）
        if self._in_tight_space:
            mult *= self.TIGHT_SPACE_MULT

        return max(self.MIN_FWD_PWM, base * mult) if base * mult > 0 else 0.0

    # ════════════════════════════════════════════════════
    # 主迴圈
    # ════════════════════════════════════════════════════
    def run_cycle(self):
        if not self._ok():
            self.motor.stop()
            return

        self._cycle += 1

        # 前置檢查
        if not self._rb:
            self.motor.stop()
            time.sleep(0.3)
            return
        if not self._rb.is_lidar_alive(getattr(config, 'ROS_LIDAR_WATCHDOG_SEC', 2.0)):
            self.motor.stop()
            self._log("warn", "[F] /scan 過期，暫停巡邏")
            time.sleep(0.3)
            return
        pose = self._rb.get_slam_pose()
        if pose is None:
            self.motor.stop()
            time.sleep(0.3)
            return

        ranges = self._rb.get_lidar_ranges() or {}
        if not ranges:
            self.motor.stop()
            time.sleep(0.2)
            return

        # 更新訪問與軌跡
        self._mark_visited_at(pose[0], pose[1])
        self._sync_heat_map_with_slam(pose)
        self._pose_trace.append((time.time(), pose[0], pose[1]))

        # 週期清理過期 escape-visited（每 32 cycle 一次）
        if (self._cycle & 31) == 0:
            self._prune_escape_visited()

        # 軌跡式 stuck 偵測（3 秒窗位移 < 8cm）為 escape 的唯一觸發
        if self._is_stuck():
            self._log("warn", "[F] 軌跡 stuck → 激進逃脫")
            self._escape_aggressive(ranges)
            return

        # 目標刷新（frontier）
        self._refresh_goal(pose)

        # 計算目標 bearing（相對當前 yaw）
        if self._current_target is not None:
            tx, ty = self._current_target[0], self._current_target[1]
            goal_abs = math.atan2(ty - pose[1], tx - pose[0])
            goal_rel = self._norm_angle(goal_abs - pose[2])
        else:
            goal_rel = 0.0   # 沒目標就朝前

        # Gap-following：找最佳可行駛方向
        best_rel_rad, best_raw_cm, front_raw_cm = self._find_best_gap(ranges, goal_rel)

        if best_rel_rad is None:
            # 整圈沒有可行駛方向
            self._log("warn", "[F] 360° 無可行駛 gap → 激進逃脫")
            self._escape_aggressive(ranges)
            return

        # 側邊餘裕（body clearance）→ 影響速度與轉向
        side_min_body, side_status = self._check_side_clearance(ranges)

        # 計算 forward / turn
        forward, turn = self._command_from_best_dir(best_rel_rad, front_raw_cm, side_status)

        # 對準但仍阻塞 → 死角
        if forward == 0.0 and abs(best_rel_rad) < math.radians(self.ALIGNED_DEG) \
                and front_raw_cm < self._DEAD_END_RAW:
            self._log("warn",
                      f"[F] 對準但前方 raw={front_raw_cm:.0f}cm 阻塞 → 激進逃脫")
            self._escape_aggressive(ranges)
            return

        # Idle 計數（僅作觀察；不再直接觸發 escape，由 pose-trace stuck 統一處理）
        if forward == 0.0:
            self._immobile_count += 1
        else:
            self._immobile_count = 0

        if side_status in ('BLOCK_L', 'BLOCK_R'):
            self._log("warn",
                      f"[F] 側邊 body={side_min_body:.0f}cm 警戒 ({side_status})")

        # 擁擠空間 log（每 30 cycle 記一次，避免洗版）
        if self._in_tight_space and (self._cycle % 30 == 0):
            self._log("info", f"[F] 擁擠空間偵測中 → 降速至 {self.TIGHT_SPACE_MULT:.0%}")

        self._drive_cmd(forward, turn)
        time.sleep(self.CYCLE_SLEEP_SEC)

        # ── SLAM 掃描暫停：靜止時 slam_toolbox scan matching 精準度提升 ──
        # 只在確實前進時計數（轉彎/靜止的 cycle 不計），避免在原地轉時多餘停頓
        if forward > 0.0:
            self._cycles_since_slam_pause += 1
            if self._cycles_since_slam_pause >= self.SLAM_PAUSE_EVERY:
                self.motor.stop()
                time.sleep(self.SLAM_PAUSE_SEC)
                self._cycles_since_slam_pause = 0
        else:
            # 非前進狀態也算 SLAM 更新機會，部分歸零避免長期不觸發
            self._cycles_since_slam_pause = max(0, self._cycles_since_slam_pause - 1)

    # ════════════════════════════════════════════════════
    # Gap-following 核心
    # ════════════════════════════════════════════════════
    def _find_best_gap(self, ranges: dict, goal_rel_rad: float):
        """掃 360° LiDAR，回傳 (best_rel_rad, best_raw_cm, front_raw_cm)。
        找不到可行駛方向回 (None, None, front_raw_cm)。"""
        # Bucket by 10°
        buckets: dict = {}
        for a, d in ranges.items():
            if d <= 0:
                continue
            raw = d * 100.0
            if raw < self._self_reflect_cutoff_cm:
                # 極近讀數視為車體自反射或貼身障礙，保守計入
                raw = max(1.0, raw)
            b = int(round(a / self.BUCKET_DEG)) * self.BUCKET_DEG
            # Normalize to (-180, 180]
            if b > 180:
                b -= 360
            elif b < -180:
                b += 360
            if b not in buckets or raw < buckets[b]:
                buckets[b] = raw

        if not buckets:
            return None, None, 999.0

        # ── 低障礙記憶：補上最近看過卻本輪可能遺漏的近距讀值 ──
        # LiDAR 對椅腳/桌腳常間歇性遺漏；保留 0.4s 內的最近讀值抗遺漏
        now = time.time()
        # 先清除過期記憶
        stale_keys = [k for k, (_, exp) in self._bucket_memory.items() if exp < now]
        for k in stale_keys:
            del self._bucket_memory[k]
        # 合併記憶到當前 buckets（取 min）
        for b, (mem_raw, _) in self._bucket_memory.items():
            if b in buckets:
                buckets[b] = min(buckets[b], mem_raw)
            else:
                buckets[b] = mem_raw
        # 將本輪近距讀值寫入記憶（相鄰 ±1 bucket 也記，涵蓋細物體游走）
        mem_ttl = now + self.BUCKET_MEM_TTL
        for b, raw in list(buckets.items()):
            if raw < self.BUCKET_MEM_THRESH:
                self._bucket_memory[b] = (raw, mem_ttl)
                # 鄰近 bucket 也記憶（細物體可能落在邊界）
                for nb in (b - self.BUCKET_DEG, b + self.BUCKET_DEG):
                    if nb > 180: nb -= 360
                    elif nb < -180: nb += 360
                    prev = self._bucket_memory.get(nb)
                    if prev is None or prev[0] > raw:
                        self._bucket_memory[nb] = (raw, mem_ttl)

        # 前方 raw（±30° 最小值）供速度控制（擴大自 ±20°，涵蓋椅輪/桌腳邊緣）
        front_vals = [r for b, r in buckets.items() if -30 <= b <= 30]
        front_raw = min(front_vals) if front_vals else 999.0

        # 擁擠空間偵測（椅下/桌下/家具群）
        self._in_tight_space = self._is_tight_space(buckets)

        # 候選 bucket：raw > _MIN_GAP_RAW（body clearance 推導）
        goal_deg = math.degrees(goal_rel_rad)
        best_b = None
        best_score = -1e9
        prev_b = self._prev_best_bucket
        for b, raw_cm in buckets.items():
            if raw_cm < self._MIN_GAP_RAW:
                continue
            # 最短角度差
            diff = ((b - goal_deg) + 180) % 360 - 180
            abs_diff = abs(diff)
            # 後向懲罰（|b| > 90 開始扣）
            backward_penalty = max(0.0, abs(b) - 90.0) * 2.5
            # 評分：越空 + 越接近目標 − 後向懲罰
            score = raw_cm * 0.6 - abs_diff * 1.8 - backward_penalty
            # Hysteresis：上一 cycle 的 bucket 若仍可行（±2 bucket 範圍內）給獎勵
            if prev_b is not None and abs(b - prev_b) <= 2 * self.BUCKET_DEG:
                score *= self.BUCKET_HYSTERESIS
            if score > best_score:
                best_score = score
                best_b = b

        if best_b is None:
            self._prev_best_bucket = None
            return None, None, front_raw
        self._prev_best_bucket = best_b
        return math.radians(best_b), buckets[best_b], front_raw

    def _command_from_best_dir(self, best_rel_rad: float, front_raw_cm: float,
                               side_status: str = 'OK'):
        """根據最佳方向、前方距離、側邊狀態計算 (forward, turn)。"""
        abs_err_deg = abs(math.degrees(best_rel_rad))

        # 自適應三檔前進
        forward = self._compute_forward_pwm(front_raw_cm, abs_err_deg, side_status)

        # 轉向：aligned 不轉，其餘依 P-gain
        if abs_err_deg < self.ALIGNED_DEG:
            turn = 0.0
        else:
            turn = self._turn_toward(best_rel_rad)

        # 側邊硬擋：禁止朝該側轉向
        if side_status == 'BLOCK_L' and turn < 0:    # 左側被擋，不可 CCW (turn<0 = left)
            turn = 0.0
        elif side_status == 'BLOCK_R' and turn > 0:  # 右側被擋，不可 CW
            turn = 0.0

        return forward, turn

    def _turn_toward(self, rel_rad: float) -> float:
        """產生朝 rel_rad 方向旋轉的馬達 r PWM。
        慣例：rel_rad > 0 = 需左轉（math CCW）→ motor r 為負。"""
        if abs(rel_rad) < math.radians(3):
            return 0.0
        mag = min(self.TURN_PWM, max(self.MIN_TURN_PWM, abs(rel_rad) * self.HEADING_P_GAIN))
        return -mag if rel_rad > 0 else mag

    # ════════════════════════════════════════════════════
    # 激進逃脫：短倒車 + 動態轉向 + TTL visited + 級聯脫困
    # ════════════════════════════════════════════════════
    def _escape_aggressive(self, ranges: dict):
        """逃脫流程，含級聯偵測：反覆卡住時啟動更激進模式。"""
        now = time.time()

        # ── 級聯偵測：統計時間窗內逃脫次數 ──
        self._escape_timestamps.append(now)
        while self._escape_timestamps and (now - self._escape_timestamps[0]) > self._ESC_CASCADE_WIN:
            self._escape_timestamps.popleft()
        is_cascade = len(self._escape_timestamps) >= self._ESC_CASCADE_LIMIT

        # ── 估計當前前方 body 餘裕（決定倒車長度）──
        front_vals = [d for a, d in ranges.items() if -20 <= a <= 20 and d > 0]
        front_raw_cm = min(front_vals) * 100.0 if front_vals else 999.0
        front_body_cm = front_raw_cm - self._lidar_front_cm
        very_close = front_body_cm < self._ESC_CLOSE_BODY

        # 選擇倒車時間 & PWM
        if is_cascade:
            reverse_sec = self._ESC_CASCADE_REV
            reverse_pwm = min(0.42, self._ESC_REVERSE_PWM * 1.3)
            rot_timeout = self._ESC_CASCADE_ROT
            self._log("warn", f"[F] 級聯逃脫（{len(self._escape_timestamps)} 次/"
                              f"{self._ESC_CASCADE_WIN:.0f}s）→ 深度脫困模式")
        elif very_close:
            reverse_sec = self._ESC_DEEP_REV
            reverse_pwm = self._ESC_REVERSE_PWM * 1.1
            rot_timeout = self._ESC_ROT_TIMEOUT
        else:
            reverse_sec = self._ESC_REVERSE_SEC
            reverse_pwm = self._ESC_REVERSE_PWM
            rot_timeout = self._ESC_ROT_TIMEOUT

        # === 1) 倒車 ===
        t0 = time.time()
        while time.time() - t0 < reverse_sec and self._ok():
            self._drive_cmd(-reverse_pwm, 0.0)
            time.sleep(0.04)
        self.motor.stop()
        time.sleep(0.1)

        # === 2) 動態轉向：選向較空一側，轉向中即時監測 clear 方向 ===
        left_vals  = [d for a, d in ranges.items() if 45 <= a <= 135 and d > 0]
        right_vals = [d for a, d in ranges.items() if -135 <= a <= -45 and d > 0]
        left_avg  = (sum(left_vals) / len(left_vals))  if left_vals  else 0.0
        right_avg = (sum(right_vals) / len(right_vals)) if right_vals else 0.0
        go_left = left_avg >= right_avg
        diff_m = abs(left_avg - right_avg)
        target_deg = 180 if (diff_m < 0.5 or is_cascade) else 120
        target_delta = math.radians(target_deg if go_left else -target_deg)
        r_sign = -1 if go_left else 1

        pose0 = self._rb.get_slam_pose() if self._rb else None
        yaw0 = pose0[2] if pose0 is not None else None
        t0 = time.time()
        accumulated = 0.0
        last_yaw = yaw0
        found_clear = False
        while time.time() - t0 < rot_timeout and self._ok():
            self._drive_cmd(0.0, r_sign * self.TURN_PWM)
            time.sleep(0.04)

            # ── 轉向中即時監測：若前方出現明顯空曠方向就立刻停止轉動 ──
            if self._rb:
                cur_ranges = self._rb.get_lidar_ranges() or {}
                if cur_ranges:
                    cur_front = [d for a, d in cur_ranges.items()
                                 if -25 <= a <= 25 and d > 0]
                    if cur_front:
                        front_now_cm = min(cur_front) * 100.0
                        # 累積角度 >= 60° 後才允許「發現空曠就停」，避免起步就停
                        if abs(accumulated) >= math.radians(60) and \
                                front_now_cm >= self._ESC_CLEAR_RAW:
                            found_clear = True
                            break

                cur = self._rb.get_slam_pose()
                if cur and last_yaw is not None:
                    d = self._norm_angle(cur[2] - last_yaw)
                    accumulated += d
                    last_yaw = cur[2]
                    if abs(accumulated) >= abs(target_delta):
                        break
        self.motor.stop()
        time.sleep(0.1)

        if found_clear:
            self._log("info",
                      f"[F] 轉向 {math.degrees(abs(accumulated)):.0f}° 後發現空曠方向 → 立即恢復巡邏")

        # === 3) 不強制前進；交由下一 cycle gap-following 處理 ===

        # === 4) 訪問格膨脹（TTL），級聯模式清空 escape_visited 允許重試 ===
        pose = self._rb.get_slam_pose() if self._rb else None
        if is_cascade:
            # 級聯：清空之前標記為 visited/unreachable 的格子，讓 frontier 可重新挑選
            self._escape_visited.clear()
            self._unreachable.clear()
            self._bucket_memory.clear()
            self._log("warn", "[F] 級聯：清空 escape_visited / unreachable / bucket_memory")
        elif pose is not None:
            kx = int(round(pose[0] / self.VISITED_CELL_M))
            ky = int(round(pose[1] / self.VISITED_CELL_M))
            expiry = time.time() + self._ESC_VIS_TTL_SEC
            r = self._ESC_VIS_RADIUS
            for dx in range(-r, r + 1):
                for dy in range(-r, r + 1):
                    self._escape_visited[(kx + dx, ky + dy)] = expiry

        # === 5) 清除當前 target、重設計數與 hysteresis ===
        if self._current_target is not None:
            self._mark_unreachable(self._current_target[0], self._current_target[1])
            self._current_target = None
        self._pose_trace.clear()
        self._immobile_count = 0
        self._prev_best_bucket = None

    def _prune_escape_visited(self):
        """移除過期的 escape-inflated visited 項目（避免無限累積）。"""
        now = time.time()
        stale = [k for k, exp in self._escape_visited.items() if exp < now]
        for k in stale:
            del self._escape_visited[k]

    def _is_stuck(self) -> bool:
        now = time.time()
        while self._pose_trace and (now - self._pose_trace[0][0]) > self.STUCK_WIN_SEC:
            self._pose_trace.popleft()
        if len(self._pose_trace) < 15:
            return False
        x0, y0 = self._pose_trace[0][1], self._pose_trace[0][2]
        max_d = 0.0
        for _, x, y in self._pose_trace:
            d = math.hypot(x - x0, y - y0)
            if d > max_d:
                max_d = d
        return max_d < self.STUCK_MOVE_MIN_M

    # ════════════════════════════════════════════════════
    # Frontier 目標管理
    # ════════════════════════════════════════════════════
    def _refresh_goal(self, pose: tuple):
        now = time.time()
        need = False
        if self._current_target is None:
            need = True
        else:
            tx, ty, _ = self._current_target
            dist = math.hypot(tx - pose[0], ty - pose[1])
            if dist < self.GOAL_REACH_M:
                self._mark_visited_at(tx, ty)
                self._current_target = None
                need = True
            elif now - self._goal_set_time > self.GOAL_REFRESH_SEC:
                need = True
            elif self._cell_visited(tx, ty) or self._is_unreachable(tx, ty):
                self._current_target = None
                need = True

        if not need:
            return

        frontiers = self._rb.get_frontier_points() if self._rb else []
        if not frontiers:
            self._no_frontier_cycles += 1
            self._current_target = None
            return

        target = self._select_frontier(frontiers, pose)
        if target is None:
            self._no_frontier_cycles += 1
            self._current_target = None
            return

        self._no_frontier_cycles = 0
        self._current_target = target
        self._goal_set_time = now
        logger.info(
            f"[F] #{self._cycle} 新目標=({target[0]:.2f},{target[1]:.2f}) size={target[2]}"
        )

    def _select_frontier(self, frontiers, pose):
        """評分 = -dist/turn_cost + log(size)*0.4 − visit_pen。
        同時用 is_cell_free 確認目標點周圍有足夠通行空間。"""
        cx, cy, yaw = pose
        best = None
        best_score = -1e9
        for f in frontiers:
            fx, fy, size = f[0], f[1], f[2] if len(f) > 2 else 1
            if self._is_unreachable(fx, fy):
                continue
            if self._cell_visited(fx, fy):
                continue
            dx, dy = fx - cx, fy - cy
            dist = math.hypot(dx, dy)
            if dist < 0.30:
                continue
            # LOS：直線必須穿過 SLAM free space
            if self._rb and not self._rb.line_of_sight(cx, cy, fx, fy):
                continue
            # 目標點周圍必須有 15cm 緩衝的 free space
            if self._rb and hasattr(self._rb, 'is_cell_free'):
                if not self._rb.is_cell_free(fx, fy, 0.15):
                    continue

            goal_yaw = math.atan2(dy, dx)
            angle_err = abs(self._norm_angle(goal_yaw - yaw))
            angle_deg = math.degrees(angle_err)
            if angle_deg > 90:
                turn_cost = 1.0 + (angle_deg - 90) / 90.0
            else:
                turn_cost = max(0.5, 1.0 - angle_deg / 180.0)
            info_bonus = math.log(max(size, 1)) * 0.4
            visit_pen = self._visit_count_at(fx, fy) * 0.8

            # 時間加權近期訪問懲罰：frontier 附近若近 N 秒內剛訪問過，重罰
            recent_pen = 0.0
            last_t = self._recent_visit_epoch_near(fx, fy)
            now_ts = time.time()
            if last_t > 0 and (now_ts - last_t) < self._RECENT_VISIT_SEC:
                age_factor = 1.0 - (now_ts - last_t) / self._RECENT_VISIT_SEC
                recent_pen = self._RECENT_VISIT_PEN * age_factor

            score = (-dist / turn_cost) + info_bonus - visit_pen - recent_pen
            if score > best_score:
                best_score = score
                best = f
        return best

    def _recent_visit_epoch_near(self, wx: float, wy: float) -> float:
        """回傳該 frontier 周圍 3×3 cell 內最近一次訪問的時間戳（0 代表從未訪問）。"""
        cx, cy = self._visited_key(wx, wy)
        latest = 0.0
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                t = self._last_visit_time.get((cx + dx, cy + dy), 0.0)
                if t > latest:
                    latest = t
        return latest

    # ════════════════════════════════════════════════════
    # Visited / Unreachable 記憶
    # ════════════════════════════════════════════════════
    def _visited_key(self, wx, wy) -> tuple:
        return (
            int(round(wx / self.VISITED_CELL_M)),
            int(round(wy / self.VISITED_CELL_M)),
        )

    def _unreach_key(self, wx, wy) -> tuple:
        return (
            int(round(wx / self.UNREACH_CELL_M)),
            int(round(wy / self.UNREACH_CELL_M)),
        )

    def _visit_count_at(self, wx, wy) -> int:
        return self._visited_cells.get(self._visited_key(wx, wy), 0)

    def _cell_visited(self, wx, wy) -> bool:
        key = self._visited_key(wx, wy)
        if self._visited_cells.get(key, 0) >= self.VISITED_SATURATE:
            return True
        expiry = self._escape_visited.get(key, 0.0)
        return time.time() < expiry

    def _is_unreachable(self, wx, wy) -> bool:
        return self._unreach_key(wx, wy) in self._unreachable

    def _mark_unreachable(self, wx, wy):
        self._unreachable.add(self._unreach_key(wx, wy))
        logger.info(f"[F] 標記不可達 ({wx:.2f},{wy:.2f})；共 {len(self._unreachable)} 筆")

    def _mark_visited_at(self, wx: float, wy: float):
        key = self._visited_key(wx, wy)
        cur = self._visited_cells.get(key, 0) + 1
        # 計數上限：單 cell 不能無限累加，否則扭曲 frontier 評分
        self._visited_cells[key] = min(cur, self._VISITED_CAP)
        self._last_visit_time[key] = time.time()
        if self._visited_cells[key] >= self.VISITED_SATURATE:
            cx, cy = key
            now_ts = time.time()
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    nk = (cx + dx, cy + dy)
                    if self._visited_cells.get(nk, 0) < self.VISITED_SATURATE:
                        self._visited_cells[nk] = self.VISITED_SATURATE
                        self._last_visit_time[nk] = now_ts

    def _sync_heat_map_with_slam(self, pose):
        if not self._hm:
            return
        if hasattr(self._hm, "update_from_slam_pose"):
            try:
                self._hm.update_from_slam_pose(pose[0], pose[1], pose[2])
            except Exception as e:
                logger.debug(f"[F] heat_map SLAM 同步失敗: {e}")

    # ════════════════════════════════════════════════════
    # 工具
    # ════════════════════════════════════════════════════
    @staticmethod
    def _norm_angle(a: float) -> float:
        while a > math.pi:
            a -= 2 * math.pi
        while a < -math.pi:
            a += 2 * math.pi
        return a
