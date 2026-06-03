"""
搜救機器人 — 掃描巡邏模組 (模式 F) V9
"""

import time
import math
import logging
import threading
import config

logger = logging.getLogger("rescue.scan_patrol")

DRIVE_SPEED = 0.45
DRIVE_SEC = 2.0
REVERSE_SPEED = 0.30
TURN_SPEED = 0.32
STRAFE_SPEED = 0.25
# 以下閾值是「車頭到障礙物」的實際淨空（已扣除 LiDAR→車頭 18cm 偏移）
OBSTACLE_CLEAR_CM = 22   # 車頭淨空 ≤22cm → 方向不可行（等效舊版 40cm raw）
BRAKE_CLEAR_CM    = 10   # 車頭淨空 ≤10cm → 緊急煞車

# 安全圓：以 LiDAR 為中心 15cm 半徑的禁區（不含車頭方向，那邊由 18cm 車身偏移處理）
# 用於防止 mecanum 平移/旋轉時側邊或後方撞到
SAFETY_CIRCLE_CM  = 15   # LiDAR 為中心的安全半徑
SAFETY_FRONT_ARC  = 45   # 車頭 ±45° 扇形 — 此區由車頭偏移規則處理，不套用安全圓
SWEEP_SEC = 8.0    # 完整 MIN→MAX→MIN 來回所需最長時間
SWEEP_STEP = 10    # 舊常數，保留給其他處引用（新掃描改用 config.PATROL_PAN_STEP）

_RAD_PER_SEC = TURN_SPEED * getattr(config, 'HEAT_MAP_RAD_PER_SPEED_SEC', 2.0)
DEG_PER_SEC = max(_RAD_PER_SEC * 180 / math.pi, 15)

scanning_lock = threading.Event()


class ScanPatrol:

    # Visited tracking grid：0.5 m cells
    VISITED_CELL_M = 0.5
    # 投影距離：看起來有多少空間就走多少（70% 安全係數，最多 2m）
    PROJECT_RATIO = 0.7
    PROJECT_MAX_M = 2.0
    # 單步前進最大距離
    STEP_DISTANCE_CM = 80
    STEP_TIMEOUT_SEC = 5.0

    def __init__(self, motor, servo, get_distance_fn,
                 get_victim_score_fn, check_active_fn,
                 add_log_fn, update_pan_fn, heat_map=None,
                 backtrack_engine=None, ros_bridge=None):
        self.motor = motor
        self.servo = servo
        self._dist = get_distance_fn
        self._score = get_victim_score_fn
        self._ok = check_active_fn
        self._log = add_log_fn
        self._pan_fn = update_pan_fn
        self._hm = heat_map
        self._bt = backtrack_engine
        self._rb = ros_bridge
        self._cycle = 0
        self._stuck = 0
        self._turn_dir = 1
        # 已訪問的世界座標格子（用 SLAM pose 更新，不受 dead reckoning 漂移）
        self._visited = set()
        # 連續脫困次數（用於升級策略）
        self._consecutive_escapes = 0
        # 上次 sweep 的 cycle 編號和 pose（決定是否要 sweep）
        self._last_sweep_cycle = -99
        self._last_sweep_pose = None

    def _front_clear_cm(self) -> float:
        """車頭實際淨空 (cm)：raw LiDAR 最近 - LiDAR→車頭偏移。
        回傳 -1 = 無資料。負值代表已穿過車頭（撞到）。"""
        if not self._rb:
            return -1
        ranges = self._rb.get_lidar_ranges()
        if not ranges:
            return -1
        half = 45  # ±45° 前方扇區
        front = [d for a, d in ranges.items() if (-half <= a <= half) and d > 0]
        if not front:
            return -1
        raw_cm = min(front) * 100
        offset = getattr(config, 'ROS_LIDAR_TO_FRONT_CM', 18)
        return round(raw_cm - offset, 1)

    # 向後相容：保留 _raw_front_cm 名稱但回傳 clearance
    def _raw_front_cm(self) -> float:
        return self._front_clear_cm()

    def _direction_clear_cm(self, angle_deg: float, ranges: dict, arc_half: float = 15) -> float:
        """該方向若機器人面朝過去，前進時的淨空 (cm)。
        **一律用車頭 18cm 偏移**——因為旋轉後「那個方向」就是新的前方。
        回傳 -1 表示無資料。"""
        dists = [d for a, d in ranges.items() if _adiff(a, angle_deg) < arc_half and d > 0]
        if not dists:
            return -1
        raw_cm = min(dists) * 100
        front_off = getattr(config, 'ROS_LIDAR_TO_FRONT_CM', 18)
        return round(raw_cm - front_off, 1)

    def _safety_circle_violations(self, ranges: dict) -> list:
        """檢查 LiDAR 周圍的禁區內是否有障礙：
        - 車頭 ±45°：由 18cm 車頭偏移處理，不在此檢查
        - 車尾 ±45° (≈ 180°)：半徑 15cm
        - 側邊 ±90°（前輪/後輪區）：**半徑 20cm**（補償 LiDAR 死角 + 輪子寬度）
        回傳 [(angle_deg, dist_cm), ...]，空 list = 安全。"""
        violations = []
        side_radius = SAFETY_CIRCLE_CM + 5  # 側邊多 5cm 補償死角
        for angle_deg, dist_m in ranges.items():
            if dist_m <= 0:
                continue
            if abs(angle_deg) < SAFETY_FRONT_ARC:
                continue  # 前方扇形：忽略，交由車頭偏移邏輯
            dist_cm = dist_m * 100
            # 側邊區 (45°~135° 和 -135°~-45°) 用加大半徑
            if SAFETY_FRONT_ARC <= abs(angle_deg) <= 135:
                threshold = side_radius
            else:
                # 車尾區（|angle| > 135）用原本 15cm
                threshold = SAFETY_CIRCLE_CM
            if dist_cm < threshold:
                violations.append((round(angle_deg, 1), round(dist_cm, 1)))
        return violations

    def run_cycle(self):
        """智慧探索巡邏迴圈：
        1. 停車 → 標記當前位置
        2. 舵機掃描（視覺偵測）
        3. 取 360° LiDAR
        4. 評分 12 個方向，選出最佳目標
        5. 旋轉對準
        6. 直行一個小步（最多 STEP_DISTANCE_CM）"""
        if not self._ok():
            self.motor.stop()
            return

        self._cycle += 1

        # ── Step 1: 停車 + 標記已訪問 ──
        self.motor.stop()
        self._mark_visited()

        # ── Step 2: 舵機掃描（每 cycle）──
        self._sweep()
        if not self._ok():
            return

        # ── Step 3: 取 LiDAR 與 pose ──
        ranges = self._rb.get_lidar_ranges() if self._rb else {}
        pose = self._rb.get_slam_pose() if self._rb else None

        if not ranges:
            logger.info(f"[F] #{self._cycle} LiDAR 無資料，前進 1.5s")
            self._drive(DRIVE_SPEED, 1.5)
            return

        # ── Step 4: 評分決策 ──
        target_angle, target_dist_m = self._pick_direction(ranges, pose)

        if target_angle is None:
            logger.warning(f"[F] #{self._cycle} 無可行方向，後退脫困")
            self._escape_reverse()
            return

        logger.info(
            f"[F] #{self._cycle} 選擇方向 {target_angle:+.0f}° "
            f"(淨空 {target_dist_m*100:.0f}cm, visited={len(self._visited)})"
        )

        # ── Step 5: 大角度時先旋轉 ──
        if abs(target_angle) > 15:
            self._rotate(target_angle)
            if not self._ok():
                return

        # ── Step 6: 前進 STEP_DISTANCE_CM（由 _drive_step 監測障礙自動煞車）──
        self._drive_step(self.STEP_DISTANCE_CM)

    # ══════════════════════════════════════════════════════════
    # 新：SLAM + LiDAR 智慧決策輔助函數
    # ══════════════════════════════════════════════════════════

    def _mark_visited(self):
        """把當前 SLAM pose 轉成世界格子座標並標記為已訪問。"""
        if not self._rb:
            return
        pose = self._rb.get_slam_pose()
        if pose is None:
            return
        cx = round(pose[0] / self.VISITED_CELL_M)
        cy = round(pose[1] / self.VISITED_CELL_M)
        # 當前格 + 八鄰域都標記為 visited（避免近距離來回）
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                self._visited.add((cx + dx, cy + dy))

    def _pick_direction(self, ranges, pose):
        """對 12 個方向評分並回傳最佳 (angle_deg, dist_m)。
        無可行方向回傳 (None, 0)。"""
        best_angle = None
        best_dist = 0
        best_score = 0

        for angle_deg in range(-180, 180, 30):
            # 車身淨空 (已扣除車邊偏移) — 這才是真正能走的距離
            clear_cm = self._direction_clear_cm(angle_deg, ranges)
            if clear_cm < 0:
                continue
            if clear_cm < OBSTACLE_CLEAR_CM:
                continue
            dist_m = clear_cm / 100.0   # 用淨空作為可行距離

            # 計算投影目標位置
            step_m = min(dist_m * self.PROJECT_RATIO, self.PROJECT_MAX_M)

            # 已訪問懲罰
            visit_factor = 1.0
            if pose is not None:
                target_rad = pose[2] + math.radians(angle_deg)
                proj_x = pose[0] + step_m * math.cos(target_rad)
                proj_y = pose[1] + step_m * math.sin(target_rad)
                cell = (
                    round(proj_x / self.VISITED_CELL_M),
                    round(proj_y / self.VISITED_CELL_M),
                )
                if cell in self._visited:
                    visit_factor = 0.25   # 大幅扣分但不完全封殺

            # 角度偏好：0° 滿分，±180° 扣 40%
            angle_bonus = 1.0 - (abs(angle_deg) / 180.0) * 0.4

            score = dist_m * angle_bonus * visit_factor
            if score > best_score:
                best_score = score
                best_angle = angle_deg
                best_dist = dist_m

        return best_angle, best_dist

    def _soft_start(self, vx, vy, omega, ramp_sec=0.15, steps=5):
        """0 → 目標速度的線性 ramp，避免 PWM 瞬間跳變造成輪子打滑。
        總時間 = ramp_sec（約 0.15s）。之後 caller 會繼續維持目標速度。"""
        for i in range(1, steps + 1):
            frac = i / steps
            self.motor.move(vx * frac, vy * frac, omega * frac)
            time.sleep(ramp_sec / steps)

    def _drive_step(self, max_cm: float):
        """前進固定距離：軟起步 + 全速巡航 + SLAM 量測距離。
        遇障礙硬煞車後退。"""
        start_pose = self._rb.get_slam_pose() if self._rb else None
        target_m = max_cm / 100.0

        # 軟起步（0.15s 從 0 加速到 DRIVE_SPEED）
        self._soft_start(0, DRIVE_SPEED, 0, ramp_sec=0.15)
        t0 = time.time()
        hit = False
        dist_travelled = 0.0

        while time.time() - t0 < self.STEP_TIMEOUT_SEC:
            if not self._ok():
                break
            time.sleep(0.1)
            # 障礙檢查
            d = self._raw_front_cm()
            if 0 < d < BRAKE_CLEAR_CM:
                hit = True
                break
            # 距離量測（SLAM 優先）
            if start_pose is not None and self._rb:
                cur = self._rb.get_slam_pose()
                if cur is not None:
                    dx = cur[0] - start_pose[0]
                    dy = cur[1] - start_pose[1]
                    dist_travelled = math.hypot(dx, dy)
                    if dist_travelled >= target_m:
                        break

        self.motor.stop()
        dt = time.time() - t0

        # 記錄給 backtrack
        if self._bt:
            self._bt.record("forward", DRIVE_SPEED, dt)
        if self._hm:
            self._hm.update_forward(DRIVE_SPEED, dt)
            self._hm.mark_scanned()

        if hit:
            logger.info(f"[F] 前進 {dist_travelled*100:.0f}cm → 遇障礙後退")
            self.motor.move(0, -REVERSE_SPEED, 0)
            time.sleep(0.3)
            self.motor.stop()
            if self._bt:
                self._bt.record("reverse", REVERSE_SPEED, 0.3)
            if self._hm:
                self._hm.update_reverse(REVERSE_SPEED, 0.3)
                self._hm.mark_obstacle()
        else:
            logger.info(f"[F] 前進 {dist_travelled*100:.0f}cm 完成")

    def _escape_reverse(self):
        """所有方向被堵 → 軟後退 + 原地旋轉 180° 找新方向"""
        self._soft_start(0, -REVERSE_SPEED, 0, ramp_sec=0.15)
        time.sleep(max(0.0, 0.8 - 0.15))
        self.motor.stop()
        if self._bt:
            self._bt.record("reverse", REVERSE_SPEED, 0.8)
        self._rotate(180)

    def _rear_clear_cm(self, ranges: dict = None) -> float:
        """車尾實際淨空 (cm)：raw LiDAR rear arc - LiDAR→車尾 5cm 偏移。
        只看 LiDAR 後方 ±45° 扇形。"""
        if ranges is None:
            if not self._rb:
                return -1
            ranges = self._rb.get_lidar_ranges()
        if not ranges:
            return -1
        # 後方 = 180° 附近 (也就是 |angle - 180| < 45 或 |angle + 180| < 45)
        rear_dists = [d for a, d in ranges.items() if _adiff(a, 180) < 45 and d > 0]
        if not rear_dists:
            return -1
        raw_cm = min(rear_dists) * 100
        rear_off = getattr(config, 'ROS_LIDAR_TO_REAR_CM', 5)
        return round(raw_cm - rear_off, 1)

    def _safety_escape(self, violation_angle_deg: float):
        """安全圓違規脫困。
        ⚠ 絕對不用 mecanum 平移（strafe 有 LiDAR 死角 + 馬達漂移問題）。
        策略：
          1. 違規在前方 → 後退（若車尾有空間）
          2. 違規在後方 → 前進（若車頭有空間）
          3. 違規在側邊 → 往違規反方向**小角度旋轉**再觀察
          4. 都沒空間 → 180° 大旋轉
        """
        if not self._rb:
            return
        ranges = self._rb.get_lidar_ranges()
        if not ranges:
            return

        front_clear = self._front_clear_cm()
        rear_clear = self._rear_clear_cm(ranges)
        is_front_violation = abs(violation_angle_deg) < 90
        is_rear_violation = abs(violation_angle_deg) >= 90

        logger.info(
            f"[F] 脫困評估：front={front_clear:.0f}cm rear={rear_clear:.0f}cm "
            f"violation @ {violation_angle_deg:.0f}°"
        )

        # 策略 1：違規在前/側前方 + 車尾有空間 → 慢速後退 2-3cm
        if is_front_violation and rear_clear >= 20:
            logger.info(f"[F] 脫困：後退 2cm (rear={rear_clear:.0f}cm)")
            self._soft_start(0, -0.20, 0, ramp_sec=0.25)
            time.sleep(0.15)   # ~2cm
            self.motor.stop()
            time.sleep(0.15)
            if self._bt:
                self._bt.record("reverse", 0.20, 0.15)
            return

        # 策略 2：違規在後/側後方 + 車頭有空間 → 慢速前進 2-3cm
        if is_rear_violation and front_clear >= 20:
            logger.info(f"[F] 脫困：前進 2cm (front={front_clear:.0f}cm)")
            self._soft_start(0, 0.20, 0, ramp_sec=0.25)
            time.sleep(0.15)
            self.motor.stop()
            time.sleep(0.15)
            if self._bt:
                self._bt.record("forward", 0.20, 0.15)
            return

        # 策略 3：前後都不太寬 → 小角度旋轉遠離違規點
        # 違規在左 (-angle) → 往右轉（+angle）；反之亦然
        turn_dir = 1 if violation_angle_deg < 0 else -1
        turn_deg = 30 * turn_dir
        logger.info(f"[F] 脫困：小角度旋轉 {turn_deg:+.0f}°")
        self._rotate(turn_deg)

    def _drive(self, speed, sec):
        self.motor.move(0, speed, 0)
        t0 = time.time()
        hit = False
        while time.time() - t0 < sec:
            time.sleep(0.1)
            d = self._raw_front_cm()
            if 0 < d < BRAKE_CLEAR_CM:
                hit = True
                break
        self.motor.stop()
        dt = time.time() - t0
        if self._bt:
            self._bt.record("forward", speed, dt)
        if self._hm:
            self._hm.update_forward(speed, dt)
            self._hm.mark_scanned()
        if hit:
            logger.info("[F] 障礙，後退")
            self.motor.move(0, -REVERSE_SPEED, 0)
            time.sleep(0.3)
            self.motor.stop()
            if self._bt:
                self._bt.record("reverse", REVERSE_SPEED, 0.3)
            if self._hm:
                self._hm.update_reverse(REVERSE_SPEED, 0.3)
                self._hm.mark_obstacle()

    def _sweep(self):
        """Mode F 水平掃描：用 sweep_pan_to 平滑步進，時序由 config 控制。"""
        step = max(1, abs(config.PATROL_PAN_STEP))
        interval = config.PATROL_SWEEP_DELAY
        t0 = time.time()

        # 先移到起點
        if self._ok():
            self.servo.sweep_pan_to(config.PATROL_PAN_MIN)
            self._pan_fn(config.PATROL_PAN_MIN)
            time.sleep(0.2)

        # 往右：MIN → MAX
        p = config.PATROL_PAN_MIN
        while p < config.PATROL_PAN_MAX:
            if not self._ok() or time.time() - t0 > SWEEP_SEC:
                break
            p = min(p + step, config.PATROL_PAN_MAX)
            self.servo.sweep_pan_to(p)
            self._pan_fn(p)
            time.sleep(interval)

        # 往左：MAX → MIN
        while p > config.PATROL_PAN_MIN:
            if not self._ok() or time.time() - t0 > SWEEP_SEC:
                break
            p = max(p - step, config.PATROL_PAN_MIN)
            self.servo.sweep_pan_to(p)
            self._pan_fn(p)
            time.sleep(interval)

        # 歸中（mode 已切換就直接 detach）
        if self._ok():
            self.servo.sweep_pan_to(0)
            self._pan_fn(0)
            time.sleep(0.2)
        self.servo.sweep_end()
        if self._hm:
            self._hm.mark_scan_fan(config.PATROL_PAN_MIN, config.PATROL_PAN_MAX)

    def _try_explore(self):
        if not self._hm:
            return
        angle, count = self._hm.get_best_exploration_angle()
        # 只在角度明顯偏離（>30°）且有足夠未探索格子時才轉
        if abs(angle) < 30 or count < 3:
            return
        # LiDAR 驗證：用車身淨空判斷
        if self._rb:
            ranges = self._rb.get_lidar_ranges()
            if ranges:
                clear = self._direction_clear_cm(angle, ranges, arc_half=30)
                if clear >= 0 and clear < OBSTACLE_CLEAR_CM:
                    return
        # 限制探索轉向最大 90°（不要在通暢時做大幅轉彎）
        angle = max(-90, min(90, angle))
        self._rotate(angle)

    def _redirect(self):
        """堵住：後退 → 用 LiDAR 找最空方向 → 轉過去 → 前進。
        若連續多次選到小角度（< 45°），判定為繞圈 → 強制大角度脫離。"""
        self.motor.stop()
        if self._hm:
            self._hm.mark_obstacle()

        # 後退
        self.motor.move(0, -REVERSE_SPEED, 0)
        time.sleep(0.5)
        self.motor.stop()
        if self._bt:
            self._bt.record("reverse", REVERSE_SPEED, 0.5)
        if self._hm:
            self._hm.update_reverse(REVERSE_SPEED, 0.5)

        # 找方向
        target = self._best_lidar_dir()
        if target is None:
            target = 90 * self._turn_dir
            self._turn_dir *= -1

        # 繞圈偵測：前一次轉向也是相同方向且 < 45° → 強制改 ±120°
        if (hasattr(self, "_last_turn_target")
                and self._last_turn_target is not None
                and abs(target) < 45
                and abs(self._last_turn_target) < 45
                and (target * self._last_turn_target > 0)):
            target = 120 * (1 if target > 0 else -1)
            logger.warning(f"[F] ⚠️ 偵測到繞圈，強制大角度轉向 {target}°")
        self._last_turn_target = target

        logger.info(f"[F] 轉向 {target}°")
        self._rotate(target)

        # 轉完前進一段
        self._drive(DRIVE_SPEED, 1.5)

    def _best_lidar_dir(self):
        """
        用 LiDAR 360° 選最佳方向：優先最遠可通行，並給「較直」的方向加分。
        熱區作為 tie-breaker（相似距離時偏好未探索方向）。
        """
        if not self._rb:
            return None
        ranges = self._rb.get_lidar_ranges()
        if not ranges:
            return None

        # 收集所有可通行方向（車身淨空 > OBSTACLE_CLEAR_CM）
        passable = {}
        for c in range(-180, 180, 30):
            clear = self._direction_clear_cm(c, ranges, arc_half=20)
            if clear >= OBSTACLE_CLEAR_CM:
                passable[c] = clear / 100.0   # 存 m 供後續評分用

        if not passable:
            return None

        # 取得熱區探索角度（可選）
        explore_angle = None
        if self._hm:
            try:
                ea, count = self._hm.get_best_exploration_angle()
                if count > 1:
                    explore_angle = ea
            except Exception:
                pass

        # 評分 = 距離（主要）+ 小角度加分（偏好直行）+ 熱區加分
        max_dist = max(passable.values())
        scored = {}
        for angle, dist in passable.items():
            # 距離歸一化（0~1）
            dist_score = dist / max_dist
            # 小角度偏好：|angle|=0 加 30%，|angle|=180 扣 10%
            angle_bonus = 1.0 - (abs(angle) / 180.0) * 0.4
            score = dist_score * angle_bonus
            # 熱區 tie-break：方向接近探索角度 → +15%
            if explore_angle is not None and _adiff(angle, explore_angle) < 30:
                score *= 1.15
            scored[angle] = (score, dist)

        # 取最高分
        best = max(scored, key=lambda a: scored[a][0])
        best_dist_m = scored[best][1]
        tag = " (未探索區域)" if explore_angle is not None and _adiff(best, explore_angle) < 30 else ""
        logger.info(f"[F] 選擇方向 {best}° (車身淨空 {best_dist_m*100:.0f}cm){tag}")
        return best

    def _escape(self):
        """連續堵死脫困：後退 + 大旋轉（取消 mecanum 平移，避免 LiDAR 死角撞擊）"""
        self._log("warn", "連續堵住，脫困")
        # 先慢速後退
        self._soft_start(0, -REVERSE_SPEED, 0, ramp_sec=0.2)
        time.sleep(0.65)
        self.motor.stop()
        time.sleep(0.1)
        if self._hm:
            self._hm.update_reverse(REVERSE_SPEED, 0.8)

        # 選最佳方向或硬轉 120°
        target = self._best_lidar_dir()
        if target is None:
            target = 120 * self._turn_dir
        self._rotate(target)
        self._turn_dir *= -1

    def _rotate(self, deg):
        """簡單開環旋轉：軟起步 + 連續旋轉 + 硬停。"""
        if abs(deg) < 8:
            return
        deg = max(-120, min(120, deg))
        d = 1 if deg > 0 else -1
        actual_speed = min(TURN_SPEED * 1.5, 0.50)
        actual_deg_per_sec = actual_speed * getattr(config, 'HEAT_MAP_RAD_PER_SPEED_SEC', 2.0) * 180 / math.pi
        actual_deg_per_sec = max(actual_deg_per_sec, 15)
        dur = abs(deg) / actual_deg_per_sec
        logger.info(f"[F] 旋轉 {deg}° dur={dur:.1f}s")
        scanning_lock.set()
        try:
            self._soft_start(0, 0, actual_speed * d, ramp_sec=0.15)
            hold = max(0.0, dur - 0.15)
            if hold > 0:
                time.sleep(hold)
            self.motor.stop()
        finally:
            scanning_lock.clear()
        new_d = self._raw_front_cm()
        logger.info(f"[F] 旋轉完成，新前方距離={new_d:.0f}cm")
        if self._bt:
            self._bt.record("turn", TURN_SPEED, dur, d)
        if self._hm:
            self._hm.update_turn(TURN_SPEED, dur, d)


def _adiff(a, b):
    d = abs(a - b) % 360
    return d if d <= 180 else 360 - d
