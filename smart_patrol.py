"""
搜救機器人 — 智慧巡邏模組 (V3 探索導航版) [DEPRECATED]
==========================================
走停掃描 + 熱區地圖導航：
  前進 → 停車掃描 → 查熱區地圖選方向 → 遇障礙轉彎。
  每次掃描後主動轉向未探索最多的方向。

DEPRECATED (V11 改動)：
  模式 E 已改由 scan_patrol.ScanPatrol.set_profile("smart") + run_cycle()
  實作，所有行進/轉彎決策統一改以 SLAM /map + TF 為權威來源。
  本類別的 heat_map dead-reckoning 決策路徑不再被 search_loop 呼叫，
  暫保留供未來實驗與參考，但不應用於生產巡邏流程。
"""

import time
import logging

import config

logger = logging.getLogger("rescue.smart_patrol")


class SmartPatrol:
    """
    災後搜救智慧巡邏。
    呼叫 run_cycle() 執行一個完整週期（前進→掃描→導航/避障）。
    """

    def __init__(self, motor, servo, get_distance_fn,
                 get_victim_score_fn, check_active_fn,
                 add_log_fn, update_pan_fn, heat_map=None,
                 ros_bridge=None):
        self.motor = motor
        self.servo = servo
        self._get_distance = get_distance_fn
        self._get_score = get_victim_score_fn
        self._active = check_active_fn
        self._log = add_log_fn
        self._update_pan = update_pan_fn
        self._heat_map = heat_map
        self._ros_bridge = ros_bridge

        self._turn_dir = 1        # 交替左右轉（避障用）
        self._consec_blocked = 0  # 連續被堵次數

        # 卡住偵測
        self._last_grid_pos = None
        self._same_pos_count = 0

    # ──────────────────────────────────────────────
    # 主入口：一個完整的巡邏週期
    # ──────────────────────────────────────────────

    def run_cycle(self):
        """停車掃描 → 導航 → 前進 → 避障。一個週期約 5~10 秒。
        掃描永遠先做，確保任何位置都能偵測周圍目標。"""

        if not self._active():
            self.motor.stop()
            return

        # ── 步驟 1：停車掃描（永遠先做）──
        self.motor.stop()
        self._scan_sweep()

        if not self._active():
            self.motor.stop()
            return

        # ── 步驟 2：用熱區地圖決定方向（掃完後才決定要往哪走）──
        dist = self._get_distance()
        if dist < 0 or (0 < dist < config.SMART_PATROL_OBSTACLE):
            # 前方有障礙 → 避障
            self._avoid()
            self._check_stuck()
            return
        else:
            self._consec_blocked = 0
            self._navigate_by_map()

        if not self._active():
            self.motor.stop()
            return

        # ── 步驟 3：前進 ──
        self._drive_forward()

        # ── 步驟 4：卡住偵測 ──
        self._check_stuck()

    # ──────────────────────────────────────────────
    # 步驟 1：前進
    # ──────────────────────────────────────────────

    def _drive_forward(self) -> bool:
        """
        前進 MOVE_SEC 秒，途中每 0.1s 檢查 LiDAR 距離。
        回傳 True = 被障礙提前打斷。
        """
        logger.info("[巡邏] 前進中...")
        self.motor.move(0, config.SMART_PATROL_SPEED, 0)

        drive_start = time.time()
        hit_obstacle = False
        steps = int(config.SMART_PATROL_MOVE_SEC / 0.1)
        for _ in range(steps):
            if not self._active():
                break
            time.sleep(0.1)
            dist = self._get_distance()

            if 0 < dist < config.SMART_PATROL_OBSTACLE:
                logger.info(f"[巡邏] 前方障礙 {dist:.0f}cm，提前停車")
                self.motor.stop()
                hit_obstacle = True
                break

        if not hit_obstacle:
            self.motor.stop()

        # 更新熱區地圖位置
        if self._heat_map:
            actual_duration = time.time() - drive_start
            self._heat_map.update_forward(config.SMART_PATROL_SPEED, actual_duration)

        return hit_obstacle

    # ──────────────────────────────────────────────
    # 步驟 2：停車掃描
    # ──────────────────────────────────────────────

    def _scan_sweep(self):
        """舵機左→右→歸中，一個完整來回。
        用 sweep_pan_to 直接寫 pan，固定 step + interval 確保平滑。
        掃描間隔內仍持續檢查分數以快速鎖定。"""
        logger.info("[巡邏] 開始水平掃描")
        step = max(1, abs(config.PATROL_PAN_STEP))
        interval = config.PATROL_SWEEP_DELAY

        def _step_and_check(target_pan):
            """移到 target_pan，sleep interval，順便查分數/active"""
            self.servo.sweep_pan_to(target_pan)
            self._update_pan(target_pan)
            # interval 期間每 20ms 查一次
            ticks = max(1, int(interval / 0.02))
            for _ in range(ticks):
                if self._get_score() >= config.VICTIM_SUSPECT_THRESHOLD:
                    return True   # 中斷掃描
                if not self._active():
                    return True
                time.sleep(0.02)
            return False

        # 先移到起點 (-MIN)
        start = config.PATROL_PAN_MIN
        self.servo.sweep_pan_to(start)
        self._update_pan(start)
        time.sleep(0.2)   # 給一個短暫時間讓舵機到達起點

        # 往右：PATROL_PAN_MIN → PATROL_PAN_MAX
        pan = start
        while pan < config.PATROL_PAN_MAX:
            pan = min(pan + step, config.PATROL_PAN_MAX)
            if _step_and_check(pan):
                break

        # 往左：PATROL_PAN_MAX → PATROL_PAN_MIN
        if self._active() and self._get_score() < config.VICTIM_SUSPECT_THRESHOLD:
            while pan > config.PATROL_PAN_MIN:
                pan = max(pan - step, config.PATROL_PAN_MIN)
                if _step_and_check(pan):
                    break

        # 歸中（mode 已切換就直接 detach，不再動 servo）
        if self._active():
            self.servo.sweep_pan_to(0)
            self._update_pan(0)
            time.sleep(0.2)
        self.servo.sweep_end()   # 無論如何都要 detach 確保靜默

        # 標記掃描區域到熱區地圖
        if self._heat_map:
            self._heat_map.mark_scan_fan(config.PATROL_PAN_MIN, config.PATROL_PAN_MAX)

    # ──────────────────────────────────────────────
    # 步驟 3a：熱區導航（無障礙時）
    # ──────────────────────────────────────────────

    def _navigate_by_map(self):
        """
        用熱區地圖找到未探索最多的方向，主動轉過去。
        已強化：
          - get_best_exploration_angle 現在偏好直行
          - 只在前方明確無可探時才轉
          - 轉彎角度最大限制在 ±90° (避免 180° 大迴轉)
        """
        if not self._heat_map:
            return

        best_angle, unexplored_count = self._heat_map.get_best_exploration_angle()

        # best_angle = 0 表示 get_best_exploration_angle 判斷應直行
        if best_angle == 0:
            return

        # 未探索格數太少 → 附近都探索過了，不需要刻意轉
        if unexplored_count < 4:
            return

        # 限制轉彎角度在 ±90°，避免 180° 迴轉
        if abs(best_angle) > 90:
            best_angle = 90 if best_angle > 0 else -90

        # LiDAR 驗證：確認目標方向確實可通行
        turn_dir = 1 if best_angle > 0 else -1
        if self._ros_bridge:
            ranges = self._ros_bridge.get_lidar_ranges()
            if ranges:
                # 檢查目標方向 ±30° 是否暢通
                target_dists = [d for a, d in ranges.items()
                                if abs(a - best_angle) < 30 or abs(a - best_angle + 360) < 30]
                if target_dists and min(target_dists) < 0.30:
                    # 目標方向被堵，改找 LiDAR 最空曠的方向
                    sector_scores = {}
                    for sector in range(-180, 180, 30):
                        dists = [d for a, d in ranges.items()
                                 if abs(a - sector) < 15 or abs(a - sector + 360) < 15]
                        if dists:
                            sector_scores[sector] = sum(dists) / len(dists)
                    if sector_scores:
                        best_angle = max(sector_scores, key=sector_scores.get)
                        turn_dir = 1 if best_angle > 0 else -1
                        logger.info(f"[巡邏] LiDAR 修正導航方向：{best_angle}°")
        # 將角度映射到轉彎時間：45° ≈ 基礎轉彎時間，90° ≈ 2倍，180° ≈ 3倍
        angle_ratio = min(abs(best_angle) / 60.0, 3.0)
        turn_duration = config.SMART_PATROL_TURN_SEC * angle_ratio

        dir_name = "右" if turn_dir > 0 else "左"
        logger.info(f"[巡邏] 熱區導航：轉{dir_name} {abs(best_angle)}° 探索未知區域")
        self._log("info", f"探索導航：轉{dir_name} {abs(best_angle)}°")

        self.motor.move(0, 0, config.PATROL_TURN_SPEED * turn_dir)
        time.sleep(turn_duration)
        self.motor.stop()

        if self._heat_map:
            self._heat_map.update_turn(
                config.PATROL_TURN_SPEED, turn_duration, turn_dir)

    # ──────────────────────────────────────────────
    # 步驟 3b：避障
    # ──────────────────────────────────────────────

    def _avoid(self):
        """後退 + 轉彎。連續被堵就升級策略。避障永遠交替方向防止卡角落。"""
        self._consec_blocked += 1
        self.motor.stop()

        # 在地圖上標記障礙物位置
        if self._heat_map:
            self._heat_map.mark_obstacle()

        turn_dir = self._turn_dir

        # LiDAR 智慧方向選擇：用 360° 掃描判斷哪邊更空曠
        if self._ros_bridge:
            ranges = self._ros_bridge.get_lidar_ranges()
            if ranges:
                left_free = sum(1 for a, d in ranges.items() if -135 <= a <= -45 and d > 0.5)
                right_free = sum(1 for a, d in ranges.items() if 45 <= a <= 135 and d > 0.5)
                if left_free != right_free:
                    turn_dir = 1 if right_free > left_free else -1
                    logger.info(f"[巡邏] LiDAR 判斷：左={left_free} 右={right_free}，轉{'右' if turn_dir > 0 else '左'}")

        if self._consec_blocked >= 5:
            # 連續 5 次堵死 → 麥克納姆側移脫困
            logger.info("[巡邏] 連續被堵 5 次 → 側移脫困")
            self._log("warn", "側移脫困")
            self.motor.move(config.STRAFE_SPEED * turn_dir, 0, 0)
            time.sleep(config.STRAFE_DURATION)
            self.motor.stop()
            if self._heat_map:
                self._heat_map.update_strafe(
                    config.STRAFE_SPEED, config.STRAFE_DURATION, turn_dir)
            self._turn_dir *= -1
            self._consec_blocked = 0

        elif self._consec_blocked >= 3:
            # 連續 3 次堵 → 大角度後退+轉彎
            logger.info("[巡邏] 連續被堵 3 次 → 大角度後退轉彎")
            self._log("warn", "大角度轉彎脫困")
            self.motor.move(0, -config.PATROL_REVERSE_SPEED, 0)
            time.sleep(config.SMART_PATROL_REVERSE_SEC * 1.5)
            self.motor.stop()
            if self._heat_map:
                self._heat_map.update_reverse(
                    config.PATROL_REVERSE_SPEED, config.SMART_PATROL_REVERSE_SEC * 1.5)
            self.motor.move(0, 0, config.PATROL_TURN_SPEED * turn_dir)
            time.sleep(config.SMART_PATROL_TURN_SEC * 2)
            self.motor.stop()
            if self._heat_map:
                self._heat_map.update_turn(
                    config.PATROL_TURN_SPEED, config.SMART_PATROL_TURN_SEC * 2, turn_dir)
            self._turn_dir *= -1

        else:
            # 一般避障 → 後退 + 轉彎
            logger.info(f"[巡邏] 避障：後退+轉{'左' if turn_dir < 0 else '右'}")
            self._log("info", f"避障轉{'左' if turn_dir < 0 else '右'}")
            self.motor.move(0, -config.PATROL_REVERSE_SPEED, 0)
            time.sleep(config.SMART_PATROL_REVERSE_SEC)
            self.motor.stop()
            if self._heat_map:
                self._heat_map.update_reverse(
                    config.PATROL_REVERSE_SPEED, config.SMART_PATROL_REVERSE_SEC)
            self.motor.move(0, 0, config.PATROL_TURN_SPEED * turn_dir)
            time.sleep(config.SMART_PATROL_TURN_SEC)
            self.motor.stop()
            if self._heat_map:
                self._heat_map.update_turn(
                    config.PATROL_TURN_SPEED, config.SMART_PATROL_TURN_SEC, turn_dir)
            self._turn_dir *= -1

    # ──────────────────────────────────────────────
    # 步驟 4：卡住偵測
    # ──────────────────────────────────────────────

    def _check_stuck(self):
        """
        如果連續 4 個週期都在同一個格位 → 機器人卡住了。
        強制大角度轉彎（用熱區地圖找最佳方向）脫困。
        """
        if not self._heat_map:
            return

        current_pos = self._heat_map.get_grid_position()

        if current_pos == self._last_grid_pos:
            self._same_pos_count += 1
        else:
            self._same_pos_count = 0
            self._last_grid_pos = current_pos

        if self._same_pos_count >= 4:
            logger.warning(f"[巡邏] 卡住偵測：同一位置 {self._same_pos_count} 個週期！強制脫困")
            self._log("warn", "卡住偵測：強制轉向脫困")

            # 用熱區地圖找最佳探索方向
            best_angle, _ = self._heat_map.get_best_exploration_angle()
            if abs(best_angle) < 45:
                best_angle = 120 * self._turn_dir  # 至少轉 120°

            turn_dir = 1 if best_angle > 0 else -1
            turn_duration = config.SMART_PATROL_TURN_SEC * 3  # 大角度轉

            # 先後退
            self.motor.move(0, -config.PATROL_REVERSE_SPEED, 0)
            time.sleep(config.SMART_PATROL_REVERSE_SEC * 2)
            self.motor.stop()
            if self._heat_map:
                self._heat_map.update_reverse(
                    config.PATROL_REVERSE_SPEED, config.SMART_PATROL_REVERSE_SEC * 2)

            # 再大角度轉彎
            self.motor.move(0, 0, config.PATROL_TURN_SPEED * turn_dir)
            time.sleep(turn_duration)
            self.motor.stop()
            if self._heat_map:
                self._heat_map.update_turn(
                    config.PATROL_TURN_SPEED, turn_duration, turn_dir)

            self._same_pos_count = 0
            self._consec_blocked = 0
            self._turn_dir *= -1
