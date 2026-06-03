"""
搜救機器人 — 熱區記憶地圖模組
===============================
格狀地圖追蹤已掃描/未掃描區域，引導巡邏優先探索未知區域。
使用航位推算 (Dead Reckoning) 估計機器人位置，無需額外硬體。

校準說明：
  HEAT_MAP_CM_PER_SPEED_SEC — 速度 1.0 每秒移動距離 (cm)
  HEAT_MAP_RAD_PER_SPEED_SEC — 速度 1.0 每秒旋轉角度 (rad)
  這兩個參數需要在實際機器人上校準以獲得最佳效果。
"""

import math
import threading
import logging
import numpy as np

import config

logger = logging.getLogger("rescue.heat_map")


class HeatMap:
    """
    格狀探索記憶地圖。
    使用航位推算追蹤機器人大致位置，標記掃描過的區域。
    引導巡邏優先探索未知區域，避免重複搜索。
    """

    def __init__(self):
        size = config.HEAT_MAP_GRID_SIZE
        self.grid = np.zeros((size, size), dtype=np.int32)
        self.cell_cm = config.HEAT_MAP_CELL_CM

        # 機器人位置 (cm) 和航向 (rad, 0 = 正前方 / +Y)
        self.pos_x = 0.0
        self.pos_y = 0.0
        self.heading = 0.0

        # 網格原點 = 中心
        self._origin = size // 2
        self._lock = threading.Lock()

        # 路徑歷史 (grid coords)
        self.path_history = []
        # 事件標記
        self.obstacles = []   # [(gx, gy), ...] 障礙物位置
        self.persons = []     # [{"gx":, "gy":, "status":}, ...] 人員位置
        self._record_position()

        logger.info(f"✅ 熱區地圖初始化 | {size}×{size} 格 | 每格 {self.cell_cm}cm")

    # ──────────────────────────────────────────────
    # 位置更新
    # ──────────────────────────────────────────────

    def update_forward(self, speed: float, duration_sec: float):
        """前進後更新位置"""
        with self._lock:
            dist_cm = speed * config.HEAT_MAP_CM_PER_SPEED_SEC * duration_sec
            self.pos_x += dist_cm * math.sin(self.heading)
            self.pos_y += dist_cm * math.cos(self.heading)
            self._record_position()

    def update_reverse(self, speed: float, duration_sec: float):
        """後退後更新位置"""
        with self._lock:
            dist_cm = speed * config.HEAT_MAP_CM_PER_SPEED_SEC * duration_sec
            self.pos_x -= dist_cm * math.sin(self.heading)
            self.pos_y -= dist_cm * math.cos(self.heading)
            self._record_position()

    def update_turn(self, turn_speed: float, duration_sec: float, direction: int):
        """轉彎後更新航向 (direction: +1=右, -1=左)"""
        with self._lock:
            angle = turn_speed * config.HEAT_MAP_RAD_PER_SPEED_SEC * duration_sec * direction
            self.heading += angle

    def update_strafe(self, speed: float, duration_sec: float, direction: int):
        """側移後更新位置 (direction: +1=右, -1=左)"""
        with self._lock:
            dist_cm = speed * config.HEAT_MAP_CM_PER_SPEED_SEC * duration_sec
            self.pos_x += dist_cm * math.cos(self.heading) * direction
            self.pos_y -= dist_cm * math.sin(self.heading) * direction
            self._record_position()

    # ──────────────────────────────────────────────
    # 掃描標記
    # ──────────────────────────────────────────────

    def mark_scanned(self, radius_cells: int = None):
        """標記當前位置周圍為已掃描（圓形區域）"""
        if radius_cells is None:
            radius_cells = config.HEAT_MAP_SCAN_RADIUS
        with self._lock:
            gx, gy = self._pos_to_grid()
            size = self.grid.shape[0]
            for dx in range(-radius_cells, radius_cells + 1):
                for dy in range(-radius_cells, radius_cells + 1):
                    if dx * dx + dy * dy <= radius_cells * radius_cells:
                        nx, ny = gx + dx, gy + dy
                        if 0 <= nx < size and 0 <= ny < size:
                            self.grid[nx, ny] += 1

    def mark_obstacle(self, dist_cm: float = None):
        """
        在機器人前方標記障礙物。
        dist_cm: 超聲波讀數（可選），用於精確定位障礙位置。
        """
        with self._lock:
            gx, gy = self._pos_to_grid()
            if dist_cm and dist_cm > 0:
                # 精確定位：用實際距離計算障礙格位
                obs_r = max(1, int(dist_cm / self.cell_cm))
                ox = gx + int(obs_r * math.sin(self.heading))
                oy = gy + int(obs_r * math.cos(self.heading))
                pos = (ox, oy)
                if pos not in self.obstacles:
                    self.obstacles.append(pos)
            else:
                # 粗略定位：前方 1~2 格
                for r in range(1, 3):
                    ox = gx + int(r * math.sin(self.heading))
                    oy = gy + int(r * math.cos(self.heading))
                    pos = (ox, oy)
                    if pos not in self.obstacles:
                        self.obstacles.append(pos)
            if len(self.obstacles) > 300:
                self.obstacles = self.obstacles[-300:]

    def mark_person(self, status: str = "detected"):
        """在機器人當前位置標記偵測到的人員"""
        with self._lock:
            gx, gy = self._pos_to_grid()
            self.persons.append({"gx": gx, "gy": gy, "status": status})
            if len(self.persons) > 50:
                self.persons = self.persons[-50:]

    def mark_lidar_scan(self, scan_data: dict):
        """
        記錄 LiDAR 360° 掃描資料到地圖。
        scan_data: {角度(度): 距離(cm)}
        - 射線沿途：標記為已掃描（確認無障礙）
        - 射線末端：標記障礙物
        """
        with self._lock:
            gx, gy = self._pos_to_grid()
            size = self.grid.shape[0]

            for angle_deg, dist_cm in scan_data.items():
                angle_rad = self.heading + math.radians(angle_deg)
                if dist_cm <= 0:
                    continue

                # 沿射線標記已掃描（從機器人到障礙之間都是空的）
                max_r = min(int(dist_cm / self.cell_cm), 10)
                for r in range(1, max_r + 1):
                    nx = gx + int(r * math.sin(angle_rad))
                    ny = gy + int(r * math.cos(angle_rad))
                    if 0 <= nx < size and 0 <= ny < size:
                        self.grid[nx, ny] = max(self.grid[nx, ny], 1)

                # 障礙位置（射線末端，只在合理距離內）
                if dist_cm < 300:
                    obs_r = max(1, int(dist_cm / self.cell_cm))
                    ox = gx + int(obs_r * math.sin(angle_rad))
                    oy = gy + int(obs_r * math.cos(angle_rad))
                    pos = (ox, oy)
                    if pos not in self.obstacles:
                        self.obstacles.append(pos)

            if len(self.obstacles) > 300:
                self.obstacles = self.obstacles[-300:]

    def mark_scan_fan(self, pan_min_deg: float, pan_max_deg: float,
                      range_cells: int = 4):
        """標記扇形掃描區域（雲台左右掃描時使用）"""
        with self._lock:
            gx, gy = self._pos_to_grid()
            size = self.grid.shape[0]
            for r in range(1, range_cells + 1):
                for pan_deg in range(int(pan_min_deg), int(pan_max_deg) + 1, 10):
                    angle = self.heading + math.radians(pan_deg)
                    nx = gx + int(r * math.sin(angle))
                    ny = gy + int(r * math.cos(angle))
                    if 0 <= nx < size and 0 <= ny < size:
                        self.grid[nx, ny] += 1

    # ──────────────────────────────────────────────
    # 導航決策
    # ──────────────────────────────────────────────

    def get_preferred_turn_direction(self) -> int:
        """
        根據未掃描區域決定轉彎方向。
        回傳 +1 (右) 或 -1 (左)，朝向未探索區域較多的方向。
        """
        with self._lock:
            gx, gy = self._pos_to_grid()
            look_range = 5
            left_unexplored = 0
            right_unexplored = 0

            for r in range(1, look_range + 1):
                for offset_deg in range(10, 91, 10):
                    # 左側
                    angle_l = self.heading - math.radians(offset_deg)
                    lx = gx + int(r * math.sin(angle_l))
                    ly = gy + int(r * math.cos(angle_l))
                    if 0 <= lx < self.grid.shape[0] and 0 <= ly < self.grid.shape[1]:
                        if self.grid[lx, ly] == 0:
                            left_unexplored += 1

                    # 右側
                    angle_r = self.heading + math.radians(offset_deg)
                    rx = gx + int(r * math.sin(angle_r))
                    ry = gy + int(r * math.cos(angle_r))
                    if 0 <= rx < self.grid.shape[0] and 0 <= ry < self.grid.shape[1]:
                        if self.grid[rx, ry] == 0:
                            right_unexplored += 1

            if left_unexplored > right_unexplored:
                return -1
            elif right_unexplored > left_unexplored:
                return 1
            else:
                return 1  # 無明顯差異時預設右轉

    def is_ahead_fully_scanned(self, look_cells: int = 3) -> bool:
        """
        檢查正前方是否已充分掃描。
        若前方 70%+ 已掃描 → 建議主動轉向。
        """
        with self._lock:
            gx, gy = self._pos_to_grid()
            scanned = 0
            total = 0
            for r in range(1, look_cells + 1):
                nx = gx + int(r * math.sin(self.heading))
                ny = gy + int(r * math.cos(self.heading))
                if 0 <= nx < self.grid.shape[0] and 0 <= ny < self.grid.shape[1]:
                    total += 1
                    if self.grid[nx, ny] > 0:
                        scanned += 1
            if total == 0:
                return False
            return scanned / total > 0.7

    def get_best_exploration_angle(self) -> tuple:
        """
        找出未探索最多的方向（相對於當前航向）。
        強烈偏好「正前方」：只要前方有未探索格子，不轉彎。
        只在前方已全被探索時才找其他方向。

        回傳 (angle_deg, unexplored_count)。
        """
        with self._lock:
            gx, gy = self._pos_to_grid()
            size = self.grid.shape[0]

            def count_unexplored(angle_deg, radius=8):
                """計算該方向上 N 格內有幾格是未探索"""
                angle_rad = self.heading + math.radians(angle_deg)
                count = 0
                for r in range(1, radius + 1):
                    nx = gx + int(r * math.sin(angle_rad))
                    ny = gy + int(r * math.cos(angle_rad))
                    if 0 <= nx < size and 0 <= ny < size:
                        if self.grid[nx, ny] == 0:
                            count += 1
                return count

            # Step 1: 前方 ±20° 如果還有未探索格 → 直行，不轉
            forward_unexplored = max(
                count_unexplored(0),
                count_unexplored(-15),
                count_unexplored(15),
            )
            if forward_unexplored >= 3:
                return 0, forward_unexplored

            # Step 2: 前方沒什麼可探 → 找四周最佳方向（偏好小角度轉彎）
            best_angle = 0
            best_score = 0
            # 掃 -180~180，每 15° 一個候選；加上「角度懲罰」偏好直行
            for angle_deg in range(-180, 180, 15):
                unexplored = count_unexplored(angle_deg)
                # 小角度加分（-20°~20° 加 30%，超過 90° 大幅扣分）
                angle_penalty = 1.0 - (abs(angle_deg) / 180.0) * 0.5
                score = unexplored * angle_penalty
                if score > best_score:
                    best_score = score
                    best_angle = angle_deg

            return best_angle, int(best_score)

    def get_grid_position(self) -> tuple:
        """回傳當前格座標（供卡住偵測用）"""
        with self._lock:
            return self._pos_to_grid()

    # ──────────────────────────────────────────────
    # 狀態查詢
    # ──────────────────────────────────────────────

    def get_coverage_percent(self) -> float:
        """回傳已探索百分比"""
        with self._lock:
            total = self.grid.size
            explored = int(np.count_nonzero(self.grid))
            return round(explored / total * 100, 1)

    def get_grid_data(self) -> dict:
        """回傳完整地圖資料供 Web UI Canvas 繪圖"""
        with self._lock:
            gx, gy = self._pos_to_grid()
            # 已掃描區域的格座標（值 > 0 的格子）
            scanned = []
            size = self.grid.shape[0]
            # 只回傳中心附近 ±15 格的已掃描格（避免傳太多資料）
            origin = self._origin
            r = 15
            for x in range(max(0, gx - r), min(size, gx + r)):
                for y in range(max(0, gy - r), min(size, gy + r)):
                    if self.grid[x, y] > 0:
                        scanned.append((x, y))

            return {
                "coverage": self.get_coverage_percent_unlocked(),
                "robot_grid_x": int(gx),
                "robot_grid_y": int(gy),
                "heading_deg": round(math.degrees(self.heading) % 360, 1),
                "grid_size": int(self.grid.shape[0]),
                "path": list(self.path_history[-300:]),
                "obstacles": list(self.obstacles),
                "persons": list(self.persons),
                "scanned": scanned,  # 已掃描格座標（供 Canvas 繪製覆蓋區域）
            }

    def get_coverage_percent_unlocked(self) -> float:
        """無鎖版本（供 get_grid_data 等已持鎖函數呼叫）"""
        total = self.grid.size
        explored = int(np.count_nonzero(self.grid))
        return round(explored / total * 100, 1)

    def update_from_slam_pose(self, x_m: float, y_m: float, yaw_rad: float):
        """用 SLAM 校正位姿取代 dead reckoning（消除累積誤差）"""
        with self._lock:
            self.pos_x = x_m * 100  # m → cm
            self.pos_y = y_m * 100
            self.heading = yaw_rad
            self._record_position()

    def reset(self):
        """重置地圖"""
        with self._lock:
            self.grid.fill(0)
            self.pos_x = 0.0
            self.pos_y = 0.0
            self.heading = 0.0
            self.path_history.clear()
            self.obstacles.clear()
            self.persons.clear()
            self._record_position()
        logger.info("🗺️ 熱區地圖已重置")

    # ──────────────────────────────────────────────
    # 內部方法
    # ──────────────────────────────────────────────

    def _pos_to_grid(self):
        """將 cm 座標轉為格座標（呼叫前需持鎖）"""
        gx = self._origin + int(self.pos_x / self.cell_cm)
        gy = self._origin + int(self.pos_y / self.cell_cm)
        size = self.grid.shape[0]
        gx = max(0, min(gx, size - 1))
        gy = max(0, min(gy, size - 1))
        return gx, gy

    def _record_position(self):
        """記錄路徑（呼叫前需持鎖）"""
        gx, gy = self._pos_to_grid()
        if not self.path_history or self.path_history[-1] != (gx, gy):
            self.path_history.append((gx, gy))
            if len(self.path_history) > 500:
                self.path_history = self.path_history[-500:]
