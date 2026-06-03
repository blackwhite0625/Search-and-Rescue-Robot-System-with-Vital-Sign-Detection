"""
搜救機器人 — ROS 2 橋接模組
============================
封裝所有 ROS 2 通訊（/cmd_vel、/scan、/map、TF）。
rclpy 在專屬 daemon thread 中 spin，不影響 Flask。
若 ROS 2 不可用則自動降級，所有方法回傳安全預設值。
"""

import os
import math
import time
import logging
import threading

import numpy as np

import config

logger = logging.getLogger("rescue.ros_bridge")

# 確保 DDS 環境與 systemd 服務一致
os.environ.setdefault("RMW_IMPLEMENTATION", "rmw_cyclonedds_cpp")
os.environ.setdefault("ROS_AUTOMATIC_DISCOVERY_RANGE", "LOCALHOST")

# ============================================================
# 嘗試載入 ROS 2
# ============================================================
try:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
    from rclpy.duration import Duration
    from geometry_msgs.msg import Twist
    from sensor_msgs.msg import LaserScan
    from nav_msgs.msg import OccupancyGrid
    import tf2_ros
    RCLPY_OK = True
    logger.info("rclpy 載入成功")
except ImportError:
    RCLPY_OK = False
    logger.warning("rclpy 不可用，ROS 橋接將以模擬模式運行")


class RosBridge:
    """ROS 2 橋接：cmd_vel 發布 + /scan、/map 訂閱 + TF 查詢"""

    def __init__(self):
        self._available = False
        self._node = None
        self._spin_thread = None
        self._map_paused = False        # 暫停 map poll（TTS 播放時減少負載）

        # 資料緩衝（各自獨立 Lock）
        self._scan_lock = threading.Lock()
        self._scan_ranges = {}          # {angle_deg: dist_m}
        self._scan_stamp = 0.0

        self._map_lock = threading.Lock()
        self._map_data = None           # 可序列化 dict
        self._map_seq = 0
        self._map_last_sent_seq = -1

        self._pose_lock = threading.Lock()
        self._slam_pose = None          # (x_m, y_m, yaw_rad)

        # Pose history：記錄行走軌跡，定期降取樣
        self._pose_history = []         # list of (x_m, y_m)
        self._pose_history_max = 1000
        self._pose_history_min_dist = 0.05
        self._last_history_stamp = 0.0

        if not RCLPY_OK or not config.ROS_BRIDGE_ENABLED:
            return

        try:
            rclpy.init(args=None)
            self._node = rclpy.create_node("rescue_app_bridge")

            # --- Publisher: /cmd_vel ---
            self._cmd_vel_pub = self._node.create_publisher(Twist, "/cmd_vel", 10)

            # --- Subscriber: /scan ---
            scan_qos = QoSProfile(
                reliability=ReliabilityPolicy.BEST_EFFORT,
                durability=DurabilityPolicy.VOLATILE,
                history=HistoryPolicy.KEEP_LAST,
                depth=1,
            )
            self._node.create_subscription(LaserScan, "/scan", self._scan_cb, scan_qos)

            # --- Subscriber: /map（slam_toolbox 用 RELIABLE + TRANSIENT_LOCAL latched） ---
            self._map_qos = QoSProfile(
                reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.TRANSIENT_LOCAL,
                history=HistoryPolicy.KEEP_LAST,
                depth=1,
            )
            self._map_sub = self._node.create_subscription(
                OccupancyGrid, "/map", self._map_cb, self._map_qos
            )

            # --- TF Listener ---
            self._tf_buffer = tf2_ros.Buffer()
            self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self._node)

            # --- Spin thread ---
            self._spin_thread = threading.Thread(target=self._spin_loop, daemon=True)
            self._spin_thread.start()

            # 等待 spin thread 建立 + 確認連線
            time.sleep(0.5)
            self._available = True

            # --- Pose cache thread：30 Hz 背景查詢 TF（必須在 _available=True 後才啟動）---
            self._pose_cache_running = True
            self._pose_thread = threading.Thread(target=self._pose_cache_loop, daemon=True)
            self._pose_thread.start()

            logger.info("ROS 2 Bridge 啟動成功")

        except Exception as e:
            logger.error(f"ROS 2 Bridge 初始化失敗: {e}")
            self._available = False

    # ============================================================
    # 公開方法
    # ============================================================

    def is_available(self) -> bool:
        return self._available

    def publish_cmd_vel(self, x: float, y: float, r: float):
        """將 motor.move(x, y, r) 的參數轉為 Twist 發布"""
        if not self._available:
            return
        try:
            twist = Twist()
            twist.linear.x = float(y) * config.ROS_CMD_VEL_LINEAR_SCALE   # y=前後
            twist.linear.y = float(x) * config.ROS_CMD_VEL_LINEAR_SCALE   # x=橫移
            twist.angular.z = float(r) * config.ROS_CMD_VEL_ANGULAR_SCALE  # r=旋轉
            self._cmd_vel_pub.publish(twist)
        except Exception:
            pass

    def get_lidar_ranges(self) -> dict:
        """取得最新 LiDAR 掃描資料 {angle_deg: dist_m}"""
        if not self._available:
            return {}
        with self._scan_lock:
            return dict(self._scan_ranges)

    def get_lidar_obstacles(self, threshold_m: float = 0.30) -> list:
        """取得距離 < threshold_m 的障礙物列表 [(angle_deg, dist_m), ...]"""
        if not self._available:
            return []
        with self._scan_lock:
            return [
                (angle, dist)
                for angle, dist in self._scan_ranges.items()
                if 0 < dist < threshold_m
            ]

    def get_front_distance_cm(self, arc_deg: float = 90) -> float:
        """
        取得前方扇形區域最近障礙物距離 (cm)，已扣除 LiDAR 到車頭偏移。
        回傳的是「車頭到障礙物」的實際距離。
        回傳 -1 表示無資料（LiDAR 離線或前方無回波）。
        """
        if not self._available:
            return -1
        half = arc_deg / 2
        offset_cm = getattr(config, 'ROS_LIDAR_TO_FRONT_CM', 0)
        with self._scan_lock:
            if not self._scan_ranges:
                return -1
            front = [
                d for a, d in self._scan_ranges.items()
                if (-half <= a <= half) and d > 0
            ]
        if not front:
            return -1
        raw_cm = min(front) * 100
        corrected = raw_cm - offset_cm
        return round(max(corrected, 0), 1)

    def check_body_collision(self, margin_cm: float = None) -> dict:
        """
        全車身防撞檢查：把車當成包覆 LiDAR 的非對稱矩形 footprint，
        對所有 /scan 點計算「該方向的車邊到障礙物」距離。

        車頭朝 0°（前），180°/-180°（後）；左 = -90°，右 = +90°。

        回傳：
            {
                "collision": bool,           # 是否任何方向已撞到/即將撞到
                "min_clear_cm": float,       # 全車身最小淨空（負值=已穿過邊界）
                "min_angle_deg": float,      # 最危險角度
                "front_clear_cm", "rear_clear_cm",
                "left_clear_cm", "right_clear_cm",
            }
        """
        if not self._available:
            return {"collision": False, "min_clear_cm": -1, "min_angle_deg": 0,
                    "front_clear_cm": -1, "rear_clear_cm": -1,
                    "left_clear_cm": -1, "right_clear_cm": -1}

        if margin_cm is None:
            margin_cm = getattr(config, 'ROS_BODY_SAFETY_MARGIN_CM', 5.0)

        front = getattr(config, 'ROS_LIDAR_TO_FRONT_CM', 18.0)
        rear  = getattr(config, 'ROS_LIDAR_TO_REAR_CM',  3.5)
        side  = getattr(config, 'ROS_LIDAR_TO_SIDE_CM',  2.5)

        with self._scan_lock:
            ranges = list(self._scan_ranges.items())

        if not ranges:
            return {"collision": False, "min_clear_cm": -1, "min_angle_deg": 0,
                    "front_clear_cm": -1, "rear_clear_cm": -1,
                    "left_clear_cm": -1, "right_clear_cm": -1}

        min_clear = float('inf')
        min_angle = 0.0
        sector_min = {"front": float('inf'), "rear": float('inf'),
                      "left": float('inf'), "right": float('inf')}

        for ang_deg, d_m in ranges:
            if d_m <= 0:
                continue
            d_cm = d_m * 100.0
            rad = math.radians(ang_deg)
            cos_a = math.cos(rad)
            sin_a = math.sin(rad)

            # 從 LiDAR 中心沿 (cos_a, sin_a) 方向，車身邊緣距離
            # 車頭 +x、車尾 -x、右 +y、左 -y（以前進方向為 +x）
            # YDLidar X4Pro 安裝慣例：0° 朝前，正角度為左？— 用 abs 處理 left/right 對稱
            half_lengths = []
            if cos_a > 1e-6:
                half_lengths.append(front / cos_a)
            elif cos_a < -1e-6:
                half_lengths.append(rear / (-cos_a))
            if abs(sin_a) > 1e-6:
                half_lengths.append(side / abs(sin_a))
            if not half_lengths:
                continue
            edge_dist_cm = min(half_lengths)

            clear = d_cm - edge_dist_cm
            if clear < min_clear:
                min_clear = clear
                min_angle = ang_deg

            # 分區（粗略）
            if -45 <= ang_deg <= 45:
                sector_min["front"] = min(sector_min["front"], clear)
            elif ang_deg >= 135 or ang_deg <= -135:
                sector_min["rear"] = min(sector_min["rear"], clear)
            elif 45 < ang_deg < 135:
                sector_min["left"] = min(sector_min["left"], clear)
            else:
                sector_min["right"] = min(sector_min["right"], clear)

        def _norm(v):
            return round(v, 1) if v != float('inf') else -1

        return {
            "collision": min_clear < margin_cm,
            "min_clear_cm": _norm(min_clear),
            "min_angle_deg": round(min_angle, 1),
            "front_clear_cm": _norm(sector_min["front"]),
            "rear_clear_cm":  _norm(sector_min["rear"]),
            "left_clear_cm":  _norm(sector_min["left"]),
            "right_clear_cm": _norm(sector_min["right"]),
        }

    def get_scan_age(self) -> float:
        """回傳最近一次 /scan 的年齡 (秒)。用於 watchdog。"""
        with self._scan_lock:
            if self._scan_stamp == 0:
                return float('inf')
            return time.time() - self._scan_stamp

    def is_lidar_alive(self, timeout_sec: float = 2.0) -> bool:
        """LiDAR watchdog：資料是否在 timeout_sec 內更新過。"""
        return self._available and self.get_scan_age() < timeout_sec

    def get_slam_map(self) -> dict:
        """取得 SLAM OccupancyGrid（可序列化 dict），地圖未更新時回傳 None"""
        if not self._available or not config.ROS_SLAM_MAP_ENABLED:
            return None
        with self._map_lock:
            if self._map_data is None:
                return None
            if self._map_seq == self._map_last_sent_seq:
                return None  # 沒有更新，前端可跳過重繪
            self._map_last_sent_seq = self._map_seq
            return dict(self._map_data)

    def get_frontier_points(self, max_points: int = 60, robot_pose=None,
                            min_cluster: int = None) -> list:
        """從 SLAM /map 找 frontier cells（free 相鄰 unknown），做 flood-fill 聚類，
        只回傳群組大小 ≥ min_cluster 的 cluster 中心。
        min_cluster=None 時依地圖解析度動態計算（至少對應
        ``config.PATROL_FRONTIER_MIN_CLUSTER_M`` 物理長度，下限 10 cells），
        避免 5cm/cell 下 3 cells 的雜訊 frontier 干擾決策。
        回傳：[(x_m, y_m, cluster_size), ...]，按距離機器人升序。"""
        if not self._available:
            return []
        with self._map_lock:
            md = self._map_data
            if md is None:
                return []
            walls = md.get('walls', [])
            free = md.get('free', [])
            res = md.get('resolution', 0.05)
            ox = md.get('origin_x', 0.0)
            oy = md.get('origin_y', 0.0)
            gw = md.get('width', 0)
            gh = md.get('height', 0)

        if not free or gw == 0:
            return []

        if min_cluster is None:
            min_m = float(getattr(config, 'PATROL_FRONTIER_MIN_CLUSTER_M', 0.5))
            min_cluster = max(10, int(min_m / max(res, 1e-6)))

        free_set = set((c[0], c[1]) for c in free)
        wall_set = set((c[0], c[1]) for c in walls)

        # Step 1: 找所有 frontier cells（free 且相鄰 unknown）
        frontier_set = set()
        for fx, fy in free_set:
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nx, ny = fx + dx, fy + dy
                if 0 <= nx < gw and 0 <= ny < gh:
                    if (nx, ny) not in free_set and (nx, ny) not in wall_set:
                        frontier_set.add((fx, fy))
                        break

        if not frontier_set:
            return []

        # Step 2: Flood-fill 成 clusters（8 鄰域）
        seen = set()
        clusters = []
        neigh8 = ((1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1))
        for start in frontier_set:
            if start in seen:
                continue
            stack = [start]
            group = []
            while stack:
                p = stack.pop()
                if p in seen:
                    continue
                seen.add(p)
                group.append(p)
                for dx, dy in neigh8:
                    n = (p[0] + dx, p[1] + dy)
                    if n in frontier_set and n not in seen:
                        stack.append(n)
            if len(group) >= min_cluster:
                cx_g = sum(p[0] for p in group) / len(group)
                cy_g = sum(p[1] for p in group) / len(group)
                wx = round(ox + cx_g * res, 3)
                wy = round(oy + cy_g * res, 3)
                clusters.append((wx, wy, len(group)))

        if not clusters:
            return []

        # Step 3: 按距離機器人升序
        if robot_pose:
            rx, ry = robot_pose[0], robot_pose[1]
        else:
            with self._pose_lock:
                if self._slam_pose:
                    rx, ry = self._slam_pose[0], self._slam_pose[1]
                else:
                    rx, ry = 0.0, 0.0
        clusters.sort(key=lambda c: (c[0] - rx) ** 2 + (c[1] - ry) ** 2)
        return clusters[:max_points]

    def line_of_sight(self, x1: float, y1: float, x2: float, y2: float) -> bool:
        """Bresenham 檢查世界座標兩點間路徑是否穿越牆壁 cell。
        True = 通暢；False = 阻擋；地圖無資料時 True（不阻擋決策）。"""
        if not self._available:
            return True
        with self._map_lock:
            md = self._map_data
            if md is None:
                return True
            walls = md.get('walls', [])
            res = md.get('resolution', 0.05)
            ox = md.get('origin_x', 0.0)
            oy = md.get('origin_y', 0.0)
            gw = md.get('width', 0)
            gh = md.get('height', 0)
        if not walls or gw == 0:
            return True
        wall_set = set((w[0], w[1]) for w in walls)

        gx1 = int((x1 - ox) / res)
        gy1 = int((y1 - oy) / res)
        gx2 = int((x2 - ox) / res)
        gy2 = int((y2 - oy) / res)
        dx = abs(gx2 - gx1)
        dy = abs(gy2 - gy1)
        sx = 1 if gx1 < gx2 else -1
        sy = 1 if gy1 < gy2 else -1
        err = dx - dy
        steps = 0
        max_steps = dx + dy + 2
        while steps < max_steps:
            if (gx1, gy1) in wall_set:
                return False
            if gx1 == gx2 and gy1 == gy2:
                return True
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                gx1 += sx
            if e2 < dx:
                err += dx
                gy1 += sy
            steps += 1
        return True

    def is_cell_free(self, x_m: float, y_m: float, radius_m: float = 0.15) -> bool:
        """檢查世界座標 (x_m, y_m) 以 radius_m 為半徑的方形區域是否為 SLAM free space
        （所有格子都在 free_set 內，且無任何 wall）。
        地圖未就緒時回傳 True（不阻擋決策）。"""
        if not self._available:
            return True
        with self._map_lock:
            md = self._map_data
            if md is None:
                return True
            walls = md.get('walls', [])
            free = md.get('free', [])
            res = md.get('resolution', 0.05)
            ox = md.get('origin_x', 0.0)
            oy = md.get('origin_y', 0.0)
            gw = md.get('width', 0)
            gh = md.get('height', 0)
        if gw == 0 or not free:
            return True
        free_set = set((c[0], c[1]) for c in free)
        wall_set = set((w[0], w[1]) for w in walls)
        gx = int((x_m - ox) / res)
        gy = int((y_m - oy) / res)
        r = max(0, int(radius_m / max(res, 1e-6)))
        for dx in range(-r, r + 1):
            for dy in range(-r, r + 1):
                cx, cy = gx + dx, gy + dy
                if cx < 0 or cy < 0 or cx >= gw or cy >= gh:
                    return False
                if (cx, cy) in wall_set:
                    return False
                if (cx, cy) not in free_set:
                    return False
        return True

    def get_slam_pose(self) -> tuple:
        """取得 SLAM 校正位姿 (x_m, y_m, yaw_rad)。從 background cache 讀取，非 blocking。"""
        if not self._available:
            return None
        with self._pose_lock:
            return self._slam_pose

    def get_pose_history(self) -> list:
        """取得行走軌跡 [(x, y), ...]，單位：公尺"""
        if not self._available:
            return []
        with self._pose_lock:
            return list(self._pose_history)

    def get_lidar_world_points(self, max_points: int = 360) -> list:
        """取得**當前 scan** 轉成世界座標的有序點列 [(x_m, y_m), ...]。
        按角度升序排列，前端可以直接畫 polyline 形成牆壁線條。
        只回傳最新一幀，不累積歷史——乾淨、即時。"""
        if not self._available:
            return []
        with self._pose_lock:
            pose = self._slam_pose
        if pose is None:
            return []
        cx, cy, yaw = pose
        with self._scan_lock:
            items = list(self._scan_ranges.items())
        if not items:
            return []
        # 按角度排序（LiDAR scan 是有角度順序的）
        items.sort(key=lambda kv: kv[0])
        # 降取樣
        step = max(1, len(items) // max_points)
        points = []
        for i in range(0, len(items), step):
            angle_deg, dist_m = items[i]
            if dist_m <= 0.15 or dist_m > 6.0:
                # 無效點：插入 None 作為「線段中斷」訊號
                points.append(None)
                continue
            theta = yaw + math.radians(angle_deg)
            wx = cx + dist_m * math.cos(theta)
            wy = cy + dist_m * math.sin(theta)
            points.append((round(wx, 3), round(wy, 3)))
        return points

    def reset_pose_history(self):
        """清空軌跡"""
        with self._pose_lock:
            self._pose_history = []

    def refresh_slam_map(self) -> bool:
        """強制刷新 SLAM 地圖：destroy + recreate /map subscription，
        slam_toolbox 在新 subscriber 連上時會重發 TRANSIENT_LOCAL latched 訊息。"""
        if not self._available or not self._node:
            return False
        try:
            old_sub = self._map_sub
            self._map_sub = self._node.create_subscription(
                OccupancyGrid, "/map", self._map_cb, self._map_qos
            )
            # 舊 subscription 還在 spin，給它一點時間收最後一次訊息，然後 destroy
            def _cleanup():
                time.sleep(1.0)
                try:
                    self._node.destroy_subscription(old_sub)
                except Exception:
                    pass
            threading.Thread(target=_cleanup, daemon=True).start()
            logger.info("📡 /map 訂閱已刷新")
            return True
        except Exception as e:
            logger.warning(f"refresh_slam_map 失敗: {e}")
            return False

    def shutdown(self):
        """關閉 ROS 2 節點"""
        if self._node:
            try:
                self._node.destroy_node()
            except Exception:
                pass
        try:
            rclpy.shutdown()
        except Exception:
            pass
        self._available = False
        logger.info("ROS 2 Bridge 已關閉")

    # ============================================================
    # 內部回呼
    # ============================================================

    def _spin_loop(self):
        """在 daemon thread 中 spin"""
        try:
            rclpy.spin(self._node)
        except Exception:
            pass

    def pause_map_poll(self):
        """TTS 播放時暫停 map poll 減少 CPU/USB 負載"""
        self._map_paused = True

    def resume_map_poll(self):
        """TTS 結束後恢復 map poll"""
        self._map_paused = False

    def _pose_cache_loop(self):
        """50 Hz 背景查詢 TF map→base_link，同時記錄行走軌跡。
        加 fallback：若 map→base_link 失敗，試 map→odom（假裝 robot 在 odom 原點）。
        加診斷 log：首次成功、首次失敗、每 5 秒失敗一次。"""
        # V11：30 → 50 Hz，降低 SLAM pose 讀取延遲（配合更頻繁的閉環旋轉校正）
        period = 1.0 / 50.0
        first_success = False
        first_fail_logged = False
        last_fail_log = 0.0
        last_error = None
        grace_until = time.time() + 3.0   # 開機 3 秒內不印 log，讓 TF 穩定

        def _parse_tf(t):
            x = t.transform.translation.x
            y = t.transform.translation.y
            q = t.transform.rotation
            siny = 2.0 * (q.w * q.z + q.x * q.y)
            cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
            return x, y, math.atan2(siny, cosy)

        # 100ms timeout 讓 buffer 有時間等最新 TF（否則在 30Hz poll 下容易撲空）
        tf_timeout = Duration(seconds=0.1)

        while self._pose_cache_running and self._available:
            pose = None
            err_msg = None
            # 主要：map → base_link
            try:
                t = self._tf_buffer.lookup_transform(
                    "map", "base_link", rclpy.time.Time(), tf_timeout
                )
                pose = _parse_tf(t)
            except Exception as e:
                err_msg = f"map→base_link: {type(e).__name__}: {e}"
                # Fallback：map → odom（假設 odom 就是機器人位置）
                try:
                    t = self._tf_buffer.lookup_transform(
                        "map", "odom", rclpy.time.Time(), tf_timeout
                    )
                    pose = _parse_tf(t)
                    err_msg += " | fallback map→odom OK"
                except Exception as e2:
                    err_msg += f" | map→odom: {type(e2).__name__}: {e2}"
                    # 再 fallback：odom → base_link
                    try:
                        t = self._tf_buffer.lookup_transform(
                            "odom", "base_link", rclpy.time.Time(), tf_timeout
                        )
                        pose = _parse_tf(t)
                        err_msg += " | fallback odom→base_link OK"
                    except Exception as e3:
                        err_msg += f" | odom→base_link: {type(e3).__name__}: {e3}"

            if pose is not None:
                if not first_success:
                    logger.info(f"✅ Pose TF 查詢成功: x={pose[0]:.2f} y={pose[1]:.2f} yaw={pose[2]:.2f}")
                    first_success = True
                x, y, yaw = pose
                with self._pose_lock:
                    self._slam_pose = (x, y, yaw)
                    # 記錄軌跡：距離上一點 > 5cm 才添加（避免靜止時塞爆）
                    if not self._pose_history:
                        self._pose_history.append((round(x, 3), round(y, 3)))
                    else:
                        lx, ly = self._pose_history[-1]
                        dx = x - lx
                        dy = y - ly
                        if (dx * dx + dy * dy) >= (self._pose_history_min_dist ** 2):
                            self._pose_history.append((round(x, 3), round(y, 3)))
                            if len(self._pose_history) > self._pose_history_max:
                                self._pose_history = (
                                    self._pose_history[:1]
                                    + self._pose_history[1::2]
                                )
            else:
                # 失敗診斷 log
                now_t = time.time()
                if now_t > grace_until:
                    if not first_fail_logged or (now_t - last_fail_log) > 5.0:
                        logger.warning(f"⚠️ Pose TF 查詢全部失敗: {err_msg}")
                        first_fail_logged = True
                        last_fail_log = now_t

            time.sleep(period)

    def _scan_cb(self, msg: "LaserScan"):
        """處理 /scan 訊息 → 轉為 {angle_deg: dist_m}"""
        ranges = {}
        angle = msg.angle_min
        for r in msg.ranges:
            angle_deg = round(math.degrees(angle), 1)
            if msg.range_min < r < msg.range_max:
                ranges[angle_deg] = round(r, 3)
            angle += msg.angle_increment
        now = time.time()
        with self._scan_lock:
            self._scan_ranges = ranges
            self._scan_stamp = now
        # 注意：不再累積到世界地圖，因前端改為只畫當前 scan polyline（乾淨）

    def _map_cb(self, msg: "OccupancyGrid"):
        """處理 /map 訊息 → 轉為可序列化 dict（numpy 加速）"""
        info = msg.info
        width = info.width
        height = info.height
        resolution = info.resolution
        origin_x = info.origin.position.x
        origin_y = info.origin.position.y

        # numpy 向量化：比 Python for-loop 快 50-100 倍
        data = np.array(msg.data, dtype=np.int8).reshape(height, width)

        wall_ys, wall_xs = np.where(data == 100)
        walls = np.column_stack([wall_xs, wall_ys]).tolist() if len(wall_xs) > 0 else []

        free_ys, free_xs = np.where(data == 0)
        free_cells = np.column_stack([free_xs, free_ys]).tolist() if len(free_xs) > 0 else []

        # 降取樣過多的 free cells（減少 JSON 傳輸量）
        if len(free_cells) > 5000:
            free_cells = free_cells[::2]

        map_dict = {
            "width": width,
            "height": height,
            "resolution": resolution,
            "origin_x": round(origin_x, 3),
            "origin_y": round(origin_y, 3),
            "walls": walls,
            "free": free_cells,
            "seq": self._map_seq + 1,
            "timestamp": time.time(),
        }

        with self._map_lock:
            self._map_data = map_dict
            self._map_seq += 1
