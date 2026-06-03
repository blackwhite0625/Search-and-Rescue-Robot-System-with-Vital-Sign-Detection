"""
搜救機器人系統 — 集中設定檔
所有 GPIO 腳位、模型路徑、閾值、搜救參數集中管理
"""

import os

# ============================================================
# 路徑設定
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")
SOUND_DIR = os.path.join(BASE_DIR, "sounds")
EVENT_DIR = os.path.join(BASE_DIR, "events")
VICTIM_MEMORY_FILE = os.path.join(EVENT_DIR, "reported_victims.json")

# ============================================================
# AI 模型路徑
# ============================================================

# --- ONNX 模型（預設使用）---
YOLO_GENERAL_MODEL = "model/yolov8n.onnx"       # 通用物件偵測 (人體)
YOLO_POSE_MODEL    = "model/yolov8n-pose.onnx"   # 姿態偵測

# 各模型推理尺寸
YOLO_GENERAL_INFER_SIZE = 320
YOLO_POSE_INFER_SIZE    = 320

# ============================================================
# AI 偵測參數
# ============================================================
GENERAL_CONFIDENCE = 0.60     # yolov8n 人體偵測置信度閾值（提高減少誤偵測）
POSE_CONFIDENCE    = 0.55     # 姿態偵測信心度閾值（提高減少物體誤判為人）
PERSON_CLASS_ID    = 0        # 'person' 在 COCO 中的 ID
AI_DETECT_EVERY_N_FRAMES = 3  # 每 N 個攝影機 loop 做一次 YOLO；中間幀用快取 overlay
AI_DISPLAY_OVERLAY_TTL_SEC = 0.45  # 快取 YOLO overlay 最多保留秒數，避免舊框殘留

# ============================================================
# GPIO 腳位配置 — 馬達驅動 (L298N x 2 塊)
# ============================================================
# 前輪 L298N: 控制左前 (FL) 與 右前 (FR)
L298N_FRONT = {
    "ena": 12, "in1": 6, "in2": 5,    # 左前輪 (FL)
    "enb": 26, "in3": 19, "in4": 13   # 右前輪 (FR)
}

# 後輪 L298N: 控制左後 (RL) 與 右後 (RR)
L298N_REAR = {
    "ena": 16, "in1": 27, "in2": 17,  # 左後輪 (RL)
    "enb": 20, "in3": 23, "in4": 22   # 右後輪 (RR)
}

# 馬達預設速度 (0.0 ~ 1.0)
DEFAULT_SPEED = 0.6

# 手動蘑菇頭控制：低延遲 + 後端失聯保護
MANUAL_CONTROL_TIMEOUT_SEC = 0.35
MANUAL_COMMAND_DEADZONE    = 0.03
MANUAL_ACCEL_RATE          = 8.0

# ============================================================
# GPIO 腳位配置 — 舵機 (攝影機雲台)
# ============================================================
SERVO_PAN_PIN  = 18     # 水平旋轉（BCM 18 = 實體 pin 12，硬體 PWM 通道 0）
SERVO_TILT_PIN = 25     # 垂直傾斜（軟體 PWM，tilt 很少動不明顯）
# 硬體 PWM 參數（RPi 5 上 dtoverlay=pwm,pin=18,func=2 建立的 pwmchip）
SERVO_PAN_HW_PWM_CHIP    = 2     # 通常是 2；若 ls /sys/class/pwm/ 顯示別的就改
SERVO_PAN_HW_PWM_CHANNEL = 0     # 單通道 overlay 為 0

# 舵機角度範圍
SERVO_MIN_ANGLE = -90
SERVO_MAX_ANGLE = 90

# 舵機脈衝寬度 (秒) — 解放完整 180° 物理極限
SERVO_MIN_PULSE = 0.0005
SERVO_MAX_PULSE = 0.0025

# 舵機預設角度
SERVO_PAN_DEFAULT  = -7.9    # 水平預設偏移
SERVO_TILT_DEFAULT = 45.0    # 垂直預設仰角

# 巡航掃描參數
SCAN_STEP  = 5       # 每次步進角度
SCAN_DELAY = 0.15    # 每步間隔 (秒)

# ============================================================
# LiDAR 安全距離 (公分) — 由 ROS 2 /scan 提供
# ============================================================
SAFE_DISTANCE_CM     = 50   # 行人優先停止距離
OBSTACLE_DISTANCE_CM = 30   # 障礙物緊急停止

# ============================================================
# 攝影機設定
# ============================================================
CAMERA_INDEX  = 1
CAMERA_WIDTH  = 640           # 降低解析度釋放 USB 頻寬，避免播音時攝影機被踢掉
CAMERA_HEIGHT = 480
CAMERA_FPS    = 30
CAMERA_DISPLAY_MAX_AGE_SEC = 0.90  # 標註畫面短暫保留，避免 YOLO 慢時骨架消失

# ============================================================
# 搜救任務狀態機
# ============================================================
# 7 階段：STANDBY / SEARCH / ANOMALY / LOCK_ON / INQUIRY / CONFIRM / REPORT
# 額外：MANUAL（手動操控）

VICTIM_SUSPECT_THRESHOLD   = 0.30   # VictimScore ≥ 此值 → ANOMALY
VICTIM_HIGH_RISK_THRESHOLD = 0.60   # VictimScore ≥ 此值 → INQUIRY/REPORT

# ============================================================
# 多模態融合權重 (VictimScore)
# ============================================================
VICTIM_SCORE_WEIGHTS = {
    "person":      0.40,
    "pose":        0.18,
    "audio":       0.18,
    "motion":      0.09,
    "distance":    0.05,
    "vital_signs": 0.10,
}

# ============================================================
# 音訊設定 (V2 啟用)
# ============================================================
MIC_DEVICE_INDEX = None    # None = 自動偵測 USB 麥克風
MIC_SAMPLE_RATE  = 48000   # 取樣率 (Hz)
MIC_CHUNK_SIZE   = 4096    # 每次讀取的樣本數
MIC_BUFFER_SEC   = 5.0     # 環形緩衝區長度 (秒)

VAD_THRESHOLD    = 0.10    # Voice Activity Detection 閾值（再降，提高靈敏度）
HELP_THRESHOLD   = 0.4     # 呼救聲分類閾值（降低，減少漏報）
KNOCK_THRESHOLD  = 0.08    # 敲擊聲偵測閾值（降低，提高靈敏度）

# ============================================================
# 搜索策略
# ============================================================
SEARCH_MODE = "F"                  # 唯一模式：SLAM 反應式掃描巡邏（D/E 已於 2026-04 淘汰）

# ============================================================
# 智慧巡邏 (模式 E) — 走走停停掃描
# ============================================================
SMART_PATROL_SPEED       = 0.40   # 前進速度 (0~1)
SMART_PATROL_MOVE_SEC    = 4.0    # 每段前進持續秒數
SMART_PATROL_OBSTACLE    = 25     # cm, 前方障礙停車距離（車頭實際距離）
SMART_PATROL_REVERSE_SEC = 0.8    # 避障後退時間 (秒)
SMART_PATROL_TURN_SEC    = 1.0    # 避障轉彎時間 (秒)
STRAFE_SPEED             = 0.25   # 麥克拉姆側移速度
STRAFE_DURATION          = 0.5    # 側移持續時間 (秒)

# ============================================================
# 警報與事件回報
# ============================================================
ALERT_COOLDOWN = 30            # 事件回報後最短停留時間（秒），確保警報播完再進入下一階段

# ============================================================
# 傷患靠近 / 已通報記憶
# ============================================================
# LOCK_ON 時使用比一般障礙物更保守的距離，避免車體壓到倒地者。
VICTIM_APPROACH_STOP_CM       = 55    # 車頭距離 <= 此值即停止並開始 HRI
VICTIM_APPROACH_REVERSE_CM    = 38    # 比此距離更近時先後退
VICTIM_APPROACH_SLOW_CM       = 110   # 進入此距離內降速靠近
VICTIM_VISUAL_STOP_Y_RATIO    = 0.72  # 倒地 bbox 底邊接近畫面下緣即停
VICTIM_VISUAL_STOP_AREA       = 0.18  # 倒地 bbox 佔畫面比例過大即停
VICTIM_VISUAL_SLOW_Y_RATIO    = 0.60
VICTIM_VISUAL_SLOW_AREA       = 0.10

# 已通報傷患以 SLAM 世界座標記憶；半徑內視為同一名傷患，避免重複通報。
VICTIM_MEMORY_MERGE_RADIUS_M  = 0.85
VICTIM_MEMORY_NEAR_RADIUS_M   = 1.10
VICTIM_MEMORY_NO_POSE_MERGE_SEC = 90  # 無 SLAM/track 時，短時間內視為同一名傷患
VICTIM_DEPART_REVERSE_SEC     = 1.4   # REPORT 後倒車離場
VICTIM_DEPART_TURN_SEC        = 0.9   # 離場後轉向回到巡邏
VICTIM_DEPART_REVERSE_PWM     = 0.30
VICTIM_DEPART_TURN_PWM        = 0.32
VICTIM_DEPART_REAR_MIN_CM     = 12
# 警報語音播放輪數。每輪須暫停相機讓出 USB 頻寬,輪數越少相機停擺越短、YOLO 中斷越少。
# 設為 1:只播一次,相機停擺壓到最低(~6s),YOLO 中斷時間最短。
ALERT_TTS_ROUNDS = 1
ALERT_SOUND = os.path.join(BASE_DIR, "參考資料/警報聲參考.wav")  # 已停用，保留路徑供 import 相容

# Telegram Bot（只從環境變數讀取，避免 Token 暴露於原始碼與 GitHub）
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID   = os.environ.get("TELEGRAM_CHAT_ID", "")

# ============================================================
# Flask 伺服器
# ============================================================
FLASK_HOST = "0.0.0.0"
FLASK_PORT = 5001

# ============================================================
# 自動巡檢 / 搜索邏輯
# ============================================================
PATROL_SPEED              = 0.3    # 前進速度 (0~1)
PATROL_TURN_DISTANCE      = 40     # 避障警戒距離 (cm)
PATROL_EMERGENCY_DISTANCE = 15     # 緊急停止距離 (cm)
PATROL_TURN_SPEED         = 0.35   # 轉彎速度（降低避免原地轉圈）
PATROL_TURN_DURATION      = 0.5    # 轉彎持續時間 (秒)（從1.2縮小）
PATROL_REVERSE_SPEED      = 0.35   # 後退速度
PATROL_REVERSE_DURATION   = 0.4    # 一般警戒後退時間 (秒)
PATROL_EMERGENCY_REVERSE  = 0.7    # 緊急距離後退時間 (秒)

PATROL_PAN_MIN     = -60    # 掃描最小角度
PATROL_PAN_MAX     = 60     # 掃描最大角度
PATROL_PAN_STEP    = 2      # 每步 2°
PATROL_SWEEP_DELAY = 0.04   # 掃描間隔 40ms
PATROL_SCAN_PAUSE  = 1.5    # 停車掃描時間 (秒)
PATROL_STUCK_THRESHOLD = 5
PATROL_MOVE_DURATION   = 2.5

# ── SLAM 驅動掃描巡邏（模式 F / E 共用） ──
# 所有行進/轉彎決策以 SLAM /map + TF(map→base_link) 為權威來源。
# 巡邏只使用前進、後退、原地轉彎，禁用麥克納姆側移（減少 odom 誤差、配合 diff-drive SLAM）。
PATROL_YAW_TOL_DEG            = 3.0    # 閉環旋轉停止容差（度）
PATROL_ROTATE_TIMEOUT_SEC     = 4.0    # 單次閉環旋轉最長時間
PATROL_VISITED_CELL_M         = 0.30   # 世界座標 visited 格子邊長 (m)（0.25→0.30 減少無謂切分）
PATROL_VISITED_SATURATE       = 2      # 訪問次數 ≥ 此值視為完全探索
PATROL_RECENT_VISIT_SEC       = 60.0   # frontier 評分：距離最近訪問若 < 此秒數則重罰
PATROL_RECENT_VISIT_PENALTY   = 2.5    # 近期訪問 frontier 的扣分幅度
PATROL_UNREACH_CELL_M         = 0.30   # 不可達 frontier 記憶格子邊長 (m)
PATROL_BLOCKED_HIT_LIMIT      = 2      # 前方連續被阻檔幾次即標為不可達
PATROL_FRONTIER_MIN_CLUSTER_M = 0.5    # frontier cluster 最小物理長度 (m)
PATROL_ALLOW_STRAFE           = False  # 巡邏時是否允許橫向側移（False = 禁用平移）


# ── 硬體安全限制（L298N + Mecanum，SLAM 友善慢速版）──
# 這些參數針對「平穩移動 + SLAM 精準掃描配對」設計。
# 值以 PWM 佔空比 (0-1) 表達，不是 m/s。
SMOOTH_MAX_LINEAR       = 0.26   # 最大線速度 PWM
SMOOTH_MIN_LINEAR       = 0.23   # 低於此值會 stall
SMOOTH_MAX_ANGULAR      = 0.30   # 最大角速度 PWM
SMOOTH_MIN_ANGULAR      = 0.23
SMOOTH_ACCEL_SEC        = 0.35   # 加速/減速 ramp 時間
SMOOTH_CONTROL_HZ       = 20     # 控制迴圈頻率
SMOOTH_POST_MOVE_DELAY  = 0.5    # 移動後停頓時間（讓 SLAM 穩定）
SMOOTH_ROTATE_TOL_DEG   = 6      # 旋轉目標容許誤差
SMOOTH_DRIVE_TOL_M      = 0.05   # 直行目標容許誤差
SMOOTH_DRIVE_DECEL_M    = 0.20   # 距離目標 m 時開始減速
SMOOTH_ROTATE_DECEL_DEG = 25     # 剩餘度數 < 此值時開始減速

# ============================================================
# 熱區記憶地圖 (Heat Map)
# ============================================================
HEAT_MAP_ENABLED          = True
HEAT_MAP_GRID_SIZE        = 40     # 格狀地圖大小 (40×40)
HEAT_MAP_CELL_CM          = 30     # 每格代表的實際距離 (cm)
HEAT_MAP_SCAN_RADIUS      = 2      # 掃描標記半徑 (格)
HEAT_MAP_CM_PER_SPEED_SEC = 30     # 速度 1.0 每秒移動距離 (cm) [需校準]
HEAT_MAP_RAD_PER_SPEED_SEC = 2.0   # 速度 1.0 每秒旋轉角度 (rad) [需校準]

# ============================================================
# ROS 2 Bridge 設定
# ============================================================
ROS_BRIDGE_ENABLED             = True
ROS_CMD_VEL_LINEAR_SCALE       = 0.30   # motor speed 1.0 = 0.30 m/s
ROS_CMD_VEL_ANGULAR_SCALE      = 2.0    # motor rotation 1.0 = 2.0 rad/s
ROS_LIDAR_OBSTACLE_THRESHOLD_M = 0.30   # LiDAR 避障觸發距離 (m)
ROS_LIDAR_WARNING_THRESHOLD_M  = 0.50   # LiDAR 警告距離 (m)
ROS_LIDAR_BRAKE_THRESHOLD_M    = 0.20   # LiDAR 前方緊急煞車距離 (m)（車頭實際距離）
ROS_LIDAR_FRONT_ARC_DEG        = 90     # 前方安全弧度 (±45°)
ROS_LIDAR_WATCHDOG_SEC         = 2.0    # LiDAR 斷線 watchdog 超時 (秒)
ROS_SLAM_MAP_ENABLED           = True   # 網頁顯示 SLAM 地圖

# ── 車體幾何：LiDAR 到各邊距離（cm） ──
# 用於 360° 全身防撞：把車體當成包覆 LiDAR 的非對稱 footprint
# 依照 參考資料/照片/樹莓派車.png 俯視圖實測
# 依 樹莓派車.png 實測：LiDAR 直徑 7cm，最外緣→車頭/車尾 15cm、最外緣→側緣 7cm
# 中心距離 = 最外緣距離 + LiDAR 半徑 (3.5cm)
ROS_LIDAR_TO_FRONT_CM = 18.5  # LiDAR 中心 → 車頭 (15 + 3.5)
ROS_LIDAR_TO_REAR_CM  = 18.5  # LiDAR 中心 → 車尾（對稱；舊值 5 嚴重低估後方防撞）
ROS_LIDAR_TO_SIDE_CM  = 10.5  # LiDAR 中心 → 車體側緣 (7 + 3.5)
# 安全餘裕：障礙物距離 < (車身邊緣 + margin) 視為碰撞
ROS_BODY_SAFETY_MARGIN_CM = 8.0

# ── 巡邏安全餘裕（以車體邊緣為基準，cm） ──
# scan_patrol 於 __init__ 將這些 body 距離轉為 raw LiDAR 距離
PATROL_FRONT_MIN_BODY_CM    = 15   # 前方硬煞（10→15 提高椅腳/桌腳餘裕）
PATROL_FRONT_BRAKE_BODY_CM  = 50   # 前方進入減速區起點（40→50 更早減速）
PATROL_FRONT_CRUISE_BODY_CM = 90   # 前方全速巡航門檻（80→90 在空曠才拉滿）
PATROL_FRONT_DEAD_END_BODY_CM = 5  # 對準但仍阻塞判定為死胡同的餘裕
PATROL_MIN_GAP_BODY_CM      = 12   # 可選 bucket 所需最小車體餘裕（8→12 拒絕太窄的椅腳縫隙）
PATROL_SIDE_MIN_BODY_CM     = 10   # 側邊硬煞（8→10 預留椅輪餘裕）
PATROL_SIDE_CAUTION_BODY_CM = 25   # 側邊減速門檻（20→25）

# ── 擁擠空間偵測（椅子下 / 桌下 / 家具群 等 LiDAR 多反射情境）──
# 當前方半圓弧有 >= TIGHT_BUCKET_COUNT 個 bucket 的 body 餘裕 < TIGHT_BODY_CM 時，
# 判定為擁擠空間，速度降為原本的 TIGHT_SPACE_MULT 倍
PATROL_TIGHT_BUCKET_COUNT   = 4
PATROL_TIGHT_BODY_CM        = 40
PATROL_TIGHT_SPACE_MULT     = 0.45

# ── 巡邏自適應速度 (PWM, motor.MAX_SPEED=0.55 為上限) ──
PATROL_CRUISE_FAST_PWM   = 0.50   # 空曠全速 ≈ 15.0 cm/s
PATROL_CRUISE_NORMAL_PWM = 0.38   # 一般巡航 ≈ 11.4 cm/s
PATROL_MIN_FWD_PWM       = 0.22   # 最低起動 PWM ≈ 6.6 cm/s
PATROL_TURN_PWM          = 0.36   # 旋轉 PWM
PATROL_MIN_TURN_PWM      = 0.28   # 最小旋轉 PWM
PATROL_SIDE_CAUTION_MULT = 0.6    # 側邊 CAUTION 時速度倍率
PATROL_BUCKET_HYSTERESIS = 1.15   # 上一 cycle bucket 分數加權 (防震盪)

# ── 低障礙（椅腳/桌腳）記憶 ──
# LiDAR 7Hz + 1° 取樣對細長物體會間歇性遺漏，此記憶保留最近一次「近距離」讀值
# 短時間內，避免下一 cycle 誤判為安全而撞上
PATROL_BUCKET_MEMORY_TTL_SEC   = 0.8   # 近距讀值記憶長度（0.4→0.8 椅腳/桌腳更可靠）
PATROL_BUCKET_MEMORY_THRESH_CM = 100   # 記憶 raw < 100cm 的讀值（60→100 擴大覆蓋）

# ── SLAM 掃描暫停（提升地圖完整度）──
# 靜止時 slam_toolbox 的 scan matching 精準度顯著高於移動中，
# 定期短暫停車讓 SLAM 精修 pose graph 與 occupancy grid
PATROL_SLAM_PAUSE_EVERY_CYCLES = 40    # 每 N cycles 暫停一次（40 × 0.08s ≈ 3.2s）
PATROL_SLAM_PAUSE_SEC          = 0.4   # 暫停時長（0.4s 足夠 slam_toolbox update）

# ── 逃脫級聯（LiDAR 死角/牆角反覆卡住時啟動更激進脫困）──
PATROL_ESCAPE_CASCADE_WINDOW_SEC = 15.0  # 統計時間窗（秒）
PATROL_ESCAPE_CASCADE_LIMIT      = 3     # 窗內 >= N 次逃脫視為級聯
PATROL_ESCAPE_CASCADE_REV_SEC    = 1.0   # 級聯模式倒車時間
PATROL_ESCAPE_CASCADE_ROT_TIMEOUT= 3.2   # 級聯模式轉向最長時間
PATROL_ESCAPE_CLEAR_RAW_CM       = 120   # 轉向中若偵測到 raw >= 此值的方向立即停止轉動
PATROL_ESCAPE_CLOSE_BODY_CM      = 15    # 前方 body 餘裕 < 此值啟動「深倒車」
PATROL_ESCAPE_DEEP_REV_SEC       = 0.7   # 深倒車時間

# ── Escape 參數（精簡版，目標總時長 ~2 秒）──
PATROL_ESCAPE_REVERSE_SEC    = 0.3    # 倒車時間
PATROL_ESCAPE_REVERSE_PWM    = 0.32   # 倒車 PWM（原 0.288）
PATROL_ESCAPE_ROTATE_TIMEOUT = 1.8    # 轉向最長時間
PATROL_ESCAPE_VISITED_RADIUS = 1      # 訪問格膨脹半徑（cell，1=3×3）
PATROL_ESCAPE_VISITED_TTL_SEC = 30.0  # 膨脹 visited 衰退秒數

# ── 高階脫困 watchdog（補 scan_patrol 內部 stuck 偵測的死角）──
# 被小物(鞋子/電線/門檻等 LiDAR 與視覺都沒抓到)卡死時,robot 可能完全不動。
# 當「SEARCH/ANOMALY 巡邏中 + 前方無人 + SLAM 位置長時間幾乎沒動」即強制脫困。
# 需 SLAM pose 可用才計時(LiDAR 死時不盲衝,交由 scan 的 LiDAR watchdog 停車)。
PATROL_STUCK_WATCHDOG_SEC        = 8.0   # 無位移 + 無人 超過此秒數 → 強制脫困
PATROL_STUCK_WATCHDOG_POSE_EPS_M = 0.05  # 位移 < 5cm 視為沒動(濾 SLAM 抖動)
PATROL_STUCK_WATCHDOG_YAW_EPS    = 0.10  # 轉動 < 0.1rad(~6°)視為沒動
PATROL_UNSTUCK_REVERSE_PWM       = 0.45  # 脫困後退 PWM(比一般避障強,掙脫卡夾)
PATROL_UNSTUCK_REVERSE_SEC       = 1.0   # 脫困後退時間
PATROL_UNSTUCK_TURN_SEC          = 1.2   # 脫困大角度轉向時間

# ============================================================
# 多人追蹤 (Multi-Person Tracking)
# ============================================================
TRACKER_MAX_LOST_FRAMES = 30       # 遺失超過此幀數則移除追蹤
TRACKER_MIN_IOU         = 0.25     # IoU 匹配最低閾值

# ============================================================
# 物件偵測擴展 (Object Detection Extension)
# ============================================================
OBJECT_DETECTION_ENABLED = True
RESCUE_OBJECT_CLASSES = {
    24: "backpack",     # 背包
    26: "handbag",      # 手提包
    28: "suitcase",     # 行李箱
    67: "cell phone",   # 手機
}

# ============================================================
# 視覺低障礙偵測 (補 LiDAR 車頂盲區)
# ============================================================
# LiDAR 安裝於車頂掃描平面,偵測不到「低於 LiDAR 高度」的物體
# (人體工學椅椅腳、桌腳、椅子下方、行李等)。
# 改用攝影機 YOLO 偵測這些常見地面障礙,依 bbox 在畫面的位置/大小估計距離,
# 與 LiDAR 防撞並行運作。
GROUND_OBSTACLE_ENABLED = True
# COCO 類別:常見會擋路但可能低於車頂 LiDAR 的家具/物品(夠大才會擋路)
# 注意:故意不含 cell phone / handbag 等小扁物,避免地上小物造成不必要煞停
GROUND_OBSTACLE_CLASSES = {
    13: "bench",          # 長椅
    24: "backpack",       # 背包(地上大型)
    28: "suitcase",       # 行李箱
    39: "bottle",         # 瓶子
    56: "chair",          # 椅子(含人體工學椅)
    57: "couch",          # 沙發
    58: "potted plant",   # 盆栽
    59: "bed",            # 床
    60: "dining table",   # 餐桌(桌腳)
    75: "vase",           # 花瓶
}
GROUND_OBSTACLE_DETECT_EVERY  = 5     # auto 模式每 N 個 detect-frame 跑一次 general 偵測
GROUND_OBSTACLE_STALE_SEC     = 1.0   # 偵測結果保留秒數(超過視為已清空)
# 危險分級(bbox 底邊 y 佔畫面高比例 + bbox 面積佔畫面比例)
GROUND_OBSTACLE_BRAKE_Y_RATIO = 0.80  # 底邊 > 畫面 80% 高 → 非常近
GROUND_OBSTACLE_BRAKE_AREA    = 0.06  # bbox 面積 > 畫面 6%
GROUND_OBSTACLE_WARN_Y_RATIO  = 0.62  # 底邊 > 62% → 警告距離(減速)
# 前方路徑中央帶(以 bbox 中心 x 佔畫面寬比例界定;只有在路徑上的才算擋路)
GROUND_OBSTACLE_CENTER_LO     = 0.18
GROUND_OBSTACLE_CENTER_HI     = 0.82

# ============================================================
# rPPG 生命跡象偵測（遠端光體積描記術）
# ============================================================
RPPG_ENABLED              = True
RPPG_BUFFER_SECONDS       = 5.0    # 滾動緩衝長度 (秒)
RPPG_UPDATE_INTERVAL      = 10     # 每 N 幀計算一次心率（更即時）
RPPG_MIN_FACE_CONFIDENCE  = 0.25   # 臉部關鍵點最低信心度（放寬：低光/側臉容忍）
RPPG_ROI_STABILITY_PX     = 40     # 幀間臉部最大位移 (px)（放寬：手持機器人晃動容忍）
RPPG_UNSTABLE_FRAMES_TO_RESET = 5  # 連續 N 幀不穩才清空 buffer（hysteresis 防誤清）
RPPG_FACE_LOSS_TOLERANCE_SEC  = 1.0  # 臉部短暫消失 N 秒內仍保留 buffer
RPPG_BANDPASS_LOW_HZ      = 0.83   # 50 bpm（避開呼吸頻率 0.2-0.5Hz 的洩漏）
RPPG_BANDPASS_HIGH_HZ     = 3.0    # 180 bpm（避開高頻噪聲）
RPPG_CONFIDENCE_MIN       = 0.35   # 結果可信度門檻（放寬：弱信號也回報）
RPPG_MIN_BUFFER_SEC       = 1.5    # 最少累積 N 秒才嘗試計算（縮短冷啟動）

# ── B1: 呼吸率偵測（rPPG 延伸）──
# 呼吸會造成皮膚顏色以 0.1-0.5 Hz（6-30 breaths/min）的慢速週期性變化
# 需比 HR 更長的緩衝（15s+）以獲得足夠頻率解析度
RPPG_RESP_ENABLED         = True
RPPG_RESP_BUFFER_SECONDS  = 15.0   # 呼吸訊號緩衝（15s 允許偵測 4 breaths/min 以上）
RPPG_RESP_LOW_HZ          = 0.1    # 6 bpm
RPPG_RESP_HIGH_HZ         = 0.5    # 30 bpm（含兒童快速呼吸）
RPPG_RESP_MIN_BUFFER_SEC  = 8.0    # 最少累積 8s 才嘗試計算
RPPG_RESP_CONFIDENCE_MIN  = 0.30

# ── A4: 小人物 ROI 再推論（提升遠距姿態偵測可靠度）──
ROI_REINFER_ENABLED        = True
ROI_REINFER_BBOX_MAX_H_PX  = 150   # bbox 高度 < 此值視為小人物
ROI_REINFER_MIN_GOOD_KPS   = 6     # 高信心關鍵點 < N 才觸發再推論
ROI_REINFER_PAD_RATIO      = 0.15  # crop 時 bbox 四周外擴比例
ROI_REINFER_CONF_GATE      = 0.5   # 判定「高信心關鍵點」的閾值

# ============================================================
# 低光增強 (CLAHE)
# ============================================================
CLAHE_MODE            = "auto"     # "auto" / "on" / "off"（預設 auto：低光環境自動啟用）
CLAHE_AUTO_THRESHOLD  = 80         # 平均亮度低於此值自動啟用 (0-255)
CLAHE_CLIP_LIMIT      = 3.0        # CLAHE 對比限制
CLAHE_TILE_SIZE       = (8, 8)     # CLAHE 網格大小

# ============================================================
# 本地硬體 GPIO 相容性（集中初始化 pigpio，避免多模組重複建立連線）
# ============================================================
try:
    import gpiozero
    from gpiozero import Device
    try:
        from gpiozero.pins.pigpio import PiGPIOFactory
        Device.pin_factory = PiGPIOFactory()
        PIGPIO_OK = True
    except Exception:
        PIGPIO_OK = False
    GPIO_AVAILABLE = True
except ImportError:
    GPIO_AVAILABLE = False
    PIGPIO_OK = False
    print("⚠️ GPIO 不可用，硬體模組將進入模擬模式")

SERVO_DEFAULT_PAN  = SERVO_PAN_DEFAULT
SERVO_DEFAULT_TILT = SERVO_TILT_DEFAULT
