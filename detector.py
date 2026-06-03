"""
搜救機器人 — AI 偵測模組 (ONNX 後端)
===================================================
人體偵測 (yolov8n) + 姿態/倒地偵測 (yolov8n-pose)
使用 ONNX (ultralytics) 推論後端
"""

import cv2
import math
import threading
import time
import logging
import numpy as np
from collections import deque
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

# 多人追蹤
try:
    from tracker import PersonTracker
    TRACKER_AVAILABLE = True
except ImportError:
    TRACKER_AVAILABLE = False

try:
    from rppg import rPPGDetector
    RPPG_AVAILABLE = True
except ImportError:
    RPPG_AVAILABLE = False

try:
    from vital_signs import VitalSignsAggregator
    VITAL_SIGNS_AVAILABLE = True
except ImportError:
    VITAL_SIGNS_AVAILABLE = False

logger = logging.getLogger("rescue.detector")

# ============================================================
# ONNX 後端 (ultralytics)
# ============================================================
import os
os.environ['YOLO_AUTOINSTALL'] = 'false'
os.environ['YOLO_VERBOSE'] = 'false'
try:
    from ultralytics import YOLO
    import ultralytics
    ultralytics.checks = lambda: None
    try:
        from ultralytics.utils import SETTINGS
        SETTINGS['sync'] = False
        SETTINGS['runs_dir'] = '/tmp/ultralytics'
    except Exception:
        pass
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

import config


@dataclass
class DetectionResult:
    """單幀偵測結果"""
    person_count: int = 0
    persons: List[dict] = field(default_factory=list)
    fallen_count: int = 0
    fallen_persons: List[dict] = field(default_factory=list)
    pose_anomaly_score: float = 0.0
    annotated_frame: np.ndarray = None
    timestamp: float = 0.0
    # 多人追蹤
    tracks: list = field(default_factory=list)
    unique_person_count: int = 0
    unreported_count: int = 0
    # 物件偵測擴展
    objects: List[dict] = field(default_factory=list)
    # 視覺低障礙 (補 LiDAR 車頂盲區): 椅子/桌腳/行李等
    ground_obstacles: List[dict] = field(default_factory=list)
    ground_obstacle_level: str = "CLEAR"   # "CLEAR" / "WARN" / "BRAKE"
    # 眼睛狀態
    eye_state: str = "UNKNOWN"   # "OPEN" / "CLOSED" / "UNKNOWN"
    # rPPG 生命跡象
    heart_rate_bpm: float = -1.0           # -1 = 未量測
    rppg_confidence: float = 0.0           # 0~1 信號品質
    rppg_signal_quality: str = "UNKNOWN"   # "GOOD" / "WEAK" / "UNKNOWN"
    # B1: 呼吸率（次/分）
    respiration_rate: float = -1.0         # -1 = 未量測
    resp_confidence: float = 0.0
    # 呼吸率/心率 buffer 累積進度（0.0~1.0），供 UI 顯示「建立中」
    rr_buffer_ratio: float = 0.0
    hr_buffer_ratio: float = 0.0
    # 眨眼 tracker warmup 進度（0.0~1.0）
    blink_warmup_ratio: float = 0.0
    # B2: 微動（0-1，人體 bbox 內光流/幀差值平滑後幅度）
    micro_motion_score: float = 0.0
    # B4: 眨眼率 + 意識狀態
    blink_rate_per_min: float = -1.0       # 近 30s 換算的每分鐘眨眼次數（-1 = 資料不足）
    consciousness_state: str = "UNKNOWN"   # "AWAKE" / "DROWSY" / "UNCONSCIOUS" / "UNKNOWN"
    # ── B5: 整合性生命跡象指標 (心率/呼吸/眨眼/意識/微動 加權融合) ──
    vital_score: float = -1.0              # 0=無生命跡象, 1=完全正常, -1=未知
    vital_status: str = "未知"              # "正常"/"微弱"/"失去意識"/"無反應"/"建立中"/"未知"
    vital_confidence: float = 0.0          # 0-1 綜合生命跡象可信度
    victim_vital_score: float = 0.0        # fusion 用:異常程度 0-1 (越高越像受困者)
    vital_components: dict = field(default_factory=dict)  # 各子分數供 UI 細節顯示


# 人體骨架連線定義
SKELETON = [
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12),
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16)
]

FALLEN_CONFIRM_FRAMES     = 3   # 連續 3 幀確認；回到較靈敏的倒地偵測
CROUCHING_CONFIRM_FRAMES  = 3   # 蜷縮需 3 幀確認，避免伸手拿東西的瞬間誤判
DISTRESSED_CONFIRM_FRAMES = 4   # 半躺/側靠判準較弱，需 4 幀確認


# ============================================================
# 姿態判定函數
# ============================================================
def _get_center(p1, p2, min_conf=0.3):
    """取得兩個關鍵點的中心座標（至少一個信心度足夠即可）"""
    if p1[2] > min_conf and p2[2] > min_conf:
        return ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)
    elif p1[2] > min_conf:
        return (p1[0], p1[1])
    elif p2[2] > min_conf:
        return (p2[0], p2[1])
    return None


def _head_orientation(keypoints, min_conf=0.3):
    """只憑頭部關鍵點（眼/耳）估計頭部與水平線的夾角（度）。
    站立正視：眼連線接近水平 → 角度 < 15°
    側躺：眼連線近垂直 → 角度 > 60°
    回傳角度或 None（資訊不足）"""
    left_eye, right_eye = keypoints[1], keypoints[2]
    left_ear, right_ear = keypoints[3], keypoints[4]

    def _line_angle(p1, p2):
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        if abs(dx) < 1 and abs(dy) < 1:
            return None
        return math.degrees(math.atan2(abs(dy), max(1.0, abs(dx))))

    # 優先用雙眼連線（距離較短但相對位置穩）
    if left_eye[2] > min_conf and right_eye[2] > min_conf:
        a = _line_angle(left_eye, right_eye)
        if a is not None:
            return a
    # 備援：雙耳連線
    if left_ear[2] > min_conf and right_ear[2] > min_conf:
        return _line_angle(left_ear, right_ear)
    return None


def _count_visible_body_kps(keypoints, min_conf=0.3):
    """計算可見的「軀幹以下」關鍵點數（肩/肘/腕/臀/膝/踝，共 12 點）。"""
    body_indices = list(range(5, 17))
    return sum(1 for i in body_indices if keypoints[i][2] > min_conf)


def _count_visible_head_kps(keypoints, min_conf=0.3):
    """計算可見的頭部關鍵點數（鼻/雙眼/雙耳，共 5 點）。"""
    return sum(1 for i in range(5) if keypoints[i][2] > min_conf)


def _body_vertical_extent(keypoints, shoulder_pt, min_conf=0.3) -> dict:
    """回傳 body 關鍵點 y 軸分布的統計：
       {shoulder_y, hip_y, knee_y, ankle_y, shoulder_width}，缺失為 None。
       用於判定躺平（腿部被壓縮到軀幹附近）。"""
    sw = None
    if keypoints[5][2] > min_conf and keypoints[6][2] > min_conf:
        sw = abs(float(keypoints[5][0]) - float(keypoints[6][0]))
    knee = _get_center(keypoints[13], keypoints[14], min_conf)
    ankle = _get_center(keypoints[15], keypoints[16], min_conf)
    return {
        "shoulder_y": shoulder_pt[1] if shoulder_pt else None,
        "knee_y": knee[1] if knee else None,
        "ankle_y": ankle[1] if ankle else None,
        "shoulder_width": sw,
    }


def _is_upright_kp_sequence(keypoints, min_conf: float = 0.3) -> bool:
    """檢查可見關鍵點是否呈現「真直立」分佈(頭→肩→臀→膝→踝)。

    真直立(站/坐/蹲):y 座標遞增 → 逆序對少 + 全身 y 跨度遠大於水平跨度
    真倒地:身體壓在同一水平 → 逆序對多,或 y 跨度被壓縮

    NEW 用途:防止「坐著前傾、彎腰駝背」被誤判為倒地。
    至少需要 3 個層級可見才作判斷;逆序對少於層級數的一半視為仍 upright。
    每對允許 8 px 容差(蹲踞時膝有時略高於臀)。
    額外保護:若身體 x 跨度大於 y 跨度的 2 倍,代表水平躺臥,直接視為非 upright。
    """
    levels = []
    # 頭 (取最上面的頭部點作代表)
    head_ys = [float(keypoints[i][1]) for i in range(5) if keypoints[i][2] > min_conf]
    if head_ys:
        levels.append(("head", min(head_ys)))
    # 肩、臀、膝、踝(中心)
    for label, (a, b) in (("shoulder", (5, 6)), ("hip", (11, 12)),
                          ("knee", (13, 14)), ("ankle", (15, 16))):
        c = _get_center(keypoints[a], keypoints[b], min_conf)
        if c:
            levels.append((label, float(c[1])))
    if len(levels) < 3:
        return False

    # NEW: x 跨度 vs y 跨度檢查 (側躺時所有 y 雖按順序排但很扁平)
    ys = [y for _, y in levels]
    y_range = max(ys) - min(ys)
    xs = [float(keypoints[i][0]) for i in (5, 6, 11, 12, 13, 14, 15, 16)
          if keypoints[i][2] > min_conf]
    if xs:
        x_range = max(xs) - min(xs)
        # 身體水平跨度 > 垂直跨度 × 2 → 必定不是直立(側躺)
        if x_range > 30 and y_range < x_range * 0.5:
            return False

    inversions = 0
    for i in range(len(levels) - 1):
        # 「上層 y」應 < 「下層 y」;允許 8px 容差
        if levels[i][1] > levels[i + 1][1] + 8:
            inversions += 1
    # 逆序對 < 半數視為 upright(站/坐/蹲都會通過,只有躺平會破壞順序)
    return inversions < len(levels) / 2


def _leg_compressed_vs_trunk(kstat: dict, spine_dy: float, keypoints=None,
                             min_conf: float = 0.3) -> bool:
    """判定整個身體是否在 image 的 y 軸上被壓縮到一條狹帶 → 躺平朝/背對鏡頭關鍵特徵。

    原理：站立/坐姿/蹲踞的人，雖然 spine_dy 可能被壓縮，
    但身體各 keypoint（頭/肩/臀/膝/踝）在 image y 軸上仍有明顯分布：
        站立  body_y_range / shoulder_width ≈ 6-8
        坐姿  ≈ 4-6 （頭在上、腳在下）
        蹲踞  ≈ 3-5
        完全躺平朝/背鏡頭  < 1.5 （所有 keypoint 壓到同一水平）

    只有最後一種應判為倒地。坐姿/蹲踞仍具可觀 y 軸延伸，不應誤觸發。
    """
    sw = kstat.get("shoulder_width")
    if sw is None or sw < 15:
        return False

    # 收集所有可見 body keypoints 的 y 座標
    # （用傳入的 keypoints 更可靠，可取得頭部 y 一併計算全身範圍）
    ys = []
    if keypoints is not None:
        # 頭部（鼻/雙眼/雙耳）
        for i in range(5):
            if keypoints[i][2] > min_conf:
                ys.append(float(keypoints[i][1]))
        # 肩/臀/膝/踝
        for i in (5, 6, 11, 12, 13, 14, 15, 16):
            if keypoints[i][2] > min_conf:
                ys.append(float(keypoints[i][1]))
    else:
        # fallback：只用 kstat 提供的 4 個點
        for key in ("shoulder_y", "hip_y", "knee_y", "ankle_y"):
            y = kstat.get(key)
            if y is not None:
                ys.append(float(y))

    # 至少需要 3 個 keypoints 才能可靠判斷
    if len(ys) < 3:
        return False

    body_y_range = max(ys) - min(ys)
    # 躺平朝/背鏡頭的關鍵閾值：全身 y 軸延伸 < 1.5× 肩寬
    # （非常嚴格，避免坐姿、蹲踞等誤觸發）
    return body_y_range < sw * 1.5


def is_fallen(keypoints, person_bbox=None) -> bool:
    """
    判斷人員是否倒地。
      Path A: 有 shoulder+hip → 脊椎向量法 + 下半身壓縮備援
      Path A.5: 只有頭部 → 頭部姿態法（眼/耳連線角度）
      Path B: 完全無關鍵點 → bbox 寬高比（保守，需頭部輔證）
    """
    min_conf = 0.3

    shoulder = _get_center(keypoints[5], keypoints[6], min_conf)
    hip = _get_center(keypoints[11], keypoints[12], min_conf)

    # ═══ Path A：有骨架 → 用 spine 向量 ═══
    if shoulder and hip:
        spine_dx = abs(hip[0] - shoulder[0])
        spine_dy = hip[1] - shoulder[1]   # 正值 = 臀在肩下方

        # NEW: 完全躺平朝/背鏡頭時，spine 看似垂直但全身 y 軸被透視壓縮到一條狹帶。
        # 使用所有可見 keypoints 的 y range 判定（區別坐姿/蹲踞與真躺平）。
        kstat = _body_vertical_extent(keypoints, shoulder, min_conf)
        leg_compressed = _leg_compressed_vs_trunk(kstat, spine_dy, keypoints, min_conf)
        # 1. 站立/坐著：hip 明顯在 shoulder 下方且 spine 接近垂直
        if spine_dy > 40 and spine_dy > spine_dx * 1.3:
            # 若下半身壓縮 → 實際是躺平面向鏡頭，非站立
            if leg_compressed:
                return True
            return False

        # 2. 橫躺判定：
        #    a) hip 在 shoulder 上方或近似同高 (spine_dy < 25)
        #    b) spine 比較水平 (spine_dx > spine_dy * 1.2)
        if spine_dy < 25 or spine_dx > spine_dy * 1.2:
            # 回到原本偏靈敏的判定：只要肩臀向量明顯水平，就視為倒地。
            # bbox 只作輔助，不再用來否決骨架的倒地結果。
            if person_bbox:
                bw = person_bbox["x2"] - person_bbox["x1"]
                bh = person_bbox["y2"] - person_bbox["y1"]
                if bh > 0:
                    ratio = bw / bh
                    if ratio > 0.55 or leg_compressed:
                        return True
            return True

        # NEW: spine 不太長、不太橫，但下半身被壓縮 → 躺平
        # （涵蓋: spine_dy 在 25-40 之間、看似「蹲踞」但其實是躺平側角度）
        if leg_compressed:
            return True

        # Path A 有資料但不構成倒地 → 直接回 False，不落到後續路徑
        return False

    # ═══ Path A.5：身體被遮擋只剩頭部 → 用頭部姿態判讀（新增）═══
    # 側躺的人若僅頭部可見，眼/耳連線會近乎垂直 → 強力倒地訊號
    head_kps = _count_visible_head_kps(keypoints, min_conf)
    body_kps = _count_visible_body_kps(keypoints, min_conf)
    if head_kps >= 2 and body_kps == 0:
        head_angle = _head_orientation(keypoints, min_conf)
        if head_angle is not None and head_angle >= 55:
            # 頭部明顯傾斜 >55° → 側躺倒地
            return True
        # 頭部近水平時無法區分站立面對鏡頭/仰躺，不作倒地結論（避免誤判）
        return False

    # ═══ Path B：無關鍵點 → 完全靠 bbox（最嚴格）═══
    if person_bbox:
        bw = person_bbox["x2"] - person_bbox["x1"]
        bh = person_bbox["y2"] - person_bbox["y1"]
        if bh > 0 and bw > 0 and (bw / bh) > 1.8:
            # 保留原寬度閾值，但要求至少一個頭部關鍵點可見，避免純 bbox 誤判
            if head_kps >= 1:
                return True

    return False


def is_crouching(keypoints) -> bool:
    """判斷人員是否蜷縮。新增直立剔除：spine_dy > 80 視為站立不蜷縮。"""
    min_conf = 0.3
    core_indices = [5, 6, 11, 12, 13, 14, 15, 16]
    valid_pts = [keypoints[i] for i in core_indices if keypoints[i][2] > min_conf]
    if len(valid_pts) < 4:
        return False

    # 直立剔除：若肩臀皆可見且 spine 仍垂直 > 80px，視為站立（非蜷縮）
    shoulder_c = _get_center(keypoints[5], keypoints[6], min_conf)
    hip_c = _get_center(keypoints[11], keypoints[12], min_conf)
    if shoulder_c and hip_c:
        spine_dy = hip_c[1] - shoulder_c[1]
        if spine_dy > 80:
            return False

    xs = [p[0] for p in valid_pts]
    ys = [p[1] for p in valid_pts]
    w = max(xs) - min(xs)
    h = max(ys) - min(ys)

    if h <= 0 or w <= 0:
        return False

    aspect = w / h
    if 0.6 < aspect < 1.6 and h < 200:
        shoulder = None
        hip = None
        if keypoints[5][2] > min_conf and keypoints[6][2] > min_conf:
            shoulder = ((keypoints[5][1] + keypoints[6][1]) / 2)
        if keypoints[11][2] > min_conf and keypoints[12][2] > min_conf:
            hip = ((keypoints[11][1] + keypoints[12][1]) / 2)
        if shoulder and hip:
            torso_vertical = abs(hip - shoulder)
            if torso_vertical < 80:
                return True
    return False


def is_distressed(keypoints, person_bbox=None) -> bool:
    """
    半躺/側靠姿勢（介於正常與完全倒地之間）。
    修訂：加入骨架 sanity check + 頭部關鍵點輔證，避免坐著或只拍到臉誤觸發。
    """
    if not person_bbox:
        return False

    bw = person_bbox["x2"] - person_bbox["x1"]
    bh = person_bbox["y2"] - person_bbox["y1"]
    if bh <= 0:
        return False

    bbox_ratio = bw / bh
    if not (1.4 < bbox_ratio <= 1.8):
        return False

    min_conf = 0.3
    shoulder = _get_center(keypoints[5], keypoints[6], min_conf)
    hip = _get_center(keypoints[11], keypoints[12], min_conf)
    body_kps = _count_visible_body_kps(keypoints, min_conf)
    head_kps = _count_visible_head_kps(keypoints, min_conf)

    # 1) 有肩有臀 → 必須確認非站立（spine 近水平或微斜）
    if shoulder and hip:
        spine_dy = hip[1] - shoulder[1]
        spine_dx = abs(hip[0] - shoulder[0])
        # 站立/正常坐姿：spine 明顯垂直
        if spine_dy > 60 and spine_dy > spine_dx * 1.2:
            return False
        # NEW: upright 序列保護,避免坐姿前傾被誤判半躺
        if _is_upright_kp_sequence(keypoints, min_conf):
            return False
        # NEW: 頭部仍正立 + bbox 不算太寬扁 → 拒絕
        head_ang = _head_orientation(keypoints, min_conf)
        if head_ang is not None and head_ang < 30 and bbox_ratio < 1.6:
            return False
        # spine 偏水平（dx ≈ dy 或 dy 偏小）→ 接受半躺判定
        return True

    # 2) 有軀幹點但不齊全（>= 3 點）→ 可接受 distressed
    if body_kps >= 3:
        return True

    # 3) 頭部 only（無軀幹）→ 需更強 bbox 訊號 (>1.6) + 頭部關鍵點
    if bbox_ratio > 1.6 and head_kps >= 2:
        return True

    # 其他情況（純 bbox + 無關鍵點、bbox ~1.4-1.6 + 無軀幹）→ 保守回 False
    return False


def detect_eye_state(frame, keypoints) -> str:
    """
    估計眼睛開閉狀態。
    用 Laplacian 邊緣方差分析眼部區域：
      開眼 → 虹膜/瞳孔邊緣多 → 方差高
      閉眼 → 皮膚平滑 → 方差低
    回傳 "OPEN" / "CLOSED" / "UNKNOWN"

    放寬策略（提升偵測率）：
      - min_conf 0.3 → 0.2（遠距/側臉也能抓）
      - nose 不存在時，改用雙眼距離推算 scale（只有雙眼可見仍可運作）
      - 只要至少一眼 conf ≥ 0.2 即啟動分析
    """
    min_conf = 0.2
    nose = keypoints[0]
    left_eye = keypoints[1]
    right_eye = keypoints[2]

    # 若 nose 可見 → 用 nose-eye 距離作 scale；否則 fallback 用 eye-eye 距離
    have_nose = nose[2] >= min_conf
    have_left = left_eye[2] >= min_conf
    have_right = right_eye[2] >= min_conf

    if not (have_left or have_right):
        return "UNKNOWN"   # 連一眼都沒可用 keypoint

    # 估算 scale：優先 nose→eye，備援 eye→eye
    scale = 0.0
    if have_nose:
        if have_left:
            scale = abs(left_eye[0] - nose[0])
        elif have_right:
            scale = abs(right_eye[0] - nose[0])
    if scale < 5 and have_left and have_right:
        scale = abs(left_eye[0] - right_eye[0]) * 0.5   # 雙眼距離的一半當近似
    if scale < 5:
        return "UNKNOWN"   # scale 無法估計（極端 case）

    h, w = frame.shape[:2]
    results = []

    for eye_kp in [left_eye, right_eye]:
        if eye_kp[2] < min_conf:
            continue
        ex, ey = int(eye_kp[0]), int(eye_kp[1])

        face_ref = max(10, scale * 0.8)
        hw = max(5, int(face_ref * 0.5))
        hh = max(3, int(face_ref * 0.25))

        x1 = max(0, ex - hw)
        x2 = min(w, ex + hw)
        y1 = max(0, ey - hh)
        y2 = min(h, ey + hh)

        if x2 - x1 < 4 or y2 - y1 < 3:
            continue

        crop = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
        edge_var = cv2.Laplacian(crop, cv2.CV_64F).var()

        results.append("OPEN" if edge_var > 100 else "CLOSED")

    if not results:
        return "UNKNOWN"
    if all(r == "CLOSED" for r in results):
        return "CLOSED"
    return "OPEN"


def compute_pose_anomaly_score(keypoints_list, fallen_ids: set,
                                crouching_ids: set, distressed_ids: set) -> float:
    """計算姿態異常分數 (0~1)。"""
    if not keypoints_list:
        return 0.0
    max_score = 0.0
    for kps in keypoints_list:
        kid = id(kps)
        if kid in fallen_ids:
            max_score = max(max_score, 1.0)
        elif kid in crouching_ids:
            max_score = max(max_score, 0.6)
        elif kid in distressed_ids:
            max_score = max(max_score, 0.4)
    return max_score


# ============================================================
# B4: 眨眼率 / 意識狀態追蹤器
# ============================================================
class BlinkRateTracker:
    """
    基於每幀 eye_state (OPEN/CLOSED/UNKNOWN) 的時序分析，
    統計近 30 秒的眨眼次數並換算每分鐘眨眼率，推估意識狀態：

        AWAKE:        15-25 blinks/min（正常）
        DROWSY:       5-14  blinks/min 或 > 30（疲憊/眨眼過多）
        UNCONSCIOUS:  < 5 blinks/min 且持續眼閉 > 10s
        UNKNOWN:      資料不足 / 未見臉

    眨眼定義：OPEN → CLOSED → OPEN 的完整轉換，且 CLOSED 持續 < 0.6s
    （長時間閉眼不算眨眼而算「失去意識」信號）
    """

    WINDOW_SEC = 30.0
    LONG_CLOSED_SEC = 10.0
    BLINK_MAX_CLOSED_SEC = 0.6
    WARMUP_SEC = 8.0               # 資料累積期，期間回傳「樂觀」預估
    CONFIDENT_SEC = 15.0           # 超過此時長才以 rate 判 DROWSY

    def __init__(self):
        self._events = deque()     # (timestamp, state) 事件序列（僅 OPEN/CLOSED 轉換）
        self._last_state = "UNKNOWN"
        self._last_change_ts = 0.0
        self._closed_start_ts = -1.0
        self._first_seen_ts = -1.0   # 第一次看到 OPEN/CLOSED 的時間（用於 warmup 計時）

    def update(self, state: str, timestamp: float) -> tuple:
        """
        每幀更新。回傳 (blink_rate_per_min, consciousness_state)。
        warmup 期間（< 8s）若眼睛可見 → 回傳 AWAKE + blink=0（而非 UNKNOWN）
        讓 UI 能即時顯示「AWAKE（建立中）」而非持續空白。
        """
        # 清除超過窗口的舊事件
        while self._events and (timestamp - self._events[0][0]) > self.WINDOW_SEC:
            self._events.popleft()

        # 首次看到眼睛記時間
        if state in ("OPEN", "CLOSED") and self._first_seen_ts < 0:
            self._first_seen_ts = timestamp

        # 狀態轉換
        if state in ("OPEN", "CLOSED") and state != self._last_state:
            self._events.append((timestamp, state))
            if state == "CLOSED":
                self._closed_start_ts = timestamp
            else:
                self._closed_start_ts = -1.0
            self._last_change_ts = timestamp
            self._last_state = state

        # 計算眨眼次數
        blink_count = 0
        events = list(self._events)
        for i in range(1, len(events) - 1):
            if events[i][1] == "CLOSED" and events[i - 1][1] == "OPEN" and events[i + 1][1] == "OPEN":
                closed_duration = events[i + 1][0] - events[i][0]
                if closed_duration < self.BLINK_MAX_CLOSED_SEC:
                    blink_count += 1

        # 若完全沒看過眼睛
        if self._first_seen_ts < 0:
            return -1.0, "UNKNOWN"

        # 從首次看到眼睛起算的持續時間
        duration = timestamp - self._first_seen_ts

        # 長期閉眼 → UNCONSCIOUS（任何階段都適用）
        if (state == "CLOSED" and self._closed_start_ts >= 0 and
                (timestamp - self._closed_start_ts) > self.LONG_CLOSED_SEC):
            return 0.0, "UNCONSCIOUS"

        # Warmup 期間（< 8s）：樂觀假設為 AWAKE，避免 UI 永久空白
        if duration < self.WARMUP_SEC:
            # 回傳「暫定」估計值（供 UI 顯示 warmup 狀態）
            est_rate = blink_count * 60.0 / max(duration, 1.0) if blink_count > 0 else -1.0
            return est_rate, "AWAKE"

        # 換算每分鐘眨眼率
        rate = blink_count * 60.0 / duration

        # 意識狀態判定（進入 CONFIDENT_SEC 以上才以 rate 嚴格判 DROWSY）
        if rate > 25:
            state_label = "DROWSY"  # 過快眨眼也可能代表不適
        elif duration < self.CONFIDENT_SEC:
            # 資料尚不完整 → 樂觀保持 AWAKE（eye 可見就不標 DROWSY）
            state_label = "AWAKE"
        elif rate < 5:
            state_label = "DROWSY"
        else:
            state_label = "AWAKE"

        return rate, state_label

    @property
    def warmup_ratio(self) -> float:
        """0.0~1.0：資料累積進度（供 UI 顯示「建立中」進度）。"""
        if self._first_seen_ts < 0:
            return 0.0
        # 以 WARMUP_SEC 為門檻，超過即 1.0
        import time as _t
        dur = _t.time() - self._first_seen_ts
        return min(1.0, max(0.0, dur / self.WARMUP_SEC))

    def reset(self):
        self._events.clear()
        self._last_state = "UNKNOWN"
        self._last_change_ts = 0.0
        self._first_seen_ts = -1.0
        self._closed_start_ts = -1.0


# ============================================================
# B2: 微動偵測器（光流/幀差）
# ============================================================
class MicroMotionDetector:
    """
    對人體 bbox 區域做幀差運算偵測微小活動（呼吸胸廓起伏、手指抽動等）。
    同時比對 bbox 外的背景變化量作為 ego-motion 參考：
      - bbox 內 high，bbox 外 low → 人正在動（生命跡象）
      - bbox 內外都 high → 機器人/鏡頭在晃，非可靠信號
      - bbox 內外都 low → 靜止（可能無生命跡象或穩定狀態）

    回傳 score 0.0-1.0（EMA 平滑），需搭配 person bbox 才啟用。
    """

    DIFF_PIXEL_THRESHOLD = 8     # 灰階差異 > 此值才算動
    DOWNSAMPLE = 2                # 幀下採樣倍率（加速 + 降噪）
    EMA_ALPHA = 0.3               # 時域平滑係數
    EGO_MOTION_CANCEL_RATIO = 1.8 # bbox 內變化需比 bbox 外高 N 倍才算真動

    def __init__(self):
        self._prev_gray = None
        self._score_ema = 0.0

    def update(self, frame: np.ndarray, persons: list) -> float:
        """回傳微動分數 (0-1)；無 person 或首幀回傳 0。"""
        if frame is None or not persons:
            self._score_ema *= 0.7   # 衰減
            return round(self._score_ema, 3)

        try:
            # 下採樣 + 灰階
            small = frame[::self.DOWNSAMPLE, ::self.DOWNSAMPLE]
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        except Exception:
            return round(self._score_ema, 3)

        if self._prev_gray is None or self._prev_gray.shape != gray.shape:
            self._prev_gray = gray
            return 0.0

        diff = cv2.absdiff(gray, self._prev_gray)
        self._prev_gray = gray

        motion_mask = diff > self.DIFF_PIXEL_THRESHOLD
        h, w = gray.shape
        total_mask = np.zeros_like(motion_mask, dtype=bool)

        # 建立 bbox 聯集 mask（已下採樣座標）
        for p in persons:
            try:
                x1 = max(0, int(p["x1"] / self.DOWNSAMPLE))
                y1 = max(0, int(p["y1"] / self.DOWNSAMPLE))
                x2 = min(w, int(p["x2"] / self.DOWNSAMPLE))
                y2 = min(h, int(p["y2"] / self.DOWNSAMPLE))
                if x2 > x1 and y2 > y1:
                    total_mask[y1:y2, x1:x2] = True
            except (KeyError, ValueError):
                continue

        bbox_pixels = int(total_mask.sum())
        if bbox_pixels < 50:
            return round(self._score_ema, 3)

        bbox_motion_pixels = int((motion_mask & total_mask).sum())
        bg_pixels = int((~total_mask).sum())
        bg_motion_pixels = int((motion_mask & ~total_mask).sum())

        in_ratio = bbox_motion_pixels / bbox_pixels
        bg_ratio = bg_motion_pixels / max(1, bg_pixels)

        # Ego-motion 抵消：bbox 內動量須顯著高於背景
        if bg_ratio * self.EGO_MOTION_CANCEL_RATIO >= in_ratio:
            # 視為機器人自身晃動造成的假動
            raw_score = 0.0
        else:
            # 對 in_ratio 做 soft cap：0.02 (2% 像素動) → 分數 1.0
            raw_score = min(in_ratio / 0.02, 1.0)

        self._score_ema = self.EMA_ALPHA * raw_score + (1 - self.EMA_ALPHA) * self._score_ema
        return round(self._score_ema, 3)

    def reset(self):
        self._prev_gray = None
        self._score_ema = 0.0


# ============================================================
# 關鍵點 EMA 平滑器（A2：減少幀間抖動）
# ============================================================
class KeypointSmoother:
    """以 bbox IoU 關聯幀間人物 → 每關鍵點 EMA 平滑。
    alpha=0.5 代表新舊各半；confidence 保留當前幀值（不平滑）。
    只有當前 + 上幀 keypoint 皆 >conf_gate 時才平滑，否則直接用當前值避免死資料汙染。"""

    def __init__(self, alpha: float = 0.5, iou_threshold: float = 0.3,
                 conf_gate: float = 0.3, stale_sec: float = 0.5):
        self._alpha = alpha
        self._iou_thresh = iou_threshold
        self._conf_gate = conf_gate
        self._stale_sec = stale_sec
        self._tracks = []      # list of dict{bbox, kps, last_seen}

    @staticmethod
    def _iou(b1: dict, b2: dict) -> float:
        x1 = max(b1["x1"], b2["x1"])
        y1 = max(b1["y1"], b2["y1"])
        x2 = min(b1["x2"], b2["x2"])
        y2 = min(b1["y2"], b2["y2"])
        if x2 <= x1 or y2 <= y1:
            return 0.0
        inter = (x2 - x1) * (y2 - y1)
        a1 = max(1.0, (b1["x2"] - b1["x1"]) * (b1["y2"] - b1["y1"]))
        a2 = max(1.0, (b2["x2"] - b2["x1"]) * (b2["y2"] - b2["y1"]))
        return inter / (a1 + a2 - inter)

    def smooth(self, persons: list, all_kps: list) -> list:
        """回傳平滑後的 keypoints（同序）；內部更新 track 狀態。"""
        now = time.time()
        # 清除過期 track
        self._tracks = [t for t in self._tracks
                        if now - t["last_seen"] < self._stale_sec]

        smoothed = []
        used_tracks = set()
        for p_bbox, cur_kps in zip(persons, all_kps):
            # 找最佳 IoU track
            best_idx = -1
            best_iou = self._iou_thresh
            for i, t in enumerate(self._tracks):
                if i in used_tracks:
                    continue
                iou = self._iou(p_bbox, t["bbox"])
                if iou > best_iou:
                    best_iou = iou
                    best_idx = i

            if best_idx >= 0:
                t = self._tracks[best_idx]
                prev_kps = t["kps"]
                new_kps = cur_kps.copy()
                n = min(len(cur_kps), len(prev_kps))
                for i in range(n):
                    cc = cur_kps[i][2]
                    pc = prev_kps[i][2]
                    if cc > self._conf_gate and pc > self._conf_gate:
                        new_kps[i, 0] = self._alpha * cur_kps[i, 0] + (1 - self._alpha) * prev_kps[i, 0]
                        new_kps[i, 1] = self._alpha * cur_kps[i, 1] + (1 - self._alpha) * prev_kps[i, 1]
                        # confidence 保留當前幀原值
                t["bbox"] = p_bbox
                t["kps"] = new_kps
                t["last_seen"] = now
                used_tracks.add(best_idx)
                smoothed.append(new_kps)
            else:
                # 新 track
                self._tracks.append({
                    "bbox": p_bbox,
                    "kps": cur_kps.copy() if hasattr(cur_kps, 'copy') else np.array(cur_kps),
                    "last_seen": now,
                })
                smoothed.append(cur_kps)
        return smoothed


# ============================================================
# 主偵測器
# ============================================================
class RescueDetector:
    """
    搜救用偵測器（ONNX 後端）
    """

    def __init__(self):
        logger.info("初始化搜救 AI 偵測器...")
        self._general_model = None
        self._pose_model = None
        self._frame_count = 0
        self._latest_result = DetectionResult()
        self._lock = threading.Lock()
        self._fallen_consecutive = 0
        self._last_fallen_list = []
        self._last_fallen_persons = []
        self._crouching_consecutive = 0
        self._distressed_consecutive = 0
        self._all_keypoints: List = []
        self._last_draw_payload = None
        self._tracker = PersonTracker() if TRACKER_AVAILABLE else None
        self._rppg = rPPGDetector() if (RPPG_AVAILABLE and config.RPPG_ENABLED) else None
        self._kp_smoother = KeypointSmoother(alpha=0.5, iou_threshold=0.3)
        self._blink_tracker = BlinkRateTracker()
        self._motion_detector = MicroMotionDetector()
        # B5: 整合性生命跡象 (心率+呼吸+眨眼+意識+微動 → 綜合分數與狀態)
        self._vital_aggregator = VitalSignsAggregator() if VITAL_SIGNS_AVAILABLE else None

        # 視覺低障礙偵測 (補 LiDAR 車頂盲區)
        # app.py 在 auto 模式時設 True,提高 general 偵測頻率以即時抓地面障礙
        self.obstacle_scan_active = False
        self._last_ground_obstacles = []
        self._last_ground_level = "CLEAR"
        self._ground_obstacle_stamp = 0.0

        if YOLO_AVAILABLE:
            self._init_onnx()
        else:
            logger.warning("ultralytics 未安裝，AI 偵測功能停用")

    @staticmethod
    def _resolve_model_path(base_path: str) -> str:
        """優先使用 *_int8.onnx（若由 tools/quantize_models_int8.py 產生）。
        INT8 在 RPi 5 CPU 推論可加速 20–40%。找不到則 fallback 原檔。"""
        if not base_path.endswith(".onnx"):
            return base_path
        int8_path = base_path[:-5] + "_int8.onnx"
        if os.path.exists(int8_path):
            logger.info(f"偵測到 INT8 量化版本：{int8_path}")
            return int8_path
        return base_path

    def _init_onnx(self):
        """初始化 ONNX/ultralytics 後端。"""
        if not YOLO_AVAILABLE:
            logger.warning("ultralytics 未安裝，AI 偵測功能停用")
            return
        try:
            gen_path = self._resolve_model_path(config.YOLO_GENERAL_MODEL)
            pose_path = self._resolve_model_path(config.YOLO_POSE_MODEL)
            logger.info(f"載入 ONNX 通用模型: {gen_path}")
            self._general_model = YOLO(gen_path, task='detect')
            logger.info(f"載入 ONNX 姿態模型: {pose_path}")
            self._pose_model = YOLO(pose_path, task='pose')
            logger.info("ONNX 雙模型載入完成")
        except Exception as e:
            logger.error(f"ONNX 模型載入失敗: {e}")

    @property
    def is_loaded(self) -> bool:
        return self._general_model is not None

    @property
    def backend_name(self) -> str:
        return "ONNX"

    @property
    def latest_result(self) -> DetectionResult:
        with self._lock:
            return self._latest_result

    def clear_transient_state(self, frame: Optional[np.ndarray] = None):
        """清除即時偵測狀態，避免 REPORT/STANDBY 暫停推論時保留上一個倒地結果。"""
        self._fallen_consecutive = 0
        self._last_fallen_list = []
        self._last_fallen_persons = []
        self._crouching_consecutive = 0
        self._distressed_consecutive = 0
        self._all_keypoints = []
        self._last_draw_payload = None
        now = time.time()
        with self._lock:
            keep_unique = self._latest_result.unique_person_count
            self._latest_result.person_count = 0
            self._latest_result.persons = []
            self._latest_result.fallen_count = 0
            self._latest_result.fallen_persons = []
            self._latest_result.pose_anomaly_score = 0.0
            self._latest_result.objects = []
            self._latest_result.ground_obstacles = []
            self._latest_result.ground_obstacle_level = "CLEAR"
            self._latest_result.tracks = []
            self._latest_result.unique_person_count = keep_unique
            self._latest_result.unreported_count = 0
            self._latest_result.eye_state = "UNKNOWN"
            self._latest_result.heart_rate_bpm = -1.0
            self._latest_result.rppg_confidence = 0.0
            self._latest_result.rppg_signal_quality = "UNKNOWN"
            self._latest_result.respiration_rate = -1.0
            self._latest_result.resp_confidence = 0.0
            self._latest_result.rr_buffer_ratio = 0.0
            self._latest_result.hr_buffer_ratio = 0.0
            self._latest_result.blink_rate_per_min = -1.0
            self._latest_result.consciousness_state = "UNKNOWN"
            self._latest_result.blink_warmup_ratio = 0.0
            self._latest_result.micro_motion_score = 0.0
            self._latest_result.vital_score = -1.0
            self._latest_result.vital_status = "未知"
            self._latest_result.vital_confidence = 0.0
            self._latest_result.victim_vital_score = 0.0
            self._latest_result.vital_components = {}
            if frame is not None:
                self._latest_result.annotated_frame = frame
            self._latest_result.timestamp = now

    def draw_cached_overlay(self, frame: np.ndarray, max_age_sec: float = 0.45):
        """在最新 raw frame 上重畫短時間內的 YOLO 標註，避免 skipped frame 造成框線閃爍。"""
        if frame is None:
            return None
        now = time.time()
        with self._lock:
            payload = self._last_draw_payload
            if not payload:
                return None
            if now - payload.get("timestamp", 0.0) > max_age_sec:
                return None
            persons = [dict(p) for p in payload.get("persons", [])]
            fallen_kps = list(payload.get("fallen_kps", []))
            crouching_kps = list(payload.get("crouching_kps", []))
            all_kps = list(payload.get("all_kps", []))
            distressed_list = list(payload.get("distressed_list", []))
            objects = [dict(o) for o in payload.get("objects", [])]
            eye_state = payload.get("eye_state", "UNKNOWN")
            rppg = dict(payload.get("rppg", {}) or {})

        try:
            return self._draw_annotations(
                frame.copy(), persons, fallen_kps, crouching_kps, all_kps,
                distressed_list=distressed_list,
                objects=objects,
                eye_state=eye_state,
                rppg=rppg,
            )
        except Exception as e:
            logger.debug(f"快取 overlay 繪製失敗: {e}")
            return None

    # ------------------------------------------------------------------
    def detect(self, frame: np.ndarray) -> DetectionResult:
        """對單幀執行 AI 偵測。"""
        if frame is None:
            return self._latest_result

        self._frame_count += 1
        now = time.time()
        h, w = frame.shape[:2]

        # ONNX 推理：pose 模型取得人員邊界框+骨架，general 模型取得物件
        objects = []
        if self._pose_model:
            all_kps = self._detect_pose_onnx(frame)
            persons = self._kps_to_persons(all_kps, h, w)
        else:
            persons, objects = self._detect_persons_onnx(frame)
            all_kps = []
        # general 模型取得物件偵測（背包、手機、椅子等）
        # auto 巡邏模式下提高頻率(每 N 幀)以即時抓地面低障礙;否則每 10 幀(省 CPU)
        _ground_enabled = getattr(config, 'GROUND_OBSTACLE_ENABLED', False)
        gen_every = (int(getattr(config, 'GROUND_OBSTACLE_DETECT_EVERY', 5))
                     if (self.obstacle_scan_active and _ground_enabled) else 10)
        _ran_general = False
        if self._general_model and self._frame_count % gen_every == 0:
            _, objects = self._detect_persons_onnx(frame)
            _ran_general = True

        # 視覺低障礙分類(補 LiDAR 車頂盲區)
        if _ground_enabled:
            if _ran_general:
                g_obs, g_level = self._classify_ground_obstacles(objects, h, w)
                self._last_ground_obstacles = g_obs
                self._last_ground_level = g_level
                self._ground_obstacle_stamp = now
            elif now - self._ground_obstacle_stamp > getattr(config, 'GROUND_OBSTACLE_STALE_SEC', 1.0):
                # 太久沒更新 → 清空,避免殘留舊障礙
                self._last_ground_obstacles = []
                self._last_ground_level = "CLEAR"

        # 多人追蹤
        if self._tracker:
            try:
                self._tracker.update(persons)
            except Exception as e:
                logger.debug(f"追蹤器錯誤: {e}")

        with self._lock:
            self._latest_result.persons = persons
            self._latest_result.person_count = len(persons)
            self._latest_result.objects = objects
            self._latest_result.ground_obstacles = self._last_ground_obstacles
            self._latest_result.ground_obstacle_level = self._last_ground_level
            if self._tracker:
                self._latest_result.tracks = self._tracker.get_tracks_info()
                self._latest_result.unique_person_count = self._tracker.total_unique_persons
                self._latest_result.unreported_count = self._tracker.get_unreported_count()

        # (2) 姿態分析
        fallen_list: List = []
        fallen_persons: List[dict] = []
        crouching_list: List = []
        distressed_list: List = []

        if not all_kps and len(persons) > 0:
            all_kps = self._detect_pose_onnx(frame)

        # A4: 小人物 ROI 放大再推論（每 cycle 最多 1 次，補遠距 / 小 bbox 關鍵點）
        if all_kps and persons and len(all_kps) == len(persons):
            all_kps = self._reinfer_small_persons_roi(frame, persons, all_kps)

        # A2: 幀間 EMA 平滑，壓制關鍵點抖動（影響 fallen/crouching/distressed 穩定度）
        if all_kps and persons and len(all_kps) == len(persons):
            all_kps = self._kp_smoother.smooth(persons, all_kps)

        self._all_keypoints = all_kps

        for i, kps in enumerate(all_kps):
            bbox = persons[i] if i < len(persons) else None
            if is_fallen(kps, bbox):
                fallen_list.append(kps)
                if bbox is not None:
                    fallen_persons.append(dict(bbox))
            elif is_crouching(kps):
                crouching_list.append(kps)
            elif is_distressed(kps, bbox):
                distressed_list.append(kps)

        # 倒地多幀確認（hysteresis：命中 +1，miss -1，避免單幀漏判歸零）
        if fallen_list:
            self._fallen_consecutive = min(self._fallen_consecutive + 1, FALLEN_CONFIRM_FRAMES + 3)
            if self._fallen_consecutive >= FALLEN_CONFIRM_FRAMES:
                if self._fallen_consecutive == FALLEN_CONFIRM_FRAMES:
                    logger.warning(f"確認倒地！({len(fallen_list)} 人)")
            else:
                fallen_list = []
                fallen_persons = []
        else:
            self._fallen_consecutive = max(0, self._fallen_consecutive - 1)
            # 仍在 hysteresis 內 → 視為仍倒地（保留上一幀的 fallen_list）
            if self._fallen_consecutive >= FALLEN_CONFIRM_FRAMES and self._last_fallen_list:
                fallen_list = self._last_fallen_list
                fallen_persons = self._last_fallen_persons
            else:
                fallen_list = []
                fallen_persons = []
        self._last_fallen_list = list(fallen_list)
        self._last_fallen_persons = list(fallen_persons)

        # 蜷縮多幀確認：需 CROUCHING_CONFIRM_FRAMES 連續命中才採用
        if crouching_list:
            self._crouching_consecutive = min(self._crouching_consecutive + 1,
                                              CROUCHING_CONFIRM_FRAMES + 2)
            if self._crouching_consecutive < CROUCHING_CONFIRM_FRAMES:
                crouching_list = []
        else:
            self._crouching_consecutive = max(0, self._crouching_consecutive - 1)
            if self._crouching_consecutive < CROUCHING_CONFIRM_FRAMES:
                crouching_list = []

        # 半躺/側靠多幀確認：判準更弱所以需更多幀（DISTRESSED_CONFIRM_FRAMES）
        if distressed_list:
            self._distressed_consecutive = min(self._distressed_consecutive + 1,
                                               DISTRESSED_CONFIRM_FRAMES + 2)
            if self._distressed_consecutive < DISTRESSED_CONFIRM_FRAMES:
                distressed_list = []
        else:
            self._distressed_consecutive = max(0, self._distressed_consecutive - 1)
            if self._distressed_consecutive < DISTRESSED_CONFIRM_FRAMES:
                distressed_list = []

        fallen_ids = {id(kps) for kps in fallen_list}
        crouching_ids = {id(kps) for kps in crouching_list}
        distressed_ids = {id(kps) for kps in distressed_list}
        pose_score = compute_pose_anomaly_score(all_kps, fallen_ids, crouching_ids, distressed_ids)

        # (2.6) 眼睛開閉偵測（距離夠近時才有意義）
        eye_state = "UNKNOWN"
        if all_kps:
            try:
                eye_state = detect_eye_state(frame, all_kps[0])
            except Exception:
                pass

        # (2.6b) B4: 眨眼率與意識狀態
        blink_rate, consciousness = self._blink_tracker.update(eye_state, now)

        # (2.6c) B2: 微動偵測（bbox 內 vs 背景，抵消 ego-motion）
        micro_motion = self._motion_detector.update(frame, persons)

        # (2.7) rPPG 心率偵測（全局：從所有人臉中選最佳 ROI，含 hysteresis 與遮擋容忍）
        rppg_result = {"bpm": -1.0, "confidence": 0.0, "signal_quality": "UNKNOWN"}
        if self._rppg:
            try:
                rppg_result = self._rppg.process_all_persons(frame, all_kps or [], now)
            except Exception as e:
                logger.debug(f"rPPG 處理錯誤: {e}")

        # (2.8) B5 整合性生命跡象 (五項指標 → 綜合分數 + 中文狀態)
        hr_bpm = rppg_result.get("bpm", -1.0)
        hr_conf = rppg_result.get("confidence", 0.0)
        rr_bpm = rppg_result.get("rr_bpm", -1.0)
        rr_conf = rppg_result.get("rr_confidence", 0.0)
        try:
            blink_warmup = float(self._blink_tracker.warmup_ratio)
        except Exception:
            blink_warmup = 0.0

        vital_score = -1.0
        vital_status = "未知"
        vital_confidence = 0.0
        victim_vital_score = 0.0
        vital_components = {}
        if self._vital_aggregator:
            try:
                vres = self._vital_aggregator.update(
                    hr_bpm=hr_bpm, hr_conf=hr_conf,
                    rr_bpm=rr_bpm, rr_conf=rr_conf,
                    blink_rate=blink_rate if blink_rate >= 0 else -1.0,
                    consciousness=consciousness,
                    micro_motion=float(micro_motion),
                    eye_state=eye_state,
                    blink_warmup=blink_warmup,
                )
                vital_score = vres.vital_score
                vital_status = vres.vital_status
                vital_confidence = getattr(vres, "confidence", 0.0)
                victim_vital_score = vres.victim_vital_score
                vital_components = vres.components or {}
            except Exception as e:
                logger.debug(f"VitalSignsAggregator 失敗: {e}")

        with self._lock:
            # 警報只看真正倒地（已通過 3 幀確認），distressed 算疑似不適但不觸發警報
            self._latest_result.fallen_count = len(fallen_list)
            self._latest_result.fallen_persons = fallen_persons
            self._latest_result.pose_anomaly_score = pose_score
            self._latest_result.eye_state = eye_state
            self._latest_result.heart_rate_bpm = hr_bpm
            self._latest_result.rppg_confidence = hr_conf
            self._latest_result.rppg_signal_quality = rppg_result.get("signal_quality", "UNKNOWN")
            self._latest_result.respiration_rate = rr_bpm
            self._latest_result.resp_confidence = rr_conf
            self._latest_result.rr_buffer_ratio = rppg_result.get("rr_buffer_ratio", 0.0)
            self._latest_result.hr_buffer_ratio = rppg_result.get("hr_buffer_ratio", 0.0)
            self._latest_result.blink_rate_per_min = round(blink_rate, 1) if blink_rate >= 0 else -1.0
            self._latest_result.consciousness_state = consciousness
            self._latest_result.blink_warmup_ratio = blink_warmup
            self._latest_result.micro_motion_score = float(micro_motion)
            # B5: 整合性生命跡象
            self._latest_result.vital_score = vital_score
            self._latest_result.vital_status = vital_status
            self._latest_result.vital_confidence = vital_confidence
            self._latest_result.victim_vital_score = victim_vital_score
            self._latest_result.vital_components = vital_components

        # (3) 繪製標註（所有參數直接傳入，不在繪製中取鎖，避免凍結）
        annotated = frame.copy()
        annotated = self._draw_annotations(
            annotated, persons, fallen_list, crouching_list,
            getattr(self, '_all_keypoints', []),
            distressed_list=distressed_list,
            objects=objects,
            eye_state=eye_state,
            rppg=rppg_result,
        )

        with self._lock:
            self._last_draw_payload = {
                "timestamp": now,
                "persons": list(persons),
                "fallen_kps": list(fallen_list),
                "crouching_kps": list(crouching_list),
                "all_kps": list(getattr(self, '_all_keypoints', [])),
                "distressed_list": list(distressed_list),
                "objects": list(objects),
                "eye_state": eye_state,
                "rppg": dict(rppg_result or {}),
            }
            self._latest_result.annotated_frame = annotated
            self._latest_result.timestamp = now

        return self._latest_result

    # ------------------------------------------------------------------
    # 視覺低障礙分類 (補 LiDAR 車頂盲區)
    # ------------------------------------------------------------------
    def _classify_ground_obstacles(self, objects, h, w):
        """從 YOLO 偵測物件中找出「地面低障礙」並依 bbox 位置/大小估計危險程度。

        原理:LiDAR 在車頂掃描平面,看不到椅腳/桌腳/行李等低物。攝影機可看到。
        沒有深度資訊,改用單眼幾何啟發式:
          - bbox 底邊越靠畫面下緣(y2 比例越大) → 物體離車越近
          - bbox 面積越大 → 越近
          - bbox 中心 x 在畫面中央帶 → 在行進路徑上才算擋路

        回傳 (obstacle_list, level)。level: "CLEAR" / "WARN" / "BRAKE"。
        """
        if not objects or h <= 0 or w <= 0:
            return [], "CLEAR"
        gclasses = getattr(config, 'GROUND_OBSTACLE_CLASSES', {})
        brake_y = float(getattr(config, 'GROUND_OBSTACLE_BRAKE_Y_RATIO', 0.80))
        brake_area = float(getattr(config, 'GROUND_OBSTACLE_BRAKE_AREA', 0.06))
        warn_y = float(getattr(config, 'GROUND_OBSTACLE_WARN_Y_RATIO', 0.62))
        cx_lo = float(getattr(config, 'GROUND_OBSTACLE_CENTER_LO', 0.18))
        cx_hi = float(getattr(config, 'GROUND_OBSTACLE_CENTER_HI', 0.82))
        frame_area = float(h * w)

        result = []
        worst = "CLEAR"
        for o in objects:
            cid = o.get("class_id")
            # 只看 GROUND_OBSTACLE_CLASSES 內的會擋路地面物;手機/手提包等小扁物不算
            if cid not in gclasses:
                continue
            try:
                x1, y1, x2, y2 = o["x1"], o["y1"], o["x2"], o["y2"]
            except KeyError:
                continue
            bw = max(1, x2 - x1)
            bh = max(1, y2 - y1)
            y2_ratio = y2 / float(h)
            cx_ratio = ((x1 + x2) / 2.0) / float(w)
            area_ratio = (bw * bh) / frame_area
            # 不在前方路徑上 → 不擋路,略過
            if not (cx_lo <= cx_ratio <= cx_hi):
                continue
            # 危險分級
            if y2_ratio >= brake_y and area_ratio >= brake_area:
                level = "BRAKE"
            elif y2_ratio >= warn_y:
                level = "WARN"
            else:
                continue  # 太遠不計
            result.append({
                "class_name": o.get("class_name", "obj"),
                "x1": int(x1), "y1": int(y1), "x2": int(x2), "y2": int(y2),
                "level": level,
                "y2_ratio": round(y2_ratio, 2),
                "area_ratio": round(area_ratio, 3),
                "cx_ratio": round(cx_ratio, 2),
            })
            if level == "BRAKE":
                worst = "BRAKE"
            elif level == "WARN" and worst != "BRAKE":
                worst = "WARN"
        return result, worst

    # ------------------------------------------------------------------
    # ONNX 後端
    # ------------------------------------------------------------------
    def _detect_persons_onnx(self, frame) -> tuple:
        """回傳 (persons, objects)"""
        if not self._general_model:
            return [], []
        persons = []
        objects = []

        # 建立偵測類別清單(人 + 救援物件 + 地面低障礙)
        detect_classes = [config.PERSON_CLASS_ID]
        if config.OBJECT_DETECTION_ENABLED:
            detect_classes.extend(config.RESCUE_OBJECT_CLASSES.keys())
        _ground_classes = getattr(config, 'GROUND_OBSTACLE_CLASSES', {}) \
            if getattr(config, 'GROUND_OBSTACLE_ENABLED', False) else {}
        if _ground_classes:
            detect_classes.extend(_ground_classes.keys())
        # 去重(rescue 與 ground 理論上無交集,保險起見)
        detect_classes = list(dict.fromkeys(detect_classes))

        try:
            results = self._general_model(
                frame, conf=config.GENERAL_CONFIDENCE,
                classes=detect_classes,
                imgsz=config.YOLO_GENERAL_INFER_SIZE, verbose=False
            )
            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    cls_id = int(box.cls[0])
                    conf = round(float(box.conf[0]), 2)

                    if cls_id == config.PERSON_CLASS_ID:
                        persons.append({
                            "x1": int(x1), "y1": int(y1),
                            "x2": int(x2), "y2": int(y2),
                            "conf": conf
                        })
                    elif config.OBJECT_DETECTION_ENABLED and cls_id in config.RESCUE_OBJECT_CLASSES:
                        objects.append({
                            "x1": int(x1), "y1": int(y1),
                            "x2": int(x2), "y2": int(y2),
                            "conf": conf,
                            "class_id": cls_id,
                            "class_name": config.RESCUE_OBJECT_CLASSES[cls_id],
                        })
                    elif cls_id in _ground_classes:
                        objects.append({
                            "x1": int(x1), "y1": int(y1),
                            "x2": int(x2), "y2": int(y2),
                            "conf": conf,
                            "class_id": cls_id,
                            "class_name": _ground_classes[cls_id],
                        })
        except Exception as e:
            logger.error(f"ONNX 偵測錯誤: {e}")
        return persons, objects

    def _detect_pose_onnx(self, frame) -> List[np.ndarray]:
        if not self._pose_model:
            return []
        all_kps = []
        try:
            results = self._pose_model(
                frame, conf=config.POSE_CONFIDENCE,
                imgsz=config.YOLO_POSE_INFER_SIZE, verbose=False
            )
            for r in results:
                if r.keypoints is not None:
                    for kp in r.keypoints.data:
                        all_kps.append(kp.cpu().numpy())
        except Exception as e:
            logger.error(f"ONNX 姿態偵測錯誤: {e}")
        return all_kps

    def _reinfer_small_persons_roi(self, frame, persons: list,
                                   all_kps: list) -> list:
        """
        A4: 對「小人物」（bbox 很小且關鍵點少）做 ROI 放大再推論。
        限制每 cycle 最多 1 個，避免 CPU 爆表。
        """
        if not getattr(config, 'ROI_REINFER_ENABLED', True):
            return all_kps
        if not self._pose_model or not persons or not all_kps:
            return all_kps
        if len(persons) != len(all_kps):
            return all_kps

        max_h = float(getattr(config, 'ROI_REINFER_BBOX_MAX_H_PX', 150))
        min_good = int(getattr(config, 'ROI_REINFER_MIN_GOOD_KPS', 6))
        pad_ratio = float(getattr(config, 'ROI_REINFER_PAD_RATIO', 0.15))
        conf_gate = float(getattr(config, 'ROI_REINFER_CONF_GATE', 0.5))

        # 挑出需要再推論的人（小 + 關鍵點少），選 bbox 最小者（資訊最不足）
        target_idx = -1
        target_h = 9999
        for i, (p, kps) in enumerate(zip(persons, all_kps)):
            bh = p["y2"] - p["y1"]
            if bh > max_h:
                continue
            good_kps = int(np.sum(np.asarray(kps)[:, 2] > conf_gate))
            if good_kps >= min_good:
                continue
            if bh < target_h:
                target_h = bh
                target_idx = i

        if target_idx < 0:
            return all_kps

        p = persons[target_idx]
        H, W = frame.shape[:2]
        bw = p["x2"] - p["x1"]
        bh = p["y2"] - p["y1"]
        pad_x = int(bw * pad_ratio)
        pad_y = int(bh * pad_ratio)
        cx1 = max(0, p["x1"] - pad_x)
        cy1 = max(0, p["y1"] - pad_y)
        cx2 = min(W, p["x2"] + pad_x)
        cy2 = min(H, p["y2"] + pad_y)
        if cx2 - cx1 < 40 or cy2 - cy1 < 40:
            return all_kps

        crop = frame[cy1:cy2, cx1:cx2]
        try:
            results = self._pose_model(
                crop, conf=config.POSE_CONFIDENCE,
                imgsz=config.YOLO_POSE_INFER_SIZE, verbose=False
            )
        except Exception as e:
            logger.debug(f"ROI 再推論失敗: {e}")
            return all_kps

        new_kps = None
        for r in results:
            if r.keypoints is not None and len(r.keypoints.data) > 0:
                # 選面積最大（最可能是該 crop 的主角）
                best_score = 0.0
                best = None
                for kp in r.keypoints.data:
                    k = kp.cpu().numpy()
                    score = float(np.sum(k[:, 2]))
                    if score > best_score:
                        best_score = score
                        best = k
                new_kps = best
                break

        if new_kps is None:
            return all_kps

        # 把 crop 內座標轉回原 frame 座標
        new_kps = new_kps.copy()
        new_kps[:, 0] += cx1
        new_kps[:, 1] += cy1

        # 合併策略：逐點取較高 confidence
        merged = np.asarray(all_kps[target_idx]).copy()
        m = min(len(merged), len(new_kps))
        for i in range(m):
            if new_kps[i, 2] > merged[i, 2]:
                merged[i] = new_kps[i]

        new_good = int(np.sum(merged[:, 2] > conf_gate))
        if new_good > int(np.sum(np.asarray(all_kps[target_idx])[:, 2] > conf_gate)):
            logger.debug(f"ROI 再推論提升：第 {target_idx} 人 "
                         f"bbox={bh:.0f}px 關鍵點 → {new_good}")
            out = list(all_kps)
            out[target_idx] = merged
            return out
        return all_kps

    def _kps_to_persons(self, all_kps, h, w) -> list:
        """從 pose keypoints 推算人員邊界框（免跑 general 模型）"""
        persons = []
        for kps in all_kps:
            visible = kps[kps[:, 2] > 0.4]
            # 至少 5 個可見關鍵點才算人（減少物體誤判）
            if len(visible) < 5:
                continue
            x1 = max(0, int(visible[:, 0].min()) - 10)
            y1 = max(0, int(visible[:, 1].min()) - 10)
            x2 = min(w, int(visible[:, 0].max()) + 10)
            y2 = min(h, int(visible[:, 1].max()) + 10)
            # 邊界框太小 → 不是真人（噪點或小物件）
            if (x2 - x1) < 40 or (y2 - y1) < 60:
                continue
            avg_conf = float(visible[:, 2].mean())
            if avg_conf < 0.45:
                continue
            persons.append({
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "conf": round(avg_conf, 2)
            })
        return persons

    # ------------------------------------------------------------------
    # 繪製標註
    # ------------------------------------------------------------------
    @property
    def tracker(self):
        """暴露追蹤器供外部存取（標記已通報等）"""
        return self._tracker

    def _draw_annotations(self, frame, persons,
                           fallen_kps=None, crouching_kps=None, all_kps=None,
                           distressed_list=None, objects=None,
                           eye_state="UNKNOWN", rppg=None):
        fallen_kps = fallen_kps or []
        crouching_kps = crouching_kps or []
        distressed_list = distressed_list or []
        all_kps = all_kps or []
        objects = objects or []

        # 預建 id 集合，O(1) 查詢取代 O(n) array_equal
        fallen_id_set = {id(kps) for kps in fallen_kps}
        crouching_id_set = {id(kps) for kps in crouching_kps}
        distressed_id_set = {id(kps) for kps in distressed_list}

        for kps in all_kps:
            is_this_fallen = id(kps) in fallen_id_set
            is_this_crouching = id(kps) in crouching_id_set
            is_this_distressed = id(kps) in distressed_id_set
            is_abnormal = is_this_fallen or is_this_crouching or is_this_distressed

            if is_abnormal:
                color = (0, 0, 255)       # 紅色：所有異常姿態
            else:
                color = (0, 255, 128)     # 綠色：正常

            for i, j in SKELETON:
                if kps[i][2] > 0.3 and kps[j][2] > 0.3:
                    # 手臂骨架用更粗的線條 + 亮色突顯
                    is_arm = (i in (5, 6, 7, 8, 9, 10) and j in (5, 6, 7, 8, 9, 10))
                    line_color = (0, 255, 255) if is_arm else color
                    line_w = 3 if is_arm else 2
                    cv2.line(frame,
                             (int(kps[i][0]), int(kps[i][1])),
                             (int(kps[j][0]), int(kps[j][1])),
                             line_color, line_w)
            valid_pts = []
            for idx, kp in enumerate(kps):
                if kp[2] > 0.3:
                    pt = (int(kp[0]), int(kp[1]))
                    valid_pts.append(pt)
                    # 手腕用大圓圈 + 標籤突顯
                    if idx in (9, 10):
                        cv2.circle(frame, pt, 8, (0, 255, 255), 2)
                        cv2.circle(frame, pt, 3, (0, 255, 255), -1)
                    else:
                        cv2.circle(frame, pt, 4, color, -1)

            if is_abnormal and valid_pts:
                top_pt = min(valid_pts, key=lambda p: p[1])
                if is_this_fallen:
                    label = "FALLEN!"
                elif is_this_crouching:
                    label = "CROUCHING"
                else:
                    label = "DISTRESSED"
                cv2.putText(frame, label, (top_pt[0] - 35, top_pt[1] - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        for p in persons:
            cv2.rectangle(frame, (p["x1"], p["y1"]), (p["x2"], p["y2"]), (0, 255, 0), 2)
            cv2.putText(frame, f"Person {p['conf']:.0%}", (p["x1"], p["y1"] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # OSD 資訊（不取鎖，直接用已知數值）
        n_persons = len(persons)
        n_abnormal = len(fallen_kps) + len(crouching_kps) + len(distressed_list)
        eye_tag = f"  Eye:{eye_state}" if eye_state != "UNKNOWN" else ""
        info = f"[ONNX] P:{n_persons}  F:{n_abnormal}  {eye_tag}"
        cv2.putText(frame, info, (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        if eye_state == "CLOSED":
            cv2.putText(frame, "EYES CLOSED!", (10, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # rPPG 心率顯示
        if rppg and rppg.get("valid"):
            bpm = rppg["bpm"]
            conf = rppg["confidence"]
            hr_color = (0, 255, 0) if 50 <= bpm <= 100 else (0, 0, 255)
            cv2.putText(frame, f"HR: {bpm:.0f} BPM ({conf:.0%})", (10, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, hr_color, 2)
        elif rppg and rppg.get("signal_quality") == "WEAK":
            cv2.putText(frame, "HR: measuring...", (10, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (128, 128, 128), 2)

        # 繪製追蹤 ID
        if self._tracker:
            for track in self._tracker.active_tracks:
                b = track.bbox
                label = f"#{track.track_id}"
                if track.reported:
                    label += " [R]"
                cv2.putText(frame, label, (b["x1"], b["y2"] + 16),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 2)

        # 繪製偵測到的災區物件（直接用傳入參數，不取鎖）
        for obj in objects:
            obj_color = (255, 165, 0)
            cv2.rectangle(frame, (obj["x1"], obj["y1"]), (obj["x2"], obj["y2"]),
                          obj_color, 2)
            label = f"{obj['class_name']} {obj['conf']:.0%}"
            cv2.putText(frame, label, (obj["x1"], obj["y1"] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, obj_color, 2)

        return frame

    def cleanup(self):
        logger.info("釋放 AI 模型資源...")
        self._general_model = None
        self._pose_model = None
