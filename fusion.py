"""
搜救機器人 — 多模態風險融合模組
================================
VictimScore = 0.40×Person + 0.18×Pose + 0.18×Audio + 0.09×Motion + 0.05×Distance + 0.10×VitalSigns
"""

import logging
from dataclasses import dataclass

import config

logger = logging.getLogger("rescue.fusion")


@dataclass
class FusionInput:
    """融合輸入資料"""
    person_detected: bool = False    # 是否偵測到人體
    person_count: int = 0            # 人數
    pose_anomaly_score: float = 0.0  # 姿態異常分數 (0~1)
    audio_help_score: float = 0.0    # 呼救聲分數 (0~1)
    audio_knock: bool = False        # 是否偵測到敲擊聲
    motion_response: float = 0.0     # 互動回應分數 (0~1)（HRI 模組提供）
    distance_cm: float = -1          # LiDAR 前方距離 (cm)
    heart_rate_bpm: float = -1.0     # rPPG 心率 (-1 = 未量測)
    rppg_confidence: float = 0.0     # rPPG 信號品質 (0~1)
    # B5: 整合性生命跡象風險分數 (心率+呼吸+眨眼+意識+微動 → 0-1, 越異常越高)
    # >= 0 時優先採用,取代舊的「僅看心率」邏輯;< 0 視為無資料 fallback 舊邏輯
    victim_vital_score: float = -1.0


@dataclass
class FusionResult:
    """融合輸出結果"""
    victim_score: float = 0.0
    risk_level: str = "LOW"          # LOW / SUSPECT / HIGH
    components: dict = None          # 各分量詳細分數


class VictimFusion:
    """多模態受困者風險評估"""

    def __init__(self):
        self.weights = config.VICTIM_SCORE_WEIGHTS
        self._suspect_threshold = config.VICTIM_SUSPECT_THRESHOLD
        self._high_risk_threshold = config.VICTIM_HIGH_RISK_THRESHOLD
        logger.info(f"✅ 多模態融合模組初始化 | 權重: {self.weights}")

    def compute(self, inp: FusionInput) -> FusionResult:
        """計算 VictimScore"""
        # 人體分數：只有偵測到姿態異常（倒地/蜷縮）時才計分
        # 正常站立的人不應觸發搜救流程
        if inp.pose_anomaly_score >= 0.5:
            person_score = 1.0   # 倒地/蜷縮者
        elif inp.person_detected and inp.pose_anomaly_score > 0:
            person_score = 0.4   # 有人但姿態只是輕微異常
        else:
            person_score = 0.0   # 正常人或無人

        # 姿態分數：直接使用 detector 的 pose_anomaly_score
        pose_score = min(inp.pose_anomaly_score, 1.0)

        # 音訊分數：呼救聲 + 敲擊聲的複合分數
        audio_score = inp.audio_help_score
        if inp.audio_knock:
            audio_score = max(audio_score, 0.7)  # 敲擊聲至少 0.7
        audio_score = min(audio_score, 1.0)

        # 互動回應分數：由 HRI 模組主動詢問後提供
        motion_score = min(inp.motion_response, 1.0)

        # 距離安全分數：越近越高（搜救場景）
        if inp.distance_cm < 0:
            distance_score = 0.0  # 無讀數
        elif inp.distance_cm < 30:
            distance_score = 1.0  # 非常近
        elif inp.distance_cm < 80:
            distance_score = 0.6  # 適中距離
        else:
            distance_score = 0.2  # 遠距

        # 生命跡象分數
        # 優先採用 B5 整合分數 (心率+呼吸+眨眼+意識+微動 加權融合);
        # 無資料時 fallback 舊「僅看心率」邏輯,保持向下相容。
        if inp.victim_vital_score >= 0:
            vital_score = float(inp.victim_vital_score)
        else:
            vital_score = 0.0
            if inp.rppg_confidence >= 0.5 and inp.heart_rate_bpm > 0:
                bpm = inp.heart_rate_bpm
                if bpm < 50 or bpm > 130:
                    vital_score = 0.7   # 異常心率（過慢/過快）
                elif bpm < 60 or bpm > 100:
                    vital_score = 0.3   # 略偏正常範圍
                else:
                    vital_score = 0.0   # 正常心率

        # 加權融合
        victim_score = (
            self.weights["person"]   * person_score +
            self.weights["pose"]     * pose_score +
            self.weights["audio"]    * audio_score +
            self.weights["motion"]   * motion_score +
            self.weights["distance"] * distance_score +
            self.weights.get("vital_signs", 0.10) * vital_score
        )
        victim_score = round(min(victim_score, 1.0), 3)

        # 風險等級
        if victim_score >= self._high_risk_threshold:
            risk_level = "HIGH"
        elif victim_score >= self._suspect_threshold:
            risk_level = "SUSPECT"
        else:
            risk_level = "LOW"

        components = {
            "person": round(person_score, 2),
            "pose": round(pose_score, 2),
            "audio": round(audio_score, 2),
            "motion": round(motion_score, 2),
            "distance": round(distance_score, 2),
            "vital_signs": round(vital_score, 2),
        }

        return FusionResult(
            victim_score=victim_score,
            risk_level=risk_level,
            components=components,
        )
