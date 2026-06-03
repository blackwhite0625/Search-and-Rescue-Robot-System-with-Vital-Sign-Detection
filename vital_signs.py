"""
搜救機器人 — 生命跡象綜合分析模組
==================================
把分散的心率、呼吸率、眨眼率、意識狀態、微動分數五項指標,
整合成一個「綜合生命跡象分數」與直觀狀態字串,
並提供 fusion 用的「受困者風險反向分數」。

設計理念:
  原本 fusion.py 的 vital_signs 只看心率 (50/130 BPM 異常),
  忽略了呼吸異常、失意識、無微動等更強的訊號。
  本模組以「正常生命跡象」為基準 1.0,各維度偏離程度線性扣分,
  再以加權平均產出最終分數,UI 與 fusion 都用同一份結果。

分數定義:
  vital_score (0~1, -1=未知):
    1.0  完全正常 (各項都在生理正常範圍)
    0.7  輕微異常 (1-2 項略偏)
    0.4  明顯微弱 (信號弱或多項偏離)
    0.0  無生命跡象 / 完全無反應
   -1.0  資料不足以判斷

  vital_status (字串):
    "正常" / "微弱" / "失去意識" / "無反應" / "建立中" / "未知"

  victim_vital_score (0~1, 給 fusion 用):
    越異常分數越高,代表「越像受困需救援」
    1.0 = 失意識或無反應, 0 = 完全正常無事
"""

import time
import logging
from collections import deque
from dataclasses import dataclass

logger = logging.getLogger("rescue.vital_signs")


@dataclass
class VitalSignsResult:
    """生命跡象綜合結果"""
    vital_score: float = -1.0          # 0-1 綜合健康程度,-1 = 未知
    vital_status: str = "未知"          # 中文狀態
    confidence: float = 0.0            # 0-1 綜合可信度
    victim_vital_score: float = 0.0    # 0-1 fusion 用的風險反向分數
    components: dict = None             # 各維度子分數(供 UI 顯示)
    contributors: int = 0               # 有效輸入維度數


class VitalSignsAggregator:
    """
    多源生命跡象整合器。

    輸入:
      - heart_rate_bpm (float, -1=未量測)
      - rppg_confidence (0-1)
      - respiration_rate (float, -1=未量測)
      - resp_confidence (0-1)
      - blink_rate_per_min (float, -1=未量測)
      - consciousness_state ("AWAKE"/"DROWSY"/"UNCONSCIOUS"/"UNKNOWN")
      - micro_motion_score (0-1)
      - eye_state ("OPEN"/"CLOSED"/"UNKNOWN")  ← 輔助

    用法:
      agg = VitalSignsAggregator()
      result = agg.update(hr=72, hr_conf=0.7, rr=15, rr_conf=0.5,
                          blink=18, consciousness="AWAKE", micro=0.2,
                          eye_state="OPEN")
      result.vital_score      # 0-1
      result.vital_status     # "正常"
      result.victim_vital_score   # 給 fusion 用 (0-1, 異常程度)
    """

    # 各維度權重 (有效輸入才計入,自動正規化)
    WEIGHTS = {
        "hr": 0.32,             # 心率最權威
        "rr": 0.22,             # 呼吸次要
        "consciousness": 0.28,  # 意識狀態強信號
        "motion": 0.18,         # 微動輔助
    }

    # 信心度門檻:低於此值視為「未量測」
    MIN_HR_CONF = 0.30
    MIN_RR_CONF = 0.30

    # 平滑歷史長度
    HISTORY_LEN = 8

    def __init__(self):
        self._history = deque(maxlen=self.HISTORY_LEN)
        self._last_result = VitalSignsResult()

    # ──────────────────────────────────────────────
    # 公開 API
    # ──────────────────────────────────────────────
    def update(self, hr_bpm: float = -1.0, hr_conf: float = 0.0,
               rr_bpm: float = -1.0, rr_conf: float = 0.0,
               blink_rate: float = -1.0, consciousness: str = "UNKNOWN",
               micro_motion: float = 0.0,
               eye_state: str = "UNKNOWN",
               blink_warmup: float = 1.0) -> VitalSignsResult:
        """整合一次性輸入並回傳結果(帶歷史平滑)"""
        # 1. 各維度健康分數(0-1, None = 未量測不計入)
        hr_score = self._heart_rate_score(hr_bpm, hr_conf)
        rr_score = self._resp_rate_score(rr_bpm, rr_conf)
        cons_score = self._consciousness_score(consciousness, blink_rate,
                                               blink_warmup, eye_state)
        motion_score = self._motion_score(micro_motion)

        # 2. 加權平均(自動跳過 None)
        contributions = [
            ("hr", hr_score),
            ("rr", rr_score),
            ("consciousness", cons_score),
            ("motion", motion_score),
        ]
        valid = [(k, s) for k, s in contributions if s is not None]
        contributors = len(valid)

        if not valid:
            # 完全無資料
            result = VitalSignsResult(
                vital_score=-1.0,
                vital_status="未知",
                confidence=0.0,
                victim_vital_score=0.0,
                components={"hr": None, "rr": None,
                            "consciousness": None, "motion": None},
                contributors=0,
            )
            self._last_result = result
            return result

        total_w = sum(self.WEIGHTS[k] for k, _ in valid)
        raw_score = sum(self.WEIGHTS[k] * s for k, s in valid) / total_w

        # 3. 歷史平滑(EMA 風格,新值權重 0.5)
        self._history.append(raw_score)
        if len(self._history) >= 2:
            smoothed = 0.5 * raw_score + 0.5 * (sum(list(self._history)[:-1])
                                                / max(1, len(self._history) - 1))
        else:
            smoothed = raw_score

        # 4. 狀態分類
        status = self._classify_status(smoothed, consciousness,
                                       hr_score, rr_score,
                                       cons_score, motion_score,
                                       contributors)

        # 5. fusion 用的風險反向分數
        victim_risk = self._compute_victim_risk(
            smoothed, consciousness, hr_bpm, hr_conf,
            rr_bpm, rr_conf, micro_motion, contributors,
            hr_score, rr_score, cons_score
        )
        confidence = self._compute_confidence(
            hr_score, rr_score, cons_score, motion_score,
            hr_conf, rr_conf, consciousness, blink_warmup,
            micro_motion, contributors
        )

        result = VitalSignsResult(
            vital_score=round(smoothed, 3),
            vital_status=status,
            confidence=round(confidence, 3),
            victim_vital_score=round(victim_risk, 3),
            components={
                "hr": round(hr_score, 2) if hr_score is not None else None,
                "rr": round(rr_score, 2) if rr_score is not None else None,
                "consciousness": round(cons_score, 2) if cons_score is not None else None,
                "motion": round(motion_score, 2) if motion_score is not None else None,
            },
            contributors=contributors,
        )
        self._last_result = result
        return result

    @property
    def latest(self) -> VitalSignsResult:
        return self._last_result

    def reset(self):
        self._history.clear()
        self._last_result = VitalSignsResult()

    # ──────────────────────────────────────────────
    # 各維度分數函數
    # ──────────────────────────────────────────────
    def _heart_rate_score(self, bpm: float, conf: float):
        """心率分數:60-100 正常 = 1.0,越偏離越低。
        無讀數或信心過低 → None (不計入)。"""
        if bpm <= 0 or conf < self.MIN_HR_CONF:
            return None
        if 60 <= bpm <= 100:
            return 1.0
        if 50 <= bpm < 60 or 100 < bpm <= 120:
            return 0.7
        if 40 <= bpm < 50 or 120 < bpm <= 140:
            return 0.4
        # 極端 (<40 或 >140)
        return 0.1

    def _resp_rate_score(self, rr_bpm: float, rr_conf: float):
        """呼吸率分數:成人正常 12-20 次/分。"""
        if rr_bpm <= 0 or rr_conf < self.MIN_RR_CONF:
            return None
        if 12 <= rr_bpm <= 20:
            return 1.0
        if 10 <= rr_bpm < 12 or 20 < rr_bpm <= 25:
            return 0.7
        if 8 <= rr_bpm < 10 or 25 < rr_bpm <= 30:
            return 0.4
        return 0.15

    def _consciousness_score(self, state: str, blink_rate: float,
                             warmup: float, eye_state: str):
        """意識分數:AWAKE=1, DROWSY=0.5, UNCONSCIOUS=0。
        warmup < 0.5 表示資料仍在累積,信心降低。"""
        if state == "AWAKE":
            base = 1.0
        elif state == "DROWSY":
            base = 0.5
        elif state == "UNCONSCIOUS":
            return 0.0   # 失意識直接 0,不打折
        else:
            # UNKNOWN:看 eye_state 救一下
            if eye_state == "OPEN":
                return 0.7   # 眼睛是開的,至少有意識
            if eye_state == "CLOSED":
                return 0.3   # 眼閉但不確定多久
            return None

        # warmup 期間信心打折(但不歸零,避免一直 None)
        if warmup < 0.5:
            return base * (0.5 + warmup)   # warmup=0 → 0.5x, warmup=0.5 → 1.0x
        return base

    def _motion_score(self, micro_motion: float):
        """微動分數:有微動 = 有生命跡象。
        分階閾值:>=0.3 滿分, 0.1-0.3 中等, 0.02-0.1 微弱, <0.02 視為靜止。"""
        if micro_motion is None or micro_motion < 0:
            return None
        if micro_motion >= 0.3:
            return 1.0
        if micro_motion >= 0.1:
            return 0.7
        if micro_motion >= 0.02:
            return 0.4
        # 完全靜止(可能無生命跡象)
        return 0.0

    # ──────────────────────────────────────────────
    # 狀態分類與風險評估
    # ──────────────────────────────────────────────
    def _classify_status(self, score: float, consciousness: str,
                         hr_score, rr_score, cons_score, motion_score,
                         contributors: int) -> str:
        """根據綜合分數與意識狀態決定 UI 顯示的中文狀態。"""
        # 強信號:意識直接優先
        if consciousness == "UNCONSCIOUS":
            return "失去意識"

        # 資料完全不足
        if contributors == 0:
            return "未知"
        # 只有 motion 單一訊號 → 不足以判定(微動本身不夠權威)
        if (contributors == 1 and hr_score is None and
                rr_score is None and cons_score is None):
            return "建立中"

        # 微動 + 心率都極低 → 無反應 (即使有讀數但都很差)
        if (motion_score is not None and motion_score <= 0.0 and
                hr_score is not None and hr_score <= 0.1):
            return "無反應"

        # 一般分類 (門檻略寬,避免單一維度未滿分就降級)
        if score >= 0.75:
            return "正常"
        if score >= 0.40:
            return "微弱"
        if score >= 0.15:
            return "微弱"   # 統一為微弱,避免過多分類
        return "無反應"

    def _compute_confidence(self, hr_score, rr_score, cons_score, motion_score,
                            hr_conf: float, rr_conf: float, consciousness: str,
                            blink_warmup: float, micro_motion: float,
                            contributors: int) -> float:
        """估計生命跡象判斷可信度。

        HR/RR 直接使用 rPPG 信號信心度；意識與微動是搜救線索，
        但單獨使用時不應過度自信，因此會依資料來源數量封頂。
        """
        parts = []
        if hr_score is not None:
            parts.append(("hr", max(0.0, min(1.0, float(hr_conf or 0.0)))))
        if rr_score is not None:
            parts.append(("rr", max(0.0, min(1.0, float(rr_conf or 0.0)))))
        if cons_score is not None:
            if consciousness in ("AWAKE", "DROWSY", "UNCONSCIOUS"):
                c = max(0.45, min(1.0, float(blink_warmup or 0.0)))
            else:
                c = 0.35
            parts.append(("consciousness", c))
        if motion_score is not None:
            mm = max(0.0, min(1.0, float(micro_motion or 0.0)))
            parts.append(("motion", min(0.55, 0.25 + mm)))

        if not parts:
            return 0.0

        total_w = sum(self.WEIGHTS[k] for k, _ in parts)
        conf = sum(self.WEIGHTS[k] * c for k, c in parts) / max(total_w, 1e-6)
        keys = {k for k, _ in parts}

        if contributors <= 1:
            conf = min(conf, 0.25 if keys == {"motion"} else 0.55)
        elif contributors == 2 and keys.issubset({"consciousness", "motion"}):
            conf = min(conf, 0.65)
        elif contributors == 2:
            conf = min(conf, 0.75)
        return max(0.0, min(1.0, conf))

    # ──────────────────────────────────────────────
    # 文字摘要 (供 Telegram / TTS / 事件紀錄共用)
    # ──────────────────────────────────────────────
    @staticmethod
    def format_telegram_summary(result, eye_state: str = "UNKNOWN") -> str:
        """產生 Telegram / 事件紀錄用的多行摘要(含中文標籤與單位)。
        result: VitalSignsResult 物件;失敗則回傳簡短訊息。"""
        if result is None or result.vital_score < 0:
            return "生命跡象: 資料建立中…"
        comp = result.components or {}
        lines = [
            f"━━━ 生命跡象 ━━━",
            f"綜合狀態: {result.vital_status}  (分數 {result.vital_score:.2f}/1.00)",
        ]
        lines.append(f"可信度: {result.confidence:.0%}")
        # 各維度子分數(只列出有效項)
        if comp.get('hr') is not None:
            lines.append(f"  心率分: {comp['hr']:.2f}")
        if comp.get('rr') is not None:
            lines.append(f"  呼吸分: {comp['rr']:.2f}")
        if comp.get('consciousness') is not None:
            lines.append(f"  意識分: {comp['consciousness']:.2f}")
        if comp.get('motion') is not None:
            lines.append(f"  微動分: {comp['motion']:.2f}")
        if eye_state and eye_state != "UNKNOWN":
            lines.append(f"  眼睛: {eye_state}")
        lines.append(f"風險評估: {result.victim_vital_score:.2f}/1.00")
        return "\n".join(lines)

    @staticmethod
    def format_telegram_full(vital_status: str, vital_score: float,
                             hr_bpm: float, hr_conf: float,
                             rr_bpm: float, rr_conf: float,
                             blink_rate: float, consciousness: str,
                             micro_motion: float, eye_state: str,
                             vital_confidence: float = 0.0) -> str:
        """從原始指標欄位產生完整 Telegram 訊息(供 app.py 直接呼叫)。
        若 vital_status 仍未知,回傳簡短訊息。"""
        if not vital_status or vital_status == "未知":
            if vital_confidence > 0:
                return f"生命跡象: 資料建立中… (可信度 {vital_confidence:.0%})"
            return "生命跡象: 資料建立中…"
        lines = [f"━━━ 生命跡象 ━━━",
                 f"綜合狀態: {vital_status}"]
        if vital_score >= 0:
            lines[1] += f"  (分數 {vital_score:.2f}/1.00)"
        lines.append(f"可信度: {vital_confidence:.0%}")
        if hr_bpm > 0 and hr_conf >= 0.3:
            lines.append(f"  心率: {hr_bpm:.0f} BPM (信心 {hr_conf:.0%})")
        if rr_bpm > 0 and rr_conf >= 0.3:
            lines.append(f"  呼吸: {rr_bpm:.0f} 次/分 (信心 {rr_conf:.0%})")
        if consciousness and consciousness != "UNKNOWN":
            zh = {"AWAKE": "清醒", "DROWSY": "嗜睡", "UNCONSCIOUS": "失去意識"}.get(consciousness, consciousness)
            lines.append(f"  意識: {zh}")
        if blink_rate > 0:
            lines.append(f"  眨眼率: {blink_rate:.0f} 次/分")
        if micro_motion is not None:
            mm_desc = "正常" if micro_motion >= 0.1 else ("微弱" if micro_motion >= 0.02 else "靜止")
            lines.append(f"  微動: {micro_motion:.2f} ({mm_desc})")
        if eye_state and eye_state != "UNKNOWN":
            lines.append(f"  眼睛: {eye_state}")
        return "\n".join(lines)

    @staticmethod
    def tts_text(vital_status: str, consciousness: str = "UNKNOWN") -> tuple:
        """根據生命跡象狀態產出 TTS 警報文字 (中文, 英文)。
        刻意保持「短」:每次警報需暫停相機讓出 USB 頻寬,文字越短相機停擺越短,
        YOLO 辨識中斷時間越少。關鍵資訊(失意識/無反應/倒地)放最前面。"""
        # 失意識:最高緊急
        if vital_status == "失去意識" or consciousness == "UNCONSCIOUS":
            return ("緊急! 發現失去意識傷患!",
                    "Critical! Unconscious victim!")
        # 無反應:幾乎等同失意識
        if vital_status == "無反應":
            return ("緊急! 發現無反應傷患!",
                    "Critical! Unresponsive victim!")
        # 微弱:有生命跡象但異常
        if vital_status == "微弱":
            return ("警報! 發現傷患 生命微弱!",
                    "Alert! Victim, weak vitals!")
        # 正常 vital 但倒地
        if vital_status == "正常":
            return ("發現人員倒地!",
                    "Person down detected!")
        # 預設(未知/建立中)
        return ("緊急! 發現人員倒地!",
                "Emergency! Person down!")

    def _compute_victim_risk(self, score: float, consciousness: str,
                             hr_bpm: float, hr_conf: float,
                             rr_bpm: float, rr_conf: float,
                             micro_motion: float, contributors: int,
                             hr_score, rr_score, cons_score) -> float:
        """fusion 用的反向風險分數:越異常越高 (0-1)。

        規則:
          - 失意識 → 1.0 (最高風險,絕對需救援)
          - 心率極端 (<40 或 >140) → 至少 0.85
          - 完全靜止 (micro_motion≈0) + 心率讀數差 → 至少 0.65
          - 一般情況: 1 - vital_score
          - 「只有 motion 單一訊號」不夠權威,直接回 0 避免誤觸發 fusion
        """
        # 失意識壓倒一切
        if consciousness == "UNCONSCIOUS":
            return 1.0

        # 無資料 → 風險未知,給 0 不誤觸發
        if score < 0:
            return 0.0

        # 沒有任何「權威訊號」(hr/rr/consciousness 都 None),只有 motion → 不足以判定
        if hr_score is None and rr_score is None and cons_score is None:
            return 0.0

        risk = 1.0 - score

        # 心率極端加重
        if hr_bpm > 0 and hr_conf >= self.MIN_HR_CONF:
            if hr_bpm < 40 or hr_bpm > 140:
                risk = max(risk, 0.85)
            elif hr_bpm < 50 or hr_bpm > 130:
                risk = max(risk, 0.65)

        # 完全靜止 + 有心率讀數 → 中等風險 (可能是昏迷)
        if (micro_motion is not None and micro_motion < 0.02 and
                hr_conf >= self.MIN_HR_CONF):
            risk = max(risk, 0.6)

        return min(risk, 1.0)
