"""
搜救機器人 — rPPG 遠端光體積描記術模組
========================================
透過攝影機分析臉部皮膚綠色通道的微小變化提取心率。
原理：血液流動造成皮膚顏色週期性變化，與心跳同步。

流程：
  1. 從 YOLOv8-pose 臉部關鍵點 (0-4) 定位臉部 ROI
  2. 提取綠色通道空間平均值（每幀一個數值）
  3. 累積滾動緩衝 (~4 秒)
  4. 帶通濾波 (0.7-3.5 Hz) + FFT → 心率 BPM
"""

import time
import logging
import numpy as np
from collections import deque

import config

logger = logging.getLogger("rescue.rppg")

# scipy 為選用依賴
try:
    from scipy import signal as sp_signal
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    logger.warning("scipy 未安裝，rPPG 功能停用（pip install scipy）")


class rPPGDetector:
    """
    遠端光體積描記術心率偵測器。
    每幀呼叫 process_frame()，內部累積信號後定期計算心率。
    """

    def __init__(self):
        fps = config.CAMERA_FPS
        buf_len = int(config.RPPG_BUFFER_SECONDS * fps)

        self._buffer = deque(maxlen=buf_len)      # 綠色通道時間序列
        self._timestamps = deque(maxlen=buf_len)   # 對應時間戳

        # B1: 呼吸率偵測專用長緩衝（共用同一 green 值）
        self._resp_enabled = bool(getattr(config, 'RPPG_RESP_ENABLED', True))
        resp_buf_len = int(getattr(config, 'RPPG_RESP_BUFFER_SECONDS', 15.0) * fps)
        self._resp_buffer = deque(maxlen=resp_buf_len)
        self._resp_timestamps = deque(maxlen=resp_buf_len)
        self._resp_min_buffer = int(fps * getattr(config, 'RPPG_RESP_MIN_BUFFER_SEC', 8.0))
        self._resp_confidence_min = float(getattr(config, 'RPPG_RESP_CONFIDENCE_MIN', 0.30))
        self._resp_low_hz = float(getattr(config, 'RPPG_RESP_LOW_HZ', 0.1))
        self._resp_high_hz = float(getattr(config, 'RPPG_RESP_HIGH_HZ', 0.5))
        self._last_resp_result = {
            "rr_bpm": -1.0,
            "rr_confidence": 0.0,
            "rr_valid": False,
        }
        self._prev_face_center = None              # 上一幀臉部中心（穩定度檢查）
        self._frame_count = 0
        self._fps = fps
        self._update_interval = config.RPPG_UPDATE_INTERVAL
        self._min_conf = config.RPPG_MIN_FACE_CONFIDENCE
        self._stability_px = config.RPPG_ROI_STABILITY_PX
        self._min_buffer = int(fps * getattr(config, 'RPPG_MIN_BUFFER_SEC', 1.5))

        # Hysteresis：避免單幀晃動清空 buffer
        self._unstable_frames = 0
        self._unstable_threshold = getattr(config, 'RPPG_UNSTABLE_FRAMES_TO_RESET', 5)
        # 臉部短暫消失容忍：上次看到臉的時間
        self._last_face_seen = 0.0
        self._face_loss_tolerance = getattr(config, 'RPPG_FACE_LOSS_TOLERANCE_SEC', 1.0)
        # 多人模式：記住「正在追蹤」的臉部中心，避免每幀換不同人
        self._tracked_center = None

        # 快取結果（兩次 FFT 之間保持上次結果）
        self._last_result = {
            "bpm": -1.0,
            "confidence": 0.0,
            "valid": False,
            "roi_stable": False,
            "signal_quality": "UNKNOWN",
        }

        # 預建帶通濾波器（避免每次重建）
        self._sos = None
        if SCIPY_AVAILABLE:
            try:
                self._sos = sp_signal.butter(
                    4,
                    [config.RPPG_BANDPASS_LOW_HZ, config.RPPG_BANDPASS_HIGH_HZ],
                    btype='bandpass',
                    fs=fps,
                    output='sos'
                )
            except Exception as e:
                logger.error(f"帶通濾波器建立失敗: {e}")

        logger.info(
            f"rPPG 偵測器初始化 | "
            f"緩衝 {buf_len} 幀 ({config.RPPG_BUFFER_SECONDS}s) | "
            f"每 {self._update_interval} 幀更新"
        )

    def process_all_persons(self, frame: np.ndarray, all_keypoints: list,
                            timestamp: float) -> dict:
        """全局 rPPG：從所有偵測到的人選最佳臉部來追蹤。
        策略：優先延續上次追蹤的臉（位置最接近 _tracked_center），
        否則挑「臉部關鍵點可見度最高」的人。"""
        if not SCIPY_AVAILABLE or self._sos is None:
            return self._last_result

        if not all_keypoints:
            return self._handle_no_face(timestamp)

        # 為每個人計算臉部分數（可見度）並算出中心
        candidates = []
        for kps in all_keypoints:
            face_kps = kps[:5]
            visible = sum(1 for kp in face_kps[:3] if kp[2] > self._min_conf)
            if visible < 2:
                continue
            center = self._get_face_center(face_kps)
            score = sum(float(kp[2]) for kp in face_kps[:3] if kp[2] > self._min_conf)
            candidates.append((center, score, face_kps))

        if not candidates:
            return self._handle_no_face(timestamp)

        # 選擇：若有 _tracked_center，挑距離最近的；否則挑分數最高的
        if self._tracked_center is not None:
            def _dist(c):
                dx = c[0][0] - self._tracked_center[0]
                dy = c[0][1] - self._tracked_center[1]
                return dx * dx + dy * dy
            candidates.sort(key=_dist)
            best_center, best_score, best_kps = candidates[0]
            # 距離太遠就放棄延續，改挑分數最高的
            if _dist((best_center, 0, None)) > (self._stability_px * 4) ** 2:
                candidates.sort(key=lambda c: -c[1])
                best_center, best_score, best_kps = candidates[0]
        else:
            candidates.sort(key=lambda c: -c[1])
            best_center, best_score, best_kps = candidates[0]

        return self.process_frame(frame, best_kps, timestamp, _face_center=best_center)

    def _handle_no_face(self, timestamp: float) -> dict:
        """臉部完全不可見時的處理：短暫消失（< tolerance）保留 buffer，
        超過容忍時間才清空。"""
        if self._last_face_seen > 0 and (timestamp - self._last_face_seen) < self._face_loss_tolerance:
            # 短暫遮擋 / 偵測失敗，保留 buffer，狀態維持上次
            merged = dict(self._last_result)
            merged.update(self._last_resp_result)
            return merged
        # 真的丟失太久 → 清空
        self._buffer.clear()
        self._timestamps.clear()
        self._resp_buffer.clear()
        self._resp_timestamps.clear()
        self._prev_face_center = None
        self._tracked_center = None
        self._unstable_frames = 0
        self._last_result["roi_stable"] = False
        self._last_result["signal_quality"] = "UNKNOWN"
        self._last_resp_result = {"rr_bpm": -1.0, "rr_confidence": 0.0, "rr_valid": False}
        merged = dict(self._last_result)
        merged.update(self._last_resp_result)
        return merged

    def process_frame(self, frame: np.ndarray, face_keypoints: np.ndarray,
                      timestamp: float, _face_center=None) -> dict:
        """
        處理一幀影像，提取臉部綠色通道並累積。
        定期執行 FFT 計算心率。

        Args:
            frame: BGR 影像
            face_keypoints: keypoints[0:5]
            timestamp: 當前時間
            _face_center: 內部使用，避免重算

        Returns:
            {"bpm": float, "confidence": float, "valid": bool,
             "roi_stable": bool, "signal_quality": str}
        """
        if not SCIPY_AVAILABLE or self._sos is None:
            return self._last_result

        self._frame_count += 1

        # 1. 檢查臉部關鍵點可見性
        visible_count = sum(1 for kp in face_keypoints[:3] if kp[2] > self._min_conf)
        if visible_count < 2:
            return self._handle_no_face(timestamp)

        self._last_face_seen = timestamp

        # 2. 計算臉部中心 + 穩定度檢查（hysteresis 防誤清）
        face_center = _face_center if _face_center is not None else self._get_face_center(face_keypoints)
        roi_stable = self._check_stability(face_center)
        self._prev_face_center = face_center
        self._tracked_center = face_center
        self._last_result["roi_stable"] = roi_stable

        if not roi_stable:
            self._unstable_frames += 1
            # 連續 N 幀都不穩才清空 → 短暫晃動不會丟資料
            if self._unstable_frames >= self._unstable_threshold:
                self._buffer.clear()
                self._timestamps.clear()
                self._unstable_frames = 0
                self._last_result["signal_quality"] = "WEAK"
                return self._last_result
            # hysteresis 期間：跳過這幀但 buffer 保留，繼續從上次的位置算
            return self._last_result
        else:
            self._unstable_frames = 0

        # 3. 提取臉部 ROI 綠色通道
        green_val = self._extract_green(frame, face_keypoints)
        if green_val is None:
            return self._last_result

        self._buffer.append(green_val)
        self._timestamps.append(timestamp)
        # B1: 呼吸率緩衝同步累積
        if self._resp_enabled:
            self._resp_buffer.append(green_val)
            self._resp_timestamps.append(timestamp)

        # 4. 定期計算心率
        if (self._frame_count % self._update_interval == 0 and
                len(self._buffer) >= self._min_buffer):
            bpm, confidence, valid = self._compute_heart_rate()
            quality = "GOOD" if valid else ("WEAK" if confidence > 0.2 else "UNKNOWN")
            # 即使不夠強也回傳 BPM，UI 可用 quality 標示弱讀數
            self._last_result = {
                "bpm": round(bpm, 1) if bpm > 0 else -1.0,
                "confidence": round(confidence, 2),
                "valid": valid,
                "roi_stable": roi_stable,
                "signal_quality": quality,
            }
            tag = "✅" if valid else "⚠️"
            if bpm > 0:
                logger.info(f"rPPG {tag}: {bpm:.0f} BPM (信心度 {confidence:.0%}, "
                            f"quality={quality}, buffer {len(self._buffer)}/{self._buffer.maxlen})")

        # 5. 呼吸率獨立計算（較慢、每 N × update_interval 才算一次以省 CPU）
        if (self._resp_enabled and
                self._frame_count % (self._update_interval * 3) == 0 and
                len(self._resp_buffer) >= self._resp_min_buffer):
            rr_bpm, rr_conf, rr_valid = self._compute_respiration_rate()
            self._last_resp_result = {
                "rr_bpm": round(rr_bpm, 1) if rr_bpm > 0 else -1.0,
                "rr_confidence": round(rr_conf, 2),
                "rr_valid": rr_valid,
            }
            if rr_valid:
                logger.info(f"rPPG 呼吸率 ✅: {rr_bpm:.0f} 次/分 (信心度 {rr_conf:.0%})")

        merged = dict(self._last_result)
        merged.update(self._last_resp_result)
        # 暴露 buffer 累積進度（0.0~1.0）供 UI 顯示「建立中 N/15s」
        min_req = max(1, self._resp_min_buffer)
        merged["rr_buffer_ratio"] = min(1.0, len(self._resp_buffer) / float(min_req))
        # HR buffer 進度亦暴露（秒數基準）
        merged["hr_buffer_ratio"] = min(1.0, len(self._buffer) / float(max(1, self._min_buffer)))
        return merged

    def _get_face_center(self, kps) -> tuple:
        """從臉部關鍵點計算中心座標"""
        valid = [(kp[0], kp[1]) for kp in kps[:3] if kp[2] > self._min_conf]
        if not valid:
            return (0, 0)
        cx = sum(p[0] for p in valid) / len(valid)
        cy = sum(p[1] for p in valid) / len(valid)
        return (cx, cy)

    def _check_stability(self, center) -> bool:
        """檢查臉部是否穩定（幀間位移 < 閾值）"""
        if self._prev_face_center is None:
            return True
        dx = center[0] - self._prev_face_center[0]
        dy = center[1] - self._prev_face_center[1]
        drift = (dx * dx + dy * dy) ** 0.5
        return drift < self._stability_px

    def _extract_green(self, frame, kps) -> float:
        """
        從臉部 ROI 提取綠色通道平均值。
        ROI：用 nose + eyes 建立矩形，向外擴展 30%。
        """
        h, w = frame.shape[:2]
        valid = [(int(kp[0]), int(kp[1])) for kp in kps[:3] if kp[2] > self._min_conf]
        if len(valid) < 2:
            return None

        xs = [p[0] for p in valid]
        ys = [p[1] for p in valid]

        # 臉部矩形 + 30% 擴展
        face_w = max(xs) - min(xs)
        face_h = max(ys) - min(ys)
        pad_x = max(int(face_w * 0.3), 10)
        pad_y = max(int(face_h * 0.3), 10)

        x1 = max(0, min(xs) - pad_x)
        y1 = max(0, min(ys) - pad_y)
        x2 = min(w, max(xs) + pad_x)
        y2 = min(h, max(ys) + pad_y)

        if x2 - x1 < 10 or y2 - y1 < 10:
            return None

        # BGR 的 G 通道 = index 1
        roi = frame[y1:y2, x1:x2, 1]
        return float(roi.mean())

    def _compute_heart_rate(self) -> tuple:
        """
        從綠色通道時間序列計算心率。
        回傳 (bpm, confidence, is_valid)。
        """
        signal_array = np.array(self._buffer, dtype=np.float64)
        n = len(signal_array)
        if n < 30:
            return -1.0, 0.0, False

        # 1. 強化 detrending：3 階多項式擬合移除緩慢漂移（光照、呼吸基頻）
        try:
            t = np.arange(n, dtype=np.float64)
            poly = np.polyfit(t, signal_array, 3)
            trend = np.polyval(poly, t)
            signal_array = signal_array - trend
        except Exception:
            signal_array = signal_array - np.mean(signal_array)

        # 標準化
        std = np.std(signal_array)
        if std < 1e-6:
            return -1.0, 0.0, False
        signal_array = signal_array / std

        # 2. 加 Hanning window 減少 spectral leakage
        signal_array = signal_array * np.hanning(n)

        # 3. 帶通濾波
        try:
            filtered = sp_signal.sosfiltfilt(self._sos, signal_array)
        except Exception:
            return -1.0, 0.0, False

        # 估算實際 FPS（用時間戳）
        if len(self._timestamps) >= 2:
            dt = self._timestamps[-1] - self._timestamps[0]
            actual_fps = (len(self._timestamps) - 1) / dt if dt > 0 else self._fps
        else:
            actual_fps = self._fps

        # 4. FFT（zero-pad 提高頻率解析度）
        nfft = max(n * 4, 256)
        fft_mag = np.abs(np.fft.rfft(filtered, n=nfft))
        freqs = np.fft.rfftfreq(nfft, d=1.0 / actual_fps)

        # 限制在生理心率範圍
        low_hz = config.RPPG_BANDPASS_LOW_HZ
        high_hz = config.RPPG_BANDPASS_HIGH_HZ
        mask = (freqs >= low_hz) & (freqs <= high_hz)

        if not np.any(mask):
            return -1.0, 0.0, False

        valid_fft = fft_mag[mask]
        valid_freqs = freqs[mask]

        if len(valid_fft) < 5:
            return -1.0, 0.0, False

        # 5. 找峰值（先排除邊界 bin 再選最大）
        # 邊界 ringing 常出現在前/後 1 bin
        if len(valid_fft) >= 4:
            inner = valid_fft[1:-1]
            inner_freqs = valid_freqs[1:-1]
            inner_peak = int(np.argmax(inner))
            peak_idx = inner_peak + 1
            peak_freq = float(inner_freqs[inner_peak])
            peak_mag = float(inner[inner_peak])
        else:
            peak_idx = int(np.argmax(valid_fft))
            peak_freq = float(valid_freqs[peak_idx])
            peak_mag = float(valid_fft[peak_idx])

        # Prominence：峰值需明顯高於帶內中位數
        median_mag = float(np.median(valid_fft))
        if peak_mag > 0:
            prominence_ratio = (peak_mag - median_mag) / peak_mag
        else:
            prominence_ratio = 0.0

        # 6. SNR：峰值 vs 帶內平均
        mean_mag = float(np.mean(valid_fft))
        snr = peak_mag / mean_mag if mean_mag > 0 else 0.0

        # 信心度：SNR + prominence 雙條件加總
        # SNR ≥ 3 → 滿分；prominence_ratio ≥ 0.5 → 滿分
        snr_score = min(snr / 3.0, 1.0)
        prom_score = min(prominence_ratio / 0.5, 1.0)
        confidence = (snr_score + prom_score) / 2.0   # 平均，不要相乘以免過度懲罰

        # BPM
        bpm = peak_freq * 60.0

        # 有效性判斷（放寬：給 UI 顯示弱讀數的機會）
        is_valid = (
            50 <= bpm <= 180 and
            confidence >= config.RPPG_CONFIDENCE_MIN
        )

        return bpm, confidence, is_valid

    def _compute_respiration_rate(self) -> tuple:
        """
        B1: 從長緩衝綠通道訊號計算呼吸率。
        原理：呼吸會造成皮膚顏色以 0.1-0.5 Hz 的慢速週期性變化（比心率慢 ~15x）。
        回傳 (rr_bpm, confidence, is_valid)
        """
        signal_array = np.array(self._resp_buffer, dtype=np.float64)
        n = len(signal_array)
        if n < 30:
            return -1.0, 0.0, False

        # 1. Detrend：移除非常緩慢的漂移（光照變化），但保留呼吸週期
        #    用 5 階多項式去趨勢（比 HR 的 3 階更激進以移除更低頻）
        try:
            t = np.arange(n, dtype=np.float64)
            poly = np.polyfit(t, signal_array, 5)
            signal_array = signal_array - np.polyval(poly, t)
        except Exception:
            signal_array = signal_array - np.mean(signal_array)

        std = np.std(signal_array)
        if std < 1e-6:
            return -1.0, 0.0, False
        signal_array = signal_array / std
        signal_array = signal_array * np.hanning(n)

        # 2. 呼吸帶通濾波（0.1-0.5 Hz）
        # 即時建構 SOS（因與 HR 帶通不同，不預建）
        if len(self._resp_timestamps) >= 2:
            dt = self._resp_timestamps[-1] - self._resp_timestamps[0]
            actual_fps = (len(self._resp_timestamps) - 1) / dt if dt > 0 else self._fps
        else:
            actual_fps = self._fps
        # Nyquist 保護：high_hz 必須 < fps/2
        high = min(self._resp_high_hz, actual_fps / 2 - 0.01)
        if high <= self._resp_low_hz:
            return -1.0, 0.0, False
        try:
            sos_resp = sp_signal.butter(4, [self._resp_low_hz, high],
                                        btype='bandpass', fs=actual_fps, output='sos')
            filtered = sp_signal.sosfiltfilt(sos_resp, signal_array)
        except Exception:
            return -1.0, 0.0, False

        # 3. FFT + 峰值搜索（與 HR 類似邏輯，但在呼吸頻段）
        nfft = max(n * 4, 512)
        fft_mag = np.abs(np.fft.rfft(filtered, n=nfft))
        freqs = np.fft.rfftfreq(nfft, d=1.0 / actual_fps)
        mask = (freqs >= self._resp_low_hz) & (freqs <= high)
        if not np.any(mask):
            return -1.0, 0.0, False
        valid_fft = fft_mag[mask]
        valid_freqs = freqs[mask]
        if len(valid_fft) < 5:
            return -1.0, 0.0, False

        peak_idx = int(np.argmax(valid_fft))
        peak_freq = float(valid_freqs[peak_idx])
        peak_mag = float(valid_fft[peak_idx])
        median_mag = float(np.median(valid_fft))
        mean_mag = float(np.mean(valid_fft))
        snr = peak_mag / mean_mag if mean_mag > 0 else 0.0
        prominence_ratio = (peak_mag - median_mag) / peak_mag if peak_mag > 0 else 0.0
        confidence = (min(snr / 3.0, 1.0) + min(prominence_ratio / 0.5, 1.0)) / 2.0

        rr_bpm = peak_freq * 60.0
        is_valid = (6 <= rr_bpm <= 30 and confidence >= self._resp_confidence_min)
        return rr_bpm, confidence, is_valid
