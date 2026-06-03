"""
CarBot 雲台舵機模組 (V6 — 直寫模式，無背景 thread)
=====================================================
模仿 servo_tune.py 的精確行為：
  - 沒有背景 _smooth_loop（消除 scheduler 競爭造成的抖動）
  - set_angle: 同步寫入 + detach
  - sweep_pan_to: 直寫不 detach（呼叫方負責掃描時序）
  - home: 同步寫入 + 等待 + detach

Pan  (水平)  → gpiozero + LGPIOFactory (軟體 PWM，RPi 5 上最穩的路徑)
Tilt (垂直)  → gpiozero + LGPIOFactory
"""

import time
import logging
import threading
import config

logger = logging.getLogger("rescue.servo")

GPIO_OK = False
try:
    from gpiozero import AngularServo, Device
    from gpiozero.pins.lgpio import LGPIOFactory
    try:
        Device.pin_factory = LGPIOFactory()
        logger.info("舵機使用 LGPIOFactory (kernel hrtimer)")
    except Exception as e:
        logger.warning(f"LGPIOFactory 設定失敗: {e}")
    GPIO_OK = config.GPIO_AVAILABLE
except ImportError:
    pass


class ServoController:
    """直寫式舵機控制器：每次 set_angle 都是一次性寫入 + detach，
    沒有背景插值迴圈，與 servo_tune 行為一致。"""

    SETTLE_TIME = 0.3      # 寫入後等待舵機到位
    LARGE_DELTA_STEP = 5   # 大幅度移動時的分段步長
    LARGE_DELTA_INTERVAL = 0.03  # 分段間隔

    def __init__(self):
        self.gpio_ok = GPIO_OK
        self._pan = None
        self._tilt = None
        self._current_pan = float(config.SERVO_DEFAULT_PAN)
        self._current_tilt = float(config.SERVO_DEFAULT_TILT)
        self._lock = threading.Lock()

        if not self.gpio_ok:
            logger.warning("⚠️ 舵機控制器：模擬模式（GPIO 不可用）")
            return

        try:
            self._pan = AngularServo(
                config.SERVO_PAN_PIN,
                min_angle=-90, max_angle=90,
                min_pulse_width=config.SERVO_MIN_PULSE,
                max_pulse_width=config.SERVO_MAX_PULSE,
            )
            self._tilt = AngularServo(
                config.SERVO_TILT_PIN,
                min_angle=-90, max_angle=90,
                min_pulse_width=config.SERVO_MIN_PULSE,
                max_pulse_width=config.SERVO_MAX_PULSE,
            )
            # 初始化到預設位置
            self._pan.angle = self._current_pan
            self._tilt.angle = self._current_tilt
            time.sleep(self.SETTLE_TIME)
            self._detach()
            logger.info(
                f"✅ 舵機控制器初始化完成 (直寫模式) | "
                f"Pan GPIO {config.SERVO_PAN_PIN}, Tilt GPIO {config.SERVO_TILT_PIN}"
            )
        except Exception as e:
            logger.error(f"❌ 舵機初始化失敗: {e}")
            self.gpio_ok = False

    def _detach(self):
        """釋放 PWM 信號，讓舵機靜默（減少軟體 PWM 抖動）"""
        for servo in (self._pan, self._tilt):
            if servo is None:
                continue
            try:
                servo.value = None
            except Exception:
                pass

    def _smooth_write_pan(self, target_pan):
        """大幅度移動時分段寫入（避免瞬間 torque spike），小幅度直接寫。"""
        if self._pan is None:
            return
        delta = target_pan - self._current_pan
        if abs(delta) <= self.LARGE_DELTA_STEP:
            self._pan.angle = target_pan
        else:
            steps = int(abs(delta) / self.LARGE_DELTA_STEP)
            step = delta / (steps + 1)
            current = self._current_pan
            for _ in range(steps):
                current += step
                self._pan.angle = int(round(current))
                time.sleep(self.LARGE_DELTA_INTERVAL)
            self._pan.angle = target_pan
        self._current_pan = target_pan

    def _smooth_write_tilt(self, target_tilt):
        if self._tilt is None:
            return
        delta = target_tilt - self._current_tilt
        if abs(delta) <= self.LARGE_DELTA_STEP:
            self._tilt.angle = target_tilt
        else:
            steps = int(abs(delta) / self.LARGE_DELTA_STEP)
            step = delta / (steps + 1)
            current = self._current_tilt
            for _ in range(steps):
                current += step
                self._tilt.angle = int(round(current))
                time.sleep(self.LARGE_DELTA_INTERVAL)
            self._tilt.angle = target_tilt
        self._current_tilt = target_tilt

    def set_angle(self, pan, tilt):
        """同步寫入 pan + tilt，到位後 detach。大幅度移動會分段以減少衝擊。"""
        pan = max(-90, min(90, int(pan)))
        tilt = max(-90, min(90, int(tilt)))
        if not self.gpio_ok:
            self._current_pan = pan
            self._current_tilt = tilt
            return
        with self._lock:
            try:
                self._smooth_write_pan(pan)
                self._smooth_write_tilt(tilt)
                time.sleep(self.SETTLE_TIME)
                self._detach()
            except Exception as e:
                logger.warning(f"舵機 set_angle 失敗: {e}")

    def sweep_pan_to(self, pan):
        """巡邏掃描專用：直接寫 pan，不 sleep 不 detach（呼叫方負責時序）。
        連續呼叫會讓 pan 持續輸出 PWM，結束後請呼叫 sweep_end()。"""
        pan = max(-90, min(90, int(pan)))
        if not self.gpio_ok:
            self._current_pan = pan
            return
        with self._lock:
            try:
                if self._pan is not None:
                    self._pan.angle = pan
                self._current_pan = pan
            except Exception as e:
                logger.warning(f"sweep_pan_to 失敗: {e}")

    def sweep_end(self):
        """掃描結束後呼叫：detach pan 讓舵機靜默。"""
        if not self.gpio_ok:
            return
        with self._lock:
            try:
                if self._pan is not None:
                    self._pan.value = None
            except Exception:
                pass

    def home(self):
        """強制歸位到 config 預設角度（同步阻塞，呼叫完成後舵機已在目標位置）。"""
        target_pan = int(config.SERVO_DEFAULT_PAN)
        target_tilt = int(config.SERVO_DEFAULT_TILT)
        if not self.gpio_ok:
            self._current_pan = target_pan
            self._current_tilt = target_tilt
            return
        with self._lock:
            try:
                # 手動調整後 _current 可能與實際不符 → 直接寫目標
                if self._pan is not None:
                    self._pan.angle = target_pan
                if self._tilt is not None:
                    self._tilt.angle = target_tilt
                self._current_pan = target_pan
                self._current_tilt = target_tilt
                time.sleep(0.4)
                self._detach()
            except Exception as e:
                logger.warning(f"舵機歸位失敗: {e}")

    def get_angles(self):
        return {"pan": self._current_pan, "tilt": self._current_tilt}

    def cleanup(self):
        # 先搶 lock 避免其他 thread 中途寫入 closed servo
        with self._lock:
            self.gpio_ok = False
            try:
                self._detach()
            except Exception:
                pass
            # 抓到 reference 後清掉 attribute，別的 thread 看到 None 會 skip
            pan, self._pan = self._pan, None
            tilt, self._tilt = self._tilt, None
        for servo in (pan, tilt):
            if servo is None:
                continue
            try:
                servo.close()
            except Exception:
                pass
        logger.info("✅ 舵機 GPIO 已釋放")
