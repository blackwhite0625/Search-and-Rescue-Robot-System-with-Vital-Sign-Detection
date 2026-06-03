"""
CarBot 車子移動模組
===================
麥克拉姆輪全向移動控制。
使用 4 個馬達 + 2 個 L298N 驅動板。
"""

import logging
import time
import config

logger = logging.getLogger("rescue.motor")

# GPIO 已在 config.py 統一初始化（含 PiGPIOFactory）
try:
    from gpiozero import PWMOutputDevice, OutputDevice
    GPIO_OK = config.GPIO_AVAILABLE
except ImportError:
    GPIO_OK = False


class MotorController:
    """麥克拉姆輪控制器（含全域速度限制 + 加速度 ramp）"""

    MAX_SPEED = 0.55          # 全域線速度上限（PWM 佔空比）
    MAX_ANGULAR = 0.60        # 全域角速度上限
    ACCEL_RATE = 3.0          # PWM per second（手動操控 ~0.17s 達 0.5 PWM）
    STOP_DECEL_RATE = 4.0     # 煞車專用更快速率（0.13s 從 0.5 歸零）

    def __init__(self):
        self.gpio_ok = GPIO_OK
        self._ros_bridge = None
        self._cur_x = 0.0
        self._cur_y = 0.0
        self._cur_r = 0.0
        self._last_cmd_time = time.time()

    def set_ros_bridge(self, bridge):
        """設定 ROS 2 橋接，讓 move() 自動發布 /cmd_vel"""
        self._ros_bridge = bridge

        if self.gpio_ok:
            try:
                f = config.L298N_FRONT
                r = config.L298N_REAR

                # 左前輪 FL
                self._fl_pwm = PWMOutputDevice(f["ena"])
                self._fl_in1 = OutputDevice(f["in1"])
                self._fl_in2 = OutputDevice(f["in2"])

                # 右前輪 FR
                self._fr_pwm = PWMOutputDevice(f["enb"])
                self._fr_in1 = OutputDevice(f["in3"])
                self._fr_in2 = OutputDevice(f["in4"])

                # 左後輪 RL
                self._rl_pwm = PWMOutputDevice(r["ena"])
                self._rl_in1 = OutputDevice(r["in1"])
                self._rl_in2 = OutputDevice(r["in2"])

                # 右後輪 RR
                self._rr_pwm = PWMOutputDevice(r["enb"])
                self._rr_in1 = OutputDevice(r["in3"])
                self._rr_in2 = OutputDevice(r["in4"])

                logger.info("✅ 馬達控制器初始化完成")
            except Exception as e:
                logger.error(f"❌ 馬達初始化失敗: {e}")
                self.gpio_ok = False
        else:
            logger.warning("⚠️ 馬達控制器：模擬模式（GPIO 不可用）")

    def _set_motor(self, pwm, in1, in2, speed):
        """控制單一馬達，速度範圍 -1.0 ~ 1.0"""
        if speed > 0.05:
            in1.on()
            in2.off()
            pwm.value = min(speed, 1.0)
        elif speed < -0.05:
            in1.off()
            in2.on()
            pwm.value = min(abs(speed), 1.0)
        else:
            in1.off()
            in2.off()
            pwm.value = 0

    def _ramp(self, cur, target, max_delta):
        delta = target - cur
        if abs(delta) <= max_delta:
            return target
        return cur + (max_delta if delta > 0 else -max_delta)

    def move(self, x, y, r, accel_rate=None):
        """麥克拉姆輪全向移動（含全域上限與 ramp）

        參數:
            x: 橫移 (-1 ~ 1)，正值向右
            y: 前後 (-1 ~ 1)，正值向前
            r: 旋轉 (-1 ~ 1)，正值順時針
            accel_rate: 單次命令使用的加速率；None 時使用預設值
        """
        # 1. 全域上限
        tx = max(-self.MAX_SPEED, min(self.MAX_SPEED, float(x)))
        ty = max(-self.MAX_SPEED, min(self.MAX_SPEED, float(y)))
        tr = max(-self.MAX_ANGULAR, min(self.MAX_ANGULAR, float(r)))

        # 2. Ramp（限制加速度）
        now = time.time()
        dt = max(0.001, min(0.1, now - self._last_cmd_time))
        self._last_cmd_time = now
        accel = self.ACCEL_RATE if accel_rate is None else max(0.1, float(accel_rate))
        rate = self.STOP_DECEL_RATE if (tx == 0 and ty == 0 and tr == 0) else accel
        max_delta = rate * dt
        self._cur_x = self._ramp(self._cur_x, tx, max_delta)
        self._cur_y = self._ramp(self._cur_y, ty, max_delta)
        self._cur_r = self._ramp(self._cur_r, tr, max_delta)

        self._apply(self._cur_x, self._cur_y, self._cur_r)

    def _apply(self, x, y, r):
        x = -x
        if not self.gpio_ok:
            if self._ros_bridge:
                self._ros_bridge.publish_cmd_vel(x, y, r)
            return

        speed_fl = y + x + r
        speed_fr = y - x - r
        speed_rl = y - x + r
        speed_rr = y + x - r

        # 前輪極性修正
        speed_fl = -speed_fl
        speed_fr = -speed_fr

        # 歸一化
        max_speed = max(abs(speed_fl), abs(speed_fr),
                        abs(speed_rl), abs(speed_rr))
        if max_speed > 1.0:
            speed_fl /= max_speed
            speed_fr /= max_speed
            speed_rl /= max_speed
            speed_rr /= max_speed

        self._set_motor(self._fl_pwm, self._fl_in1, self._fl_in2, speed_fl)
        self._set_motor(self._fr_pwm, self._fr_in1, self._fr_in2, speed_fr)
        self._set_motor(self._rl_pwm, self._rl_in1, self._rl_in2, speed_rl)
        self._set_motor(self._rr_pwm, self._rr_in1, self._rr_in2, speed_rr)

        # ROS 2 橋接：同步發布 /cmd_vel（維持原行為，x 為已翻號後的值）
        if self._ros_bridge:
            self._ros_bridge.publish_cmd_vel(x, y, r)

    def stop(self):
        """立即停車：繞過 ramp 直接歸零（煞車）"""
        self._cur_x = 0.0
        self._cur_y = 0.0
        self._cur_r = 0.0
        self._last_cmd_time = time.time()
        self._apply(0.0, 0.0, 0.0)

    def cleanup(self):
        self.stop()
        if self.gpio_ok:
            for name in ['_fl_pwm', '_fl_in1', '_fl_in2',
                         '_fr_pwm', '_fr_in1', '_fr_in2',
                         '_rl_pwm', '_rl_in1', '_rl_in2',
                         '_rr_pwm', '_rr_in1', '_rr_in2']:
                dev = getattr(self, name, None)
                if dev:
                    try:
                        dev.close()
                    except Exception:
                        pass
            logger.info("✅ 馬達 GPIO 已釋放")
