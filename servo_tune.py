"""
舵機校正工具 — servo_tune.py (Path Y: lgpio 軟體 PWM)
======================================================
用 gpiozero + LGPIOFactory 控制舵機，走 lgpio kernel hrtimer 路徑。
不需要 hardware PWM overlay，不需要 sudo，直接用 config.py 的腳位。

Pan  接腳由 config.SERVO_PAN_PIN  決定（目前 = GPIO 18）
Tilt 接腳由 config.SERVO_TILT_PIN 決定（目前 = GPIO 25）

執行：
    pkill -f app.py
    python3 servo_tune.py

操作鍵（每次按 Enter 確認）：
    a/d  Pan  左 / 右   ±1°
    A/D  Pan  左 / 右   ±5°
    w/s  Tilt 上 / 下   ±1°
    W/S  Tilt 上 / 下   ±5°
    0    機械置中 (0, 0)
    c    回到 config 預設值
    p    印出目前角度
    r    寫入 config.py
    h    鎖定保持（不 detach，觀察抖動用）【預設已鎖定】
    H    解除鎖定（detach PWM，舵機會鬆弛）
    1    🐢 慢速 180° 自動掃描 (step 1°, 50ms) — 按 Enter 停止
    2    🐇 快速 180° 自動掃描 (step 5°, 15ms) — 按 Enter 停止
    3    🐌 極慢速 180° 自動掃描 (step 1°, 150ms)
    q    離開

【v2 改動】預設 LOCKED：舵機持續吃 PWM、命令下達後立刻到位並保持
          （舊版每指令後 detach，小角度變化常看不出移動）
"""

import sys
import time
import threading
import config

try:
    from gpiozero import AngularServo, Device
    from gpiozero.pins.lgpio import LGPIOFactory
except ImportError as e:
    print(f"❌ gpiozero / lgpio 未安裝: {e}")
    print("   執行: sudo apt install python3-gpiozero python3-lgpio")
    sys.exit(1)

# 強制 lgpio（RPi 5 必備）
Device.pin_factory = LGPIOFactory()


def clamp(v, lo=-90, hi=90):
    return max(lo, min(hi, v))


def main():
    print("=" * 50)
    print("  舵機校正工具 (lgpio 軟體 PWM)")
    print("=" * 50)
    print(f"Pan  GPIO: {config.SERVO_PAN_PIN}")
    print(f"Tilt GPIO: {config.SERVO_TILT_PIN}")
    print(f"Pulse 範圍: {config.SERVO_MIN_PULSE*1e6:.0f}μs ~ {config.SERVO_MAX_PULSE*1e6:.0f}μs")
    print(f"Pin Factory: LGPIOFactory")
    print()
    print(f"目前 config 預設值:")
    print(f"  SERVO_PAN_DEFAULT  = {config.SERVO_PAN_DEFAULT}")
    print(f"  SERVO_TILT_DEFAULT = {config.SERVO_TILT_DEFAULT}")
    print()

    try:
        pan_servo = AngularServo(
            config.SERVO_PAN_PIN,
            min_angle=-90, max_angle=90,
            min_pulse_width=config.SERVO_MIN_PULSE,
            max_pulse_width=config.SERVO_MAX_PULSE,
        )
        tilt_servo = AngularServo(
            config.SERVO_TILT_PIN,
            min_angle=-90, max_angle=90,
            min_pulse_width=config.SERVO_MIN_PULSE,
            max_pulse_width=config.SERVO_MAX_PULSE,
        )
    except Exception as e:
        print(f"❌ 舵機初始化失敗: {e}")
        print("   1. 確認沒有其他程式佔用 GPIO（pkill -f app.py）")
        print("   2. 確認接線（Pan→GPIO {}, Tilt→GPIO {}）".format(
            config.SERVO_PAN_PIN, config.SERVO_TILT_PIN))
        sys.exit(1)

    pan = float(config.SERVO_PAN_DEFAULT)
    tilt = float(config.SERVO_TILT_DEFAULT)
    # v2: 預設鎖定，PWM 連續輸出才能讓小角度（±1°）命令實際移動到位
    # 舊版預設 unlocked → 每指令後 detach，0.3s 脈衝 + 舵機鬆弛 → 幾乎看不出動作
    locked = True
    sweep_stop = threading.Event()
    sweep_thread = [None]  # 用 list 包起來才能在閉包內改

    def write(p, t):
        try:
            pan_servo.angle = p
            tilt_servo.angle = t
        except Exception as e:
            print(f"⚠️  寫入失敗: {e}")

    def write_pan_only(p):
        try:
            pan_servo.angle = p
        except Exception:
            pass

    def detach():
        try:
            pan_servo.value = None
            tilt_servo.value = None
        except Exception:
            pass

    def sweep_loop(step_deg, interval_sec, label):
        """背景 180° 自動掃描（-90 ↔ +90），按 Enter 或任何指令觸發 stop 事件"""
        nonlocal pan
        print(f"\n{label} 開始（按 Enter 停止）")
        direction = 1
        pan = -90.0
        write_pan_only(pan)
        time.sleep(0.3)
        try:
            while not sweep_stop.is_set():
                pan += step_deg * direction
                if pan >= 90:
                    pan = 90.0
                    direction = -1
                elif pan <= -90:
                    pan = -90.0
                    direction = 1
                write_pan_only(pan)
                if sweep_stop.wait(interval_sec):
                    break
        except Exception as e:
            print(f"sweep 錯誤: {e}")
        print(f"{label} 已停止，最終 pan={pan:.1f}")

    def start_sweep(step_deg, interval_sec, label):
        # 停掉舊的
        stop_sweep()
        sweep_stop.clear()
        t = threading.Thread(
            target=sweep_loop,
            args=(step_deg, interval_sec, label),
            daemon=True,
        )
        sweep_thread[0] = t
        t.start()

    def stop_sweep():
        if sweep_thread[0] is not None and sweep_thread[0].is_alive():
            sweep_stop.set()
            sweep_thread[0].join(timeout=1.0)
        sweep_thread[0] = None

    write(pan, tilt)
    time.sleep(0.5)
    # v2: 預設鎖定 → 不 detach，舵機持續保持目標角度
    if not locked:
        detach()
    print(f"✅ 已置位於 config 預設角度: pan={pan}, tilt={tilt}")
    print(f"🔒 PWM 已鎖定保持（輸入 H 可解除鎖定）")
    print()
    print("操作: a/d=Pan±1  A/D=Pan±5  w/s=Tilt±1  W/S=Tilt±5")
    print("      0=機械置中  c=config預設  p=印目前  r=寫回config")
    print("      h=鎖定持續輸出  H=解除鎖定  q=離開")
    print()

    try:
        while True:
            sweeping = sweep_thread[0] is not None and sweep_thread[0].is_alive()
            status = "🔁" if sweeping else ("🔒" if locked else "  ")
            prompt = f"{status}[pan={pan:6.1f}, tilt={tilt:6.1f}] > "
            cmd = input(prompt).strip()

            # 掃描中按 Enter（空輸入）→ 停止掃描
            if sweeping and not cmd:
                stop_sweep()
                continue
            if not cmd:
                continue

            # 任何其他指令進來前，先停掉掃描
            if sweeping:
                stop_sweep()

            ch = cmd[0]
            if ch == 'q':
                break
            elif ch == 'a':
                pan = clamp(pan - 1)
            elif ch == 'A':
                pan = clamp(pan - 5)
            elif ch == 'd':
                pan = clamp(pan + 1)
            elif ch == 'D':
                pan = clamp(pan + 5)
            elif ch == 'w':
                tilt = clamp(tilt + 1)
            elif ch == 'W':
                tilt = clamp(tilt + 5)
            elif ch == 's':
                tilt = clamp(tilt - 1)
            elif ch == 'S':
                tilt = clamp(tilt - 5)
            elif ch == '0':
                pan, tilt = 0.0, 0.0
                print("→ 機械置中 (0, 0)")
            elif ch == 'c':
                pan = float(config.SERVO_PAN_DEFAULT)
                tilt = float(config.SERVO_TILT_DEFAULT)
                print(f"→ 回到 config 預設 ({pan}, {tilt})")
            elif ch == 'p':
                print(f"→ pan={pan}, tilt={tilt}")
                continue
            elif ch == 'r':
                update_config(pan, tilt)
                continue
            elif ch == 'h':
                locked = True
                print("→ 鎖定持續輸出 PWM（可觀察抖動）")
                write(pan, tilt)
                continue
            elif ch == 'H':
                locked = False
                detach()
                print("→ 解除鎖定（已 detach PWM）")
                continue
            elif ch == '1':
                start_sweep(step_deg=1, interval_sec=0.05, label="🐢 慢速掃描 (1°/50ms)")
                continue
            elif ch == '2':
                start_sweep(step_deg=5, interval_sec=0.015, label="🐇 快速掃描 (5°/15ms)")
                continue
            elif ch == '3':
                start_sweep(step_deg=1, interval_sec=0.15, label="🐌 極慢速掃描 (1°/150ms)")
                continue
            else:
                print("未知指令")
                continue

            write(pan, tilt)
            # 鎖定模式下 PWM 連續輸出 → 不需長 settle；舵機會持續到位並保持
            if locked:
                time.sleep(0.05)
            else:
                time.sleep(0.3)
                detach()
    except KeyboardInterrupt:
        print()
    finally:
        try:
            stop_sweep()
        except Exception:
            pass
        try:
            detach()
            pan_servo.close()
            tilt_servo.close()
        except Exception:
            pass
        print(f"\n最終角度: pan={pan}, tilt={tilt}")
        print("✅ 已脫離 PWM 並釋放 GPIO")


def update_config(pan, tilt):
    """直接改寫 config.py 中的 SERVO_PAN_DEFAULT / SERVO_TILT_DEFAULT"""
    import re
    path = "config.py"
    try:
        with open(path, "r", encoding="utf-8") as f:
            src = f.read()
        new = re.sub(r"SERVO_PAN_DEFAULT\s*=\s*[-\d.]+",
                     f"SERVO_PAN_DEFAULT  = {pan}", src)
        new = re.sub(r"SERVO_TILT_DEFAULT\s*=\s*[-\d.]+",
                     f"SERVO_TILT_DEFAULT = {tilt}", new)
        if new == src:
            print("⚠️  config.py 未變更")
            return
        with open(path, "w", encoding="utf-8") as f:
            f.write(new)
        print(f"✅ 已寫入 config.py: pan={pan}, tilt={tilt}")
    except Exception as e:
        print(f"❌ 寫入失敗: {e}")


if __name__ == "__main__":
    main()
