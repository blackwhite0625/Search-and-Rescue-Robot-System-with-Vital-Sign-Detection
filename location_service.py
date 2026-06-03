"""
搜救機器人 — 多源定位服務 (V2)
================================
融合以下定位來源，提供最佳位置估計：
  1. GPS 模組（UART / USB，最精確）
  2. Wi-Fi RSSI 掃描（BSSID + 信號強度，室內定位輔助）
  3. IP 地理定位（最快取得，粗略）
  4. 室內區域標記（可配置的固定標籤）

回傳格式化的定位字串供 Telegram 通報使用。
"""

import logging
import subprocess
import platform
import re
import os
import time

logger = logging.getLogger("rescue.location")

try:
    import requests
    REQUESTS_OK = True
except ImportError:
    REQUESTS_OK = False

# 嘗試載入 serial（GPS NMEA 解析）
try:
    import serial
    SERIAL_OK = True
except ImportError:
    SERIAL_OK = False


class LocationService:
    """多源融合定位服務"""

    # 可在 config 或環境變數中覆蓋
    GPS_PORT = os.environ.get("GPS_PORT", "/dev/ttyAMA0")
    GPS_BAUD = int(os.environ.get("GPS_BAUD", "9600"))
    INDOOR_ZONE = os.environ.get("INDOOR_ZONE", "")

    @staticmethod
    def get_location() -> str:
        """
        取得目前地理資訊（多源融合）。
        優先順序：GPS → Wi-Fi + IP → 純 IP
        """
        logger.info("開始定位掃描...")
        parts = []

        # ── 1. GPS 模組（最精確）──
        gps = LocationService._read_gps()
        if gps:
            parts.append(f"📍 GPS: {gps['lat']:.6f}, {gps['lon']:.6f}")
            if gps.get("alt"):
                parts.append(f"   海拔: {gps['alt']:.1f}m")
            if gps.get("speed"):
                parts.append(f"   速度: {gps['speed']:.1f} km/h")
            parts.append(f"   衛星數: {gps.get('sats', '?')}")

        # ── 2. Wi-Fi RSSI 掃描 ──
        wifi_info = LocationService._scan_wifi_with_rssi()
        if wifi_info:
            top3 = wifi_info[:3]
            ap_str = " / ".join([f"{w['ssid']}({w['rssi']}dBm)" for w in top3])
            parts.append(f"📶 Wi-Fi: {ap_str}")
            parts.append(f"   掃描到 {len(wifi_info)} 個基地台")

        # ── 3. IP 地理定位（備案）──
        ip_loc = LocationService._ip_geolocation()
        if ip_loc:
            parts.append(f"🌐 IP定位: {ip_loc['loc']}")
            parts.append(f"   城市: {ip_loc['city']}, {ip_loc['country']}")

        # ── 4. 室內區域標記 ──
        zone = LocationService.INDOOR_ZONE
        if zone:
            parts.append(f"🏢 室內: {zone}")

        if not parts:
            return "位置: 無法取得定位資訊"

        return "\n".join(parts)

    # ──────────────────────────────────────────────
    # GPS NMEA 解析
    # ──────────────────────────────────────────────
    @staticmethod
    def _read_gps() -> dict:
        """讀取 GPS 模組的 NMEA $GPGGA / $GPRMC 句子"""
        if not SERIAL_OK:
            return {}

        port = LocationService.GPS_PORT
        if not os.path.exists(port):
            return {}

        try:
            ser = serial.Serial(port, LocationService.GPS_BAUD, timeout=2)
            deadline = time.time() + 3  # 最多等 3 秒
            result = {}

            while time.time() < deadline:
                line = ser.readline().decode("ascii", errors="ignore").strip()

                # $GPGGA — 定位品質 + 經緯度 + 衛星
                if line.startswith("$GPGGA") or line.startswith("$GNGGA"):
                    fields = line.split(",")
                    if len(fields) >= 10 and fields[2] and fields[4]:
                        result["lat"] = LocationService._nmea_to_deg(fields[2], fields[3])
                        result["lon"] = LocationService._nmea_to_deg(fields[4], fields[5])
                        result["sats"] = fields[7] or "?"
                        try:
                            result["alt"] = float(fields[9])
                        except (ValueError, IndexError):
                            pass

                # $GPRMC — 速度
                if line.startswith("$GPRMC") or line.startswith("$GNRMC"):
                    fields = line.split(",")
                    if len(fields) >= 8 and fields[7]:
                        try:
                            result["speed"] = float(fields[7]) * 1.852  # knots → km/h
                        except ValueError:
                            pass

                if "lat" in result:
                    ser.close()
                    return result

            ser.close()
        except Exception as e:
            logger.debug(f"GPS 讀取失敗: {e}")

        return {}

    @staticmethod
    def _nmea_to_deg(value: str, direction: str) -> float:
        """NMEA 格式 (ddmm.mmmm) 轉十進制度"""
        try:
            if len(value) < 4:
                return 0.0
            dot = value.index(".")
            deg = int(value[:dot - 2])
            minutes = float(value[dot - 2:])
            result = deg + minutes / 60.0
            if direction in ("S", "W"):
                result = -result
            return result
        except (ValueError, IndexError):
            return 0.0

    # ──────────────────────────────────────────────
    # Wi-Fi RSSI 掃描
    # ──────────────────────────────────────────────
    @staticmethod
    def _scan_wifi_with_rssi() -> list:
        """
        掃描附近 Wi-Fi 基地台，回傳 [{ssid, bssid, rssi}, ...] 依信號強度排序。
        """
        results = []
        try:
            os_name = platform.system()

            if os_name == "Linux":
                # 優先用 nmcli（不需 sudo）
                try:
                    proc = subprocess.run(
                        ["nmcli", "-t", "-f", "SSID,BSSID,SIGNAL", "dev", "wifi", "list"],
                        capture_output=True, text=True, timeout=5
                    )
                    for line in proc.stdout.strip().split("\n"):
                        parts = line.split(":")
                        if len(parts) >= 3:
                            ssid = parts[0] or "(hidden)"
                            bssid = ":".join(parts[1:-1]) if len(parts) > 3 else parts[1]
                            try:
                                signal = int(parts[-1])
                                # nmcli 回傳 0~100 的品質值，轉換為近似 dBm
                                rssi = signal // 2 - 100
                            except ValueError:
                                rssi = -99
                            results.append({"ssid": ssid, "bssid": bssid, "rssi": rssi})
                except Exception:
                    # 備案：iwlist（需 sudo）
                    try:
                        proc = subprocess.run(
                            ["sudo", "iwlist", "wlan0", "scan"],
                            capture_output=True, text=True, timeout=5
                        )
                        cells = proc.stdout.split("Cell ")
                        for cell in cells[1:]:
                            bssid_m = re.search(r"Address:\s*([\w:]+)", cell)
                            ssid_m = re.search(r'ESSID:"([^"]*)"', cell)
                            signal_m = re.search(r"Signal level=(-?\d+)", cell)
                            if bssid_m:
                                results.append({
                                    "ssid": ssid_m.group(1) if ssid_m else "(hidden)",
                                    "bssid": bssid_m.group(1),
                                    "rssi": int(signal_m.group(1)) if signal_m else -99,
                                })
                    except Exception:
                        pass

            elif os_name == "Darwin":
                try:
                    proc = subprocess.run(
                        ["/System/Library/PrivateFrameworks/Apple80211.framework/"
                         "Versions/Current/Resources/airport", "-s"],
                        capture_output=True, text=True, timeout=5
                    )
                    for line in proc.stdout.strip().split("\n")[1:]:
                        parts = line.split()
                        if len(parts) >= 3:
                            rssi = int(parts[-3]) if parts[-3].lstrip("-").isdigit() else -99
                            results.append({
                                "ssid": parts[0],
                                "bssid": parts[1] if ":" in parts[1] else "",
                                "rssi": rssi,
                            })
                except Exception:
                    pass

        except Exception as e:
            logger.debug(f"Wi-Fi 掃描錯誤: {e}")

        results.sort(key=lambda x: x["rssi"], reverse=True)
        return results

    # ──────────────────────────────────────────────
    # IP 地理定位
    # ──────────────────────────────────────────────
    @staticmethod
    def _ip_geolocation() -> dict:
        """透過公開 API 取得 IP 粗略定位"""
        if not REQUESTS_OK:
            return {}
        try:
            resp = requests.get("https://ipinfo.io/json", timeout=3)
            data = resp.json()
            return {
                "loc": data.get("loc", "0.0,0.0"),
                "city": data.get("city", "未知"),
                "country": data.get("country", "未知"),
            }
        except Exception as e:
            logger.debug(f"IP 定位失敗: {e}")
            return {}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print(LocationService.get_location())
