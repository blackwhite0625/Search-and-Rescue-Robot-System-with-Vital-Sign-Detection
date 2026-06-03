"""
CarBot 攝影機模組 V3
========================
安全的 pause/resume：真正 release VideoCapture 釋放 USB，TTS 後重新開啟。
用 RLock 保護 _cap，避免 read_loop 與 pause 競爭。
"""

import threading
import time
import logging
import os
import config

logger = logging.getLogger("rescue.camera")

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False


class Camera:
    def __init__(self):
        self._cap = None
        self._cap_lock = threading.RLock()    # 保護 _cap 存取
        self._frame = None
        self._frame_lock = threading.Lock()   # 保護 _frame
        self._display_frame = None
        self._jpeg_cache = None
        self._frame_seq = 0                   # 每次 _frame 更新遞增；generate_mjpeg 用來跳過重複編碼
        self._display_seq = 0                 # _display_frame 更新遞增
        self._display_source_seq = 0          # set_display_frame 時對應到的 raw frame seq
        self._display_ts = 0.0
        self._running = False
        self._active = threading.Event()       # 讀幀迴圈是否應該工作
        self._active.set()                     # 預設啟用
        self._placeholder_jpeg = None          # pause 期間給 MJPEG 用的占位 JPEG

        if CV2_AVAILABLE:
            self._open_capture()
            if self._cap:
                self._running = True
                self._thread = threading.Thread(target=self._read_loop, daemon=True)
                self._thread.start()
        else:
            logger.warning("⚠️ 攝影機：模擬模式（opencv 不可用）")

    def _open_capture(self) -> bool:
        """建立 VideoCapture（必須在 _cap_lock 內呼叫）。
        嘗試 V4L2 backend，失敗 fallback 到預設。
        強制 MJPEG 節省 USB 頻寬。實際解析度與 config 不同也接受。"""
        cap = None
        try:
            # 1. 先試 V4L2 backend
            try:
                cap = cv2.VideoCapture(config.CAMERA_INDEX, cv2.CAP_V4L2)
                if not cap.isOpened():
                    try:
                        cap.release()
                    except Exception:
                        pass
                    cap = None
            except Exception:
                cap = None
            # 2. fallback：預設 backend
            if cap is None:
                cap = cv2.VideoCapture(config.CAMERA_INDEX)

            if not cap.isOpened():
                logger.error("❌ 無法開啟攝影機")
                try:
                    cap.release()
                except Exception:
                    pass
                self._cap = None
                return False

            # 3. 強制 MJPEG（USB camera + USB audio 共存的關鍵）
            try:
                fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
                cap.set(cv2.CAP_PROP_FOURCC, fourcc)
            except Exception:
                pass
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_WIDTH)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)
            cap.set(cv2.CAP_PROP_FPS, config.CAMERA_FPS)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            self._cap = cap
            if w != config.CAMERA_WIDTH or h != config.CAMERA_HEIGHT:
                logger.warning(
                    f"⚠️ 攝影機實際 {w}×{h}（要求 {config.CAMERA_WIDTH}×{config.CAMERA_HEIGHT}），接受"
                )
            else:
                logger.info(f"✅ 攝影機開啟成功（{w}×{h}）")
            return True
        except Exception as e:
            logger.error(f"❌ 攝影機開啟錯誤: {e}")
            if cap is not None:
                try:
                    cap.release()
                except Exception:
                    pass
            self._cap = None
            return False

    def _release_capture(self):
        """釋放 VideoCapture（必須在 _cap_lock 內呼叫）"""
        if self._cap is not None:
            # 靜默 release 避免 V4L2 噴洗
            _devnull = os.open(os.devnull, os.O_WRONLY)
            old_stderr = os.dup(2)
            os.dup2(_devnull, 2)
            try:
                self._cap.release()
            except Exception:
                pass
            finally:
                os.dup2(old_stderr, 2)
                os.close(old_stderr)
                os.close(_devnull)
            self._cap = None

    def pause(self):
        """
        硬暫停：釋放 VideoCapture 以讓出 USB 頻寬給音訊裝置。
        TTS / 警報播放期間呼叫，避免 audio + camera USB 衝突。
        generate_mjpeg 看到 _active 清除會吐 placeholder。
        """
        self._active.clear()
        with self._cap_lock:
            self._release_capture()
        # 清掉暫存 frame；若 resume 後讀幀尚未補上，瀏覽器繼續保留 placeholder
        # 而非回到暫停前的凍結畫面
        with self._frame_lock:
            self._frame = None
            self._display_frame = None
            self._jpeg_cache = None
        logger.info("攝影機已暫停（USB 已釋放）")

    def resume(self):
        """恢復：重新開啟 VideoCapture。

        aplay 播完後 V4L2 有時需要數秒才真正釋放 USB 頻寬；_open_capture() 單次
        呼叫可能 fail。這裡用退避重試 4 次（累計 ~3 秒）吸收延遲釋放；若最終仍
        失敗，仍設定 _active 讓 _read_loop 背景以 1.5s 週期繼續重連，確保連續
        多次警報後攝影機 stream 能自癒。"""
        success = False
        for attempt in range(4):
            with self._cap_lock:
                if self._cap is not None and self._cap.isOpened():
                    success = True
                    break
                # 若已有殘留 handle（pause 未完成）先保守釋放再開新
                if self._cap is not None:
                    self._release_capture()
                if self._open_capture():
                    success = True
                    break
            time.sleep(0.3 * (attempt + 1))   # 0.3 / 0.6 / 0.9 / 1.2

        # 無論成敗都要 set：成功 → read_loop 立即取幀；失敗 → read_loop 背景重試
        self._active.set()
        if success:
            logger.info("攝影機已恢復")
        else:
            logger.warning("攝影機 resume 重試 4 次未開啟，交由 _read_loop 背景續試")

    def _read_loop(self):
        """背景讀幀（active=False 時暫停；_cap_lock 保護）"""
        _read_fails = 0
        _MAX_FAILS = 15  # 連續失敗 15 幀 (~0.75s) 後強制重連
        _last_reopen_attempt = 0.0
        _REOPEN_COOLDOWN = 1.5  # 兩次重連嘗試最少間隔（秒）

        while self._running:
            # 等待 active（pause 期間阻塞）
            if not self._active.wait(timeout=0.5):
                _read_fails = 0
                continue

            with self._cap_lock:
                if self._cap is None or not self._cap.isOpened():
                    # 非 pause 造成的 cap 遺失 → 嘗試重連（節流）
                    now = time.time()
                    if self._active.is_set() and (now - _last_reopen_attempt) >= _REOPEN_COOLDOWN:
                        logger.warning("攝影機意外斷線，嘗試重連...")
                        _last_reopen_attempt = now
                        self._open_capture()
                    if self._cap is None:
                        time.sleep(_REOPEN_COOLDOWN)
                        continue

                try:
                    ret, frame = self._cap.read()
                except Exception:
                    ret = False
                    frame = None

                # 連續讀取失敗超過閾值 → 強制釋放舊連線，下一輪自動重建
                if not ret or frame is None:
                    _read_fails += 1
                    if _read_fails >= _MAX_FAILS:
                        logger.warning(
                            f"攝影機連續 {_read_fails} 幀讀取失敗，強制重連..."
                        )
                        self._release_capture()
                        _read_fails = 0
                else:
                    _read_fails = 0

            if ret and frame is not None:
                with self._frame_lock:
                    self._frame = frame
                    self._frame_seq += 1
            else:
                time.sleep(0.05)

            time.sleep(0.001)

    def get_frame(self):
        with self._frame_lock:
            return self._frame

    def set_display_frame(self, frame):
        if frame is None:
            return
        try:
            ret, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 65])
            if ret:
                with self._frame_lock:
                    self._display_frame = frame
                    self._jpeg_cache = jpeg.tobytes()
                    self._display_seq += 1
                    self._display_source_seq = self._frame_seq
                    self._display_ts = time.time()
        except Exception:
            pass

    def _make_placeholder_jpeg(self):
        """生成「警報播報中」占位 JPEG（中文），避免暫停期間瀏覽器看到凍結畫面"""
        try:
            import numpy as np
            from PIL import Image, ImageDraw, ImageFont
            w, h = config.CAMERA_WIDTH, config.CAMERA_HEIGHT
            # PIL RGB 暗紅底
            img = Image.new('RGB', (w, h), (60, 15, 15))
            draw = ImageDraw.Draw(img)

            # 嘗試載入中文字型（Ubuntu 預設可能有 Noto Sans CJK）
            font_paths = [
                "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
                "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
                "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",  # fallback
            ]
            font_big = None
            font_mid = None
            font_sml = None
            for fp in font_paths:
                try:
                    if os.path.exists(fp):
                        font_big = ImageFont.truetype(fp, 48)
                        font_mid = ImageFont.truetype(fp, 24)
                        font_sml = ImageFont.truetype(fp, 18)
                        break
                except Exception:
                    continue
            if font_big is None:
                font_big = ImageFont.load_default()
                font_mid = font_big
                font_sml = font_big

            def _center_text(text, y, font, color):
                bbox = draw.textbbox((0, 0), text, font=font)
                tw = bbox[2] - bbox[0]
                draw.text(((w - tw) // 2, y), text, font=font, fill=color)

            _center_text("⚠ 警報播報中 ⚠", h // 2 - 70, font_big, (255, 220, 80))
            _center_text("攝影機暫停以播放語音", h // 2 + 5, font_mid, (220, 220, 220))
            _center_text("數秒後將自動恢復畫面", h // 2 + 45, font_sml, (170, 170, 170))

            # PIL → OpenCV → JPEG bytes
            arr = np.array(img)
            arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
            ret, jpeg = cv2.imencode('.jpg', arr, [cv2.IMWRITE_JPEG_QUALITY, 75])
            if ret:
                return jpeg.tobytes()
        except Exception as e:
            logger.debug(f"placeholder JPEG 生成失敗: {e}")
        return None

    def generate_mjpeg(self):
        """MJPEG 串流：
        - display_jpeg（detection_loop 編好的標註 JPEG）夠新（落後 <= MAX_LAG 幀）→ 直接用，零編碼成本
        - 否則（自動巡邏時 YOLO 慢、display 落後）→ 即時編碼 raw frame 確保串流流暢
        確保自動巡邏時不因 detection 降速而卡頓。"""
        STREAM_PERIOD = 0.045  # ~22 fps
        PAUSE_PERIOD  = 0.5    # 占位畫面 2 fps
        DISPLAY_MAX_LAG = 4    # display 落後超過 4 幀就改用 raw
        DISPLAY_MAX_AGE = float(getattr(config, "CAMERA_DISPLAY_MAX_AGE_SEC", 0.90))

        last_raw_frame_seq = -1
        last_raw_jpeg_bytes = None
        while True:
            if not self._active.is_set():
                # 暫停中：吐 placeholder frame
                if self._placeholder_jpeg is None:
                    self._placeholder_jpeg = self._make_placeholder_jpeg()
                if self._placeholder_jpeg is not None:
                    yield (
                        b'--frame\r\n'
                        b'Content-Type: image/jpeg\r\n\r\n'
                        + self._placeholder_jpeg
                        + b'\r\n'
                    )
                last_raw_frame_seq = -1
                last_raw_jpeg_bytes = None
                time.sleep(PAUSE_PERIOD)
                continue

            # ── 單鎖抓 frame + seq + display 快取 ──
            with self._frame_lock:
                display_jpeg = self._jpeg_cache
                display_source_seq = self._display_source_seq
                display_ts = self._display_ts
                frame = self._frame
                frame_seq = self._frame_seq

            jpeg_bytes = None
            lag = frame_seq - display_source_seq   # display 對應的 raw frame 落後幾幀
            age = time.time() - display_ts if display_ts else 999.0

            # 決策：display 夠新就用（省編碼 + 含標註）；否則用 raw
            if display_jpeg is not None and (lag <= DISPLAY_MAX_LAG or age <= DISPLAY_MAX_AGE):
                jpeg_bytes = display_jpeg
            elif frame is not None:
                # raw frame：同一 seq 重用上次編碼
                if frame_seq == last_raw_frame_seq and last_raw_jpeg_bytes is not None:
                    jpeg_bytes = last_raw_jpeg_bytes
                else:
                    try:
                        ok, jpeg = cv2.imencode('.jpg', frame,
                                                [cv2.IMWRITE_JPEG_QUALITY, 60])
                        if ok:
                            last_raw_frame_seq = frame_seq
                            last_raw_jpeg_bytes = jpeg.tobytes()
                            jpeg_bytes = last_raw_jpeg_bytes
                    except Exception:
                        pass

            if jpeg_bytes is not None:
                yield (
                    b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n'
                    + jpeg_bytes
                    + b'\r\n'
                )

            time.sleep(STREAM_PERIOD)

    def is_opened(self):
        with self._cap_lock:
            return self._cap is not None and self._cap.isOpened()

    def cleanup(self):
        self._running = False
        self._active.set()  # 解除任何 wait
        time.sleep(0.1)
        with self._cap_lock:
            self._release_capture()
        logger.info("✅ 攝影機已釋放")
