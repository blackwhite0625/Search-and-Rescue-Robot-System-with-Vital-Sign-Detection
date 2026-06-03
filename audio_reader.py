"""
搜救機器人 — USB 麥克風音訊讀取模組
====================================
背景執行緒連續讀取麥克風輸入，環形緩衝區保存最近 N 秒音訊。
使用 sounddevice 庫（跨平台，安裝簡單）。
"""

import threading
import time
import logging
import numpy as np

logger = logging.getLogger("rescue.audio_reader")

try:
    import sounddevice as sd
    SD_AVAILABLE = True
except (ImportError, OSError) as e:
    SD_AVAILABLE = False
    logger.warning(f"sounddevice 不可用（{e}）。音訊功能已停用。")
    logger.warning("請安裝: sudo apt install portaudio19-dev && pip install sounddevice")

import config


class AudioReader:
    """USB 麥克風連續讀取器（環形緩衝）"""

    def __init__(self):
        self._sample_rate = config.MIC_SAMPLE_RATE
        self._chunk_size = config.MIC_CHUNK_SIZE
        self._buffer_sec = config.MIC_BUFFER_SEC
        self._buffer_size = int(self._sample_rate * self._buffer_sec)

        self._buffer = np.zeros(self._buffer_size, dtype=np.float32)
        self._write_pos = 0
        self._lock = threading.Lock()
        self._device_lock = threading.Lock()
        self._running = False
        self._stream = None
        self._available = False

        # 最新一塊 chunk（供即時偵測用）
        self._latest_chunk = np.zeros(self._chunk_size, dtype=np.float32)
        self._chunk_ready = threading.Event()

        if SD_AVAILABLE:
            self._open()
        else:
            logger.warning("音訊讀取器：模擬模式（sounddevice 不可用）")

    def _open(self):
        """開啟麥克風串流"""
        try:
            device_index = config.MIC_DEVICE_INDEX
            if device_index is None:
                device_index = self._find_usb_mic()

            self._stream = sd.InputStream(
                samplerate=self._sample_rate,
                channels=1,
                dtype='float32',
                blocksize=self._chunk_size,
                device=device_index,
                callback=self._audio_callback,
            )
            self._stream.start()
            self._running = True
            self._available = True
            dev_name = sd.query_devices(device_index)['name'] if device_index else "預設裝置"
            logger.info(f"✅ 麥克風開啟成功（{dev_name}, {self._sample_rate}Hz）")
        except Exception as e:
            logger.error(f"❌ 麥克風開啟失敗: {e}")
            self._available = False

    def _find_usb_mic(self):
        """自動偵測 USB 麥克風"""
        try:
            devices = sd.query_devices()
            for i, dev in enumerate(devices):
                if dev['max_input_channels'] > 0:
                    name = dev['name'].lower()
                    if 'usb' in name or 'mic' in name:
                        logger.info(f"🎤 偵測到 USB 麥克風: [{i}] {dev['name']}")
                        return i
            # 找不到 USB 麥克風，使用預設輸入裝置
            default = sd.default.device[0]
            if default is not None and default >= 0:
                logger.info(f"🎤 使用預設輸入裝置: [{default}] {sd.query_devices(default)['name']}")
                return default
        except Exception as e:
            logger.warning(f"麥克風偵測失敗: {e}")
        return None

    def _audio_callback(self, indata, frames, time_info, status):
        """sounddevice 回呼函數（在音訊執行緒中執行）"""
        if status:
            logger.debug(f"音訊狀態: {status}")

        audio = indata[:, 0]  # 單聲道

        with self._lock:
            # 寫入環形緩衝
            end = self._write_pos + len(audio)
            if end <= self._buffer_size:
                self._buffer[self._write_pos:end] = audio
            else:
                first = self._buffer_size - self._write_pos
                self._buffer[self._write_pos:] = audio[:first]
                self._buffer[:len(audio) - first] = audio[first:]
            self._write_pos = end % self._buffer_size

            # 更新即時 chunk
            self._latest_chunk = audio.copy()

        self._chunk_ready.set()

    @property
    def is_available(self) -> bool:
        return self._available

    def get_latest_chunk(self) -> np.ndarray:
        """取得最新一塊音訊 chunk"""
        with self._lock:
            return self._latest_chunk.copy()

    def get_audio_buffer(self, duration_sec: float = None) -> np.ndarray:
        """取得指定長度的音訊緩衝（從最新時間往前）"""
        duration = duration_sec or self._buffer_sec
        samples = min(int(self._sample_rate * duration), self._buffer_size)

        with self._lock:
            end = self._write_pos
            start = end - samples
            if start >= 0:
                return self._buffer[start:end].copy()
            else:
                return np.concatenate([
                    self._buffer[start % self._buffer_size:],
                    self._buffer[:end]
                ]).copy()

    def get_write_pos(self) -> int:
        """取得目前環形緩衝寫入位置（給對講串流用來抓「新」資料）。"""
        with self._lock:
            return self._write_pos

    def get_audio_since(self, last_pos: int, max_duration_sec: float = 1.0) -> tuple:
        """回傳自 last_pos 以來的新音訊資料與新位置。
        last_pos = -1 或超出緩衝表示第一次呼叫，回傳最近 0.3s。
        返回 (audio_ndarray, new_pos)。不會重疊亦不會漏掉（環繞邊界正確處理）。"""
        with self._lock:
            cur = self._write_pos
            if last_pos < 0:
                # 第一次呼叫 → 回傳最近 0.3s 當作起始片段
                fallback_samples = int(self._sample_rate * 0.3)
                start = cur - fallback_samples
                if start >= 0:
                    seg = self._buffer[start:cur].copy()
                else:
                    seg = np.concatenate([
                        self._buffer[start % self._buffer_size:],
                        self._buffer[:cur],
                    ]).copy()
                return seg, cur

            if last_pos == cur:
                # 無新資料 → 空陣列（避免重複送相同片段）
                return np.zeros(0, dtype=np.float32), cur

            # 計算 last_pos → cur 的新樣本數（考慮環繞）
            new_samples = (cur - last_pos) % self._buffer_size
            if new_samples == 0:
                return np.zeros(0, dtype=np.float32), cur

            # 上限保護：避免一次回傳過多（> max_duration_sec）
            max_samples = int(self._sample_rate * max_duration_sec)
            if new_samples > max_samples:
                new_samples = max_samples
                last_pos = (cur - new_samples) % self._buffer_size

            start = last_pos
            end = cur
            if start < end:
                seg = self._buffer[start:end].copy()
            else:
                seg = np.concatenate([
                    self._buffer[start:],
                    self._buffer[:end],
                ]).copy()
            return seg, cur

    def wait_for_audio(self, timeout_sec: float = 5.0) -> bool:
        """等待新的音訊資料到達"""
        self._chunk_ready.clear()
        return self._chunk_ready.wait(timeout=timeout_sec)

    def get_rms_level(self) -> float:
        """取得當前音訊 RMS 音量（0~1）"""
        chunk = self.get_latest_chunk()
        return float(np.sqrt(np.mean(chunk ** 2)))

    def pause(self):
        """暫停麥克風串流（TTS 播放時用，避免 ALSA 衝突）。
        sounddevice.stream.stop() 在 Linux ALSA 上偶爾會卡死 → 改用 abort() +
        背景 thread + timeout，保證主流程不被吊死。"""
        def _do_pause():
            try:
                with self._device_lock:
                    if self._stream is not None:
                        # abort() 不等 buffer drain，比 stop() 快很多
                        self._stream.abort()
            except Exception as e:
                logger.debug(f"pause 失敗: {e}")
        t = threading.Thread(target=_do_pause, daemon=True)
        t.start()
        t.join(timeout=0.5)
        if t.is_alive():
            logger.warning("麥克風 pause 逾時（ALSA 阻塞），跳過")

    def resume(self):
        """恢復麥克風串流。

        背景：sounddevice/PortAudio 在 ALSA 後端下，多次 abort()/start() 循環後
        stream 可能進入 **callback 靜默殭屍狀態**：`.start()` 不拋錯、`.active=True`，
        但 callback 實際上不再被呼叫 → 環形緩衝 `write_pos` 停滯 → `/intercom/listen`
        永遠 204。

        可靠的活性檢測：監控 `_write_pos` 是否在 200ms 內增加（正常 48kHz/chunk 4096
        應每 ~85ms 觸發一次 callback）。若未增加 → 重建整個 InputStream。
        """
        def _do_resume():
            try:
                need_rebuild = False
                with self._device_lock:
                    if self._stream is not None:
                        try:
                            self._stream.start()
                        except Exception as e:
                            logger.info(f"stream.start() 失敗（{e}），標記重建")
                            need_rebuild = True
                    else:
                        need_rebuild = True

                if not need_rebuild:
                    # 最可靠檢測：callback 是否實際觸發（write_pos 前進）
                    pos_before = self._write_pos
                    time.sleep(0.25)   # 應至少觸發 2 次 callback（~85ms × 2）
                    with self._device_lock:
                        pos_after = self._write_pos
                        active = (self._stream is not None and self._stream.active)
                    if pos_before == pos_after or not active:
                        logger.info(
                            f"stream callback 靜默（pos {pos_before}→{pos_after}, "
                            f"active={active}）→ 重建 stream"
                        )
                        need_rebuild = True

                if need_rebuild:
                    self._rebuild_stream()
            except Exception as e:
                logger.debug(f"resume 失敗: {e}")

        t = threading.Thread(target=_do_resume, daemon=True)
        t.start()
        # 0.25s 檢測 + 重建（PortAudio 初始化 ~1s）= 最壞 1.5s，留 2.5s 餘裕
        t.join(timeout=2.5)
        if t.is_alive():
            logger.warning("麥克風 resume 逾時（ALSA 阻塞），跳過")

    def _rebuild_stream(self):
        """徹底關閉舊 stream 並重新建立 InputStream（應對 abort/start 循環失效）。"""
        if not SD_AVAILABLE:
            return
        with self._device_lock:
            # 先關舊的
            if self._stream is not None:
                try:
                    self._stream.close()
                except Exception:
                    pass
                self._stream = None

            # 重新建立
            try:
                device_index = config.MIC_DEVICE_INDEX
                if device_index is None:
                    device_index = self._find_usb_mic()
                self._stream = sd.InputStream(
                    samplerate=self._sample_rate,
                    channels=1,
                    dtype='float32',
                    blocksize=self._chunk_size,
                    device=device_index,
                    callback=self._audio_callback,
                )
                self._stream.start()
                self._available = True
                logger.info("✅ 麥克風 stream 已重建並啟動")
            except Exception as e:
                logger.error(f"❌ 麥克風 stream 重建失敗: {e}")
                self._stream = None
                self._available = False

    def cleanup(self):
        self._running = False
        if self._stream:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            logger.info("✅ 麥克風已釋放")
