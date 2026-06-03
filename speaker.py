"""
CarBot 喇叭模組（stub）
=======================
警報音效已移除，所有警報統一由 tts_utils + notifier 走 TTS 路徑。
此檔保留為 import 相容 stub —— `play_alert()` 已成 no-op。
"""

import logging

logger = logging.getLogger("rescue.speaker")


class Speaker:
    """USB 喇叭控制（stub，警報音效已停用）"""

    def __init__(self):
        self._playing = False
        logger.info("✅ 喇叭模組就緒（警報音效已停用，僅保留 TTS）")

    def play_alert(self):
        """已停用 — 警報統一走 TTS。保留方法名稱以維持相容性。"""
        return
