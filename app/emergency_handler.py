"""
SentinelAI – emergency_handler.py
On EMERGENCY_CONFIRMED:
  1. Trigger video recording immediately
  2. Play alarm
  3. Send instant email NOW
  4. Send video email once clip is ready
  5. Save log entry
"""

import json
import time
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional

from app.video_buffer  import VideoBuffer
from app.alert_handler import AlertHandler

LOG_FILE   = Path("emergency_log.json")
ALARM_FREQ = 880


class EmergencyHandler:

    def __init__(self, settings_manager):
        self._settings_manager = settings_manager
        self._alert_handler    = AlertHandler(settings_manager)
        self._alarm_stop       = threading.Event()
        self._video_buffer     : Optional[VideoBuffer] = None

    def set_video_buffer(self, buf: VideoBuffer):
        self._video_buffer = buf

    def handle(self, risk_score: float, fall: bool, audio: bool,
               torso_angle: float = 0.0, fall_confidence: float = 0.0):

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"\n[Emergency] 🚨 FALL CONFIRMED at {timestamp}")
        print(f"[Emergency] Angle={torso_angle:.1f}° Confidence={fall_confidence:.0%}")

        # 1. Save log
        self._save_log(timestamp, torso_angle, fall_confidence, risk_score)

        # 2. Trigger video recording immediately
        clip_path = None
        if self._video_buffer:
            clip_path = self._video_buffer.trigger_save()
            print(f"[Emergency] Recording video → {clip_path}")

        # 3. Start alarm
        self._alarm_stop.clear()
        threading.Thread(target=self._play_alarm, daemon=True, name="Alarm").start()

        # 4. Send emails (instant now + video when ready)
        dispatched = self._alert_handler.send_alert(
            clip_path       = clip_path,
            torso_angle     = torso_angle,
            fall_confidence = fall_confidence,
            risk_score      = risk_score,
        )
        if not dispatched:
            print("[Emergency] Alert not sent — check Settings page for email config.")

        # Stop alarm after 30s
        threading.Timer(30.0, self._alarm_stop.set).start()

    def _play_alarm(self):
        try:
            import numpy as np
            import sounddevice as sd
            sr    = 44100
            t     = np.linspace(0, 0.4, int(sr * 0.4), False)
            beep  = (np.sin(2 * np.pi * ALARM_FREQ * t) * 0.8).astype(np.float32)
            pause = np.zeros(int(sr * 0.15), dtype=np.float32)
            tone  = np.concatenate([beep, pause])
            while not self._alarm_stop.is_set():
                sd.play(tone, sr)
                sd.wait()
        except Exception as e:
            print(f"[Alarm] {e}")

    def _save_log(self, timestamp, torso_angle, fall_confidence, risk_score):
        entry = {
            "timestamp":       timestamp,
            "torso_angle":     round(torso_angle, 1),
            "fall_confidence": round(fall_confidence, 2),
            "risk_score":      round(risk_score, 2),
        }
        try:
            log = json.loads(LOG_FILE.read_text()) if LOG_FILE.exists() else []
            log.append(entry)
            LOG_FILE.write_text(json.dumps(log, indent=2))
        except Exception as e:
            print(f"[Emergency] Log error: {e}")