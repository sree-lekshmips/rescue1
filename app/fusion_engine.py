"""
SentinelAI – fusion_engine.py
State machine: SAFE → FALL_DETECTED → EMERGENCY_CONFIRMED
No recovery window. Fall confirmed = alert fires immediately.
"""

import threading
import time
from enum import Enum
from dataclasses import dataclass
from typing import Optional

from app.fall_detector     import FallDetector
from app.scream_detector   import ScreamDetector
from app.emergency_handler import EmergencyHandler
from app.video_buffer      import VideoBuffer


class SystemState(str, Enum):
    SAFE                = "SAFE"
    FALL_DETECTED       = "FALL_DETECTED"
    EMERGENCY_CONFIRMED = "EMERGENCY_CONFIRMED"


@dataclass
class _State:
    state           : SystemState = SystemState.SAFE
    risk_score      : float       = 0.0
    fall_detected   : bool        = False
    fall_confidence : float       = 0.0
    audio_detected  : bool        = False
    countdown       : int         = 0
    alert_triggered : bool        = False
    last_movement   : float       = 0.0


FALL_CONFIRM_FRAMES = 2
LOOP_INTERVAL       = 0.10
VB_PRE_SECONDS      = 5.0
VB_POST_SECONDS     = 5.0
VB_CLIPS_FOLDER     = "fall_clips"


class FusionEngine:

    def __init__(self, settings_manager):
        self._settings_manager  = settings_manager
        self._lock              = threading.Lock()
        self._running           = False
        self._fall_streak       = 0
        self._state             = _State()

        self._fall_detector     = FallDetector()
        self._scream_detector   = ScreamDetector()
        self._emergency_handler = EmergencyHandler(settings_manager)
        self._video_buffer      = self._make_video_buffer()
        self._emergency_handler.set_video_buffer(self._video_buffer)

    def _make_video_buffer(self) -> VideoBuffer:
        return VideoBuffer(
            fps          = 20.0,
            pre_seconds  = VB_PRE_SECONDS,
            post_seconds = VB_POST_SECONDS,
            clips_folder = VB_CLIPS_FOLDER,
        )

    # ── Properties ────────────────────────────────────────────────────────────
    @property
    def is_running(self) -> bool:
        return self._running

    # ── Stop — called when monitoring button pressed or on reset ───────────────
    def stop(self):
        self._running = False
        self._emergency_handler.stop_alarm()
        self._scream_detector.stop()
        self._fall_detector.release_camera()
        self._video_buffer.stop()

    # ── Reset — clears state only, does NOT restart ────────────────────────────
    def reset(self):
        # 1. Stop alarm immediately
        self._emergency_handler.stop_alarm()

        # 2. Clear detection state
        with self._lock:
            self._state       = _State()
            self._fall_streak = 0

        # 3. Reset fall detector internal state (keeps camera closed)
        self._fall_detector.reset()

    # ── Main loop ──────────────────────────────────────────────────────────────
    def run_loop(self, stop_event: threading.Event):
        self._running     = True
        self._fall_streak = 0

        # Fresh detectors each start
        self._fall_detector   = FallDetector()
        self._scream_detector = ScreamDetector()
        self._video_buffer    = self._make_video_buffer()
        self._emergency_handler.set_video_buffer(self._video_buffer)

        with self._lock:
            self._state = _State()

        self._fall_detector.open_camera()
        self._scream_detector.start()
        self._video_buffer.start()

        while not stop_event.is_set() and self._running:
            self._fall_detector.process_frame()

            # ── Feed every frame into the video buffer ──────────────────────
            raw = self._fall_detector.get_latest_raw_frame()
            if raw is not None:
                self._video_buffer.push(raw)

            self._tick()
            time.sleep(LOOP_INTERVAL)

        self.stop()

    def _tick(self):
        fall       = self._fall_detector.fall_detected
        confidence = self._fall_detector.fall_confidence
        audio      = self._scream_detector.loud_detected
        movement   = self._fall_detector.movement_magnitude()
        risk_score = round(min(confidence + (0.2 if audio else 0.0), 1.0), 2)

        with self._lock:
            self._state.fall_detected   = fall
            self._state.fall_confidence = confidence
            self._state.audio_detected  = audio
            self._state.risk_score      = risk_score
            self._state.last_movement   = movement
            current = self._state.state

            # ── SAFE ───────────────────────────────────────────────────────────
            if current == SystemState.SAFE:
                self._fall_streak = (self._fall_streak + 1) if fall else 0
                if self._fall_streak >= FALL_CONFIRM_FRAMES:
                    self._state.state           = SystemState.FALL_DETECTED
                    self._state.alert_triggered = False
                    self._fall_streak           = 0

            # ── FALL_DETECTED → fire alert immediately ─────────────────────────
            elif current == SystemState.FALL_DETECTED:
                if not self._state.alert_triggered:
                    self._state.state           = SystemState.EMERGENCY_CONFIRMED
                    self._state.alert_triggered = True
                    threading.Thread(
                        target = self._emergency_handler.handle,
                        kwargs = {
                            "risk_score":      risk_score,
                            "fall":            fall,
                            "audio":           audio,
                            "torso_angle":     self._fall_detector.torso_angle,
                            "fall_confidence": confidence,
                        },
                        daemon = True,
                        name   = "EmergencyDispatch",
                    ).start()

            # ── EMERGENCY_CONFIRMED — stays here until reset ───────────────────
            elif current == SystemState.EMERGENCY_CONFIRMED:
                pass

    # ── Cancel (not used in this flow but kept for safety) ────────────────────
    def cancel_emergency(self):
        with self._lock:
            self._state       = _State()
            self._fall_streak = 0
        self._fall_detector.reset()

    # ── Status for API ─────────────────────────────────────────────────────────
    def get_status(self) -> dict:
        with self._lock:
            return {
                "state":           self._state.state.value,
                "risk_score":      self._state.risk_score,
                "fall":            self._state.fall_detected,
                "fall_confidence": self._state.fall_confidence,
                "audio":           self._state.audio_detected,
                "countdown":       self._state.countdown,
                "running":         self._running,
                "body_angle":      self._fall_detector.torso_angle,
                "movement":        self._state.last_movement,
                "rms_level":       self._scream_detector.rms_level,
            }

    def get_latest_frame(self) -> Optional[bytes]:
        return self._fall_detector.get_latest_frame()