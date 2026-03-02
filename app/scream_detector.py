"""
SentinelAI – Scream / Loud Sound Detector
Uses sounddevice to compute RMS of mic input. Sustained loudness triggers alert.
"""

import math
import threading
import time
from dataclasses import dataclass
from typing import Optional


@dataclass
class AudioState:
    loud_detected: bool = False
    rms_level: float = 0.0
    breach_start: Optional[float] = None


class ScreamDetector:
    RMS_THRESHOLD = 0.08         # 0.0–1.0 normalized amplitude
    SUSTAIN_SECONDS = 0.5        # must be sustained to avoid single pops
    SAMPLE_RATE = 16_000
    BLOCK_SIZE = 512             # frames per callback

    def __init__(self):
        self._state = AudioState()
        self._lock = threading.Lock()
        self._stream = None
        self._available = False
        self._init_sounddevice()

    # ── Setup ──────────────────────────────────────────────────────────────────
    def _init_sounddevice(self):
        try:
            import sounddevice as sd
            import numpy as np
            self._sd = sd
            self._np = np
            self._available = True
        except (ImportError, OSError):
            print("[ScreamDetector] sounddevice not available – using simulation mode.")
            self._available = False

    def start(self):
        if not self._available:
            return
        self._stream = self._sd.InputStream(
            samplerate=self.SAMPLE_RATE,
            channels=1,
            blocksize=self.BLOCK_SIZE,
            callback=self._audio_callback,
            dtype="float32",
        )
        self._stream.start()

    def stop(self):
        if self._stream:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None

    # ── Callback (runs in sounddevice thread) ──────────────────────────────────
    def _audio_callback(self, indata, frames, time_info, status):
        rms = float(self._np.sqrt(self._np.mean(indata ** 2)))
        now = time.time()
        with self._lock:
            self._state.rms_level = rms
            if rms > self.RMS_THRESHOLD:
                if self._state.breach_start is None:
                    self._state.breach_start = now
                elapsed = now - self._state.breach_start
                self._state.loud_detected = elapsed >= self.SUSTAIN_SECONDS
            else:
                self._state.breach_start = None
                self._state.loud_detected = False

    # ── Public accessors ───────────────────────────────────────────────────────
    @property
    def loud_detected(self) -> bool:
        with self._lock:
            return self._state.loud_detected

    @property
    def rms_level(self) -> float:
        with self._lock:
            return round(self._state.rms_level, 4)

    def reset(self):
        with self._lock:
            self._state = AudioState()