"""
SentinelAI – fall_detector.py
================================
Best-of-both integration:
  - Friend's clean angle calculation with visibility checks (pixel coords)
  - Our multi-factor confidence scoring (angle + head drop + hip position)
  - Our frame history buffer + EMA smoothing
  - Our MJPEG streaming frame buffer for web dashboard
  - Friend's dataclass-style clean API: update() + is_fallen + torso_angle
"""

import math
import time
import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import mediapipe as mp

# ── Tuneable parameters ────────────────────────────────────────────────────────
FALL_ANGLE_THRESHOLD : float = 25.0   # degrees from vertical
CONFIRM_SECONDS      : float = 0.5    # must stay above threshold this long
HEAD_DROP_THRESH     : float = 0.06   # normalized Y drop per frame
HIP_LOW_THRESH       : float = 0.70   # hip Y > 70% = bottom 30% of frame
SMOOTHING_ALPHA      : float = 0.4    # EMA (lower = smoother)
HISTORY_SIZE         : int   = 15     # frame history buffer size
CONFIDENCE_THRESHOLD : float = 0.20   # min confidence to call it a fall
# ──────────────────────────────────────────────────────────────────────────────

_mp_pose    = mp.solutions.pose
_mp_drawing = mp.solutions.drawing_utils


@dataclass
class _FrameSnap:
    timestamp : float
    angle     : float
    head_y    : float
    hip_y     : float
    landmarks : Optional[list] = None   # [(x,y), ...] for movement calc


class FallDetector:
    """
    Stateful fall detector.
    Call process_frame() in detection loop.
    Read fall_detected, fall_confidence, torso_angle.
    get_latest_frame() returns MJPEG bytes for web stream.
    """

    def __init__(self):
        self._lock        = threading.Lock()
        self._frame_lock  = threading.Lock()

        # detection state
        self._is_fallen       : bool  = False
        self._fall_confidence : float = 0.0
        self._torso_angle     : float = 0.0
        self._fall_start_time : Optional[float] = None

        # history + smoothing
        self._history        : deque = deque(maxlen=HISTORY_SIZE)
        self._smoothed_angle : float = 0.0

        # streaming
        self._latest_frame     : Optional[bytes] = None
        self._latest_raw_frame = None   # raw BGR numpy frame for VideoBuffer

        # mediapipe
        self._pose = _mp_pose.Pose(
            static_image_mode        = False,
            model_complexity         = 1,
            smooth_landmarks         = True,
            min_detection_confidence = 0.5,
            min_tracking_confidence  = 0.5,
        )
        self._cap       = None
        self._available = True

        try:
            import cv2
            self._cv2 = cv2
        except ImportError:
            print("[FallDetector] OpenCV not available.")
            self._available = False

    # ── Camera ─────────────────────────────────────────────────────────────────
    def open_camera(self, index: int = 0) -> bool:
        if not self._available:
            return False
        self._cap = self._cv2.VideoCapture(index)
        return self._cap.isOpened()

    def release_camera(self):
        if self._cap and self._cap.isOpened():
            self._cap.release()
        self._cap = None
        with self._frame_lock:
            self._latest_frame = None

    # ── Main processing ────────────────────────────────────────────────────────
    def process_frame(self):
        """Grab one frame, run detection, update state, store MJPEG."""
        if not self._available or self._cap is None:
            self._set_safe(); return

        ret, frame = self._cap.read()
        if not ret:
            return

        cv2 = self._cv2
        frame = cv2.flip(frame, 1)
        h, w  = frame.shape[:2]

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = self._pose.process(rgb)
        rgb.flags.writeable = True

        now = time.time()

        if results.pose_landmarks:
            lm = results.pose_landmarks

            # Always draw skeleton when landmarks exist
            _mp_drawing.draw_landmarks(
                frame, results.pose_landmarks, _mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=_mp_drawing.DrawingSpec(
                    color=(0, 212, 255), thickness=2, circle_radius=3),
                connection_drawing_spec=_mp_drawing.DrawingSpec(
                    color=(0, 200, 150), thickness=2),
            )

            # ── Angle (visibility check) ──
            angle = self._compute_angle(lm, w, h)

            if angle is None:
                # Landmarks visible but low confidence — show pose, skip angle
                cv2.rectangle(frame, (8, 8), (300, 42), (10, 15, 20), -1)
                cv2.putText(frame, "Pose found — move closer", (12, 32),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (80, 160, 80), 1)
                self._encode_frame(frame)
                return

            # EMA smoothing
            self._smoothed_angle = (SMOOTHING_ALPHA * angle
                                    + (1 - SMOOTHING_ALPHA) * self._smoothed_angle)

            snap = _FrameSnap(
                timestamp = now,
                angle     = self._smoothed_angle,
                head_y    = lm.landmark[_mp_pose.PoseLandmark.NOSE].y,
                hip_y     = (lm.landmark[_mp_pose.PoseLandmark.LEFT_HIP].y
                             + lm.landmark[_mp_pose.PoseLandmark.RIGHT_HIP].y) / 2,
                landmarks = [(p.x, p.y) for p in lm.landmark],
            )
            self._history.append(snap)

            confidence = self._compute_confidence(snap)
            self._update_fall_state(confidence, now)
            self._draw_overlay(frame, results, confidence)

        else:
            self._history.append(_FrameSnap(now, 0, 0, 0, None))
            self._fall_start_time = None
            with self._lock:
                self._is_fallen       = False
                self._fall_confidence = 0.0
                self._torso_angle     = 0.0
            cv2.rectangle(frame, (8, 8), (240, 42), (10, 15, 20), -1)
            cv2.putText(frame, "No pose detected", (12, 32),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (60, 60, 60), 1)

        self._encode_frame(frame)

    # ── Angle calculation (friend's method — visibility-aware, pixel coords) ───
    def _compute_angle(self, landmarks, w: int, h: int) -> Optional[float]:
        lm = landmarks.landmark
        L  = _mp_pose.PoseLandmark

        required = [L.LEFT_SHOULDER, L.RIGHT_SHOULDER, L.LEFT_HIP, L.RIGHT_HIP]
        if any(lm[p].visibility < 0.3 for p in required):
            return None

        def px(p): return lm[p].x * w, lm[p].y * h

        ls_x, ls_y = px(L.LEFT_SHOULDER)
        rs_x, rs_y = px(L.RIGHT_SHOULDER)
        lh_x, lh_y = px(L.LEFT_HIP)
        rh_x, rh_y = px(L.RIGHT_HIP)

        shoulder_mid = ((ls_x + rs_x) / 2, (ls_y + rs_y) / 2)
        hip_mid      = ((lh_x + rh_x) / 2, (lh_y + rh_y) / 2)

        dx = shoulder_mid[0] - hip_mid[0]
        dy = shoulder_mid[1] - hip_mid[1]

        return math.degrees(math.atan2(abs(dx), abs(dy)))

    # ── Multi-factor confidence (our scoring) ──────────────────────────────────
    def _compute_confidence(self, snap: _FrameSnap) -> float:
        score = 0.0

        # Factor 1: torso angle (max 0.5)
        if snap.angle > FALL_ANGLE_THRESHOLD:
            score += 0.5 * min((snap.angle - FALL_ANGLE_THRESHOLD) / 20.0, 1.0)

        # Factor 2: sudden head drop (max 0.3)
        if len(self._history) >= 3:
            prev = [s.head_y for s in list(self._history)[-3:]]
            drop = snap.head_y - (sum(prev) / len(prev))
            if drop > HEAD_DROP_THRESH:
                score += min(0.3, drop * 3.0)

        # Factor 3: hips in bottom 30% (max 0.2)
        if snap.hip_y > HIP_LOW_THRESH:
            score += 0.2 * min((snap.hip_y - HIP_LOW_THRESH) / 0.3, 1.0)

        return round(min(score, 1.0), 2)

    # ── Time-gated fall state (friend's confirm_seconds logic) ────────────────
    def _update_fall_state(self, confidence: float, now: float):
        with self._lock:
            self._fall_confidence = confidence
            self._torso_angle     = round(self._smoothed_angle, 1)

            if confidence >= CONFIDENCE_THRESHOLD:
                if self._fall_start_time is None:
                    self._fall_start_time = now
                elif now - self._fall_start_time >= CONFIRM_SECONDS:
                    self._is_fallen = True
            else:
                self._fall_start_time = None
                self._is_fallen       = False

    # ── Overlay drawing ────────────────────────────────────────────────────────
    def _draw_overlay(self, frame, results, confidence: float):
        cv2    = self._cv2
        h, _w  = frame.shape[:2]
        fallen = self._is_fallen

        badge_col = (0, 0, 220) if fallen else (0, 180, 120)
        label = (f"Angle:{self._smoothed_angle:.0f}  "
                 f"Conf:{confidence:.2f}  "
                 f"{'FALL!' if fallen else 'OK'}")
        cv2.rectangle(frame, (8, 8), (370, 42), (10, 15, 20), -1)
        cv2.putText(frame, label, (12, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, badge_col, 2)

        bar_h = int(confidence * h)
        cv2.rectangle(frame, (0, h - bar_h), (6, h),
                      (0, 0, 220) if fallen else (0, 200, 120), -1)

    # ── MJPEG encode ───────────────────────────────────────────────────────────
    def _encode_frame(self, frame):
        with self._frame_lock:
            self._latest_raw_frame = frame.copy()   # store raw for VideoBuffer
        ok, buf = self._cv2.imencode(
            ".jpg", frame, [self._cv2.IMWRITE_JPEG_QUALITY, 72])
        if ok:
            with self._frame_lock:
                self._latest_frame = buf.tobytes()

    def _set_safe(self):
        with self._lock:
            self._is_fallen       = False
            self._fall_confidence = 0.0
            self._torso_angle     = 0.0

    # ── Movement magnitude (used by fusion engine inactivity check) ────────────
    def movement_magnitude(self) -> float:
        hist = list(self._history)
        if len(hist) < 2:
            return 0.0
        prev = hist[-2].landmarks
        curr = hist[-1].landmarks
        if prev is None or curr is None:
            return 0.0
        return round(sum(
            math.sqrt((c[0]-p[0])**2 + (c[1]-p[1])**2)
            for c, p in zip(curr, prev)
        ), 4)

    # ── Public read-only properties ────────────────────────────────────────────
    @property
    def fall_detected(self) -> bool:
        with self._lock: return self._is_fallen

    @property
    def fall_confidence(self) -> float:
        with self._lock: return self._fall_confidence

    @property
    def torso_angle(self) -> float:
        with self._lock: return self._torso_angle

    @property
    def body_angle(self) -> float:
        return self.torso_angle

    def get_latest_frame(self) -> Optional[bytes]:
        with self._frame_lock: return self._latest_frame

    def get_latest_raw_frame(self):
        """Return latest raw BGR numpy frame for VideoBuffer.push()."""
        with self._frame_lock: return self._latest_raw_frame

    def reset(self):
        with self._lock:
            self._is_fallen       = False
            self._fall_confidence = 0.0
            self._torso_angle     = 0.0
            self._fall_start_time = None
        self._history.clear()
        self._smoothed_angle = 0.0