"""Run this directly to diagnose what's broken."""
import sys
print(f"Python: {sys.version}")
print()

# Test 1: OpenCV
try:
    import cv2
    print(f"[OK] OpenCV {cv2.__version__}")
    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        ret, frame = cap.read()
        print(f"[OK] Camera opened — frame read: {ret}, shape: {frame.shape if ret else 'N/A'}")
        cap.release()
    else:
        print("[FAIL] Camera failed to open — try index 1 or check if another app is using it")
except ImportError:
    print("[FAIL] OpenCV not installed — run: pip install opencv-python")
except Exception as e:
    print(f"[FAIL] OpenCV error: {e}")

print()

# Test 2: MediaPipe
try:
    import mediapipe as mp
    print(f"[OK] MediaPipe {mp.__version__}")
    pose = mp.solutions.pose.Pose()
    pose.close()
    print("[OK] MediaPipe Pose initialized")
except ImportError:
    print("[FAIL] MediaPipe not installed — run: pip install mediapipe")
except Exception as e:
    print(f"[FAIL] MediaPipe error: {e}")

print()

# Test 3: sounddevice
try:
    import sounddevice as sd
    import numpy as np
    devices = sd.query_devices()
    print(f"[OK] sounddevice installed")
    default = sd.default.device
    print(f"[OK] Default mic: {default}")
except ImportError:
    print("[FAIL] sounddevice not installed — run: pip install sounddevice")
except Exception as e:
    print(f"[FAIL] sounddevice error: {e}")

print()

# Test 4: FastAPI imports
try:
    from app.fall_detector import FallDetector
    print("[OK] fall_detector imports OK")
    fd = FallDetector()
    print(f"[OK] FallDetector created, available={fd._available}")
except Exception as e:
    print(f"[FAIL] fall_detector: {e}")

try:
    from app.scream_detector import ScreamDetector
    print("[OK] scream_detector imports OK")
except Exception as e:
    print(f"[FAIL] scream_detector: {e}")

try:
    from app.video_buffer import VideoBuffer
    print("[OK] video_buffer imports OK")
except Exception as e:
    print(f"[FAIL] video_buffer: {e}")

try:
    from app.alert_handler import AlertHandler
    print("[OK] alert_handler imports OK")
except Exception as e:
    print(f"[FAIL] alert_handler: {e}")

try:
    from app.emergency_handler import EmergencyHandler
    print("[OK] emergency_handler imports OK")
except Exception as e:
    print(f"[FAIL] emergency_handler: {e}")

try:
    from app.fusion_engine import FusionEngine
    print("[OK] fusion_engine imports OK")
except Exception as e:
    print(f"[FAIL] fusion_engine: {e}")

print()
print("Diagnosis complete.")