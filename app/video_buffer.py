"""
SentinelAI – video_buffer.py
==============================
Extracted from friend's project — unchanged except:
  - Moved into app/ package
  - Removed alert_config import; config passed via constructor
  - Added stop() for clean shutdown
"""

import os
import queue
import threading
import collections
from datetime import datetime
from typing import Optional

import cv2


class VideoBuffer:

    def __init__(
        self,
        fps          : float = 20.0,
        pre_seconds  : float = 5.0,
        post_seconds : float = 5.0,
        clips_folder : str   = "fall_clips",
        frame_size   : tuple = (640, 480),
    ):
        self.fps          = fps
        self.pre_seconds  = pre_seconds
        self.post_seconds = post_seconds
        self.clips_folder = clips_folder
        self.frame_size   = frame_size

        max_pre = int(fps * pre_seconds)
        self._pre_buffer : collections.deque = collections.deque(maxlen=max_pre)

        self._post_needed   = int(fps * post_seconds)
        self._post_left     = 0
        self._post_buffer   : list = []
        self._capturing_post = False

        self._last_clip_path : Optional[str] = None
        self._pending_path   : Optional[str] = None

        self._save_queue  = queue.Queue()
        self._save_thread = threading.Thread(
            target=self._save_worker, daemon=True, name="VideoBuffer-Save")

        os.makedirs(clips_folder, exist_ok=True)

    def start(self):
        self._save_thread.start()

    def stop(self):
        self._save_queue.put(None)

    def push(self, frame) -> None:
        self._pre_buffer.append(frame.copy())
        if self._capturing_post:
            self._post_buffer.append(frame.copy())
            self._post_left -= 1
            if self._post_left <= 0:
                self._flush_clip()

    def trigger_save(self) -> str:
        """Call when fall confirmed. Returns path being written to."""
        if self._capturing_post:
            return self._last_clip_path

        ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"fall_{ts}.avi"
        path     = os.path.join(self.clips_folder, filename)

        self._pending_path   = path
        self._post_buffer    = []
        self._post_left      = self._post_needed
        self._capturing_post = True
        self._last_clip_path = path

        print(f"[VideoBuffer] Recording → {path}")
        return path

    def _flush_clip(self):
        pre  = list(self._pre_buffer)
        post = list(self._post_buffer)
        path = self._pending_path
        self._save_queue.put((pre, post, path))
        self._capturing_post = False
        self._post_buffer    = []
        self._post_left      = 0

    def _save_worker(self):
        while True:
            item = self._save_queue.get()
            if item is None:
                break

            pre, post, path = item
            frames = pre + post
            if not frames:
                print("[VideoBuffer] No frames to save.")
                continue

            out_w, out_h = 480, 270
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            writer = cv2.VideoWriter(path, fourcc, self.fps, (out_w, out_h))

            if not writer.isOpened():
                fourcc = cv2.VideoWriter_fourcc(*"XVID")
                writer = cv2.VideoWriter(path, fourcc, self.fps, (out_w, out_h))

            for f in frames:
                writer.write(cv2.resize(f, (out_w, out_h)))
            writer.release()

            mb = os.path.getsize(path) / 1_048_576
            print(f"[VideoBuffer] Saved: {path}  ({mb:.1f} MB, {len(frames)} frames)")
            self._save_queue.task_done()

    @property
    def last_clip_path(self) -> Optional[str]:
        return self._last_clip_path

    @property
    def is_capturing(self) -> bool:
        return self._capturing_post