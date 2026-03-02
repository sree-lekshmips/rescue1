"""
SentinelAI – alert_handler.py
Sends 2 emails on EMERGENCY_CONFIRMED:
  1. Instant text alert (fires immediately)
  2. Video clip attached (fires after clip is ready)
Credentials read from settings.json (saved via settings page).
"""

import os
import time
import smtplib
import threading
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
from datetime import datetime
from typing import Optional

ALERT_COOLDOWN_SECONDS = 10
MAX_VIDEO_MB           = 23.0


class AlertHandler:

    def __init__(self, settings_manager):
        self._settings_manager = settings_manager
        self._last_alert_time  = 0.0
        self._lock             = threading.Lock()
        self.is_sending        = False

    def send_alert(self, clip_path=None, torso_angle=0.0,
                   fall_confidence=0.0, risk_score=0.0) -> bool:
        with self._lock:
            now = time.time()
            if now - self._last_alert_time < ALERT_COOLDOWN_SECONDS:
                remaining = ALERT_COOLDOWN_SECONDS - (now - self._last_alert_time)
                print(f"[AlertHandler] Cooldown active — {remaining:.0f}s remaining.")
                return False
            self._last_alert_time = now

        threading.Thread(
            target=self._dispatch,
            args=(clip_path, torso_angle, fall_confidence, risk_score),
            daemon=True,
        ).start()
        return True

    def _dispatch(self, clip_path, torso_angle, fall_confidence, risk_score):
        self.is_sending = True
        try:
            settings  = self._settings_manager.load()
            to_email  = settings.get("email", "").strip()
            sender    = (os.environ.get("GMAIL_SENDER", "")
                         or settings.get("gmail_sender", "")).strip()
            app_pass  = (os.environ.get("GMAIL_APP_PASSWORD", "")
                         or settings.get("gmail_apppass", "")).strip()
            name      = settings.get("name", "Emergency Contact")

            if not all([to_email, sender, app_pass]):
                print("[AlertHandler] Missing email config — check Settings page.")
                print(f"  to_email : '{to_email}'")
                print(f"  sender   : '{sender}'")
                print(f"  app_pass : '{'SET' if app_pass else 'MISSING'}'")
                return

            timestamp = datetime.now().strftime("%d %b %Y at %I:%M %p")
            print(f"[AlertHandler] Sending alert to {name} <{to_email}>")

            # Email 1 — instant text alert
            self._send_instant(sender, app_pass, to_email, name, timestamp,
                               torso_angle, fall_confidence, risk_score)

            # Email 2 — video attachment (wait for clip to finish writing)
            if clip_path:
                ready = self._wait_for_clip(clip_path, timeout=45)
                if ready:
                    self._send_video(sender, app_pass, to_email, name,
                                     timestamp, clip_path)
                else:
                    print("[AlertHandler] Clip not ready — video email skipped.")
        finally:
            self.is_sending = False

    # ── Email 1: instant alert ─────────────────────────────────────────────────
    def _send_instant(self, sender, app_pass, to_email, name, timestamp,
                      torso_angle, fall_confidence, risk_score):
        body = (
            f"Dear {name},\n\n"
            f"A FALL HAS BEEN DETECTED by SentinelAI.\n\n"
            f"Time          : {timestamp}\n"
            f"Torso Angle   : {torso_angle:.1f}°\n"
            f"Confidence    : {fall_confidence:.0%}\n"
            f"Risk Score    : {risk_score:.0%}\n"
            f"Location      : Home Monitoring Camera\n\n"
            f"NO MOVEMENT was detected for 10 seconds after the fall.\n"
            f"Please check on your family member IMMEDIATELY.\n\n"
            f"A second email with the video clip will arrive shortly.\n\n"
            f"Emergency : 112\n"
            f"Ambulance : 108\n\n"
            f"— SentinelAI (automated alert)"
        )
        msg            = MIMEText(body, "plain")
        msg["From"]    = sender
        msg["To"]      = to_email
        msg["Subject"] = f"🚨 FALL ALERT — SentinelAI — {timestamp}"

        self._smtp_send(sender, app_pass, to_email, msg, "Instant alert")

    # ── Email 2: video attachment ──────────────────────────────────────────────
    def _send_video(self, sender, app_pass, to_email, name, timestamp, clip_path):
        mb = os.path.getsize(clip_path) / 1_048_576
        print(f"[AlertHandler] Attaching video ({mb:.1f} MB)...")

        if mb > MAX_VIDEO_MB:
            print(f"[AlertHandler] Clip too large ({mb:.1f} MB) — skipping.")
            return

        body = (
            f"Dear {name},\n\n"
            f"Attached is the video clip of the fall incident.\n\n"
            f"Time     : {timestamp}\n"
            f"Duration : ~10 seconds (5s before + 5s after fall)\n\n"
            f"— SentinelAI (automated alert)"
        )
        msg            = MIMEMultipart()
        msg["From"]    = sender
        msg["To"]      = to_email
        msg["Subject"] = f"Fall Video Clip — SentinelAI — {timestamp}"
        msg.attach(MIMEText(body, "plain"))

        with open(clip_path, "rb") as f:
            part = MIMEBase("application", "octet-stream")
            part.set_payload(f.read())
        encoders.encode_base64(part)
        part.add_header("Content-Disposition",
                        f'attachment; filename="{os.path.basename(clip_path)}"')
        msg.attach(part)

        self._smtp_send(sender, app_pass, to_email, msg, "Video email")

    # ── SMTP (friend's working method — STARTTLS on port 587) ─────────────────
    def _smtp_send(self, sender, app_pass, to_email, msg, label="Email"):
        try:
            print(f"[AlertHandler] Connecting to Gmail SMTP...")
            server = smtplib.SMTP("smtp.gmail.com", 587, timeout=120)
            server.ehlo()
            server.starttls()
            server.ehlo()
            server.login(sender, app_pass)
            server.send_message(msg)
            server.quit()
            print(f"[AlertHandler] {label} sent ✓ → {to_email}")
        except smtplib.SMTPAuthenticationError:
            print("[AlertHandler] Gmail auth FAILED — check sender email and App Password in Settings.")
        except Exception as e:
            print(f"[AlertHandler] {label} error: {type(e).__name__}: {e}")

    # ── Wait for clip to finish writing ────────────────────────────────────────
    @staticmethod
    def _wait_for_clip(path: str, timeout: float = 45.0) -> bool:
        print("[AlertHandler] Waiting for video clip to finish writing...")
        deadline  = time.time() + timeout
        prev_size = -1
        while time.time() < deadline:
            if os.path.exists(path):
                size = os.path.getsize(path)
                if size > 0 and size == prev_size:
                    print(f"[AlertHandler] Clip ready: {size/1_048_576:.1f} MB")
                    return True
                prev_size = size
            time.sleep(1)
        print("[AlertHandler] Clip timed out.")
        return False