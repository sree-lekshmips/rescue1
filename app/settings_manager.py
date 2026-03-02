"""
SentinelAI – Settings Manager
"""
import json
import threading
from pathlib import Path

SETTINGS_FILE = Path("settings.json")

DEFAULT_SETTINGS = {
    "name":          "",
    "email":         "",
    "gmail_sender":  "",
    "gmail_apppass": "",
}


class SettingsManager:
    def __init__(self):
        self._lock = threading.Lock()
        if not SETTINGS_FILE.exists():
            SETTINGS_FILE.write_text(json.dumps(DEFAULT_SETTINGS, indent=2))
        else:
            try:
                existing = json.loads(SETTINGS_FILE.read_text())
                merged   = {**DEFAULT_SETTINGS, **existing}
                SETTINGS_FILE.write_text(json.dumps(merged, indent=2))
            except Exception:
                SETTINGS_FILE.write_text(json.dumps(DEFAULT_SETTINGS, indent=2))

    def load(self) -> dict:
        with self._lock:
            try:
                data = json.loads(SETTINGS_FILE.read_text())
                return {**DEFAULT_SETTINGS, **data}
            except Exception:
                return dict(DEFAULT_SETTINGS)

    def save(self, data: dict) -> bool:
        try:
            current = self.load()
            merged  = {**current,
                       **{k: str(v) for k, v in data.items() if k in DEFAULT_SETTINGS}}
            with self._lock:
                SETTINGS_FILE.write_text(json.dumps(merged, indent=2))
            # Print to terminal so you can verify
            print(f"[Settings] Saved: name={merged['name']} "
                  f"email={merged['email']} "
                  f"sender={merged['gmail_sender']} "
                  f"apppass={'SET' if merged['gmail_apppass'] else 'MISSING'}")
            return True
        except Exception as e:
            print(f"[Settings] Save error: {e}")
            return False

    def is_email_configured(self) -> bool:
        s = self.load()
        return all([s.get("email"), s.get("gmail_sender"), s.get("gmail_apppass")])