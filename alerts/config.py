from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# 🔑 Firebase
FIREBASE_KEY_PATH = BASE_DIR / "alerts" / "firebase_key.json"

# 📱 Device tokens (can be DB later)
DEVICE_TOKENS = [
    "fXvRVU5mRpW2xVpAeC0Vfx:APA91bHfFP4xECFmSpXLCcWhSnPRPIiwcThduxJQQvgnMO08GoJleIJ7xvpc7vCUITja3alemEbzzVql_-_zasgOn0F8T13t_DLJBmcieqMLzjJFoG8nWRU"
]
# 🔔 Alert settings
ALERT_TITLE_LIVE = "🚨 Sentry LIVE Alert"
ALERT_TITLE_OFFLINE = "🚨 Sentry OFFLINE Alert"
ALERT_TITLE_POSE = "🚨 Sentry POSE Alert"
