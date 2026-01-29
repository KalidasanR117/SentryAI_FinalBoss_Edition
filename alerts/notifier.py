import os
import cv2
from alerts.firebase_client import send_fcm_data_message
from alerts.config import (
    DEVICE_TOKENS,
    ALERT_TITLE_LIVE,
    ALERT_TITLE_OFFLINE,
    ALERT_TITLE_POSE
)
# 🔥 Import the new Telegram Bot
from alerts.telegram_notifier import telegram_bot

def _build_body(event: dict) -> str:
    label = event.get("type", "Unknown Event")
    severity = event.get("severity", event.get("final", "unknown")).upper()
    confidence = event.get("confidence")

    body = f"{label}\nSeverity: {severity}"

    if confidence is not None:
        body += f"\nConfidence: {confidence:.2f}"

    cause = event.get("cause", {})
    desc = cause.get("description")
    if desc:
        body += f"\n{desc}"

    return body

def send_critical_alert(*, event: dict, report_path: str, mode: str = "LIVE"):
    """
    Sends alerts to BOTH Firebase (Mobile App) and Telegram.
    """
    
    # ==========================
    # 1. TELEGRAM ALERT (NEW)
    # ==========================
    # We need to load the image from disk because Telegram needs the actual bytes
    # report_path here usually points to the screenshot.jpg for live alerts
    frame = None
    if report_path and os.path.exists(report_path):
        # Only try to load if it looks like an image
        if report_path.lower().endswith(('.jpg', '.jpeg', '.png')):
            try:
                frame = cv2.imread(report_path)
            except Exception as e:
                print(f"[NOTIFIER] Could not read image for Telegram: {e}")

    # Prepare message
    event_type = event.get("type", "Unknown")
    severity = event.get("severity", "CRITICAL")
    message = f"Mode: {mode}\nType: {event_type}"
    
    # Send to Telegram (Async inside the class)
    telegram_bot.send_alert(frame, severity, message)


    # ==========================
    # 2. FIREBASE ALERT (EXISTING)
    # ==========================
    if not DEVICE_TOKENS:
        # If no tokens, we just log and return (Telegram already sent above)
        # print("[ALERT] No device tokens configured")
        return

    if mode == "OFFLINE":
        title = ALERT_TITLE_OFFLINE
    elif mode == "POSE":
        title = ALERT_TITLE_POSE
    else:
        title = ALERT_TITLE_LIVE

    body = _build_body(event)

    screenshots = []
    if "screenshots" in event:
        screenshots = event["screenshots"]
    elif "screenshot" in event and event["screenshot"]:
        screenshots = [event["screenshot"]]

    for token in DEVICE_TOKENS:
        try:
            send_fcm_data_message(
                title=title,
                body=body,
                token=token,
                report_path=report_path,
                screenshots=screenshots,
                extra={
                    "severity": event.get("severity", ""),
                    "event_type": event.get("type", ""),
                    "source": mode
                }
            )
            print("[FCM] Sent to device:", token[:12], "...")
        except Exception as e:
            print("[FCM ERROR]", e)