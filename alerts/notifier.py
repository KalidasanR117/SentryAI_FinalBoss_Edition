from alerts.firebase_client import send_fcm_data_message
from alerts.config import (
    DEVICE_TOKENS,
    ALERT_TITLE_LIVE,
    ALERT_TITLE_OFFLINE,
    ALERT_TITLE_POSE
)


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
    if not DEVICE_TOKENS:
        print("[ALERT] No device tokens configured")
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
            print("[ALERT] Sent to device:", token[:12], "...")
        except Exception as e:
            print("[ALERT ERROR]", e)
