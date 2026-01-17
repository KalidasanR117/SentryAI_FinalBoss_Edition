# sentry/reports/event_adapter.py
import os
import cv2

def severity_to_final(severity):
    if severity in ["CRITICAL", "HIGH"]:
        return "danger"
    if severity == "MEDIUM":
        return "suspicious"
    return "normal"


def save_event_screenshot(frame, event_id, out_dir="reports/screenshots"):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"event_{event_id}.png")
    cv2.imwrite(path, frame)
    return path


def adapt_events_for_pdf(events, frame_store):
    pdf_events = []

    for e in events:
        pdf_events.append({
            "frame": e.get("start_time"),
            "type": e.get("type"),
            "final": (
                "danger" if e["severity"] in ["CRITICAL", "HIGH"]
                else "suspicious"
            ),
            "confidence": e.get("confidence"),
            "persons": e.get("persons"),
            "cause": e.get("cause"),
            "screenshot": e.get("screenshot")  # ✅ THIS IS THE KEY
        })

    return pdf_events
