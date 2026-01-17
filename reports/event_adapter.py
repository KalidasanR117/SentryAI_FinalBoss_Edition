# sentry/reports/event_adapter.py
import os
import cv2
SKIP_EVENT_TYPES = {
    "Tracking",
    "Normal",
    "Normal Motion"
}

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
        # 🚫 Skip non-reportable runtime states
        if e.get("type") in SKIP_EVENT_TYPES:
            continue

        pdf_events.append({
            "frame": e.get("start_time"),
            "type": e.get("type"),
            "final": severity_to_final(e["severity"]),
            "confidence": e.get("confidence"),
            "persons": e.get("persons"),
            "cause": e.get("cause"),
            "screenshot": e.get("screenshot")
        })

    return pdf_events

    return pdf_events
def pick_representative_frames(event, max_frames=3):
    frames = event.get("frames", [])
    if not frames:
        return []

    if len(frames) <= max_frames:
        return frames

    return [
        frames[0],                 # start
        frames[len(frames)//2],    # middle
        frames[-1]                 # end
    ]
