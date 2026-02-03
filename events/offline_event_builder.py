import os
import cv2

def build_offline_events(
    frames,
    labels,
    scores,
    fps,
    screenshot_dir
):
    events = []
    current = None

    for i, label in enumerate(labels):
        time_sec = round(i / fps, 2)

        if label == "Normal":
            if current:
                current["end_time"] = time_sec
                events.append(current)
                current = None
            continue

        severity = "danger" if "Fight" in label else "suspicious"

        if current is None:
            screenshot_path = os.path.join(
                screenshot_dir,
                f"offline_event_{i}.jpg"
            )
            cv2.imwrite(screenshot_path, frames[i])

            current = {
                "frame": i,
                "start_time": time_sec,
                "end_time": time_sec,
                "type": label,
                "final": severity,
                "confidence": round(scores[i], 2),
                "persons": ["Unknown"],
                "screenshot": screenshot_path,
                "cause": {
                    "trigger": "VIDEOMAE",
                    "rule_name": "VideoMAE-Violence",
                    "description": "Transformer-based spatiotemporal violence detection",
                    "joints_involved": [],
                    "metrics": {
                        "violence_score": f"{round(scores[i], 2):.2f}",
                        "confidence": f"{round(scores[i] * 100, 1)}%"
                    }
                }
            }
        else:
            current["end_time"] = time_sec

    if current:
        events.append(current)

    return events