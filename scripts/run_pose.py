# sentry/scripts/run_pose.py

import cv2
import time
import sys
import numpy as np

from sentry.core.detector import Detector
from sentry.core.pose_buffer import PoseBuffer
from sentry.core.rule_engine import RuleEngine
from sentry.utils.visualization import draw_pose
from sentry.core.theft_detector import TheftDetector


def run(source=0, display_fps=True, pose_model_path=None, obj_model_path=None):
    # initialize detector with optional custom weights
    detector = Detector(
        pose_model=pose_model_path if pose_model_path else None,
        obj_model_path=obj_model_path if obj_model_path else None
    )

    pose_buffer = PoseBuffer(max_len=30)
    rule_engine = RuleEngine(history=30)
    theft_detector = TheftDetector(static_thresh=8, corr_thresh=60)

    cap = cv2.VideoCapture(source)
    prev_time = 0
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print(f"[INFO] End of stream or cannot open: {source}")
            break

        frame_idx += 1

        # === RUN MODEL INFERENCE ===
        persons, objects = detector.infer(frame)

        # update pose buffer
        for p in persons:
            pose_buffer.update(p["track_id"], p["keypoints"])

        # update rule engine
        results = rule_engine.update(persons, objects)

        # update and detect theft
        events = theft_detector.detect(persons, objects, frame_idx)

        # === VISUALIZATION ===

        # Draw pose skeletons + IDs
        frame_out = draw_pose(frame.copy(), [
            {
                "keypoints": p["keypoints"],
                "confidence": np.ones(len(p["keypoints"])),
                "track_id": p["track_id"]
            } for p in persons
        ])

        # Draw action labels
        for p in persons:
            tid = p["track_id"]
            if tid in results:
                r = results[tid]
                x, y = int(p["keypoints"][0][0]), int(p["keypoints"][0][1])

                cv2.putText(frame_out, r["action"], (x, y - 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            r["color"], 2)

        # Draw object bounding boxes
        for obj in objects:
            x1, y1, x2, y2 = map(int, obj["bbox"])
            label = f"{obj['cls']} ID{obj['track_id']} {obj['conf']:.2f}"

            cv2.rectangle(frame_out, (x1, y1), (x2, y2),
                          (255, 255, 0), 2)
            cv2.putText(frame_out, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        # Draw theft alerts
        for e in events:
            pid = e["person_id"]
            oid = e["object_id"]

            alert_text = f"THEFT! P{pid} took O{oid}"
            cv2.putText(frame_out, alert_text, (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                        (0, 0, 255), 3)

            print(f"[THEFT] Frame {e['frame']} ({e['timestamp']}): "
                  f"Person {pid} took object {oid}")

        # Draw FPS
        if display_fps:
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time + 1e-6)
            prev_time = curr_time
            cv2.putText(frame_out, f"FPS: {int(fps)}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Sentry", frame_out)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # allow passing video path + custom model weights
    if len(sys.argv) >= 2:
        input_src = sys.argv[1]
        pose_model = None
        obj_model = None

        # optional args: pose model second, object model third
        if len(sys.argv) >= 3:
            pose_model = sys.argv[2]
        if len(sys.argv) >= 4:
            obj_model = sys.argv[3]

        print(f"[INFO] Running on: {input_src} | pose: {pose_model} | obj: {obj_model}")
        run(input_src, pose_model_path=pose_model, obj_model_path=obj_model)
    else:
        print("[INFO] Running on webcam (0)")
        run(0)
