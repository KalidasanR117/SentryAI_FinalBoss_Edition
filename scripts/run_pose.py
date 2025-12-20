# sentry/scripts/run_pose.py
import cv2
import time
import sys
import numpy as np

from sentry.core.detector import Detector
from sentry.core.pose_buffer import PoseBuffer
from sentry.core.rule_engine import RuleEngine
from sentry.utils.visualization import draw_pose
from sentry.core.theft_detector import TheftDetector  # fixed import

def run(source=0, display_fps=True, pose_model_path=None):
    # initialize models
    if pose_model_path:
        detector = Detector(pose_model=pose_model_path)
    else:
        detector = Detector()

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

        # --- INFERENCE ---
        persons, objects = detector.infer(frame)

        # update pose buffer
        for p in persons:
            pose_buffer.update(p["track_id"], p["keypoints"])

        # update rule engine
        results = rule_engine.update(persons, objects)

        # update and detect theft
        events = theft_detector.detect(persons, objects)

        # --- VISUALIZATION ---
        # draw pose with IDs
        frame_out = draw_pose(frame.copy(), [
            {
                "keypoints": p["keypoints"],
                "confidence": np.ones(len(p["keypoints"])),
                "track_id": p["track_id"]
            } for p in persons
        ])

        # draw actions with severity labels
        for p in persons:
            tid = p["track_id"]
            if tid in results:
                r = results[tid]
                x, y = int(p["keypoints"][0][0]), int(p["keypoints"][0][1])

                label = f"{r['action']}"
                cv2.putText(frame_out, label, (x, y - 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            r["color"], 2)

        # draw object boxes
        for obj in objects:
            x1, y1, x2, y2 = obj["bbox"]
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            cv2.rectangle(frame_out, (x1, y1), (x2, y2),
                          (255, 255, 0), 2)
            cv2.putText(frame_out, str(obj["cls"]), (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)

        # draw theft alerts
        events = theft_detector.detect(persons, objects, frame_idx)
        for e in events:
            print(f"[THEFT] Frame {e['frame']} ({e['timestamp']}): "
                f"Person {e['person_id']} took object {e['object_id']}")

            pid = e["person_id"]
            oid = e["object_id"]
            cv2.putText(frame_out, f"THEFT DETECTED! P{pid} took O{oid}",
                        (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                        (0, 0, 255), 3)
            print(f"[THEFT CONFIRMED] Frame {frame_idx}: Person {pid} stole object {oid}")

        # draw FPS
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
    if len(sys.argv) > 1:
        input_src = sys.argv[1]
        pose_model_arg = None
        if len(sys.argv) > 2:
            pose_model_arg = sys.argv[2]

        print(f"[INFO] Running on: {input_src} with model: {pose_model_arg}")
        run(input_src, pose_model_path=pose_model_arg)
    else:
        print("[INFO] Running on webcam (0)")
        run(0)
