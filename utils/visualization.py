# sentry/utils/visualization.py
import cv2
import numpy as np

# COCO-17 skeleton
SKELETON = [
    (5,7),(7,9),(6,8),(8,10),
    (5,6),(11,12),(11,13),
    (12,14),(13,15),(14,16)
]

def draw_pose(frame, detections, conf_threshold=0.3):
    """
    Draw pose skeleton + track IDs
    """
    for det in detections:
        kp = det["keypoints"]
        conf = det["confidence"]
        tid = det["track_id"]

        # joints
        for i, (x, y) in enumerate(kp):
            if conf[i] >= conf_threshold:
                cv2.circle(frame, (int(x), int(y)), 3, (0,255,0), -1)

        # skeleton
        for a, b in SKELETON:
            if conf[a] >= conf_threshold and conf[b] >= conf_threshold:
                pt1 = tuple(kp[a].astype(int))
                pt2 = tuple(kp[b].astype(int))
                cv2.line(frame, pt1, pt2, (255,0,0), 2)

        # ID label
        cx, cy = kp.mean(axis=0).astype(int)
        cv2.putText(frame, f"ID:{tid}", (cx, cy-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

    return frame
