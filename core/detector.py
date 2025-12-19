# sentry/core/detector.py
from ultralytics import YOLO
import numpy as np

class PoseDetector:
    def __init__(self,
                 model_path=r"D:\\Sentry_Final_Form\\sentry\\models\\yolo11l-pose.pt",
                 device="cuda"):
        """
        YOLOv11 Pose Detector with ByteTrack
        """
        self.model = YOLO(model_path).to(device)
        self.model.fuse()
        self.device = device

    def infer(self, frame):
        """
        Returns list of dicts:
        {
          "track_id": int,
          "keypoints": (17,2),
          "confidence": (17,)
        }
        """
        results = self.model.track(
            frame,
            tracker="bytetrack.yaml",
            persist=True,
            verbose=False
        )

        detections = []

        for r in results:
            if r.keypoints is None or r.boxes.id is None:
                continue

            keypoints = r.keypoints.xy.cpu().numpy()     # (N,17,2)
            confs     = r.keypoints.conf.cpu().numpy()   # (N,17)
            track_ids = r.boxes.id.cpu().numpy()         # (N,)

            for i in range(len(track_ids)):
                detections.append({
                    "track_id": int(track_ids[i]),
                    "keypoints": keypoints[i],
                    "confidence": confs[i]
                })

        return detections
