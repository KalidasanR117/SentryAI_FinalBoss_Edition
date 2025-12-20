# sentry/core/detector.py
from ultralytics import YOLO
import numpy as np

class Detector:
    def __init__(self,
                 pose_model="D:\Sentry_Final_Form\sentry\models\yolo11l-pose.pt",
                 obj_model="D:\Sentry_Final_Form\sentry\models\yolo11m.pt",
                 device="cuda"):
        self.pose_model = YOLO(pose_model).to(device)
        self.obj_model  = YOLO(obj_model).to(device)
        self.pose_model.fuse()
        self.obj_model.fuse()

    def infer(self, frame):
        persons, objects = [], []

        # ==== POSE ====
        pose_results = self.pose_model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            verbose=False
        )

        for r in pose_results:
            if r.keypoints is None or r.boxes.id is None:
                continue

            for i in range(len(r.boxes.id)):
                kpts = r.keypoints.xy[i].cpu().numpy()
                conf = r.keypoints.conf[i].cpu().numpy()
                tid  = int(r.boxes.id[i])

                persons.append({
                    "track_id": tid,
                    "keypoints": kpts,
                    "confidence": conf
                })

        # ==== OBJECTS ====
        obj_results = self.obj_model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            verbose=False
        )

        for r in obj_results:
            if r.boxes.id is None:
                continue
            for i in range(len(r.boxes.id)):
                objects.append({
                    "track_id": int(r.boxes.id[i]),
                    "cls": int(r.boxes.cls[i]),
                    "conf": float(r.boxes.conf[i]),
                    "bbox": r.boxes.xyxy[i].cpu().numpy()
                })

        return persons, objects
