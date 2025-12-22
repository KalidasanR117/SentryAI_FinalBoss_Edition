# sentry/core/detector.py

from ultralytics import YOLO
import numpy as np

# Object class names of your custom model
ALL_OBJ_CLASSES = {
    0: "fire",
    1: "gun",
    2: "helmet",
    3: "knife",
    4: "mask",
    5: "person"
}

# Only want to keep these
TARGET_OBJ_CLASSES = ["gun", "knife"]

class Detector:
    def __init__(self,
                 pose_model=None,
                 obj_model_path=None,
                 device="cuda"):
        """
        Unified detector with object tracking integration.
        """

        # ==== POSE MODEL ====
        if pose_model is None:
            pose_model = "D:/Sentry_Final_Form/sentry/models/yolo11n-pose.pt"
        self.pose_model = YOLO(pose_model).to(device)
        self.pose_model.fuse()

        # ==== OBJECT MODEL ====
        if obj_model_path is None:
            obj_model_path = "D:/Sentry_Final_Form/sentry/models/object_yolo.pt"
        self.obj_model = YOLO(obj_model_path).to(device)
        self.obj_model.fuse()

        # ByteTrack config
        self.obj_tracker_config = "bytetrack.yaml"

    def infer(self, frame, obj_conf=0.25):
        """
        Runs tracked inference.

        Returns:
            persons: list of dicts with keys track_id, keypoints, confidence
            objects: list of dicts with keys track_id, cls, conf, bbox
        """

        persons = []
        objects = []

        # === POSE + PERSON TRACKING ===
        pose_results = self.pose_model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",  # use ByteTrack for persons too
            verbose=False
        )

        for r in pose_results:
            if r.keypoints is None or r.boxes.id is None:
                continue
            for i in range(len(r.boxes.id)):
                tid = int(r.boxes.id[i])
                kpts = r.keypoints.xy[i].cpu().numpy()
                conf = r.keypoints.conf[i].cpu().numpy()
                persons.append({
                    "track_id": tid,
                    "keypoints": kpts,
                    "confidence": conf
                })

        # === OBJECT DETECTION + TRACKING ===
        obj_results = self.obj_model.track(
            frame,
            persist=True,
            tracker=self.obj_tracker_config,
            verbose=False
        )

        for r in obj_results:
            if r.boxes.id is None:
                continue
            for i in range(len(r.boxes.id)):
                tid = int(r.boxes.id[i])
                cls_id = int(r.boxes.cls[i])
                conf = float(r.boxes.conf[i])
                bbox = r.boxes.xyxy[i].cpu().numpy()

                # Map class id → name
                cls_name = ALL_OBJ_CLASSES.get(cls_id, f"cls_{cls_id}")

                # Only keep knife and gun
                if cls_name not in TARGET_OBJ_CLASSES:
                    continue

                objects.append({
                    "track_id": tid,
                    "cls": cls_name,
                    "conf": conf,
                    "bbox": bbox
                })

        return persons, objects
