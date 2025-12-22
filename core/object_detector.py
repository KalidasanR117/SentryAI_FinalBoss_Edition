# sentry/core/object_detector.py

from ultralytics import YOLO
import numpy as np

# Replace with your trained YOLOv12 weights path
MODEL_PATH = "D:\\Sentry_Final_Form\\sentry\\models\\object_yolo.pt"

# The complete class list your model outputs
ALL_OBJ_CLASSES = {
    0: "fire",
    1: "gun",
    2: "helmet",
    3: "knife",
    4: "mask",
    5: "person"
}

# Only want to detect gun and knife
TARGET_CLASSES = ["gun", "knife"]


class ObjectDetector:
    def __init__(self, model_path=MODEL_PATH, device="cuda"):
        """
        Loads the custom YOLO model.
        """
        self.model = YOLO(model_path).to(device)
        self.model.fuse()

    def infer(self, frame, conf_thresh=0.25):
        """
        Runs object detection on a single frame.

        Returns:
          objects: list of dict with keys:
            - bbox: [x1, y1, x2, y2]
            - cls: class_name
            - conf: confidence score
            - track_id: None (no tracker here)
        """
        results = self.model(frame)[0]  # single image

        objects = []
        boxes = results.boxes.xyxy.cpu().numpy()
        cls_ids = results.boxes.cls.cpu().numpy().astype(int)
        confs = results.boxes.conf.cpu().numpy()

        for bbox, cls_id, conf in zip(boxes, cls_ids, confs):

            if conf < conf_thresh:
                continue

            # map class id → class name
            cls_name = ALL_OBJ_CLASSES.get(cls_id, f"cls_{cls_id}")

            # skip classes not in your target list
            if cls_name not in TARGET_CLASSES:
                continue

            objects.append({
                "bbox": bbox,
                "cls": cls_name,
                "conf": float(conf),
                "track_id": None
            })

        return objects
