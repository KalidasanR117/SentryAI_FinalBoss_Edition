import cv2
import argparse
import sys
import time
import numpy as np
import warnings

# --- Sentry Core Imports ---
# Ensure 'sentry' is in your python path
from core.detector import Detector
from core.pose_buffer import PoseBuffer
from core.rule_engine import RuleEngine
from core.theft_detector import TheftDetector
from utils.visualization import draw_pose

# --- Face Analysis Imports ---
# Adjust these imports based on your specific folder structure if needed
try:
    from facial_analysis.models import SCRFD, Attribute
    from facial_analysis.utils.helpers import Face, draw_face_info
except ImportError:
    # Fallback if running directly inside facial_analysis folder (less likely for unified)
    from facial_analysis.models import SCRFD, Attribute
    from facial_analysis.utils.helpers import Face, draw_face_info

warnings.filterwarnings("ignore")

def load_face_models(det_path, attr_path):
    """Initialize face analysis models."""
    print(f"[INFO] Loading Face Models...\n\tDetection: {det_path}\n\tAttributes: {attr_path}")
    try:
        det_model = SCRFD(model_path=det_path)
        attr_model = Attribute(model_path=attr_path)
        return det_model, attr_model
    except Exception as e:
        print(f"[ERROR] Failed to load face models: {e}")
        sys.exit(1)

def main(args):
    # 1. Initialize Sentry (Pose & Object) Models
    print(f"[INFO] Loading Sentry Models...")
    sentry_detector = Detector(
        pose_model=args.pose_weights, 
        obj_model_path=args.obj_weights
    )
    pose_buffer = PoseBuffer(max_len=30)
    rule_engine = RuleEngine(history=30)
    theft_detector = TheftDetector(static_thresh=8, corr_thresh=60)

    # 2. Initialize Face Analysis Models
    face_det_model, face_attr_model = load_face_models(
        args.face_det_weights, 
        args.face_attr_weights
    )

    # 3. Setup Input Source
    input_source = args.source
    # Check if source is an integer (webcam index) or string (video path)
    if input_source.isdigit():
        input_source = int(input_source)
        is_video = True
    elif input_source.lower().endswith(('.jpg', '.png', '.jpeg')):
        is_video = False
    else:
        is_video = True

    cap = None
    image = None

    if is_video:
        cap = cv2.VideoCapture(input_source)
        if not cap.isOpened():
            print(f"[ERROR] Failed to open video source: {input_source}")
            return
    else:
        image = cv2.imread(input_source)
        if image is None:
            print(f"[ERROR] Failed to load image: {input_source}")
            return

    # 4. Setup Output Writer (Optional)
    out = None
    if args.output and is_video:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') # or 'XVID'
        out = cv2.VideoWriter(args.output, fourcc, fps, (width, height))

    print("[INFO] Starting Inference Loop...")
    
    prev_time = 0
    frame_idx = 0

    while True:
        if is_video:
            ret, frame = cap.read()
            if not ret:
                print("[INFO] End of stream.")
                break
        else:
            frame = image.copy()

        frame_idx += 1
        
        # We work on a copy for visualization to keep clean data for inference if needed
        vis_frame = frame.copy()

        # ===========================
        # PIPELINE 1: SENTRY (Pose + Theft)
        # ===========================
        persons, objects = sentry_detector.infer(frame)

        # Update tracking buffers
        for p in persons:
            pose_buffer.update(p["track_id"], p["keypoints"])
        
        # Run Logic Engines
        rule_results = rule_engine.update(persons, objects)
        theft_events = theft_detector.detect(persons, objects, frame_idx)

        # ===========================
        # PIPELINE 2: FACE ANALYSIS
        # ===========================
        # Detect faces
        boxes_list, points_list = face_det_model.detect(frame)

        # ===========================
        # VISUALIZATION MERGE
        # ===========================

        # --- Draw Sentry Pose & Skeleton ---
        vis_frame = draw_pose(vis_frame, [
            {
                "keypoints": p["keypoints"],
                "confidence": np.ones(len(p["keypoints"])), # dummy conf
                "track_id": p["track_id"]
            } for p in persons
        ])

        # --- Draw Sentry Actions ---
        for p in persons:
            tid = p["track_id"]
            if tid in rule_results:
                r = rule_results[tid]
                # Draw action text slightly above head
                if len(p["keypoints"]) > 0:
                    x, y = int(p["keypoints"][0][0]), int(p["keypoints"][0][1])
                    cv2.putText(vis_frame, r["action"], (x, y - 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, r["color"], 2)

        # --- Draw Sentry Objects ---
        for obj in objects:
            x1, y1, x2, y2 = map(int, obj["bbox"])
            label = f"{obj['cls']} ID{obj['track_id']} {obj['conf']:.2f}"
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
            cv2.putText(vis_frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        # --- Draw Sentry Alerts ---
        for e in theft_events:
            pid = e["person_id"]
            oid = e["object_id"]
            alert_text = f"ALERT! THEFT DETECTED: P{pid} <-> O{oid}"
            # Draw big red text
            cv2.putText(vis_frame, alert_text, (50, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

        # --- Draw Face Analysis Info ---
        # (Gender/Age boxes)
        if boxes_list is not None:
            for boxes, keypoints in zip(boxes_list, points_list):
                # boxes contains [x1, y1, x2, y2, score]
                *bbox, conf_score = boxes
                
                # Run attribute model on the specific face crop area
                # Note: passing the full frame + bbox usually works for these models
                gender, age = face_attr_model.get(frame, bbox)
                
                # Use helper to draw
                face_obj = Face(kps=keypoints, bbox=bbox, age=age, gender=gender)
                draw_face_info(vis_frame, face_obj)

        # --- FPS & Display ---
        if is_video:
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time + 1e-6)
            prev_time = curr_time
            cv2.putText(vis_frame, f"FPS: {int(fps)}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Save frame if needed
        if out:
            out.write(vis_frame)
        elif args.output and not is_video:
            cv2.imwrite(args.output, vis_frame)
            print(f"[INFO] Image saved to {args.output}")

        # Show frame
        cv2.imshow("Unified Sentry & Face Analysis", vis_frame)

        # Handle Exit
        if not is_video:
            cv2.waitKey(0)
            break
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    if cap: cap.release()
    if out: out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified Sentry Pose & Face Analysis")
    
    # Common Arguments
    parser.add_argument('--source', type=str, default="0", 
                        help='Video file path, image path, or camera index (0)')
    parser.add_argument('--output', type=str, default=None, 
                        help='Path to save output video/image')

    # Face Model Arguments
    parser.add_argument('--face-det-weights', type=str, default="D:\\Sentry_Final_Form\\sentry\\facial_analysis\\weights\\det_500m.onnx",
                        help='Path to face detection ONNX weights')
    parser.add_argument('--face-attr-weights', type=str, default="D:\\Sentry_Final_Form\\sentry\\facial_analysis\\weights\\genderage.onnx",
                        help='Path to face attribute ONNX weights')

    # Sentry/Pose Model Arguments
    parser.add_argument('--pose-weights', type=str, default=None,
                        help='Path to Pose estimation model (optional)')
    parser.add_argument('--obj-weights', type=str, default=None,
                        help='Path to Object detection model (optional)')

    args = parser.parse_args()
    
    main(args)