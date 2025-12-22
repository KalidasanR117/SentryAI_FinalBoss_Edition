import os
import cv2
import argparse
import warnings
import numpy as np
import onnxruntime as ort
import pickle
import time

from core.detector import Detector
from core.pose_buffer import PoseBuffer
from core.rule_engine import RuleEngine
from core.theft_detector import TheftDetector
from utils.visualization import draw_pose

from facial_analysis.models import SCRFD
from facial_analysis.utils.helpers import Face, draw_face_info



warnings.filterwarnings("ignore")

# ===========================
# FACE DATABASE BUILDER
# ===========================

def preprocess_arcface(face_img):
    """Prepare a face crop for ArcFace (112×112 RGB normalized)."""
    face_resized = cv2.resize(face_img, (112, 112))
    face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
    face_norm = (face_rgb - 127.5) / 128.0
    face_input = np.transpose(face_norm, (2, 0, 1)).astype(np.float32)
    face_input = np.expand_dims(face_input, 0)
    return face_input

def build_face_db_from_gallery(scrfd_model, arc_session, gallery_dir="D:/Sentry_Final_Form/sentry/facial_analysis/face_gallery"):
    """
    Build an ArcFace embedding database from a gallery folder.
    """
    face_db = {}
    arc_input_name = arc_session.get_inputs()[0].name

    print(f"[INFO] Building face database from folder: {gallery_dir}")

    if not os.path.isdir(gallery_dir):
        print(f"[WARN] Gallery folder does not exist: {gallery_dir}")
        return face_db

    for person_name in os.listdir(gallery_dir):
        person_dir = os.path.join(gallery_dir, person_name)
        if not os.path.isdir(person_dir):
            continue

        embeddings = []
        for img_name in os.listdir(person_dir):
            img_path = os.path.join(person_dir, img_name)
            img = cv2.imread(img_path)
            if img is None:
                print(f"[WARN] Failed to read {img_path}")
                continue

            boxes_list, _ = scrfd_model.detect(img)
            if len(boxes_list) == 0:
                print(f"[WARN] No face detected in {img_path}")
                continue

            # Use the first detected face
            box = boxes_list[0]
            x1, y1, x2, y2 = map(int, box[:4])
            face_crop = img[y1:y2, x1:x2]
            if face_crop.size == 0:
                print(f"[WARN] Empty crop for {img_path}")
                continue

            arc_input = preprocess_arcface(face_crop)
            emb = arc_session.run(None, {arc_input_name: arc_input})[0].flatten()
            emb /= np.linalg.norm(emb + 1e-10)
            embeddings.append(emb)

        if embeddings:
            face_db[person_name] = embeddings
            print(f"[INFO] Added {len(embeddings)} embeddings for {person_name}")

    # Save database
    with open("face_db.pkl", "wb") as f:
        pickle.dump(face_db, f)
    print("[INFO] Saved face_db.pkl")

    return face_db

# ===========================
# NAME MATCHING
# ===========================

def find_name_for_embedding(emb, face_db, threshold=0.5):
    best_name, best_score = None, threshold
    for name, refs in face_db.items():
        for ref in refs:
            score = float(np.dot(emb, ref) / (np.linalg.norm(emb)*np.linalg.norm(ref)+1e-8))
            if score > best_score:
                best_score, best_name = score, name
    return best_name, best_score

# ===========================
# MODEL LOADER
# ===========================

def load_models(scrfd_path, arcface_path):
    scrfd_model = SCRFD(model_path=scrfd_path)
    arcface_session = ort.InferenceSession(
        arcface_path, providers=["CUDAExecutionProvider","CPUExecutionProvider"]
    )
    print(f"[INFO] Loaded SCRFD and ArcFace")
    return scrfd_model, arcface_session

# ===========================
# FACE + RECOGNITION
# ===========================

def process_face_recognition(frame, scrfd_model, arcface_session, face_db):
    boxes_list, points_list = scrfd_model.detect(frame)
    arc_input_name = arcface_session.get_inputs()[0].name

    for boxes, keypoints in zip(boxes_list, points_list):
        *bbox, conf = boxes
        x1,y1,x2,y2 = map(int,bbox)

        if x2 <= x1 or y2 <= y1:
            continue
        if x1 < 0 or y1 < 0:
            continue

        face_crop = frame[y1:y2, x1:x2]
        if face_crop.size == 0:
            continue

        arc_input = preprocess_arcface(face_crop)
        emb = arcface_session.run(None, {arc_input_name: arc_input})[0].flatten()
        emb /= np.linalg.norm(emb+1e-8)

        name, score = find_name_for_embedding(emb, face_db)
        label = f"{name} ({score:.2f})" if name else "Unknown"

        face = Face(kps=keypoints, bbox=bbox, age=None, gender=None, embedding=emb.tolist())
        draw_face_info(frame, face)
        cv2.putText(frame, label, (x1, y1 - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    return frame

# ===========================
# RUN
# ===========================

def run(source=0,
        scrfd_weights=None,
        arcface_weights=None,
        pose_model_path=None,
        obj_model_path=None):

    if isinstance(source, str) and source.isnumeric():
        source = int(source)

    scrfd_model, arcface_session = load_models(scrfd_weights, arcface_weights)

    # 1) Always rebuild face_db.pkl
    face_db = build_face_db_from_gallery(scrfd_model, arcface_session)

    # 2) Initialize Sentry
    detector     = Detector(pose_model=pose_model_path, obj_model_path=obj_model_path)
    pose_buffer  = PoseBuffer(max_len=30)
    rule_engine  = RuleEngine(history=30)
    theft_detector = TheftDetector(static_thresh=8, corr_thresh=60)

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open source: {source}")
        return

    prev_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[INFO] End of stream")
            break

        persons, objects = detector.infer(frame)
        for p in persons:
            pose_buffer.update(p["track_id"], p["keypoints"])
        rule_results = rule_engine.update(persons, objects)
        theft_events = theft_detector.detect(persons, objects, )

        frame = process_face_recognition(frame, scrfd_model, arcface_session, face_db)

        frame_out = draw_pose(frame.copy(), [
            {"keypoints":p["keypoints"], "confidence":np.ones(len(p["keypoints"])), "track_id":p["track_id"]}
            for p in persons
        ])

        for p in persons:
            tid = p["track_id"]
            if tid in rule_results:
                r = rule_results[tid]
                x,y = int(p["keypoints"][0][0]), int(p["keypoints"][0][1])
                cv2.putText(frame_out, r["action"], (x, y - 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, r["color"], 2)

        for obj in objects:
            x1,y1,x2,y2 = map(int, obj["bbox"])
            label = f"{obj['cls']} ID{obj['track_id']} {obj['conf']:.2f}"
            cv2.rectangle(frame_out, (x1,y1), (x2,y2),(255,255,0),2)
            cv2.putText(frame_out, label, (x1,y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,0),2)

        for ev in theft_events:
            pid,oid = ev["person_id"], ev["object_id"]
            txt = f"THEFT! P{pid} took O{oid}"
            cv2.putText(frame_out, txt, (50,50),
                        cv2.FONT_HERSHEY_SIMPLEX,1.0,(0,0,255),3)

        if True:
            curr = time.time()
            fps = 1/(curr - prev_time +1e-8)
            prev_time = curr
            cv2.putText(frame_out, f"FPS: {int(fps)}", (10,30),
                        cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)

        cv2.imshow("Unified Sentry+Recognition", frame_out)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# ===========================
# ARGPARSE
# ===========================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, default="0")
    parser.add_argument("--scrfd", type=str, default="sentry/facial_analysis/weights/det_500m.onnx")
    parser.add_argument("--arcface", type=str, default="sentry/facial_analysis/weights/w600k_r50.onnx")
    parser.add_argument("--pose", type=str, default=None)
    parser.add_argument("--obj", type=str, default=None)

    args = parser.parse_args()

    run(source=args.source,
        scrfd_weights=args.scrfd,
        arcface_weights=args.arcface,
        pose_model_path=args.pose,
        obj_model_path=args.obj)
