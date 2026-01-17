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
import json
from pathlib import Path

warnings.filterwarnings("ignore")

# ============================================================
# CONFIG (SAFE – DOES NOT AFFECT RECOGNITION)
# ============================================================

FACE_GALLERY = "sentry/facial_analysis/face_gallery"

WHITELIST = {
    "Akshay",
    "kalidasan",
    "Mridul"
}

BLACKLIST = {
    "Abhishek",
    "Ajay"
}

# ============================================================
# ArcFace helpers (UNCHANGED)
# ============================================================

def get_arcface_input_details(session):
    meta = session.get_inputs()[0]
    return meta.name, meta.shape

def preprocess_for_arcface(face_img, expected_shape):
    _, shape = expected_shape

    if shape[1] == 3:  # NCHW
        H, W = shape[2], shape[3]
        face = cv2.resize(face_img, (W, H))
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB).astype(np.float32)
        face = (face - 127.5) / 128.0
        face = np.transpose(face, (2, 0, 1))
        return np.expand_dims(face, 0)

    if shape[3] == 3:  # NHWC
        H, W = shape[1], shape[2]
        face = cv2.resize(face_img, (W, H))
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB).astype(np.float32)
        face = (face - 127.5) / 128.0
        return np.expand_dims(face, 0)

    raise ValueError("Unsupported ArcFace input shape")

# ============================================================
# Face DB builder (ORIGINAL STRUCTURE)
# ============================================================

def build_face_db(scrfd, arcface, gallery_dir):
    face_db = {}
    input_name, input_shape = get_arcface_input_details(arcface)

    print(f"[INFO] Building face DB from: {gallery_dir}")

    for person in os.listdir(gallery_dir):
        pdir = os.path.join(gallery_dir, person)
        if not os.path.isdir(pdir):
            continue

        embs = []
        for img_name in os.listdir(pdir):
            img = cv2.imread(os.path.join(pdir, img_name))
            if img is None:
                continue

            boxes, _ = scrfd.detect(img)
            if boxes is None or len(boxes) == 0:
                continue

            x1, y1, x2, y2 = map(int, boxes[0][:4])
            face = img[y1:y2, x1:x2]
            if face.size == 0:
                continue

            inp = preprocess_for_arcface(face, (input_name, input_shape))
            emb = arcface.run(None, {input_name: inp})[0].flatten()
            emb /= np.linalg.norm(emb + 1e-8)
            embs.append(emb)

        if embs:
            face_db[person] = embs
            print(f"[DB] {person}: {len(embs)} faces")

    return face_db
# ============================================================
# Face DB CACHE (NEW)
# ============================================================

def compute_face_gallery_signature(gallery_dir):
    sig = {}
    gallery_dir = Path(gallery_dir)

    for person in gallery_dir.iterdir():
        if not person.is_dir():
            continue

        sig[person.name] = {}
        for img in person.glob("*.*"):
            st = img.stat()
            sig[person.name][img.name] = {
                "size": st.st_size,
                "mtime": st.st_mtime
            }
    return sig


def load_or_build_face_db(scrfd, arcface, gallery_dir):
    gallery_dir = Path(gallery_dir)
    db_path = gallery_dir / "face_db.pkl"
    sig_path = gallery_dir / "face_index.json"

    current_sig = compute_face_gallery_signature(gallery_dir)

    if db_path.exists() and sig_path.exists():
        with open(sig_path, "r") as f:
            saved_sig = json.load(f)

        if saved_sig == current_sig:
            print("[FACE] Using cached face DB")
            with open(db_path, "rb") as f:
                return pickle.load(f)

    # ---------- rebuild ----------
    print("[FACE] Rebuilding face DB (gallery changed)")
    face_db = build_face_db(scrfd, arcface, gallery_dir)

    with open(db_path, "wb") as f:
        pickle.dump(face_db, f)

    with open(sig_path, "w") as f:
        json.dump(current_sig, f, indent=2)

    return face_db

# ============================================================
# Recognition helper (UNCHANGED)
# ============================================================

def find_name_for_embedding(emb, face_db, threshold=0.5):
    best_name, best_score = None, threshold
    for name, refs in face_db.items():
        for ref in refs:
            score = float(np.dot(emb, ref))
            if score > best_score:
                best_score, best_name = score, name
    return best_name, best_score

# ============================================================
# IoU (ORIGINAL)
# ============================================================

def iou(boxA, boxB):
    ax1, ay1, ax2, ay2 = boxA
    bx1, by1, bx2, by2 = boxB
    interX1 = max(ax1, bx1)
    interY1 = max(ay1, by1)
    interX2 = min(ax2, bx2)
    interY2 = min(ay2, by2)
    if interX2 <= interX1 or interY2 <= interY1:
        return 0.0
    interArea = (interX2 - interX1) * (interY2 - interY1)
    boxAArea = (ax2 - ax1) * (ay2 - ay1)
    boxBArea = (bx2 - bx1) * (by2 - by1)
    return interArea / (boxAArea + boxBArea - interArea + 1e-10)

# ============================================================
# Face recognition (ORIGINAL + WL/BL OVERLAY)
# ============================================================

def process_face_recognition(frame, scrfd, arcface, face_db, cache, next_face_id):
    input_name, input_shape = get_arcface_input_details(arcface)
    boxes_list, points_list = scrfd.detect(frame)

    new_cache = {}
    used_ids = set()
    detected_names = []   # ✅ FIX: track all detected names

    for boxes, kps in zip(boxes_list, points_list):
        *bbox, _ = boxes
        x1, y1, x2, y2 = map(int, bbox)
        if x2 <= x1 or y2 <= y1:
            continue

        face_crop = frame[y1:y2, x1:x2]
        if face_crop.size == 0:
            continue

        matched_id = None
        for fid, data in cache.items():
            if fid in used_ids:
                continue
            if iou(data["bbox"], bbox) > 0.5:
                matched_id = fid
                used_ids.add(fid)
                break

        if matched_id is not None:
            emb = cache[matched_id]["emb"]
            name = cache[matched_id]["name"]
        else:
            inp = preprocess_for_arcface(face_crop, (input_name, input_shape))
            emb = arcface.run(None, {input_name: inp})[0].flatten()
            emb /= np.linalg.norm(emb + 1e-8)
            match, score = find_name_for_embedding(emb, face_db)
            name = match if match else "Unknown"
            matched_id = next_face_id
            next_face_id += 1

        detected_names.append(name)  # ✅ FIX: collect names

        new_cache[matched_id] = {
            "bbox": bbox,
            "emb": emb,
            "name": name
        }

        # -------- Identity annotation --------
        if name in BLACKLIST:
            color = (0, 0, 255)
            tag = "BLACKLIST"
        elif name in WHITELIST:
            color = (255, 255, 255)
            tag = "WHITELIST"
        else:
            color = (0, 255, 255)
            tag = "UNKNOWN"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            frame,
            f"{name} [{tag}]",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2
        )

    # ✅ SAFE blacklist extraction
    blacklisted_faces = list({n for n in detected_names if n in BLACKLIST})

    return frame, new_cache, next_face_id, blacklisted_faces


# ============================================================
# MAIN LOOP
# ============================================================

def run(source=0, scrfd_weights=None, arcface_weights=None,
        pose_model_path=None, obj_model_path=None):

    if isinstance(source, str) and source.isnumeric():
        source = int(source)

    scrfd = SCRFD(model_path=scrfd_weights)
    arcface = ort.InferenceSession(
        arcface_weights,
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
    )

    face_db = load_or_build_face_db(scrfd, arcface, FACE_GALLERY)


    detector = Detector(pose_model=pose_model_path, obj_model_path=obj_model_path)
    pose_buffer = PoseBuffer(max_len=30)
    rule_engine = RuleEngine(history=30)
    theft_detector = TheftDetector(static_thresh=8, corr_thresh=60)

    cache = {}
    next_face_id = 0

    cap = cv2.VideoCapture(source)
    prev_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        persons, objects = detector.infer(frame)
        for p in persons:
            pose_buffer.update(p["track_id"], p["keypoints"])

        rule_results = rule_engine.update(persons, objects)
        theft_events = theft_detector.detect(persons, objects)

        frame, cache, next_face_id = process_face_recognition(
            frame, scrfd, arcface, face_db, cache, next_face_id
        )

        frame_out = draw_pose(frame.copy(), [
            {
                "keypoints": p["keypoints"],
                "confidence": np.ones(len(p["keypoints"])),
                "track_id": p["track_id"]
            } for p in persons
        ])

        for p in persons:
            tid = p["track_id"]
            if tid in rule_results:
                r = rule_results[tid]
                x, y = map(int, p["keypoints"][0])
                cv2.putText(
                    frame_out,
                    r["action"],
                    (x, y - 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    r["color"],
                    2
                )

        for ev in theft_events:
            cv2.putText(
                frame_out,
                "THEFT DETECTED",
                (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                3
            )

        fps = 1 / (time.time() - prev_time + 1e-8)
        prev_time = time.time()
        cv2.putText(
            frame_out,
            f"FPS: {int(fps)}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        cv2.imshow("Sentry (Stable Face Recognition)", frame_out)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

# ============================================================
# ENTRY
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="0")
    parser.add_argument("--scrfd", default="sentry/facial_analysis/weights/det_500m.onnx")
    parser.add_argument("--arcface", default="sentry/facial_analysis/weights/arc.onnx")
    parser.add_argument("--pose", default=None)
    parser.add_argument("--obj", default=None)
    args = parser.parse_args()

    run(
        source=args.source,
        scrfd_weights=args.scrfd,
        arcface_weights=args.arcface,
        pose_model_path=args.pose,
        obj_model_path=args.obj
    )
