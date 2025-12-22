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

# ──────────────────────────────────────────────────────────────────────────
# Flexible ArcFace Support (detect NCHW or NHWC ONNX)
# ──────────────────────────────────────────────────────────────────────────

def get_arcface_input_details(session):
    input_meta = session.get_inputs()[0]
    return input_meta.name, input_meta.shape

def preprocess_for_arcface(face_img, expected_shape):
    """
    Preprocess face crop for ONNX ArcFace.
    Supports both NCHW (1,3,H,W) and NHWC (1,H,W,3)
    """
    _, shape = expected_shape
    # NCHW
    if len(shape) == 4 and shape[1] == 3:
        H, W = shape[2], shape[3]
        face = cv2.resize(face_img, (W, H))
        face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB).astype(np.float32)
        face_norm = (face_rgb - 127.5) / 128.0
        face_input = np.transpose(face_norm, (2,0,1))
        face_input = np.expand_dims(face_input, 0)
        return face_input
    # NHWC
    if len(shape) == 4 and shape[3] == 3:
        H, W = shape[1], shape[2]
        face = cv2.resize(face_img, (W, H))
        face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB).astype(np.float32)
        face_norm = (face_rgb - 127.5) / 128.0
        face_input = np.expand_dims(face_norm, 0)
        return face_input
    raise ValueError(f"Unsupported ArcFace input shape: {shape}")

# ──────────────────────────────────────────────────────────────────────────
# Rebuild Face Database
# ──────────────────────────────────────────────────────────────────────────

def build_face_db_from_gallery(scrfd_model, arcface_session, gallery_dir="D:/Sentry_Final_Form/sentry/facial_analysis/face_gallery"):
    face_db = {}
    arc_input_name, arc_input_shape = get_arcface_input_details(arcface_session)

    print(f"[INFO] Building face database from: {gallery_dir}")

    if not os.path.isdir(gallery_dir):
        print(f"[WARN] Gallery folder not found: {gallery_dir}")
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
                continue

            boxes_list, _ = scrfd_model.detect(img)
            if len(boxes_list) == 0:
                continue

            box = boxes_list[0]
            x1,y1,x2,y2 = map(int, box[:4])
            face_crop = img[y1:y2, x1:x2]
            if face_crop.size == 0:
                continue

            arc_input = preprocess_for_arcface(face_crop, (arc_input_name, arc_input_shape))
            emb = arcface_session.run(None, {arc_input_name: arc_input})[0].flatten()
            emb /= np.linalg.norm(emb + 1e-8)
            embeddings.append(emb)

        if embeddings:
            face_db[person_name] = embeddings
            print(f"[DB] Added {len(embeddings)} embeddings for {person_name}")

    with open("face_db.pkl", "wb") as f:
        pickle.dump(face_db, f)
    print("[DB] Saved face_db.pkl")

    return face_db

# ──────────────────────────────────────────────────────────────────────────
# Recognition Helper
# ──────────────────────────────────────────────────────────────────────────

def find_name_for_embedding(emb, face_db, threshold=0.5):
    best_name, best_score = None, threshold
    for name, refs in face_db.items():
        for ref in refs:
            score = float(np.dot(emb, ref) / (np.linalg.norm(emb)*np.linalg.norm(ref)+1e-10))
            if score > best_score:
                best_score, best_name = score, name
    return best_name, best_score

# ──────────────────────────────────────────────────────────────────────────
# Model Loader
# ──────────────────────────────────────────────────────────────────────────

def load_models(scrfd_path, arcface_path):
    scrfd_model = SCRFD(model_path=scrfd_path)
    arcface_session = ort.InferenceSession(
        arcface_path, providers=["CUDAExecutionProvider","CPUExecutionProvider"]
    )
    print("[INFO] Loaded SCRFD & ArcFace")
    return scrfd_model, arcface_session

# ──────────────────────────────────────────────────────────────────────────
# Face Recognition Logic (with Cache / Tracking)
# ──────────────────────────────────────────────────────────────────────────

def iou(boxA, boxB):
    # Intersection Over Union
    ax1, ay1, ax2, ay2 = boxA
    bx1, by1, bx2, by2 = boxB
    interX1 = max(ax1, bx1)
    interY1 = max(ay1, by1)
    interX2 = min(ax2, bx2)
    interY2 = min(ay2, by2)
    if interX2 <= interX1 or interY2 <= interY1:
        return 0.0
    interArea = (interX2-interX1)*(interY2-interY1)
    boxAArea = (ax2-ax1)*(ay2-ay1)
    boxBArea = (bx2-bx1)*(by2-by1)
    return interArea / (boxAArea + boxBArea - interArea + 1e-10)

def process_face_recognition(frame, scrfd_model, arcface_session, face_db, cache, next_face_id):
    arc_input_name, arc_input_shape = get_arcface_input_details(arcface_session)
    boxes_list, points_list = scrfd_model.detect(frame)

    new_cache = {}
    used_ids = set()

    for boxes, keypoints in zip(boxes_list, points_list):
        *bbox, conf = boxes
        x1,y1,x2,y2 = map(int,bbox)

        if x2<=x1 or y2<=y1:
            continue
        if x1<0 or y1<0:
            continue

        face_crop = frame[y1:y2, x1:x2]
        if face_crop.size == 0:
            continue

        # find matching cached face by IoU
        matched_id = None
        for fid, data in cache.items():
            old_box = data["bbox"]
            if iou(old_box, bbox) > 0.5 and fid not in used_ids:
                matched_id = fid
                used_ids.add(fid)
                break

        if matched_id is not None:
            # reuse cached info
            emb  = cache[matched_id]["emb"]
            name = cache[matched_id]["name"]
        else:
            # new face → compute ArcFace
            arc_input = preprocess_for_arcface(face_crop, (arc_input_name, arc_input_shape))
            emb_raw  = arcface_session.run(None, {arc_input_name: arc_input})[0].flatten()
            emb      = emb_raw / np.linalg.norm(emb_raw + 1e-8)
            match_name, score = find_name_for_embedding(emb, face_db)
            name = f"{match_name} ({score:.2f})" if match_name else "Unknown"

            # assign new id
            matched_id = next_face_id
            next_face_id += 1

        # store updated cache entry
        new_cache[matched_id] = {
            "bbox": bbox,
            "name": name,
            "emb": emb
        }

        # draw
        face_obj = Face(kps=keypoints, bbox=bbox, age=None, gender=None, embedding=emb.tolist())
        draw_face_info(frame, face_obj)
        cv2.putText(frame, name, (x1, y1-15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0),2)

    return frame, new_cache, next_face_id

# ──────────────────────────────────────────────────────────────────────────
# Unified Main Loop
# ──────────────────────────────────────────────────────────────────────────

def run(source=0,
        scrfd_weights=None,
        arcface_weights=None,
        pose_model_path=None,
        obj_model_path=None):

    if isinstance(source,str) and source.isnumeric():
        source=int(source)

    scrfd_model,arcface_session=load_models(scrfd_weights,arcface_weights)

    # always rebuild face_db
    face_db = build_face_db_from_gallery(scrfd_model,arcface_session)

    detector     = Detector(pose_model=pose_model_path,obj_model_path=obj_model_path)
    pose_buffer  = PoseBuffer(max_len=30)
    rule_engine  = RuleEngine(history=30)
    theft_detector=TheftDetector(static_thresh=8,corr_thresh=60)

    cache = {}
    next_face_id = 0


    cap=cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open source:{source}")
        return

    prev_time=time.time()

    while True:
        ret,frame=cap.read()
        if not ret:
            print("[INFO] End of stream")
            break

        persons,objects=detector.infer(frame)
        for p in persons:
            pose_buffer.update(p["track_id"],p["keypoints"])
        rule_results=rule_engine.update(persons,objects)
        theft_events=theft_detector.detect(persons,objects)

        frame, cache, next_face_id = process_face_recognition(
    frame, scrfd_model, arcface_session, face_db, cache, next_face_id
)


        frame_out=draw_pose(frame.copy(),[
            {"keypoints":p["keypoints"],"confidence":np.ones(len(p["keypoints"])),"track_id":p["track_id"]}
            for p in persons
        ])

        for p in persons:
            tid=p["track_id"]
            if tid in rule_results:
                r=rule_results[tid]
                x,y=int(p["keypoints"][0][0]),int(p["keypoints"][0][1])
                cv2.putText(frame_out,r["action"],(x,y-25),
                            cv2.FONT_HERSHEY_SIMPLEX,0.7,r["color"],2)

        for obj in objects:
            x1,y1,x2,y2=map(int,obj["bbox"])
            label=f"{obj['cls']} ID{obj['track_id']} {obj['conf']:.2f}"
            cv2.rectangle(frame_out,(x1,y1),(x2,y2),(255,255,0),2)
            cv2.putText(frame_out,label,(x1,y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,0),2)

        for ev in theft_events:
            pid,oid=ev["person_id"],ev["object_id"]
            txt=f"THEFT! P{pid} took O{oid}"
            cv2.putText(frame_out,txt,(50,50),
                        cv2.FONT_HERSHEY_SIMPLEX,1.0,(0,0,255),3)

        curr=time.time()
        fps=1/(curr-prev_time+1e-8)
        prev_time=curr
        cv2.putText(frame_out,f"FPS: {int(fps)}",(10,30),
                    cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)

        cv2.imshow("Unified Sentry+FaceRec",frame_out)
        if cv2.waitKey(1)&0xFF==ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

# ──────────────────────────────────────────────────────────────────────────
# Entry Point
# ──────────────────────────────────────────────────────────────────────────

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--source",type=str,default="0")
    parser.add_argument("--scrfd",type=str,
                        default="sentry/facial_analysis/weights/det_500m.onnx")
    parser.add_argument("--arcface",type=str,
                        default="sentry/facial_analysis/weights/arc.onnx")
    parser.add_argument("--pose",type=str,default=None)
    parser.add_argument("--obj",type=str,default=None)
    args=parser.parse_args()

    run(source=args.source,
        scrfd_weights=args.scrfd,
        arcface_weights=args.arcface,
        pose_model_path=args.pose,
        obj_model_path=args.obj)
