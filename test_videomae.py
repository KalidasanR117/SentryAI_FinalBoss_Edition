# ================== HARD BLOCK TENSORFLOW ==================
import os
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"
# ===========================================================

import cv2
import numpy as np
import torch
import onnxruntime as ort
from pathlib import Path
from transformers import VideoMAEImageProcessor, VideoMAEForVideoClassification
from torch.nn.functional import cosine_similarity

# ========================= FACE PIPELINE (COPIED FROM main.py) =========================
from facial_analysis.models import SCRFD

FACE_GALLERY = "sentry/facial_analysis/face_gallery"

WHITELIST = {"kalidasan", "Akshay", "Mridul"}
BLACKLIST = {"Abhishek", "Ajay"}

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

def build_face_db(scrfd, arcface, gallery_dir):
    face_db = {}
    input_name, input_shape = get_arcface_input_details(arcface)

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

    return face_db

def find_name_for_embedding(emb, face_db, threshold=0.5):
    best_name, best_score = None, threshold
    for name, refs in face_db.items():
        for ref in refs:
            score = float(np.dot(emb, ref))
            if score > best_score:
                best_score, best_name = score, name
    return best_name, best_score

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

def process_face_recognition(frame, scrfd, arcface, face_db, cache, next_face_id):
    input_name, input_shape = get_arcface_input_details(arcface)
    boxes_list, points_list = scrfd.detect(frame)

    new_cache = {}
    used_ids = set()

    for boxes, _ in zip(boxes_list, points_list):
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
            match, _ = find_name_for_embedding(emb, face_db)
            name = match if match else "Unknown"
            matched_id = next_face_id
            next_face_id += 1

        new_cache[matched_id] = {
            "bbox": bbox,
            "emb": emb,
            "name": name
        }

        if name in BLACKLIST:
            color = (0, 0, 255)
        elif name in WHITELIST:
            color = (255, 255, 255)
        else:
            color = (0, 255, 255)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, name, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    return frame, new_cache, next_face_id
# ======================================================================================


# ========================= CONFIG =========================
VIDEO_PATH = "D:/Sentry_Final_Form/sentry/tests/dance/flamenco.mp4"
DANCE_DIR = "D:/Sentry_Final_Form/sentry/tests/dance"

SCRFD_MODEL = "sentry/facial_analysis/weights/det_500m.onnx"
ARCFACE_MODEL = "sentry/facial_analysis/weights/arc.onnx"

MODEL_NAME = "DanJoshua/videomae-base-finetuned-rwf2000-subset"
NUM_FRAMES = 16
STRIDE = 8
FRAME_SIZE = (224, 224)
FIGHT_THRESHOLD = 0.6
DANCE_SIM_THRESHOLD = 0.85
# =========================================================


# ========================= LOAD MODELS =========================
scrfd = SCRFD(model_path=SCRFD_MODEL)
arcface = ort.InferenceSession(
    ARCFACE_MODEL,
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
)
face_db = build_face_db(scrfd, arcface, FACE_GALLERY)

processor = VideoMAEImageProcessor.from_pretrained(MODEL_NAME)
model = VideoMAEForVideoClassification.from_pretrained(MODEL_NAME).eval()
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

fight_idx = [k for k, v in model.config.id2label.items() if v.lower() == "fight"][0]
# =========================================================


# ========================= VIDEO LOAD =========================
cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS)
frames = []
while True:
    ret, f = cap.read()
    if not ret:
        break
    frames.append(f)
cap.release()
# =========================================================


# ========================= DANCE EMBEDDINGS =========================
def extract_embedding(clip):
    inputs = processor(clip, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True)
    emb = out.hidden_states[-1][:, 0, :]
    return (emb / emb.norm(dim=1, keepdim=True)).squeeze(0)

dance_embs = []
for vid in Path(DANCE_DIR).glob("*.mp4"):
    cap = cv2.VideoCapture(str(vid))
    dframes = []
    while True:
        ret, f = cap.read()
        if not ret:
            break
        dframes.append(f)
    cap.release()

    for i in range(0, len(dframes) - NUM_FRAMES, STRIDE):
        clip = [
            cv2.resize(cv2.cvtColor(x, cv2.COLOR_BGR2RGB), FRAME_SIZE)
            for x in dframes[i:i + NUM_FRAMES]
        ]
        dance_embs.append(extract_embedding(clip).cpu())

dance_embs = torch.stack(dance_embs).to(device)
# =========================================================


# ========================= VIDEO ANALYSIS =========================
labels = ["NonFight"] * len(frames)
scores = np.zeros(len(frames))

for i in range(0, len(frames) - NUM_FRAMES, STRIDE):
    clip = [
        cv2.resize(cv2.cvtColor(x, cv2.COLOR_BGR2RGB), FRAME_SIZE)
        for x in frames[i:i + NUM_FRAMES]
    ]

    emb = extract_embedding(clip)
    sim = cosine_similarity(emb.unsqueeze(0), dance_embs).max().item()

    inputs = processor(clip, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        out = model(**inputs)

    fight_score = torch.softmax(out.logits, dim=-1)[0][fight_idx].item()

    for j in range(i, i + NUM_FRAMES):
        scores[j] = max(scores[j], fight_score)
        if fight_score > FIGHT_THRESHOLD and sim > DANCE_SIM_THRESHOLD:
            labels[j] = "Dance (Suppressed)"
        elif fight_score > FIGHT_THRESHOLD:
            labels[j] = "Fight"
# =========================================================


# ========================= PLAYBACK =========================
cap = cv2.VideoCapture(VIDEO_PATH)
cache = {}
next_face_id = 0
idx = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame, cache, next_face_id = process_face_recognition(
        frame, scrfd, arcface, face_db, cache, next_face_id
    )

    label = labels[idx]
    score = scores[idx]

    color = (0, 255, 0)
    if label == "Fight":
        color = (0, 0, 255)
    elif "Dance" in label:
        color = (255, 200, 0)

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], frame.shape[0]), color, -1)
    frame = cv2.addWeighted(overlay, 0.25, frame, 0.75, 0)

    cv2.putText(frame, f"{label} | {score:.2f}", (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 3)

    cv2.imshow("Sentry Offline (Face + VideoMAE)", frame)
    if cv2.waitKey(int(1000 / fps)) & 0xFF == ord("q"):
        break

    idx += 1

cap.release()
cv2.destroyAllWindows()
# =========================================================
