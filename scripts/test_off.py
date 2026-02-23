import cv2
import numpy as np
from pathlib import Path
from insightface.app import FaceAnalysis
from facial_analysis.identity_store import load_identity_map

# ================= CONFIG =================
MODEL_NAME = "buffalo_l"
GALLERY_PATH = Path("facial_analysis/face_gallery")

# 🔥 FIX 1: INCREASE THRESHOLD
# 0.45 is too loose (allows strangers). 
# 0.60 is standard for strict identity verification.
MATCH_THRESHOLD = 0.60  

# ================= LOAD MODEL =================
print("[TEST] Initializing FaceAnalysis...")
app = FaceAnalysis(
    name=MODEL_NAME,
    providers=["CUDAExecutionProvider"]
)
app.prepare(ctx_id=0, det_size=(640, 640))

# ================= LOAD IDENTITY MAP =================
identity_map = load_identity_map()

# ================= BUILD FACE DB =================
print("[TEST] Loading face gallery...")
face_db = {}

for person_dir in GALLERY_PATH.iterdir():
    if not person_dir.is_dir():
        continue

    name = person_dir.name
    embeddings = []

    for img_path in person_dir.glob("*.*"):
        img = cv2.imread(str(img_path))
        if img is None: continue

        faces = app.get(img)
        if faces:
            # Get largest face
            face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
            
            # 🔥 FIX 2: NORMALIZE DB EMBEDDINGS
            # Ensure the vector has a length of 1.0 for accurate Cosine Similarity
            norm_emb = face.embedding / np.linalg.norm(face.embedding)
            embeddings.append(norm_emb)

    if embeddings:
        face_db[name] = embeddings
        print(f"  ✔ Loaded {name} ({len(embeddings)} refs)")

if not face_db:
    raise RuntimeError("❌ No faces loaded from gallery!")

print("[TEST] Face DB ready.\n")

# ================= MATCH FUNCTION =================
def match_face(embedding):
    # 🔥 FIX 3: NORMALIZE INPUT EMBEDDING
    embedding = embedding / np.linalg.norm(embedding)
    
    best_name = "Unknown"
    best_score = -1.0

    for name, refs in face_db.items():
        for ref in refs:
            # Dot product of normalized vectors = Cosine Similarity
            score = np.dot(embedding, ref)
            if score > best_score:
                best_score = score
                best_name = name

    # Only return name if score beats the stricter threshold
    if best_score >= MATCH_THRESHOLD:
        return best_name, best_score
        
    return "Unknown", best_score

# ================= CAMERA LOOP =================
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("❌ Camera not accessible")

print("[TEST] Camera started. Press 'q' to quit.\n")

while True:
    ret, frame = cap.read()
    if not ret: break

    faces = app.get(frame)

    for face in faces:
        name, score = match_face(face.embedding)
        status = identity_map.get(name.lower(), "unknown")

        # Color Logic
        if name == "Unknown":
            color = (0, 255, 255) # Yellow for Unknown
            label_text = f"Unknown ({score:.2f})"
        else:
            if status == "blacklist":
                color = (0, 0, 255) # Red
            elif status == "whitelist":
                color = (0, 255, 0) # Green
            else:
                color = (255, 0, 0) # Blue for Known but no status
            label_text = f"{name} ({score:.2f})"

        # Draw Box
        x1, y1, x2, y2 = face.bbox.astype(int)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Draw Label
        cv2.rectangle(frame, (x1, y1 - 25), (x2, y1), color, -1)
        cv2.putText(
            frame,
            label_text,
            (x1 + 5, y1 - 7),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0), # Black text
            2,
        )

    cv2.imshow("Face Recognition Test", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
print("[TEST] Finished.")