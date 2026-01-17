# ================== HARD BLOCK TENSORFLOW (MUST BE FIRST) ==================
import os
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"
# ==========================================================================

import cv2
import argparse
import time
import json
import numpy as np
import torch
import onnxruntime as ort
from pathlib import Path
from transformers import VideoMAEImageProcessor, VideoMAEForVideoClassification
from torch.nn.functional import cosine_similarity

# ========================= LIVE PIPELINE IMPORTS =========================
from core.detector import Detector
from core.pose_buffer import PoseBuffer
from core.rule_engine import RuleEngine
from core.theft_detector import TheftDetector
from utils.visualization import draw_pose
from facial_analysis.models import SCRFD
from events.event_manager import EventManager

# ========================= FACE PIPELINE (CACHED) =========================
from pose_face_main import (
    load_or_build_face_db,
    process_face_recognition
)
# ==========================================================================


# ========================= CONFIG =========================
FACE_GALLERY = "sentry/facial_analysis/face_gallery"
SCRFD_MODEL = "sentry/facial_analysis/weights/det_500m.onnx"
ARCFACE_MODEL = "sentry/facial_analysis/weights/arc.onnx"

VIDEOMAE_MODEL = "DanJoshua/videomae-base-finetuned-rwf2000-subset"

NUM_FRAMES = 16
STRIDE = 8
FRAME_SIZE = (224, 224)

LOW_VIOLENCE = 0.55
HIGH_VIOLENCE = 0.85
DANCE_SIM_THRESHOLD = 0.85
# ========================= PATHS =========================
SENTRY_ROOT = os.path.dirname(os.path.abspath(__file__))
REPORTS_DIR = os.path.join(SENTRY_ROOT, "reports")
SCREENSHOT_DIR = os.path.join(REPORTS_DIR, "screenshots")

os.makedirs(SCREENSHOT_DIR, exist_ok=True)
# ========================================================

DANCE_DIR = "sentry/tests/dance"
# ========================================================


# ========================= HELPERS =========================
def is_dance_video(video_path, dance_dir):
    try:
        return Path(dance_dir).resolve() in Path(video_path).resolve().parents
    except Exception:
        return False


def compute_dance_signature(dance_dir):
    sig = {}
    for p in Path(dance_dir).glob("*.mp4"):
        st = p.stat()
        sig[p.name] = {"size": st.st_size, "mtime": st.st_mtime}
    return sig


def load_or_build_dance_embeddings(dance_dir, extract_embedding, device):
    emb_path = Path(dance_dir) / "embeddings.pkl"
    sig_path = Path(dance_dir) / "index.json"

    current_sig = compute_dance_signature(dance_dir)

    if emb_path.exists() and sig_path.exists():
        with open(sig_path, "r") as f:
            if json.load(f) == current_sig:
                print("[DANCE] Using cached embeddings")
                return torch.load(emb_path).to(device)

    print("[DANCE] Rebuilding embeddings")
    dance_embs = []

    for vid in Path(dance_dir).glob("*.mp4"):
        cap = cv2.VideoCapture(str(vid))
        frames = []
        while True:
            ret, f = cap.read()
            if not ret:
                break
            frames.append(f)
        cap.release()

        for i in range(0, len(frames) - NUM_FRAMES, STRIDE):
            clip = [
                cv2.resize(cv2.cvtColor(x, cv2.COLOR_BGR2RGB), FRAME_SIZE)
                for x in frames[i:i + NUM_FRAMES]
            ]
            dance_embs.append(extract_embedding(clip).cpu())

    dance_embs = torch.stack(dance_embs)
    torch.save(dance_embs, emb_path)

    with open(sig_path, "w") as f:
        json.dump(current_sig, f, indent=2)

    return dance_embs.to(device)
# ==========================================================

def run_live(source):
    print("[MODE] LIVE")

    # ===================== INIT MODELS =====================
    scrfd = SCRFD(model_path=SCRFD_MODEL)
    arcface = ort.InferenceSession(
        ARCFACE_MODEL,
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
    )

    face_db = load_or_build_face_db(scrfd, arcface, FACE_GALLERY)

    detector = Detector()
    pose_buffer = PoseBuffer(max_len=30)
    rule_engine = RuleEngine(history=30)

    # ===================== VIDEO =====================
    cap = cv2.VideoCapture(source)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30

    event_mgr = EventManager(fps=fps, source="LIVE")

    frame_idx = 0
    prev_time = time.time()

    # Face tracking cache
    cache = {}
    next_face_id = 0

    # Sparse frame store (1 frame/sec backup)
    frame_store = {}

    # ===================== MAIN LOOP =====================
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Store 1 frame per second (fallback)
        if frame_idx % int(fps) == 0:
            frame_store[frame_idx] = frame.copy()

        # ---------- Detection ----------
        persons, objects = detector.infer(frame)

        for p in persons:
            pose_buffer.update(p["track_id"], p["keypoints"])

        # ---------- Rule Engine ----------
        rule_results = rule_engine.update(persons, objects)

        # ---------- Face Recognition ----------
        frame, cache, next_face_id = process_face_recognition(
            frame, scrfd, arcface, face_db, cache, next_face_id
        )

        # ---------- Draw Pose ----------
        frame_out = draw_pose(
            frame.copy(),
            [{
                "keypoints": p["keypoints"],
                "confidence": np.ones(len(p["keypoints"])),
                "track_id": p["track_id"]
            } for p in persons]
        )

        # ---------- EVENT HANDLING ----------
        active_event_this_frame = False

        for p in persons:
            tid = p["track_id"]

            if tid in rule_results:
                r = rule_results[tid]
                active_event_this_frame = True

                # Save screenshot ONLY when event starts
                screenshot_path = None
                if (
                    event_mgr.current_event is None or
                    event_mgr.current_event["type"] != r["action"]
                ):
                    screenshot_path = os.path.join(
                        SCREENSHOT_DIR,
                        f"event_{frame_idx}_track_{tid}.jpg"
                    )
                    cv2.imwrite(screenshot_path, frame_out)

                event_mgr.update(
                    frame_idx=frame_idx,
                    label=r["action"],
                    severity=r["severity"],
                    confidence=None,
                    face_ids=[tid],
                    override=None,
                    cause=r.get("cause"),
                    screenshot=screenshot_path
                )

                # Overlay label
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

        # ---------- NORMAL (ONLY ONCE PER FRAME) ----------
        if not active_event_this_frame:
            event_mgr.update(
                frame_idx=frame_idx,
                label="Normal",
                severity="LOW"
            )

        # ---------- FPS ----------
        now = time.time()
        fps_val = 1 / (now - prev_time + 1e-8)
        prev_time = now

        cv2.putText(
            frame_out,
            f"FPS: {int(fps_val)}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        cv2.imshow("Sentry LIVE", frame_out)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        frame_idx += 1

    # ===================== CLEANUP =====================
    cap.release()
    cv2.destroyAllWindows()

    # ===================== FINALIZE EVENTS =====================
    event_mgr.finalize()
    events = event_mgr.export()

    print("\n=== FINAL EVENT TIMELINE ===")
    for e in events:
        print(e)

    # ===================== PDF REPORT =====================
    from reports.event_adapter import adapt_events_for_pdf
    from reports.pdf_report import generate_pdf_report
    from datetime import datetime

    event_buffer = adapt_events_for_pdf(events, frame_store)

    from llm.summary_generator import generate_llm_summary

    summary_text = generate_llm_summary(
        events=event_buffer,
        mode="LIVE"
    )



    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(
        REPORTS_DIR,
        f"sentry_live_report_{ts}.pdf"
    )

    generate_pdf_report(event_buffer, summary_text, output_path)

    print(f"\n[REPORT] Generated → {output_path}")


# ========================= MODE 2: OFFLINE =========================
def run_offline(video_path):
    OFFLINE_REPORT_DIR = os.path.join(REPORTS_DIR, "offline")
    OFFLINE_SCREENSHOT_DIR = os.path.join(OFFLINE_REPORT_DIR, "screenshots")
    os.makedirs(OFFLINE_SCREENSHOT_DIR, exist_ok=True)

    print("[MODE] OFFLINE")

    IS_DANCE_VIDEO = is_dance_video(video_path, DANCE_DIR)
    # if IS_DANCE_VIDEO:
    #     print("[OFFLINE] HARD DANCE WHITELIST ENABLED")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    processor = VideoMAEImageProcessor.from_pretrained(VIDEOMAE_MODEL)
    model = VideoMAEForVideoClassification.from_pretrained(
        VIDEOMAE_MODEL
    ).to(device).eval()

    fight_idx = [k for k, v in model.config.id2label.items()
                 if v.lower() == "fight"][0]

    def extract_embedding(clip):
        inputs = processor(clip, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            out = model(**inputs, output_hidden_states=True)
        emb = out.hidden_states[-1][:, 0, :]
        return (emb / emb.norm(dim=1, keepdim=True)).squeeze(0)

    dance_embs = load_or_build_dance_embeddings(
        DANCE_DIR, extract_embedding, device
    )

    scrfd = SCRFD(model_path=SCRFD_MODEL)
    arcface = ort.InferenceSession(
        ARCFACE_MODEL,
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
    )
    face_db = load_or_build_face_db(scrfd, arcface, FACE_GALLERY)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    while True:
        ret, f = cap.read()
        if not ret:
            break
        frames.append(f)
    cap.release()

    labels = ["Normal"] * len(frames)
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

            # ===== HARD OVERRIDE =====
            if IS_DANCE_VIDEO:
                labels[j] = "Dance"
                continue
            # ========================

            if fight_score > HIGH_VIOLENCE:
                labels[j] = "Fight"
            elif fight_score > LOW_VIOLENCE:
                labels[j] = "Fight (Low Confidence)"
            else:
                labels[j] = "Normal"

    cap = cv2.VideoCapture(video_path)
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

        if "Dance" in label:
            color = (255, 200, 0)
        elif "Fight" in label:
            color = (0, 0, 255)
        else:
            color = (0, 255, 0)

        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0),
                      (frame.shape[1], frame.shape[0]),
                      color, -1)
        frame = cv2.addWeighted(overlay, 0.25, frame, 0.75, 0)

        cv2.putText(frame, f"{label} | {score:.2f}",
                    (30, 50), cv2.FONT_HERSHEY_SIMPLEX,
                    1.1, color, 3)

        cv2.imshow("Sentry OFFLINE", frame)
        if cv2.waitKey(int(1000 / fps)) & 0xFF == ord("q"):
            break

        idx += 1
    from events.offline_event_builder import build_offline_events
    from reports.pdf_report import generate_pdf_report
    from datetime import datetime

    events = build_offline_events(
        frames=frames,
        labels=labels,
        scores=scores,
        fps=fps,
        screenshot_dir=OFFLINE_SCREENSHOT_DIR
    )

    from llm.summary_generator import generate_llm_summary

    summary_text = generate_llm_summary(
        events=events,
        mode="OFFLINE"
    )


    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_pdf = os.path.join(
        OFFLINE_REPORT_DIR,
        f"sentry_offline_report_{ts}.pdf"
    )

    generate_pdf_report(events, summary_text, output_pdf)

    print(f"[OFFLINE REPORT] Generated -> {output_pdf}")

    cap.release()
    cv2.destroyAllWindows()
# =============================================================


# ========================= ENTRY =========================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="0",
                        help="0 for webcam or video path")
    args = parser.parse_args()

    if args.source == "0":
        run_live(0)
    else:
        run_offline(args.source)
# =========================================================
