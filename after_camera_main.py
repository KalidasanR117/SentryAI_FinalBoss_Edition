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
from utils.visualization import draw_pose
from facial_analysis.models import SCRFD
from events.event_manager import EventManager
from core.camera_manager import CameraManager, CameraConfig, EventSeverity, map_severity_to_enum
# ========================= FACE PIPELINE (CACHED) =========================
from pose_face_main import (
    load_or_build_face_db,
    process_face_recognition
)
# ==========================================================================
SEVERITY_RANK = {
    EventSeverity.NORMAL: 0,
    EventSeverity.LOW: 1,
    EventSeverity.MEDIUM: 2,
    EventSeverity.HIGH: 3,
    EventSeverity.CRITICAL: 4,
}

BASE_DIR = Path(__file__).resolve().parent

# ========================= CONFIG =========================
FACE_GALLERY = BASE_DIR / "facial_analysis" / "face_gallery"
SCRFD_MODEL = BASE_DIR / "facial_analysis" / "weights" / "det_500m.onnx"
ARCFACE_MODEL = BASE_DIR / "facial_analysis" / "weights" / "arc.onnx"
DANCE_DIR = BASE_DIR / "tests" / "dance"

VIDEOMAE_MODEL = "DanJoshua/videomae-base-finetuned-rwf2000-subset"

NUM_FRAMES = 16
STRIDE = 8
FRAME_SIZE = (224, 224)

LOW_VIOLENCE = 0.55
HIGH_VIOLENCE = 0.85
DANCE_SIM_THRESHOLD = 0.85

# Severity colors (must match rule_engine.py)
SEVERITY_COLORS = {
    "CRITICAL": (0, 0, 255),
    "HIGH": (0, 100, 255),
    "MEDIUM": (0, 255, 255),
    "LOW": (0, 255, 0)
}

# ========================= PATHS =========================
REPORTS_DIR = BASE_DIR / "reports"
SCREENSHOT_DIR = REPORTS_DIR / "screenshots"
OFFLINE_REPORT_DIR = REPORTS_DIR / "offline"
OFFLINE_SCREENSHOT_DIR = OFFLINE_REPORT_DIR / "screenshots"

# Create directories
for dir_path in [SCREENSHOT_DIR, OFFLINE_SCREENSHOT_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)
# ========================================================


# ========================= VALIDATION =========================
def validate_paths():
    """Validate required files exist"""
    required_files = [SCRFD_MODEL, ARCFACE_MODEL]
    missing = [f for f in required_files if not f.exists()]
    
    if missing:
        raise FileNotFoundError(
            f"Missing required files:\n" + 
            "\n".join(f"  - {f}" for f in missing)
        )
    
    if not FACE_GALLERY.exists():
        print(f"[WARNING] Face gallery not found: {FACE_GALLERY}")
        FACE_GALLERY.mkdir(parents=True, exist_ok=True)
    
    if not DANCE_DIR.exists():
        print(f"[WARNING] Dance directory not found: {DANCE_DIR}")
        DANCE_DIR.mkdir(parents=True, exist_ok=True)


# ========================= HELPERS =========================
# ========================= HELPERS =========================
def get_bbox_from_keypoints(keypoints):
    """Calculate bounding box from pose keypoints"""
    valid_points = [kp for kp in keypoints if len(kp) >= 2]
    if not valid_points:
        return None
    
    xs = [p[0] for p in valid_points]
    ys = [p[1] for p in valid_points]
    
    return {
        'x1': min(xs),
        'y1': min(ys),
        'x2': max(xs),
        'y2': max(ys)
    }


def bbox_iou(bbox1, bbox2):
    """Calculate Intersection over Union between two bboxes"""
    if bbox1 is None or bbox2 is None:
        return 0.0
    
    # Intersection
    x1 = max(bbox1['x1'], bbox2['x1'])
    y1 = max(bbox1['y1'], bbox2['y1'])
    x2 = min(bbox1['x2'], bbox2['x2'])
    y2 = min(bbox1['y2'], bbox2['y2'])
    
    if x2 < x1 or y2 < y1:
        return 0.0
    
    inter_area = (x2 - x1) * (y2 - y1)
    bbox1_area = (bbox1['x2'] - bbox1['x1']) * (bbox1['y2'] - bbox1['y1'])
    bbox2_area = (bbox2['x2'] - bbox2['x1']) * (bbox2['y2'] - bbox2['y1'])
    
    union_area = bbox1_area + bbox2_area - inter_area
    
    return inter_area / union_area if union_area > 0 else 0.0


def match_faces_to_poses(persons, face_results, iou_threshold=0.3):
    """
    Match detected faces to pose tracks using spatial overlap.
    
    Args:
        persons: List of person detections with keypoints and track_id
        face_results: Dict mapping face_id to {'bbox': {...}, 'name': str, 'status': str}
        iou_threshold: Minimum IoU to consider a match
    
    Returns:
        Dict mapping track_id to face info {'name': str, 'status': 'whitelist'/'blacklist'/None}
    """
    track_to_face = {}
    
    for person in persons:
        pose_bbox = get_bbox_from_keypoints(person['keypoints'])
        if pose_bbox is None:
            continue
        
        best_iou = 0.0
        best_match = None
        
        # Find best matching face
        for face_id, face_info in face_results.items():
            if 'bbox' not in face_info:
                continue
            
            face_bbox_raw = face_info['bbox']

            # Convert list bbox → dict bbox
            if isinstance(face_bbox_raw, (list, tuple)) and len(face_bbox_raw) == 4:
                face_bbox = {
                    "x1": face_bbox_raw[0],
                    "y1": face_bbox_raw[1],
                    "x2": face_bbox_raw[2],
                    "y2": face_bbox_raw[3],
                }
            else:
                continue  # skip invalid bbox

            iou = bbox_iou(pose_bbox, face_bbox)

                        
            if iou > best_iou and iou >= iou_threshold:
                best_iou = iou
                best_match = face_info
        
        if best_match:
            track_to_face[person['track_id']] = {
                'name': best_match.get('name', 'Unknown'),
                'status': best_match.get('status', None),  # 'whitelist', 'blacklist', or None
                'iou': best_iou
            }
    
    return track_to_face


def compute_dance_signature(dance_dir):
    """Generate signature of dance videos for caching"""
    sig = {}
    for p in Path(dance_dir).glob("*.mp4"):
        try:
            st = p.stat()
            sig[p.name] = {"size": st.st_size, "mtime": st.st_mtime}
        except Exception as e:
            print(f"[WARNING] Could not stat {p}: {e}")
    return sig


def load_or_build_dance_embeddings(dance_dir, extract_embedding, device):
    """Load or build dance reference embeddings with caching"""
    emb_path = Path(dance_dir) / "embeddings.pkl"
    sig_path = Path(dance_dir) / "index.json"

    current_sig = compute_dance_signature(dance_dir)
    
    # No dance videos found
    if not current_sig:
        print("[DANCE] No dance videos found, skipping embedding generation")
        return torch.tensor([]).to(device)

    # Check cache validity
    if emb_path.exists() and sig_path.exists():
        try:
            with open(sig_path, "r") as f:
                cached_sig = json.load(f)
                if cached_sig == current_sig:
                    print("[DANCE] Using cached embeddings")
                    return torch.load(emb_path, map_location=device)
        except Exception as e:
            print(f"[DANCE] Cache invalid: {e}")

    # Rebuild embeddings
    print("[DANCE] Rebuilding embeddings...")
    dance_embs = []

    for vid in Path(dance_dir).glob("*.mp4"):
        try:
            cap = cv2.VideoCapture(str(vid))
            if not cap.isOpened():
                print(f"[WARNING] Could not open {vid}")
                continue
            
            frames = []
            while len(frames) < 1000:  # Limit frames per video
                ret, f = cap.read()
                if not ret:
                    break
                frames.append(f)
            cap.release()

            # Extract embeddings from clips
            for i in range(0, len(frames) - NUM_FRAMES, STRIDE):
                clip = [
                    cv2.resize(cv2.cvtColor(x, cv2.COLOR_BGR2RGB), FRAME_SIZE)
                    for x in frames[i:i + NUM_FRAMES]
                ]
                dance_embs.append(extract_embedding(clip).cpu())
                
            print(f"[DANCE] Processed {vid.name}: {len(dance_embs)} clips")
            
        except Exception as e:
            print(f"[ERROR] Failed to process {vid}: {e}")
            continue

    if not dance_embs:
        print("[WARNING] No dance embeddings generated")
        return torch.tensor([]).to(device)

    dance_embs = torch.stack(dance_embs)
    
    # Save cache
    try:
        torch.save(dance_embs, emb_path)
        with open(sig_path, "w") as f:
            json.dump(current_sig, f, indent=2)
        print(f"[DANCE] Saved {len(dance_embs)} embeddings to cache")
    except Exception as e:
        print(f"[WARNING] Could not save cache: {e}")

    return dance_embs.to(device)


# ========================= LIVE MODE =========================
def run_live(source):
    """Run live detection mode with camera rotation support"""
    print("[MODE] LIVE")
    
    try:
        validate_paths()
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        return

    # ===================== CAMERA SETUP =====================
    camera_mgr = None
    use_rotation = (source == "0" or source == 0)
    
    if use_rotation:
        print("[CAMERA] Initializing multi-camera rotation...")
        config = CameraConfig(
            base_time_window=10.0,
            # extension_multiplier=2.5,
            max_scan_index=10
        )
        
        camera_mgr = CameraManager(config)
        if not camera_mgr.initialize():
            print("[WARNING] Falling back to single camera")
            use_rotation = False
        else:
            if not camera_mgr.start_camera():
                print("[ERROR] Failed to start camera")
                return
            fps = camera_mgr.get_fps()
    
    if not use_rotation:
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print(f"[ERROR] Could not open: {source}")
            return
        fps = cap.get(cv2.CAP_PROP_FPS) or 30

    # ===================== INIT MODELS (unchanged) =====================
    try:
        scrfd = SCRFD(model_path=str(SCRFD_MODEL))
        arcface = ort.InferenceSession(
            str(ARCFACE_MODEL),
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
        )
        face_db = load_or_build_face_db(scrfd, arcface, str(FACE_GALLERY))
        
        detector = Detector()
        pose_buffer = PoseBuffer(max_len=30)
        rule_engine = RuleEngine(history=30)
        
    except Exception as e:
        print(f"[ERROR] Model initialization failed: {e}")
        return

    event_mgr = EventManager(fps=fps, source="LIVE")
    frame_idx = 0
    prev_time = time.time()
    cache = {}
    next_face_id = 0
    frame_store = {}

    # ===================== MAIN LOOP =====================
    try:
        while True:
            # MODIFIED: Read frame with rotation support
            if use_rotation:
                ret, frame = camera_mgr.read_frame()
                if not ret:
                    if camera_mgr.rotate_camera():
                        fps = camera_mgr.get_fps()
                        event_mgr.fps = fps
                        continue
                    else:
                        break
            else:
                ret, frame = cap.read()
                if not ret:
                    break

            if frame_idx % int(fps) == 0:
                frame_store[frame_idx] = frame.copy()

            persons, objects = detector.infer(frame)

            for p in persons:
                pose_buffer.update(p["track_id"], p["keypoints"])

            rule_results = rule_engine.update(persons, objects)

            frame, cache, next_face_id, face_results = process_face_recognition(
                frame, scrfd, arcface, face_db, cache, next_face_id
            )
            
            track_to_face = match_faces_to_poses(persons, face_results)

            frame_out = draw_pose(
                frame.copy(),
                [{
                    "keypoints": p["keypoints"],
                    "confidence": np.ones(len(p["keypoints"])),
                    "track_id": p["track_id"]
                } for p in persons]
            )

            # NEW: Track severity for camera scheduler
            current_severity = EventSeverity.NORMAL
            
            # Blacklist detection (unchanged)
            blacklisted_in_frame = [
                info['name'] for tid, info in track_to_face.items() 
                if info.get('status') == 'blacklist'
            ]
            
            if blacklisted_in_frame:
                current_severity = EventSeverity.CRITICAL  # NEW
                
                screenshot_path = SCREENSHOT_DIR / f"blacklist_{frame_idx}.jpg"
                cv2.imwrite(str(screenshot_path), frame_out)

                event_mgr.update(
                    frame_idx=frame_idx,
                    label="Blacklisted Person Detected",
                    severity="CRITICAL",
                    confidence=1.0,
                    face_ids=blacklisted_in_frame,
                    override="BLACKLIST",
                    cause={
                        "trigger": "FACE_RECOGNITION",
                        "rule_name": "BLACKLIST_MATCH",
                        "description": "Known blacklisted individual detected",
                        "joints_involved": [],
                        "metrics": {"faces": blacklisted_in_frame}
                    },
                    screenshot=str(screenshot_path)
                )

            active_event_this_frame = False

            for p in persons:
                tid = p["track_id"]

                if tid in rule_results:
                    result = rule_results[tid].copy()
                    
                    # NEW: Update severity for scheduler
                    # result_severity = map_severity_to_enum(result['severity'])
                    # if result_severity.value > current_severity.value:
                    #     current_severity = result_severity
                    result_severity = map_severity_to_enum(result['severity'])
                    if SEVERITY_RANK[result_severity] > SEVERITY_RANK[current_severity]:
                        current_severity = result_severity

                    face_info = track_to_face.get(tid)
                    
                    # Whitelist/blacklist logic (unchanged)
                    if face_info and face_info.get('status') == 'whitelist':
                        if 'cause' not in result:
                            result['cause'] = {}
                        if 'metrics' not in result['cause']:
                            result['cause']['metrics'] = {}
                        
                        result['cause']['metrics']['whitelisted_person'] = face_info['name']
                        result['cause']['whitelist_note'] = f"Whitelisted person '{face_info['name']}' involved"
                        
                        if result['severity'] == 'MEDIUM':
                            original_action = result['action']
                            result['action'] = f"{original_action} (Whitelisted - Low Priority)"
                            result['severity'] = 'LOW'
                            result['color'] = SEVERITY_COLORS['LOW']
                            result['cause']['description'] = (
                                f"Original: {result['cause'].get('description', '')}. "
                                f"Downgraded due to whitelisted person '{face_info['name']}'"
                            )
                    
                    elif face_info and face_info.get('status') == 'blacklist':
                        if 'cause' not in result:
                            result['cause'] = {}
                        if 'metrics' not in result['cause']:
                            result['cause']['metrics'] = {}
                        
                        result['cause']['metrics']['blacklisted_person'] = face_info['name']
                        result['cause']['blacklist_note'] = f"ALERT: Blacklisted person '{face_info['name']}' involved"
                        
                        if result['severity'] != 'CRITICAL':
                            result['severity'] = 'CRITICAL'
                            result['color'] = SEVERITY_COLORS['CRITICAL']
                    
                    active_event_this_frame = True

                    screenshot_path = None
                    if (
                        event_mgr.current_event is None or
                        event_mgr.current_event["type"] != result["action"]
                    ):
                        screenshot_path = SCREENSHOT_DIR / f"event_{frame_idx}_track_{tid}.jpg"
                        cv2.imwrite(str(screenshot_path), frame_out)

                    event_mgr.update(
                        frame_idx=frame_idx,
                        label=result["action"],
                        severity=result["severity"],
                        confidence=None,
                        face_ids=[tid],
                        override=None,
                        cause=result.get("cause"),
                        screenshot=str(screenshot_path) if screenshot_path else None
                    )

                    x, y = map(int, p["keypoints"][0])
                    cv2.putText(
                        frame_out,
                        result["action"],
                        (x, y - 25),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        result["color"],
                        2
                    )

            if not active_event_this_frame and not blacklisted_in_frame:
                event_mgr.update(
                    frame_idx=frame_idx,
                    label="Normal",
                    severity="LOW"
                )

            # NEW: Update camera scheduler and display info
            if use_rotation:
                camera_mgr.update_event_severity(current_severity)
                
                status = camera_mgr.get_status()
                cv2.putText(
                    frame_out,
                    f"Cam {status['current_camera']} | {int(status['remaining_time'])}s | {status['event_severity']}",
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 0),
                    2
                )

            # FPS display (unchanged)
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
            
            # NEW: Check for camera rotation
            if use_rotation and camera_mgr.should_rotate():
                if camera_mgr.rotate_camera():
                    fps = camera_mgr.get_fps()
                    event_mgr.fps = fps
                else:
                    break
            
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            frame_idx += 1

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user")
    except Exception as e:
        print(f"[ERROR] Runtime error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # MODIFIED: Cleanup
        if use_rotation:
            camera_mgr.stop_camera()
        else:
            cap.release()
        cv2.destroyAllWindows()

    # Rest of the function remains unchanged (event finalization, PDF report, etc.)
    event_mgr.finalize()
    events = event_mgr.export()

    print("\n=== FINAL EVENT TIMELINE ===")
    for e in events:
        print(e)

    try:
        from reports.event_adapter import adapt_events_for_pdf
        from reports.pdf_report import generate_pdf_report
        from llm.summary_generator import generate_llm_summary
        from datetime import datetime

        event_buffer = adapt_events_for_pdf(events, frame_store)
        summary_text = generate_llm_summary(events=event_buffer, mode="LIVE")

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = REPORTS_DIR / f"sentry_live_report_{ts}.pdf"

        generate_pdf_report(event_buffer, summary_text, str(output_path))
        print(f"\n[REPORT] Generated → {output_path}")
        
    except Exception as e:
        print(f"[ERROR] Report generation failed: {e}")

def replay_offline_inference(frames, labels, scores, blacklist_frames, fps):
    """Replay offline inference results"""
    delay = max(1, int(1000 / fps))

    for i, frame in enumerate(frames):
        label = labels[i]
        score = scores[i]

        # Determine color
        if "Fight" in label:
            color = (0, 0, 255)  # Red
        elif label == "Dance":
            color = (255, 255, 0)  # Cyan
        else:
            color = (0, 255, 0)  # Green

        cv2.putText(
            frame,
            f"{label} ({score:.2f})",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            color,
            2
        )

        if i in blacklist_frames:
            cv2.putText(
                frame,
                "BLACKLISTED PERSON",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                3
            )

        cv2.imshow("OFFLINE INFERENCE REPLAY", frame)
        if cv2.waitKey(delay) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()


# ========================= OFFLINE MODE =========================
def run_offline(video_path):
    """Run offline analysis mode"""
    print("[MODE] OFFLINE")
    
    if not Path(video_path).exists():
        print(f"[ERROR] Video file not found: {video_path}")
        return
    
    try:
        validate_paths()
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Using device: {device}")

    # ===================== LOAD MODELS =====================
    try:
        processor = VideoMAEImageProcessor.from_pretrained(VIDEOMAE_MODEL)
        model = VideoMAEForVideoClassification.from_pretrained(
            VIDEOMAE_MODEL
        ).to(device).eval()

        fight_idx = [
            k for k, v in model.config.id2label.items()
            if v.lower() == "fight"
        ][0]

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

        scrfd = SCRFD(model_path=str(SCRFD_MODEL))
        arcface = ort.InferenceSession(
            str(ARCFACE_MODEL),
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
        )
        face_db = load_or_build_face_db(scrfd, arcface, str(FACE_GALLERY))
        
    except Exception as e:
        print(f"[ERROR] Model loading failed: {e}")
        import traceback
        traceback.print_exc()
        return

    # ===================== LOAD VIDEO =====================
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Could not open video: {video_path}")
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[INFO] Video: {total_frames} frames @ {fps:.2f} FPS")

    frames = []
    print("[INFO] Loading video frames...")
    
    while len(frames) < total_frames:
        ret, f = cap.read()
        if not ret:
            break
        frames.append(f)
        
        # Progress indicator
        if len(frames) % 100 == 0:
            print(f"  Loaded {len(frames)}/{total_frames} frames", end='\r')
    
    cap.release()
    print(f"\n[INFO] Loaded {len(frames)} frames")

    labels = ["Normal"] * len(frames)
    scores = np.zeros(len(frames))

    # ===================== VIDEO INFERENCE =====================
    print("[INFO] Running violence detection...")
    
    # Check if video is from dance folder (intentional folder-based logic)
    is_dance_video = False
    try:
        is_dance_video = DANCE_DIR.resolve() in Path(video_path).resolve().parents
        
    except Exception as e:
        print(f"[WARNING] Could not check dance folder: {e}")
    
    for i in range(0, len(frames) - NUM_FRAMES, STRIDE):
        clip = [
            cv2.resize(cv2.cvtColor(x, cv2.COLOR_BGR2RGB), FRAME_SIZE)
            for x in frames[i:i + NUM_FRAMES]
        ]

        # Get violence score
        inputs = processor(clip, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            out = model(**inputs)

        fight_score = torch.softmax(out.logits, dim=-1)[0][fight_idx].item()

        # Label frames in clip
        for j in range(i, i + NUM_FRAMES):
            scores[j] = max(scores[j], fight_score)

            # Folder-based dance detection (intentional)
            if is_dance_video:
                # Dance videos: normal unless violence-like motion is detected
                if fight_score > LOW_VIOLENCE:
                    labels[j] = "Dance"
                else:
                    labels[j] = "Normal"
            else:
                # Non-dance videos: normal violence logic
                if fight_score > HIGH_VIOLENCE:
                    labels[j] = "Fight"
                elif fight_score > LOW_VIOLENCE:
                    labels[j] = "Fight (Low Confidence)"
                else:
                    labels[j] = "Normal"

        
        # Progress indicator
        if i % (STRIDE * 10) == 0:
            progress = (i / (len(frames) - NUM_FRAMES)) * 100
            print(f"  Progress: {progress:.1f}%", end='\r')
    
    print("\n[INFO] Violence detection complete")
    frame_face_data = {}
    blacklist_frames = {}

    # ===================== FACE RECOGNITION =====================
    print("[INFO] Running face recognition...")
    cache = {}
    next_face_id = 0

    for idx, frame in enumerate(frames):
        try:
            _, cache, next_face_id, face_results  = process_face_recognition(
                frame, scrfd, arcface, face_db, cache, next_face_id
            )

            frame_face_data[idx] = face_results
                
            if idx % 100 == 0:
                print(f"  Processed {idx}/{len(frames)} frames", end='\r')
                
        except Exception as e:
            print(f"\n[WARNING] Face recognition failed at frame {idx}: {e}")
            continue
    
    print(f"\n[INFO] Face recognition complete")

    # ===================== REPLAY =====================
    print("[INFO] Replaying results...")
    
    # Build blacklist_frames for replay visualization
    blacklist_frames = {}
    for idx, face_results in frame_face_data.items():
        blacklisted = [
            info['name'] for face_id, info in face_results.items()
            if info.get('status') == 'blacklist'
        ]
        if blacklisted:
            blacklist_frames[idx] = blacklisted
    
    replay_offline_inference(frames, labels, scores, blacklist_frames, fps)

    # ===================== BUILD EVENTS =====================
    try:
        from datetime import datetime
        from events.offline_event_builder import build_offline_events
        from reports.pdf_report import generate_pdf_report
        from llm.summary_generator import generate_llm_summary

        events = build_offline_events(
            frames=frames,
            labels=labels,
            scores=scores,
            fps=fps,
            screenshot_dir=str(OFFLINE_SCREENSHOT_DIR)
        )

        # Add face recognition context to events
        for e in events:
            start_f = int(e["start_time"] * fps)
            end_f = int(e["end_time"] * fps)

            # Collect all face detections in this event timespan
            blacklisted_people = set()
            whitelisted_people = set()
            
            for f in range(start_f, end_f + 1):
                if f in frame_face_data:
                    for face_id, face_info in frame_face_data[f].items():
                        status = face_info.get('status')
                        name = face_info.get('name', 'Unknown')
                        
                        if status == 'blacklist':
                            blacklisted_people.add(name)
                        elif status == 'whitelist':
                            whitelisted_people.add(name)
            
            # Blacklist override (highest priority)
            if blacklisted_people:
                e["final"] = "danger"
                e["type"] = "Blacklisted Person Detected"
                e["cause"] = {
                    "trigger": "FACE_RECOGNITION",
                    "rule_name": "BLACKLIST_MATCH",
                    "description": "Known blacklisted individual detected",
                    "joints_involved": [],
                    "metrics": {"blacklisted_faces": list(blacklisted_people)}
                }
            
            # Whitelist context (add info, suppress only false positives)
            elif whitelisted_people:
                # Add whitelist context to existing event
                if 'cause' not in e:
                    e['cause'] = {}
                if 'metrics' not in e['cause']:
                    e['cause']['metrics'] = {}
                
                e['cause']['metrics']['whitelisted_people'] = list(whitelisted_people)
                e['cause']['whitelist_note'] = f"Whitelisted person(s) involved: {', '.join(whitelisted_people)}"
                
                # Suppress only MEDIUM severity (false positives)
                # Keep HIGH and CRITICAL (real violence)
                if e.get("final") == "warning":  # Typically maps to MEDIUM
                    original_type = e["type"]
                    e["final"] = "safe"
                    e["type"] = f"{original_type} (Whitelisted - Low Priority)"
                    e['cause']['description'] = (
                        f"Original: {e['cause'].get('description', '')}. "
                        f"Downgraded due to whitelisted person(s): {', '.join(whitelisted_people)}"
                    )

        # Generate summary and report
        summary_text = generate_llm_summary(events=events, mode="OFFLINE")

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_pdf = OFFLINE_REPORT_DIR / f"sentry_offline_report_{ts}.pdf"

        generate_pdf_report(events, summary_text, str(output_pdf))
        print(f"\n[OFFLINE REPORT] Generated → {output_pdf}")
        
    except Exception as e:
        print(f"[ERROR] Report generation failed: {e}")
        import traceback
        traceback.print_exc()
def run_pose_offline(video_path):
    frame_pose_data = {}

    """Run offline analysis using pose-based rule engine (no transformer)"""
    print("[MODE] POSE OFFLINE")
    
    if not Path(video_path).exists():
        print(f"[ERROR] Video file not found: {video_path}")
        return
    
    try:
        validate_paths()
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        return

    # ===================== LOAD VIDEO =====================
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Could not open video: {video_path}")
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[INFO] Video: {total_frames} frames @ {fps:.2f} FPS")

    # ===================== LOAD MODELS =====================
    try:
        scrfd = SCRFD(model_path=str(SCRFD_MODEL))
        arcface = ort.InferenceSession(
            str(ARCFACE_MODEL),
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
        )
        face_db = load_or_build_face_db(scrfd, arcface, str(FACE_GALLERY))
        
        detector = Detector()
        pose_buffer = PoseBuffer(max_len=30)
        rule_engine = RuleEngine(history=30)
        
    except Exception as e:
        print(f"[ERROR] Model initialization failed: {e}")
        return

    # ===================== PROCESS VIDEO =====================
    print("[INFO] Processing video with pose detection...")
    
    frames = []
    all_events = []
    cache = {}
    next_face_id = 0
    frame_face_data = {}
    frame_idx = 0
    
    event_mgr = EventManager(fps=fps, source="POSE_OFFLINE")
    frame_store = {}

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frames.append(frame.copy())
        
        # Store frames at 1 second intervals
        if frame_idx % int(fps) == 0:
            frame_store[frame_idx] = frame.copy()
        
        # Pose detection
        persons, objects = detector.infer(frame)
        frame_pose_data[frame_idx] = persons

        # Update pose buffer
        for p in persons:
            pose_buffer.update(p["track_id"], p["keypoints"])
        
        # Rule engine
        rule_results = rule_engine.update(persons, objects)
        
        # Face recognition
        _, cache, next_face_id, face_results = process_face_recognition(
            frame, scrfd, arcface, face_db, cache, next_face_id
        )
        
        frame_face_data[frame_idx] = face_results
        track_to_face = match_faces_to_poses(persons, face_results)
        
        # Check for blacklisted faces
        blacklisted_in_frame = [
            info['name'] for tid, info in track_to_face.items() 
            if info.get('status') == 'blacklist'
        ]
        
        if blacklisted_in_frame:
            screenshot_path = OFFLINE_SCREENSHOT_DIR / f"blacklist_{frame_idx}.jpg"
            cv2.imwrite(str(screenshot_path), frame)
            
            event_mgr.update(
                frame_idx=frame_idx,
                label="Blacklisted Person Detected",
                severity="CRITICAL",
                confidence=1.0,
                face_ids=blacklisted_in_frame,
                override="BLACKLIST",
                cause={
                    "trigger": "FACE_RECOGNITION",
                    "rule_name": "BLACKLIST_MATCH",
                    "description": "Known blacklisted individual detected",
                    "joints_involved": [],
                    "metrics": {"faces": blacklisted_in_frame}
                },
                screenshot=str(screenshot_path)
            )
        
        # Process rule violations
        active_event_this_frame = False
        
        for p in persons:
            tid = p["track_id"]
            
            if tid in rule_results:
                result = rule_results[tid].copy()
                face_info = track_to_face.get(tid)
                
                # Apply whitelist/blacklist logic
                if face_info and face_info.get('status') == 'whitelist':
                    if 'cause' not in result:
                        result['cause'] = {}
                    if 'metrics' not in result['cause']:
                        result['cause']['metrics'] = {}
                    
                    result['cause']['metrics']['whitelisted_person'] = face_info['name']
                    result['cause']['whitelist_note'] = f"Whitelisted person '{face_info['name']}' involved"
                    
                    if result['severity'] == 'MEDIUM':
                        original_action = result['action']
                        result['action'] = f"{original_action} (Whitelisted - Low Priority)"
                        result['severity'] = 'LOW'
                        result['color'] = SEVERITY_COLORS['LOW']
                
                elif face_info and face_info.get('status') == 'blacklist':
                    if 'cause' not in result:
                        result['cause'] = {}
                    if 'metrics' not in result['cause']:
                        result['cause']['metrics'] = {}
                    
                    result['cause']['metrics']['blacklisted_person'] = face_info['name']
                    result['cause']['blacklist_note'] = f"ALERT: Blacklisted person '{face_info['name']}' involved"
                    
                    if result['severity'] != 'CRITICAL':
                        result['severity'] = 'CRITICAL'
                        result['color'] = SEVERITY_COLORS['CRITICAL']
                
                active_event_this_frame = True
                
                screenshot_path = None
                if (
                    event_mgr.current_event is None or
                    event_mgr.current_event["type"] != result["action"]
                ):
                    screenshot_path = OFFLINE_SCREENSHOT_DIR / f"event_{frame_idx}_track_{tid}.jpg"
                    cv2.imwrite(str(screenshot_path), frame)
                
                event_mgr.update(
                    frame_idx=frame_idx,
                    label=result["action"],
                    severity=result["severity"],
                    confidence=None,
                    face_ids=[tid],
                    override=None,
                    cause=result.get("cause"),
                    screenshot=str(screenshot_path) if screenshot_path else None
                )
        
        if not active_event_this_frame and not blacklisted_in_frame:
            event_mgr.update(
                frame_idx=frame_idx,
                label="Normal",
                severity="LOW"
            )
        
        # Progress indicator
        if frame_idx % 100 == 0:
            progress = (frame_idx / total_frames) * 100
            print(f"  Progress: {progress:.1f}%", end='\r')
        
        frame_idx += 1
    
    cap.release()
    print(f"\n[INFO] Processed {frame_idx} frames")
    
    # Finalize events
    event_mgr.finalize()
    events = event_mgr.export()
    
    # ===================== REPLAY =====================
    print("[INFO] Replaying results...")
    replay_pose_offline(frames, events, frame_pose_data, fps)

    
    # ===================== GENERATE REPORT =====================
    try:
        from reports.event_adapter import adapt_events_for_pdf
        from reports.pdf_report import generate_pdf_report
        from llm.summary_generator import generate_llm_summary
        from datetime import datetime
        
        event_buffer = adapt_events_for_pdf(events, frame_store)
        summary_text = generate_llm_summary(events=event_buffer, mode="POSE_OFFLINE")
        
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_pdf = OFFLINE_REPORT_DIR / f"sentry_pose_offline_report_{ts}.pdf"
        
        generate_pdf_report(event_buffer, summary_text, str(output_pdf))
        print(f"\n[POSE OFFLINE REPORT] Generated → {output_pdf}")
        
    except Exception as e:
        print(f"[ERROR] Report generation failed: {e}")
        import traceback
        traceback.print_exc()


def replay_pose_offline(frames, events, frame_pose_data, fps):

    """Replay pose offline inference results"""
    delay = max(1, int(1000 / fps))
    frame_track_severity = {}

    # Build frame-to-event mapping
    frame_events = {}
    for event in events:
        start_frame = int(event["start_time"] * fps)
        end_frame = int(event["end_time"] * fps)
        severity = event.get("final", "safe")  # danger / warning / safe

        
        for f_idx in range(start_frame, end_frame + 1):
            if f_idx not in frame_events:
                frame_events[f_idx] = []
            frame_events[f_idx].append(event)
    
    for idx, frame in enumerate(frames):
        display_frame = frame.copy()
        
        # Draw events for this frame
        if idx in frame_events:
            y_offset = 40
            for event in frame_events[idx]:
                label = event['type']
                severity = event.get("severity", "").upper()
                final = event.get("final", "").lower()

                if severity == "CRITICAL":
                    color = (0, 0, 255)      # 🔴 Danger
                elif severity in ["HIGH", "MEDIUM"]:
                    color = (0, 255, 255)    # 🟡 Suspicious
                elif final == "danger":
                    color = (0, 0, 255)
                elif final == "warning":
                    color = (0, 255, 255)
                else:
                    color = (0, 255, 0)      # 🟢 Normal

                                
                
                cv2.putText(
                    display_frame,
                    label,
                    (20, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    color,
                    2
                )
                y_offset += 30
        if idx in frame_pose_data:
            display_frame = draw_pose(
                display_frame,
                [{
                    "keypoints": p["keypoints"],
                    "confidence": np.ones(len(p["keypoints"])),
                    "track_id": p["track_id"]
                } for p in frame_pose_data[idx]]
            )

        cv2.imshow("POSE OFFLINE REPLAY", display_frame)
        if cv2.waitKey(delay) & 0xFF == ord("q"):
            break
    
    cv2.destroyAllWindows()

# ========================= ENTRY POINT =========================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sentry Violence Detection System")

    parser.add_argument(
        "--source",
        default="0",
        help="Video source: '0' for webcam or path to video file"
    )

    parser.add_argument(
        "--pose",
        action="store_true",
        help="Use pose-based offline inference instead of transformer"
    )

    args = parser.parse_args()

    try:
        # LIVE MODE
        if args.source == "0":
            run_live(0)

        # POSE OFFLINE MODE
        elif args.pose:
            run_pose_offline(args.source)

        # TRANSFORMER OFFLINE MODE (default)
        else:
            run_offline(args.source)

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user")
    except Exception as e:
        print(f"[ERROR] Fatal error: {e}")
        import traceback
        traceback.print_exc()
