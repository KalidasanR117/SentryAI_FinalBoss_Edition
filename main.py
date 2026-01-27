# ================== HARD BLOCK TENSORFLOW (MUST BE FIRST) ==================
import os
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
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
from alerts.notifier import send_critical_alert

# ========================= LIVE PIPELINE IMPORTS =========================
from core.detector import Detector
from core.pose_buffer import PoseBuffer
from core.rule_engine import RuleEngine
from utils.visualization import draw_pose
from facial_analysis.models import SCRFD
from events.event_manager import EventManager
from core.camera_manager import CameraManager, CameraConfig, EventSeverity, map_severity_to_enum
from pose_face_main import load_or_build_face_db, process_face_recognition

# ========================= CONFIG & GLOBALS =========================
SEVERITY_RANK = {
    EventSeverity.NORMAL: 0, EventSeverity.LOW: 1,
    EventSeverity.MEDIUM: 2, EventSeverity.HIGH: 3, EventSeverity.CRITICAL: 4,
}

API_MODE = True
STREAM_FRAME = None
import threading
STREAM_LOCK = threading.Lock()
API_ONLY = True
RUN_LIVE = False
PAUSE_LIVE = False
STOP_LIVE = False
CAMERA_MANAGER = None
ACTIVE_CAMERA_MGR = None
CURRENT_MODE = "IDLE"  

BASE_DIR = Path(__file__).resolve().parent
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

SEVERITY_COLORS = {
    "CRITICAL": (0, 0, 255), "HIGH": (0, 100, 255),
    "MEDIUM": (0, 255, 255), "LOW": (0, 255, 0)
}

# Directories
REPORTS_DIR = BASE_DIR / "reports"
SCREENSHOT_DIR = REPORTS_DIR / "screenshots"
OFFLINE_REPORT_DIR = REPORTS_DIR / "offline"
OFFLINE_SCREENSHOT_DIR = OFFLINE_REPORT_DIR / "screenshots"
for d in [SCREENSHOT_DIR, OFFLINE_SCREENSHOT_DIR]: d.mkdir(parents=True, exist_ok=True)

# ========================= HELPERS =========================
def create_onnx_session(model_path):
    providers = [
        ("CUDAExecutionProvider", {
            "device_id": 0,
            "arena_extend_strategy": "kNextPowerOfTwo",
            "gpu_mem_limit": 4 * 1024 * 1024 * 1024,
            "cudnn_conv_algo_search": "EXHAUSTIVE"
        }),
        "CPUExecutionProvider"
    ]

    sess = ort.InferenceSession(str(model_path), providers=providers)
    print(f"[ONNX] {model_path.name} → {sess.get_providers()}")
    return sess

def push_frame(frame):
    global STREAM_FRAME
    with STREAM_LOCK:
        STREAM_FRAME = frame.copy()

def validate_paths():
    required = [SCRFD_MODEL, ARCFACE_MODEL]
    missing = [f for f in required if not f.exists()]
    if missing: raise FileNotFoundError(f"Missing: {missing}")
    if not FACE_GALLERY.exists(): FACE_GALLERY.mkdir(parents=True, exist_ok=True)

def get_bbox_from_keypoints(keypoints):
    valid = [kp for kp in keypoints if len(kp) >= 2]
    if not valid: return None
    xs = [p[0] for p in valid]; ys = [p[1] for p in valid]
    return {'x1': min(xs), 'y1': min(ys), 'x2': max(xs), 'y2': max(ys)}

def bbox_iou(bbox1, bbox2):
    if not bbox1 or not bbox2: return 0.0
    x1 = max(bbox1['x1'], bbox2['x1']); y1 = max(bbox1['y1'], bbox2['y1'])
    x2 = min(bbox1['x2'], bbox2['x2']); y2 = min(bbox1['y2'], bbox2['y2'])
    if x2 < x1 or y2 < y1: return 0.0
    inter = (x2 - x1) * (y2 - y1)
    union = ((bbox1['x2']-bbox1['x1'])*(bbox1['y2']-bbox1['y1'])) + \
            ((bbox2['x2']-bbox2['x1'])*(bbox2['y2']-bbox2['y1'])) - inter
    return inter / union if union > 0 else 0.0

def match_faces_to_poses(persons, face_results, iou_threshold=0.3):
    track_to_face = {}
    for person in persons:
        pose_bbox = get_bbox_from_keypoints(person['keypoints'])
        if not pose_bbox: continue
        best_iou = 0.0
        best_match = None
        for _, info in face_results.items():
            if 'bbox' not in info: continue
            fb = info['bbox']
            f_bbox = {"x1": fb[0], "y1": fb[1], "x2": fb[2], "y2": fb[3]} if isinstance(fb, (list, tuple)) else fb
            iou = bbox_iou(pose_bbox, f_bbox)
            if iou > best_iou and iou >= iou_threshold:
                best_iou = iou; best_match = info
        if best_match:
            track_to_face[person['track_id']] = {
                'name': best_match.get('name', 'Unknown'),
                'status': best_match.get('status', None)
            }
    return track_to_face

# 🔥 NEW HELPER: Letterboxing (Fit video to box with black bars)
def resize_with_padding(image, target_size=(640, 480)):
    h, w = image.shape[:2]
    th, tw = target_size
    scale = min(tw/w, th/h)
    
    nw, nh = int(w * scale), int(h * scale)
    resized = cv2.resize(image, (nw, nh))
    
    # Create black canvas
    canvas = np.zeros((th, tw, 3), dtype=np.uint8)
    
    # Center image
    x_off = (tw - nw) // 2
    y_off = (th - nh) // 2
    canvas[y_off:y_off+nh, x_off:x_off+nw] = resized
    
    return canvas, scale, x_off, y_off
def send_offline_summary_alert(events, mode):
    if not events:
        return

    summary = {}
    highest_severity = "LOW"

    for e in events:
        label = e["type"]
        severity = e.get("severity", "LOW")

        summary[label] = summary.get(label, 0) + 1

        if severity == "CRITICAL":
            highest_severity = "CRITICAL"
        elif severity == "HIGH" and highest_severity != "CRITICAL":
            highest_severity = "HIGH"

    # Build summary text
    lines = [f"{k} × {v}" for k, v in summary.items()]
    summary_text = " | ".join(lines)

    send_critical_alert(
        event={
            "type": "Offline Replay Summary",
            "severity": highest_severity,
            "confidence": 1.0,
            "cause": {
                "description": "Offline analysis completed",
                "summary": summary_text
            }
        },
        report_path=None,
        mode=mode
    )

# ========================= 🔥 REPLAY ENGINE (FIT TO BOX) =========================
def stream_replay_to_browser(video_path, frame_data_map, events, fps, mode="POSE"):
    global RUN_LIVE, STOP_LIVE

    # Ensure flags are set so the loop starts
    RUN_LIVE = True
    STOP_LIVE = False

    print(f"\n[REPLAY] Streaming result to browser ({fps} FPS loop)...")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("[REPLAY] Failed to open video")
        return

    # 1. Calculate Scaling Factors
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    target_w, target_h = 640, 480
    scale_x = target_w / orig_w
    scale_y = target_h / orig_h

    # 2. Build Event Lookup (Optimization)
    frame_events = {}
    for e in events:
        start_f = int(e["start_time"] * fps)
        end_f = int(e["end_time"] * fps)
        for f in range(start_f, end_f + 1):
            frame_events.setdefault(f, []).append(e)

    frame_idx = 0
    delay = 1.0 / fps

    while RUN_LIVE and not STOP_LIVE:
        start_time = time.time()

        if PAUSE_LIVE:
            time.sleep(0.1)
            continue

        ret, frame = cap.read()
        
        # Loop Video
        if not ret:
            if STOP_LIVE:
                break
            frame_idx = 0
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue


        # Force Resize to fit Camera Box
        display_frame = cv2.resize(frame, (target_w, target_h))

        # ===================== POSE MODE SPECIFIC =====================
        if mode == "POSE" and frame_idx in frame_data_map:
            persons, rule_results = frame_data_map[frame_idx]
            scaled_persons = []

            for p in persons:
                # Deep copy and scale keypoints
                kp = p["keypoints"].copy()
                kp[:, 0] *= scale_x
                kp[:, 1] *= scale_y

                # Determine Action Label
                action = "Normal"
                label_color = (0, 255, 0) # Green

                if p["track_id"] in rule_results:
                    r = rule_results[p["track_id"]]
                    action = r["action"]
                    if r["severity"] == "CRITICAL":
                        label_color = (0, 0, 255) # Red
                    elif r["severity"] == "HIGH":
                        label_color = (0, 165, 255) # Orange

                # Draw Label above head
                if len(kp) > 0:
                    head_x = int(kp[0][0])
                    head_y = int(kp[0][1]) - 15
                    if head_y < 20: head_y = 20
                    
                    # Outline + Text
                    cv2.putText(display_frame, action, (head_x, head_y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
                    cv2.putText(display_frame, action, (head_x, head_y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, label_color, 2)

                scaled_persons.append({
                    "keypoints": kp,
                    "confidence": np.ones(len(kp)),
                    "track_id": p["track_id"]
                })

            display_frame = draw_pose(display_frame, scaled_persons)

        # ===================== SHARED STATUS BAR (BOTTOM) =====================
        # Defaults
        main_label = "Normal"
        main_score = 0.0
        bar_color = (0, 255, 0) # Green
        is_alert = False

        # 1. Check Transformer Data
        if mode == "TRANSFORMER" and frame_idx in frame_data_map:
            lbl, scr, _ = frame_data_map[frame_idx]
            main_label = lbl
            main_score = scr
            
            if "Fight" in lbl: 
                bar_color = (0, 0, 255) # Red
                is_alert = True
            elif "Dance" in lbl: 
                bar_color = (255, 255, 0) # Cyan
                is_alert = True
            elif "Suspicious" in lbl: 
                bar_color = (0, 165, 255) # Orange
                is_alert = True

        # 2. Check Event Timeline (Overrides Transformer)
        if frame_idx in frame_events:
            e = frame_events[frame_idx][0] # Take first event
            main_label = e["type"]
            is_alert = True
            if e.get("final") == "danger":
                bar_color = (0, 0, 255) # Red
            else:
                bar_color = (0, 255, 255) # Yellow/Cyan

        # Draw the Bar
        status_text = f"ALERT: {main_label}" if is_alert else f"STATUS: {main_label}"
        if main_score > 0: status_text += f" ({main_score:.2f})"

        # ===================== SAFE STATUS BAR (BOTTOM) =====================
        h, w = display_frame.shape[:2]

        BAR_HEIGHT = 40
        BAR_MARGIN = 80   # <-- move bar UP safely (this is the key)

        bar_top = h - BAR_MARGIN - BAR_HEIGHT
        bar_bottom = h - BAR_MARGIN

        # Draw background
        cv2.rectangle(
            display_frame,
            (0, bar_top),
            (w, bar_bottom),
            (0, 0, 0),
            -1
        )

        # Draw text
        cv2.putText(
            display_frame,
            status_text,
            (20, bar_bottom - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            bar_color,
            2
        )



        # ===================== STREAM =====================
        push_frame(display_frame)
        frame_idx += 1

        # FPS Pacing
        process_time = time.time() - start_time
        sleep_time = delay - process_time
        if sleep_time > 0:
            time.sleep(sleep_time)

    cap.release()
    RUN_LIVE = False # Reset flag when stopped
    print("[REPLAY] Stopped.")

# ========================= OFFLINE POSE (NO FACE REC) =========================
def run_pose_offline(video_path):
    print("[MODE] POSE OFFLINE (Face Rec: OFF)")
    if not Path(video_path).exists(): return

    try:
        # NOTE: Only loading Detector & Rule Engine. NO FACE MODELS.
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        detector = Detector()
        pose_buffer = PoseBuffer(max_len=30)
        rule_engine = RuleEngine(history=30)
    except Exception as e: print(f"[INIT ERROR] {e}"); return

    print(f"[INFO] Analyzing {total_frames} frames...")
    frame_idx = 0
    frame_data_map = {}
    frame_store = {}
    event_mgr = EventManager(fps=fps, source="POSE_OFFLINE")

    while True:
        ret, frame = cap.read()
        if not ret: break
        
        if frame_idx % int(fps) == 0: frame_store[frame_idx] = frame.copy()

        # Inference
        persons, objects = detector.infer(frame)
        for p in persons: pose_buffer.update(p["track_id"], p["keypoints"])
        rule_results = rule_engine.update(persons, objects)
        
        # Save results for replay
        frame_data_map[frame_idx] = (persons, rule_results)

        active = False
        for p in persons:
            if p["track_id"] in rule_results:
                res = rule_results[p["track_id"]]
                # Log severe events
                if res["severity"] != "NORMAL":
                    event_mgr.update(frame_idx=frame_idx, label=res["action"], severity=res["severity"], face_ids=[p["track_id"]])
                    active = True
        
        if not active:
            event_mgr.update(frame_idx=frame_idx, label="Normal", severity="LOW")

        if frame_idx % 50 == 0: print(f"   Progress: {int(frame_idx/total_frames*100)}%", end='\r')
        frame_idx += 1

    cap.release()
    event_mgr.finalize()
    events = event_mgr.export()
    
    # Reports
    try:
        from reports.event_adapter import adapt_events_for_pdf
        from reports.pdf_report import generate_pdf_report
        from llm.summary_generator import generate_llm_summary
        from datetime import datetime
        buf = adapt_events_for_pdf(events, frame_store)
        txt = generate_llm_summary(events=buf, mode="POSE_OFFLINE")
        out = OFFLINE_REPORT_DIR / f"pose_report_{datetime.now().strftime('%H%M%S')}.pdf"
        generate_pdf_report(buf, txt, str(out))
    except: pass

    stream_replay_to_browser(video_path, frame_data_map, events, fps, mode="POSE")
    send_offline_summary_alert(events, mode="OFFLINE")
# ========================= OFFLINE TRANSFORMER (NO FACE REC) =========================
def run_offline(video_path):
    print("[MODE] TRANSFORMER OFFLINE (Face Rec: OFF)")
    if not Path(video_path).exists(): return

    try:
        # NOTE: Only loading VideoMAE. NO FACE MODELS.
        device = "cuda" if torch.cuda.is_available() else "cpu"
        processor = VideoMAEImageProcessor.from_pretrained(VIDEOMAE_MODEL)
        model = VideoMAEForVideoClassification.from_pretrained(VIDEOMAE_MODEL).to(device).eval()
        fight_idx = [k for k, v in model.config.id2label.items() if v.lower() == "fight"][0]
    except Exception as e: print(f"[INIT ERROR] {e}"); return

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frames = []
    print("[INFO] Loading video into RAM...")
    while True:
        ret, f = cap.read()
        if not ret: break
        frames.append(f)
    cap.release()
    total_frames = len(frames)

    # Dance Logic
    is_dance_video = False
    try:
        if "dance" in Path(video_path).name.lower() or "dance" in str(Path(video_path).parent).lower():
            is_dance_video = True
            print(f"[INFO] 💃 Dance Mode ACTIVATED")
    except: pass

    # Initialize all as Normal
    labels = ["Normal"] * total_frames
    scores = np.zeros(total_frames)
    frame_store = {}

    print("[INFO] Running VideoMAE...")
    for i in range(0, total_frames - NUM_FRAMES, STRIDE):
        if i % int(fps) == 0: frame_store[i] = frames[i].copy()

        clip = [cv2.resize(cv2.cvtColor(x, cv2.COLOR_BGR2RGB), FRAME_SIZE) for x in frames[i:i+NUM_FRAMES]]
        inputs = processor(clip, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad(): out = model(**inputs)
        
        fight_score = torch.softmax(out.logits, dim=-1)[0][fight_idx].item()
        
        for j in range(i, i+NUM_FRAMES):
            scores[j] = max(scores[j], fight_score)
            
            if is_dance_video:
                if fight_score > HIGH_VIOLENCE: labels[j] = "Normal"
                elif fight_score > LOW_VIOLENCE: labels[j] = "Fight (Low)"
            else:
                if fight_score >= HIGH_VIOLENCE:
                    labels[j] = "Fight"
                elif fight_score >= LOW_VIOLENCE:
                    labels[j] = "Fight"
                else:
                    labels[j] = "Normal"

        
        if i % 100 == 0: print(f"   MAE Progress: {int(i/total_frames*100)}%", end='\r')

    # Prepare Replay Data
    frame_data_map = {}
    for i in range(total_frames):
        # Empty list for blacklist, Face Rec is OFF
        frame_data_map[i] = (labels[i], scores[i], [])

    # Build Events & Sync
    from events.offline_event_builder import build_offline_events
    events = build_offline_events(frames, labels, scores, fps, str(OFFLINE_SCREENSHOT_DIR))
    # 🔥 FIX: Ensure severity exists for all transformer events
    for e in events:
        if "severity" not in e:
            if "Fight" in e["type"]:
                e["severity"] = "CRITICAL" if e.get("confidence", 0) >= HIGH_VIOLENCE else "HIGH"
            else:
                e["severity"] = "LOW"
        
    temp_mgr = EventManager(fps=fps, source="OFFLINE_TRANSFORMER")
    for e in events:
        temp_mgr.update(
            frame_idx=int(e["start_time"] * fps),
            label=e["type"],
            severity=e.get("severity", "LOW"),
            confidence=0.95,
            face_ids=[], 
            cause=e.get("cause")
        )

    # Report
    try:
        from reports.event_adapter import adapt_events_for_pdf
        from reports.pdf_report import generate_pdf_report
        from llm.summary_generator import generate_llm_summary
        from datetime import datetime
        buf = adapt_events_for_pdf(events, frame_store)
        txt = generate_llm_summary(events=buf, mode="OFFLINE")
        out = OFFLINE_REPORT_DIR / f"transformer_report_{datetime.now().strftime('%H%M%S')}.pdf"
        generate_pdf_report(buf, txt, str(out))
    except Exception as e:
        print("[OFFLINE TRANSFORMER REPORT ERROR]", e)


    stream_replay_to_browser(video_path, frame_data_map, events, fps, mode="TRANSFORMER")
    send_offline_summary_alert(events, mode="OFFLINE")
# ========================= LIVE MODE =========================
def run_live(source):
    global RUN_LIVE, PAUSE_LIVE, STOP_LIVE
    RUN_LIVE = True
    PAUSE_LIVE = False
    # STOP_LIVE = False
    alert_sent_for_event = False

    """Run live detection mode with camera rotation support"""
    print("[MODE] LIVE")
    print("CUDA available:", torch.cuda.is_available())
    print("CUDA device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")

    try:
        validate_paths()
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        return

    # ===================== CAMERA SETUP =====================
    camera_mgr = None
    # Handle "0" string or int 0
    use_rotation = (str(source) == "0" or source == 0) and (source != "1") 
    
    if use_rotation:
        print("[CAMERA] Initializing multi-camera rotation...")
        config = CameraConfig(
            base_time_window=10.0,
            max_scan_index=10
        )
        
        global CAMERA_MANAGER
        CAMERA_MANAGER = CameraManager(config)
        camera_mgr = CAMERA_MANAGER

        if not camera_mgr.initialize():
            print("[WARNING] Falling back to single camera")
            use_rotation = False
        else:
            if not camera_mgr.start_camera():
                print("[ERROR] Failed to start camera")
                return
            fps = camera_mgr.get_fps()
        global ACTIVE_CAMERA_MGR
        ACTIVE_CAMERA_MGR = camera_mgr


    if not use_rotation:
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print(f"[ERROR] Could not open: {source}")
            return
        fps = cap.get(cv2.CAP_PROP_FPS) or 30

    # ===================== INIT MODELS =====================
    try:
        scrfd = SCRFD(
            model_path=str(SCRFD_MODEL)
        )

        
        # 🔥 DEBUG: CHECK ONNX DEVICE
        import onnxruntime
        print(f"ONNX Runtime Device: {onnxruntime.get_device()}")
        
        
        arcface = create_onnx_session(ARCFACE_MODEL)
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
    
    # 🔥 OPTIMIZATION STATE
    last_face_results = {}

    # ===================== MAIN LOOP =====================
    try:
        while RUN_LIVE  and not STOP_LIVE:
            if STOP_LIVE:
                print("[LIVE] Stop requested → breaking loop")
                break
                
            if PAUSE_LIVE:
                time.sleep(0.05)
                continue

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

            # 🔥 OPTIMIZATION 1: RESIZE INPUT
            # Resize large frames (e.g., 1080p) to 640px width.
            # This drastically reduces CPU load for Face Rec and WebRTC encoding.
            height, width = frame.shape[:2]
            if width > 640:
                scale = 640 / width
                new_height = int(height * scale)
                frame = cv2.resize(frame, (640, new_height))

            if frame_idx % 30 == 0:
                 # Debug print to ensure resizing is working
                 print(f"✅ [DEBUG] Processing Frame {frame_idx} | Shape: {frame.shape}")     

            if frame_idx % int(fps) == 0:
                frame_store[frame_idx] = frame.copy()

            # 1. POSE DETECTION
            persons, objects = detector.infer(frame)

            for p in persons:
                pose_buffer.update(p["track_id"], p["keypoints"])

            # 2. RULE ENGINE
            rule_results = rule_engine.update(persons, objects)

            # 3. 🔥 OPTIMIZATION 2: THROTTLE FACE RECOGNITION
            # Run Face Rec only every 5 frames. Reuse results in between.
            if frame_idx % 5 == 0:
                frame, cache, next_face_id, face_results = process_face_recognition(
                    frame, scrfd, arcface, face_db, cache, next_face_id
                )
                last_face_results = face_results # Save for next frames
            else:
                face_results = last_face_results # Reuse cached results
            
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
            if blacklisted_in_frame and not alert_sent_for_event:
                send_critical_alert(
                    event={
                        "type": "Blacklisted Person Detected",
                        "severity": "CRITICAL",
                        "confidence": 1.0,
                        "cause": {
                            "description": "Known blacklisted individual detected"
                        }
                    },
                    report_path=None,
                    mode="LIVE"
                )
                alert_sent_for_event = True

            for p in persons:
                tid = p["track_id"]

                if tid in rule_results:
                    result = rule_results[tid].copy()
                    
                    result_severity = map_severity_to_enum(result['severity'])
                    if SEVERITY_RANK[result_severity] > SEVERITY_RANK[current_severity]:
                        current_severity = result_severity

                    face_info = track_to_face.get(tid)
                    
                    
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
                    if result["severity"] == "CRITICAL" and not alert_sent_for_event:
                        send_critical_alert(
                            event={
                                "type": result["action"],
                                "severity": result["severity"],
                                "confidence": result.get("confidence"),
                                "cause": result.get("cause"),
                            },
                            report_path=None,   # 🔥 NO REPORT (as you decided)
                            mode="LIVE"
                        )
                        alert_sent_for_event = True
                    
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
                alert_sent_for_event = False

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

           # stream at max 15 FPS
            if API_MODE:
                push_frame(frame_out)

            # NEW: Check for camera rotation
            if STOP_LIVE:
                break

            if use_rotation and camera_mgr.should_rotate():
                if camera_mgr.rotate_camera():

                    fps = camera_mgr.get_fps()
                    event_mgr.fps = fps
                else:
                    break
            
            frame_idx += 1

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user")
    except Exception as e:
        print(f"[ERROR] Runtime error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        
        PAUSE_LIVE = False

        if use_rotation:
            camera_mgr.stop_camera()
        else:
            cap.release()

        # 🔥 VERY IMPORTANT
        global STREAM_FRAME
        with STREAM_LOCK:
            STREAM_FRAME = None

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

# ========================= ENTRY POINT =========================
if __name__ == "__main__":
    if API_ONLY:
        print("[INFO] main.py running in API mode")
    else:
        pass