
import os
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


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
from alerts.telegram_notifier import telegram_bot

from core.detector import Detector
from core.pose_buffer import PoseBuffer
from core.rule_engine import RuleEngine
from utils.visualization import draw_pose
from facial_analysis.models import SCRFD
from events.event_manager import EventManager
from core.camera_manager import CameraManager, CameraConfig, EventSeverity, map_severity_to_enum
from pose_face_main import load_or_build_face_db, process_face_recognition


DISPLAY_WIDTH = 1280
DISPLAY_HEIGHT = 720
BLACKLIST_HOLD_FRAMES = 15  
blacklist_hold_counter = 0

SEVERITY_RANK = {
    EventSeverity.NORMAL: 0, EventSeverity.LOW: 1,
    EventSeverity.MEDIUM: 2, EventSeverity.HIGH: 3, EventSeverity.CRITICAL: 4,
}

API_MODE = True
STREAM_FRAME = None
CURRENT_FPS = 0  
import threading
STREAM_LOCK = threading.Lock()
API_ONLY = True
RUN_LIVE = False
PAUSE_LIVE = False
STOP_LIVE = False
CAMERA_MANAGER = None
ACTIVE_CAMERA_MGR = None
CURRENT_MODE = "IDLE"  


OFFLINE_STATUS = {
    "progress": 0,
    "status": "IDLE",
    "mode": None
}

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


REPORTS_DIR = BASE_DIR / "reports"
SCREENSHOT_DIR = REPORTS_DIR / "screenshots"
OFFLINE_REPORT_DIR = REPORTS_DIR / "offline"
OFFLINE_SCREENSHOT_DIR = OFFLINE_REPORT_DIR / "screenshots"
for d in [SCREENSHOT_DIR, OFFLINE_SCREENSHOT_DIR]: d.mkdir(parents=True, exist_ok=True)


def update_offline_progress(prog, status_msg):
    """Helper to update global status safely"""
    global OFFLINE_STATUS
    OFFLINE_STATUS["progress"] = prog
    OFFLINE_STATUS["status"] = status_msg

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

def resize_with_padding(image, target_size=(640, 480)):
    h, w = image.shape[:2]
    th, tw = target_size
    scale = min(tw/w, th/h)
    
    nw, nh = int(w * scale), int(h * scale)
    resized = cv2.resize(image, (nw, nh))
    
    canvas = np.zeros((th, tw, 3), dtype=np.uint8)
    
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

    update_offline_progress(100, "Playback Started")
    
    RUN_LIVE = True
    STOP_LIVE = False

    print(f"\n[REPLAY] Streaming result to browser ({fps} FPS loop)...")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("[REPLAY] Failed to open video")
        return

    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    target_w, target_h = 640, 480
    scale_x = target_w / orig_w
    scale_y = target_h / orig_h

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
        
        if not ret:
            if STOP_LIVE:
                break
            frame_idx = 0
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        display_frame = cv2.resize(frame, (target_w, target_h))

        if mode == "POSE" and frame_idx in frame_data_map:
            persons, rule_results = frame_data_map[frame_idx]
            scaled_persons = []

            for p in persons:
                kp = p["keypoints"].copy()
                kp[:, 0] *= scale_x
                kp[:, 1] *= scale_y

                action = "Normal"
                label_color = (0, 255, 0)

                if p["track_id"] in rule_results:
                    r = rule_results[p["track_id"]]
                    action = r["action"]
                    if r["severity"] == "CRITICAL":
                        label_color = (0, 0, 255)
                    elif r["severity"] == "HIGH":
                        label_color = (0, 165, 255)

                if len(kp) > 0:
                    head_x = int(kp[0][0])
                    head_y = int(kp[0][1]) - 15
                    if head_y < 20: head_y = 20
                    
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

        main_label = "Normal"
        main_score = 0.0
        bar_color = (0, 255, 0)
        is_alert = False

        if mode == "TRANSFORMER" and frame_idx in frame_data_map:
            lbl, scr, face_results = frame_data_map[frame_idx]
            main_label = lbl
            main_score = scr
            
            # 🔥 Draw recognized faces over Transformer output
            if face_results:
                for _, info in face_results.items():
                    if 'bbox' in info:
                        name = info.get('name', 'Unknown')
                        # 🔥 ONLY DRAW IF KNOWN
                        if name != 'Unknown':
                            fb = info['bbox']
                            x1, y1, x2, y2 = map(int, fb)
                            x1, y1, x2, y2 = int(x1*scale_x), int(y1*scale_y), int(x2*scale_x), int(y2*scale_y)
                            status = info.get('status', 'unknown')
                            
                            color = (255, 0, 0)
                            if status == 'blacklist': color = (0, 0, 255)
                            elif status == 'whitelist': color = (0, 255, 0)
                            
                            cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
                            cv2.putText(display_frame, f"{name}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            if "Fight" in lbl: 
                bar_color = (0, 0, 255)
                is_alert = True
            elif "Dance" in lbl: 
                bar_color = (255, 255, 0)
                is_alert = True
            elif "Suspicious" in lbl: 
                bar_color = (0, 165, 255)
                is_alert = True

        if frame_idx in frame_events:
            e = frame_events[frame_idx][0]
            main_label = e["type"]
            is_alert = True
            if e.get("final") == "danger":
                bar_color = (0, 0, 255)
            else:
                bar_color = (0, 255, 255)

        status_text = f"ALERT: {main_label}" if is_alert else f"STATUS: {main_label}"
        if main_score > 0: status_text += f" ({main_score:.2f})"

        h, w = display_frame.shape[:2]
        BAR_HEIGHT = 40
        BAR_MARGIN = 80

        bar_top = h - BAR_MARGIN - BAR_HEIGHT
        bar_bottom = h - BAR_MARGIN

        cv2.rectangle(display_frame, (0, bar_top), (w, bar_bottom), (0, 0, 0), -1)
        cv2.putText(display_frame, status_text, (20, bar_bottom - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, bar_color, 2)

        push_frame(display_frame)
        frame_idx += 1

        process_time = time.time() - start_time
        sleep_time = delay - process_time
        if sleep_time > 0:
            time.sleep(sleep_time)

    cap.release()
    RUN_LIVE = False
    print("[REPLAY] Stopped.")

# ========================= OFFLINE POSE (NO FACE REC) =========================
def run_pose_offline(video_path):
    print("[MODE] POSE OFFLINE (Face Rec: OFF)")
    update_offline_progress(0, "Initializing Pose Models...")
    
    if not Path(video_path).exists(): return

    try:
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

        persons, objects = detector.infer(frame)
        for p in persons: pose_buffer.update(p["track_id"], p["keypoints"])
        rule_results = rule_engine.update(persons, objects)
        
        frame_data_map[frame_idx] = (persons, rule_results)

        active = False
        for p in persons:
            if p["track_id"] in rule_results:
                res = rule_results[p["track_id"]]
                if res["severity"] != "NORMAL":
                    
                    screenshot_path = None
                    if event_mgr.is_new_event(p["track_id"], res["action"]):
                        screenshot_path = OFFLINE_SCREENSHOT_DIR / f"pose_event_{frame_idx}_track_{p['track_id']}.jpg"
                        cv2.imwrite(str(screenshot_path), frame)

                    event_mgr.update(
                        frame_idx=frame_idx,
                        label=res["action"],
                        severity=res["severity"],
                        face_ids=[p["track_id"]],
                        screenshot=str(screenshot_path) if screenshot_path else None,
                        cause=res.get("cause")
                    )
                    active = True
        
        if not active:
            event_mgr.update(frame_idx=frame_idx, label="Normal", severity="LOW")

        if frame_idx % 10 == 0:
            progress = int((frame_idx / total_frames) * 90)
            update_offline_progress(progress, f"Analyzing Movements ({progress}%)")
            print(f"   Progress: {progress}%", end='\r')

        frame_idx += 1

    cap.release()
    update_offline_progress(90, "Generating Final Report...")

    event_mgr.finalize()
    events = event_mgr.export()
    
    try:
        from reports.event_adapter import adapt_events_for_pdf
        from reports.pdf_report import generate_pdf_report
        from llm.summary_generator import generate_llm_summary
        from datetime import datetime
        buf = adapt_events_for_pdf(events, frame_store)
        txt = generate_llm_summary(events=buf, mode="POSE_OFFLINE")
        out = OFFLINE_REPORT_DIR / f"pose_report_{datetime.now().strftime('%H%M%S')}.pdf"
        generate_pdf_report(buf, txt, str(out))
        telegram_bot.send_report(str(out), txt)
    except: pass

    update_offline_progress(100, "Starting Replay...")
    stream_replay_to_browser(video_path, frame_data_map, events, fps, mode="POSE")
    send_offline_summary_alert(events, mode="OFFLINE")

# ========================= OFFLINE TRANSFORMER (WITH CONSTANT FACE REC) =========================
def run_offline(video_path):
    print("[MODE] TRANSFORMER OFFLINE (Face Rec: ON - CONSTANT)")
    update_offline_progress(0, "Initializing Transformer & Face Models...")
    
    if not Path(video_path).exists(): return

    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        processor = VideoMAEImageProcessor.from_pretrained(VIDEOMAE_MODEL)
        model = VideoMAEForVideoClassification.from_pretrained(VIDEOMAE_MODEL).to(device).eval()
        fight_idx = [k for k, v in model.config.id2label.items() if v.lower() == "fight"][0]

        # 🔥 Init Face Recognition Models
        scrfd = SCRFD(model_path=str(SCRFD_MODEL))
        arcface = create_onnx_session(ARCFACE_MODEL)
        face_db = load_or_build_face_db(scrfd, arcface, str(FACE_GALLERY))
    except Exception as e: print(f"[INIT ERROR] {e}"); return

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frames = []
    
    update_offline_progress(5, "Loading Video Frames...")
    print("[INFO] Loading video into RAM...")
    while True:
        ret, f = cap.read()
        if not ret: break
        frames.append(f)
    cap.release()
    total_frames = len(frames)

    is_dance_video = False
    try:
        if "dance" in Path(video_path).name.lower() or "dance" in str(Path(video_path).parent).lower():
            is_dance_video = True
    except: pass

    labels = ["Normal"] * total_frames
    scores = np.zeros(total_frames)
    frame_store = {}
    face_map = {} 
    
    # 🔥 PHASE 1: CONSTANT FACE RECOGNITION (EVERY FRAME)
    print("[INFO] Scanning for faces...")
    update_offline_progress(10, "Scanning Faces...")

    cache = {}
    next_face_id = 0

    for i in range(total_frames):
        # Use a copy so process_face_recognition's default "Unknown" drawing doesn't permanently mark the frame
        temp_frame = frames[i].copy()
        _, cache, next_face_id, face_results = process_face_recognition(
            temp_frame, scrfd, arcface, face_db, cache, next_face_id
        )
        
        # Manually draw ONLY KNOWN faces onto the REAL frame used for screenshots
        for tid, info in face_results.items():
            name = info.get('name', 'Unknown')
            if name != "Unknown":
                bbox = info['bbox']
                x1, y1, x2, y2 = map(int, bbox)
                status = info.get('status', 'unknown')
                
                color = (255, 0, 0) # Default Blue for known
                if status == 'blacklist': color = (0, 0, 255)
                elif status == 'whitelist': color = (0, 255, 0)
                
                cv2.rectangle(frames[i], (x1, y1), (x2, y2), color, 2)
                cv2.putText(frames[i], name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        face_map[i] = face_results

        if i % 15 == 0:
            current_pct = 5 + int((i / total_frames) * 45) # Progress 5% to 50%
            update_offline_progress(current_pct, f"Scanning Faces ({current_pct}%)")

    
    print("[INFO] Running VideoMAE...")
    update_offline_progress(50, "Starting VideoMAE Analysis...")

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
                elif fight_score > LOW_VIOLENCE: labels[j] = "Fight"
            else:
                if fight_score >= HIGH_VIOLENCE:
                    labels[j] = "Fight"
                elif fight_score >= LOW_VIOLENCE:
                    labels[j] = "Fight"
                else:
                    labels[j] = "Normal"
        
        if i % 100 == 0: 
            current_pct = 50 + int((i / total_frames) * 40) # Progress 50% to 90%
            update_offline_progress(current_pct, f"Analyzing Violence ({current_pct}%)")

    update_offline_progress(90, "Building Event Timeline...")

    frame_data_map = {}
    for i in range(total_frames):
        faces_in_frame = face_map.get(i, None)
        frame_data_map[i] = (labels[i], scores[i], faces_in_frame)

    # 🔥 Build Timeline Events
    event_mgr = EventManager(fps=fps, source="OFFLINE_TRANSFORMER")
    
    for frame_idx in range(total_frames):
        current_label = labels[frame_idx]
        current_score = scores[frame_idx]
        
        blacklisted_faces = []
        if frame_idx in face_map:
            blacklisted_faces = [d["name"] for d in face_map[frame_idx].values() if d.get("status") == "blacklist"]

        if blacklisted_faces:
            screenshot_path = None
            # Only save a screenshot if it's a NEW encounter (prevents 100 screenshots of the same face)
            if event_mgr.is_new_event("blacklist_offline", "Blacklisted Person Detected"):
                screenshot_path = OFFLINE_SCREENSHOT_DIR / f"transformer_blacklist_{frame_idx}.jpg"
                cv2.imwrite(str(screenshot_path), frames[frame_idx])
            
            event_mgr.update(
                frame_idx=frame_idx,
                label="Blacklisted Person Detected",
                severity="CRITICAL",
                confidence=1.0,
                face_ids=blacklisted_faces,
                override="BLACKLIST",
                screenshot=str(screenshot_path) if screenshot_path else None,
                cause={
                    "trigger": "FACE_RECOGNITION",
                    "rule_name": "BLACKLIST_MATCH",
                    "metrics": {"faces": blacklisted_faces}
                }
            )
        elif "Fight" in current_label:
            severity = "CRITICAL" if current_score >= HIGH_VIOLENCE else "HIGH"
            screenshot_path = None
            if event_mgr.is_new_event("video_fight", current_label):
                screenshot_path = OFFLINE_SCREENSHOT_DIR / f"transformer_fight_{frame_idx}.jpg"
                cv2.imwrite(str(screenshot_path), frames[frame_idx])

            event_mgr.update(
                frame_idx=frame_idx,
                label=current_label,
                severity=severity,
                confidence=current_score,
                face_ids=[],
                screenshot=str(screenshot_path) if screenshot_path else None
            )
        else:
            event_mgr.update(frame_idx=frame_idx, label="Normal", severity="LOW")

    event_mgr.finalize()
    final_events = event_mgr.export()

    update_offline_progress(95, "Generating Final Report...")

    try:
        from reports.event_adapter import adapt_events_for_pdf
        from reports.pdf_report import generate_pdf_report
        from llm.summary_generator import generate_llm_summary
        from datetime import datetime
        buf = adapt_events_for_pdf(final_events, frame_store)
        txt = generate_llm_summary(events=buf, mode="OFFLINE_WITH_FACES")
        out = OFFLINE_REPORT_DIR / f"transformer_report_{datetime.now().strftime('%H%M%S')}.pdf"
        generate_pdf_report(buf, txt, str(out))
        telegram_bot.send_report(str(out), txt)
    except Exception as e:
        print("[OFFLINE TRANSFORMER REPORT ERROR]", e)

    update_offline_progress(100, "Starting Replay...")
    stream_replay_to_browser(video_path, frame_data_map, final_events, fps, mode="TRANSFORMER")
    send_offline_summary_alert(final_events, mode="OFFLINE")

# ========================= 🔥 FIXED LIVE MODE WITH COOLDOWN =========================
def run_live(source):
    global RUN_LIVE, PAUSE_LIVE, STOP_LIVE
    RUN_LIVE = True
    PAUSE_LIVE = False
    
    # 🔥 COOLDOWN CONFIG
    BLACKLIST_ALERT_COOLDOWN = 15
    last_blacklist_alert_time = 0
    
    blacklist_active = False
    blacklist_hold_counter = 0
    last_blacklisted_faces = []

    print("[MODE] LIVE")

    try:
        validate_paths()
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        return

    # Camera Init
    camera_mgr = None
    use_rotation = (str(source) == "0" or source == 0) and (source != "1") 
    
    if use_rotation:
        print("[CAMERA] Initializing multi-camera rotation...")
        config = CameraConfig(base_time_window=10.0, max_scan_index=10)
        global CAMERA_MANAGER
        CAMERA_MANAGER = CameraManager(config)
        camera_mgr = CAMERA_MANAGER
        if not camera_mgr.initialize():
            use_rotation = False
        else:
            if not camera_mgr.start_camera(): return
            fps = camera_mgr.get_fps()
        global ACTIVE_CAMERA_MGR
        ACTIVE_CAMERA_MGR = camera_mgr

    if not use_rotation:
        cap = cv2.VideoCapture(source)
        if not cap.isOpened(): return
        fps = cap.get(cv2.CAP_PROP_FPS) or 30

    # Model Init
    try:
        scrfd = SCRFD(model_path=str(SCRFD_MODEL))
        arcface = create_onnx_session(ARCFACE_MODEL)
        face_db = load_or_build_face_db(scrfd, arcface, str(FACE_GALLERY))
        detector = Detector()
        pose_buffer = PoseBuffer(max_len=30)
        rule_engine = RuleEngine(history=30)
    except Exception as e:
        print(f"[ERROR] Model init failed: {e}")
        return

    event_mgr = EventManager(fps=fps, source="LIVE")
    frame_idx = 0
    prev_time = time.time()
    cache = {}
    next_face_id = 0
    frame_store = {}
    last_face_results = {}
    orig_width = None
    orig_height = None

    # ===================== MAIN LOOP =====================
    try:
        while RUN_LIVE and not STOP_LIVE:
            if STOP_LIVE: break
            if PAUSE_LIVE:
                time.sleep(0.05)
                continue

            if use_rotation:
                ret, frame = camera_mgr.read_frame()
                if not ret:
                    if camera_mgr.rotate_camera():
                        fps = camera_mgr.get_fps()
                        event_mgr.fps = fps
                        continue
                    else: break
            else:
                ret, frame = cap.read()
                if not ret: break

            if orig_width is None:
                orig_height, orig_width = frame.shape[:2]

            if frame_idx % int(fps) == 0:
                frame_store[frame_idx] = frame.copy()

            # 1. POSE
            persons, objects = detector.infer(frame)
            for p in persons: pose_buffer.update(p["track_id"], p["keypoints"])

            # 2. RULES
            rule_results = rule_engine.update(persons, objects)

            # 3. FACE
            frame, cache, next_face_id, face_results = process_face_recognition(
                    frame, scrfd, arcface, face_db, cache, next_face_id
                )
            
            track_to_face = match_faces_to_poses(persons, face_results)

            # 4. DISPLAY SCALING
            scale = min(DISPLAY_WIDTH / orig_width, DISPLAY_HEIGHT / orig_height)
            new_width = int(orig_width * scale)
            new_height = int(orig_height * scale)
            display_frame = cv2.resize(frame, (new_width, new_height))
            canvas = np.zeros((DISPLAY_HEIGHT, DISPLAY_WIDTH, 3), dtype=np.uint8)
            x_offset = (DISPLAY_WIDTH - new_width) // 2
            y_offset = (DISPLAY_HEIGHT - new_height) // 2
            canvas[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = display_frame
            display_frame = canvas
            scale_x = scale; scale_y = scale

            scaled_persons = []
            for p in persons:
                kp = p["keypoints"].copy()
                kp[:, 0] = kp[:, 0] * scale_x + x_offset
                kp[:, 1] = kp[:, 1] * scale_y + y_offset
                scaled_persons.append({"keypoints": kp, "confidence": np.ones(len(kp)), "track_id": p["track_id"]})

            frame_out = draw_pose(display_frame, scaled_persons)
            
            out_h, out_w = frame_out.shape[:2]
            if out_w != DISPLAY_WIDTH or out_h != DISPLAY_HEIGHT:
                frame_out = cv2.resize(frame_out, (DISPLAY_WIDTH, DISPLAY_HEIGHT))

            # 7. EVENT TRACKING & ALERTS
            current_severity = EventSeverity.NORMAL
            
            blacklisted_faces = [
                data["name"] for data in face_results.values()
                if data.get("status") == "blacklist"
            ]

            if blacklisted_faces:
                blacklist_hold_counter = BLACKLIST_HOLD_FRAMES
                last_blacklisted_faces = blacklisted_faces[:]
            else:
                blacklist_hold_counter = max(0, blacklist_hold_counter - 1)

            blacklist_active = blacklist_hold_counter > 0   

            if blacklist_active:
                names_to_report = blacklisted_faces if blacklisted_faces else last_blacklisted_faces
                
                # Update Dashboard
                event_mgr.update(
                    frame_idx=frame_idx,
                    label="Blacklisted Person Detected",
                    severity="CRITICAL",
                    confidence=1.0,
                    face_ids=names_to_report,
                    override="BLACKLIST",
                    cause={
                        "trigger": "FACE_RECOGNITION",
                        "rule_name": "BLACKLIST_MATCH",
                        "metrics": {"faces": names_to_report}
                    }
                )

                # 🔥 TRIGGER ALERT (Direct Check)
                if (time.time() - last_blacklist_alert_time) > BLACKLIST_ALERT_COOLDOWN:
                    print(f"[ALERT] 🚨 Sending Blacklist Alert for: {names_to_report}")
                    try:
                        send_critical_alert(
                            event={
                                "type": "Blacklisted Person Detected",
                                "severity": "CRITICAL",
                                "confidence": 1.0,
                                "cause": {"faces": names_to_report}
                            },
                            report_path=None,
                            mode="LIVE"
                        )
                        last_blacklist_alert_time = time.time()
                    except Exception as e:
                        print(f"[ALERT ERROR] Could not send email: {e}")

            else:
                if event_mgr.current_event and event_mgr.current_event["type"] == "Blacklisted Person Detected":
                    event_mgr.end_current_event(frame_idx)

            # --- GENERAL RULE LOGIC ---
            for p in persons:
                tid = p["track_id"]
                if tid in rule_results:
                    result = rule_results[tid].copy()
                    result_severity = map_severity_to_enum(result['severity'])
                    if SEVERITY_RANK[result_severity] > SEVERITY_RANK[current_severity]:
                        current_severity = result_severity

                    face_info = track_to_face.get(tid)
                    
                    if face_info and face_info.get('status') == 'whitelist':
                        if 'cause' not in result: result['cause'] = {}
                        if 'metrics' not in result['cause']: result['cause']['metrics'] = {}
                        result['cause']['metrics']['whitelisted_person'] = face_info['name']
                        if result['severity'] == 'MEDIUM':
                            result['action'] += " (Whitelisted)"
                            result['severity'] = 'LOW'
                            result['color'] = SEVERITY_COLORS['LOW']
                    
                    elif face_info and face_info.get('status') == 'blacklist':
                        result['severity'] = 'CRITICAL'
                        result['color'] = SEVERITY_COLORS['CRITICAL']
                
                    screenshot_path = None
                    if (event_mgr.current_event is None or event_mgr.current_event["type"] != result["action"]):
                        screenshot_path = SCREENSHOT_DIR / f"event_{frame_idx}_track_{tid}.jpg"
                        cv2.imwrite(str(screenshot_path), frame_out)

                    event_mgr.update(
                        frame_idx=frame_idx,
                        label=result["action"],
                        severity=result["severity"],
                        face_ids=[tid],
                        cause=result.get("cause"),
                        screenshot=str(screenshot_path) if screenshot_path else None
                    )

                    if result["severity"] in ["HIGH", "CRITICAL"]:
                        if event_mgr.is_new_event(tid, result["action"]):
                            send_critical_alert(
                                event={
                                    "type": result["action"],
                                    "severity": result["severity"],
                                    "confidence": 0.95,
                                    "cause": result.get("cause", {})
                                },
                                report_path=str(screenshot_path) if screenshot_path else None,
                                mode="LIVE"
                            )

                    for sp in scaled_persons:
                        if sp["track_id"] == tid and len(sp["keypoints"]) > 0:
                            x, y = map(int, sp["keypoints"][0])
                            cv2.putText(frame_out, result["action"], (x, y - 25),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, result["color"], 2)

            if not blacklist_active and not rule_results:
                event_mgr.update(frame_idx, "Normal", "LOW")

            now = time.time()
            fps_val = 1 / (now - prev_time + 1e-8)
            prev_time = now
            global CURRENT_FPS
            CURRENT_FPS = int(fps_val * 0.2 + CURRENT_FPS * 0.8)

            if use_rotation: camera_mgr.update_event_severity(current_severity)
            if API_MODE: push_frame(frame_out)
            if STOP_LIVE: break

            if use_rotation and camera_mgr.should_rotate():
                if camera_mgr.rotate_camera():
                    fps = camera_mgr.get_fps()
                    event_mgr.fps = fps
                    orig_width = None
                else: break
            
            frame_idx += 1

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted")
    except Exception as e:
        print(f"[ERROR] Runtime: {e}")
        import traceback; traceback.print_exc()
    finally:
        PAUSE_LIVE = False
        if use_rotation: camera_mgr.stop_camera()
        else: cap.release()
        global STREAM_FRAME
        with STREAM_LOCK: STREAM_FRAME = None

    event_mgr.finalize()
    events = event_mgr.export()
    
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
        telegram_bot.send_report(str(output_path), summary_text)
    except Exception as e:
        print(f"[ERROR] Report generation failed: {e}")

# ========================= ENTRY POINT =========================
if __name__ == "__main__":
    if API_ONLY:
        print("[INFO] main.py running in API mode")
    else:
        pass