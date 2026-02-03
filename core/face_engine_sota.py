import cv2
import numpy as np
import time
import os
import imageio  # <--- USES FFMPEG (Robust Video Writing)
from pathlib import Path
from datetime import datetime
from insightface.app import FaceAnalysis
from facial_analysis.identity_store import load_identity_map

# Imports for Reporting
from reports.pdf_report import generate_pdf_report
from llm.summary_generator import generate_llm_summary

# ================= CONFIG =================
MODEL_NAME = "buffalo_l" 
MATCH_THRESHOLD = 0.60  # 🔥 STRICTER THRESHOLD

class SOTAFaceAnalyzer:
    def __init__(self, gallery_path="facial_analysis/face_gallery"):
        print(f"[SOTA] Initializing {MODEL_NAME} on GPU (RTX 4050)...")
        
        # 🔥 FORCE GPU EXECUTION
        self.app = FaceAnalysis(
            name=MODEL_NAME, 
            providers=['CUDAExecutionProvider'] 
        )
        self.app.prepare(ctx_id=0, det_size=(640, 640))
        
        self.gallery_path = Path(gallery_path)
        # We don't build DB here anymore to allow dynamic refreshing
        self.face_db = {}
        self.identity_map = {}
        
        # Ensure screenshot directory exists
        self.screenshot_dir = Path("reports/screenshots")
        self.screenshot_dir.mkdir(parents=True, exist_ok=True)
        
        print("[SOTA] Engine Ready.")

    def _build_face_db(self):
        """Pre-computes NORMALIZED embeddings for the whitelist/blacklist."""
        print("[SOTA] Building Vector Database (Fresh Scan)...")
        db = {}
        if not self.gallery_path.exists(): return db

        for person_dir in self.gallery_path.iterdir():
            if not person_dir.is_dir(): continue
            name = person_dir.name
            embeddings = []
            
            for img_path in person_dir.glob("*.*"):
                img = cv2.imread(str(img_path))
                if img is None: continue
                faces = self.app.get(img)
                if faces:
                    # Take the largest face
                    face = max(faces, key=lambda x: (x.bbox[2]-x.bbox[0]) * (x.bbox[3]-x.bbox[1]))
                    
                    # 🔥 NORMALIZE DATABASE VECTOR (Critical for Cosine Similarity)
                    norm_emb = face.embedding / np.linalg.norm(face.embedding)
                    embeddings.append(norm_emb)
            
            if embeddings:
                db[name] = embeddings
                print(f"   -> Loaded {name} ({len(embeddings)} refs)")
        return db

    def _match_face(self, target_emb):
        # 🔥 NORMALIZE INPUT VECTOR
        target_emb = target_emb / np.linalg.norm(target_emb)
        
        best_name = "Unknown"
        best_score = -1.0

        for name, ref_embs in self.face_db.items():
            for ref_emb in ref_embs:
                # Dot Product of normalized vectors == Cosine Similarity
                score = np.dot(target_emb, ref_emb)
                if score > best_score:
                    best_score = score
                    best_name = name
        
        if best_score > MATCH_THRESHOLD:
            return best_name, best_score
        return "Unknown", best_score

    def process_video(self, video_path, output_path, report_path, update_callback=None):
        # 🔥 STEP 1: REFRESH DATABASE
        self.face_db = self._build_face_db()
        self.identity_map = load_identity_map()

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # 🔥 USE IMAGEIO WRITER (Web-Compatible MP4)
        writer = imageio.get_writer(
            output_path, 
            fps=fps, 
            codec='libx264', 
            quality=8, 
            pixelformat='yuv420p',
            macro_block_size=None
        )

        frame_idx = 0
        start_time = time.time()
        
        detections_log = [] 
        seen_identities = set()

        print(f"[SOTA] Processing {total_frames} frames...")

        while True:
            ret, frame = cap.read()
            if not ret: break

            faces = self.app.get(frame)
            
            # Helper to track if we need to save a screenshot for this frame
            frame_events = [] 
            
            for face in faces:
                name, score = self._match_face(face.embedding)
                status = self.identity_map.get(name.lower(), "unknown")
                
                # 🔥 COLOR LOGIC
                if name == "Unknown":
                    color = (0, 255, 255) # Yellow
                    label_text = f"Unknown ({score:.2f})"
                else:
                    if status == "blacklist": 
                        color = (0, 0, 255) # Red
                    elif status == "whitelist": 
                        color = (0, 255, 0) # Green
                    else: 
                        color = (255, 0, 0) # Blue
                    label_text = f"{name} ({score:.2f})"

                # Draw Box
                bbox = face.bbox.astype(int)
                x1, y1, x2, y2 = bbox
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                # Draw Label
                cv2.rectangle(frame, (x1, y1-25), (x2, y1), color, -1)
                cv2.putText(frame, label_text, (x1 + 5, y1 - 8), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)
                
                # Draw Landmarks
                if face.kps is not None:
                    for kp in face.kps:
                        cv2.circle(frame, (int(kp[0]), int(kp[1])), 2, (255, 255, 255), -1)

                # 🔥 PREPARE EVENT (If Known Person & Sample Rate)
                if frame_idx % 10 == 0 and name != "Unknown":
                    timestamp = time.strftime('%H:%M:%S', time.gmtime(frame_idx / fps))
                    
                    frame_events.append({
                        "name": name,
                        "status": status,
                        "score": score,
                        "time": timestamp
                    })
                    seen_identities.add(name)

            # 🔥 SAVE SCREENSHOT (If events happened in this frame)
            if frame_events:
                # Create unique filename
                shot_name = f"sota_{int(time.time())}_{frame_idx}.jpg"
                shot_path = self.screenshot_dir / shot_name
                
                # Save the frame (with boxes drawn!)
                cv2.imwrite(str(shot_path), frame)
                
                # Add events to log WITH screenshot path
                for e in frame_events:
                    detections_log.append({
                        "type": f"Face Detected: {e['name']}",
                        "severity": "CRITICAL" if e['status'] == "blacklist" else "LOW",
                        "time": e['time'],
                        "details": f"Identity: {e['name']} ({e['status'].upper()}) - Conf: {e['score']:.2f}",
                        "screenshot": str(shot_path) # <--- SAVED PATH
                    })

            # 🔥 Convert BGR (OpenCV) to RGB (ImageIO)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            writer.append_data(frame_rgb)
            
            frame_idx += 1
            if frame_idx % 10 == 0 and update_callback:
                progress = int((frame_idx / total_frames) * 100)
                update_callback(progress)
                print(f"   Progress: {progress}%", end="\r")

        cap.release()
        writer.close() 
        
        # --- REPORT GENERATION ---
        print("\n[SOTA] Generating Report...")
        try:
            summary_text = f"Deep Face Analysis Report\n"
            summary_text += f"Processed File: {Path(video_path).name}\n"
            summary_text += f"Total Faces Identified: {len(seen_identities)}\n"
            summary_text += f"Identities Found: {', '.join(seen_identities)}\n\n"
            
            summary_text += "Timeline of Detections:\n"
            for d in detections_log[:30]:
                summary_text += f"[{d['time']}] {d['details']}\n"

            # Pass events to PDF generator
            formatted_events = []
            for d in detections_log:
                formatted_events.append({
                    "time": d["time"],
                    "type": d["type"],
                    "severity": d["severity"],
                    "screenshot": d["screenshot"] # <--- NOW HAS VALUE
                })
            
            generate_pdf_report(formatted_events, summary_text, report_path)
            print(f"[SOTA] Report saved: {report_path}")
            
        except Exception as e:
            print(f"[SOTA REPORT ERROR] {e}")

        print(f"[SOTA] Completed in {time.time() - start_time:.1f}s")
        return output_path

# Global Instance
engine = None

def run_analysis(video_path, result_path, report_path, progress_callback):
    global engine
    if engine is None:
        engine = SOTAFaceAnalyzer()
    
    return engine.process_video(video_path, result_path, report_path, progress_callback)