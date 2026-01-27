from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import threading
import shutil
import os
import time
from pathlib import Path
from datetime import datetime, date

# Core imports
import main
from events.event_manager import EventManager
from aiortc import RTCPeerConnection, RTCSessionDescription
from webrtc.video_track import SentryVideoTrack

app = FastAPI(title="Sentry API")

# Create Uploads Directory
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===============================
# Runtime state
# ===============================
live_thread = None
PEER_CONNECTIONS = set()
EVENT_BUFFER = []
TRACK_IDS = set()

# ===============================
# 🔥 NEW: FILE UPLOAD & ANALYSIS ENDPOINT
# ===============================
@app.post("/api/analyze/upload")
async def upload_and_analyze(
    file: UploadFile = File(...),
    mode: str = Form(...)  # "pose" or "transformer"
):
    try:
        # 1. Save File
        file_path = UPLOAD_DIR / file.filename
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        print(f"[API] Video saved to: {file_path}")

        # 2. Stop LIVE mode ONLY
        main.RUN_LIVE = False
        main.STOP_LIVE = True
        main.PAUSE_LIVE = False
        main.API_MODE = True

        time.sleep(0.4)  # allow LIVE loop to exit

        # 🔥 DO NOT RESET STOP_LIVE HERE
        # main.STOP_LIVE = False   ❌ REMOVE THIS

        with main.STREAM_LOCK:
            main.STREAM_FRAME = None

        # 3. Select offline function
        if mode.lower() == "pose":
            main.CURRENT_MODE = "OFFLINE_POSE"
            target_func = main.run_pose_offline
        else:
            main.CURRENT_MODE = "OFFLINE_TRANSFORMER"
            target_func = main.run_offline

        # 4. Start offline analysis
        threading.Thread(
            target=target_func,
            args=(str(file_path),),
            daemon=True
        ).start()

        # 🔥 Replay needs RUN_LIVE = True for WebRTC
        main.RUN_LIVE = True
        main.STOP_LIVE = False   # reset ONLY after replay starts

        return {"status": "started", "mode": mode, "file": file.filename}

    except Exception as e:
        print(f"[ERROR] Upload failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ===============================
# WebRTC & Standard API (Existing)
# ===============================
@app.post("/api/webrtc/offer")
async def webrtc_offer(offer: dict):
    pc = RTCPeerConnection()
    PEER_CONNECTIONS.add(pc)

    @pc.on("iceconnectionstatechange")
    async def on_ice_connection_state_change():
        if pc.iceConnectionState == "failed":
            await pc.close()
            PEER_CONNECTIONS.discard(pc)

    # Attach Video Track
    pc._video_track = SentryVideoTrack(fps=30)
    pc.addTrack(pc._video_track)

    await pc.setRemoteDescription(
        RTCSessionDescription(sdp=offer["sdp"], type=offer["type"])
    )

    answer = await pc.createAnswer()
    
    # Force VP8
    if "VP8" in answer.sdp:
        print("✅ VP8 Codec active")
        
    await pc.setLocalDescription(answer)

    return {
        "sdp": pc.localDescription.sdp,
        "type": pc.localDescription.type,
    }

@app.get("/api/events")
def get_events():
    return {"events": list(reversed(EVENT_BUFFER))}

@app.get("/api/dashboard/summary")
def dashboard_summary():
    today = date.today().isoformat()
    critical = sum(1 for e in EVENT_BUFFER if e["severity"] in ("CRITICAL", "HIGH"))
    today_events = sum(1 for e in EVENT_BUFFER if e["time"].startswith(today))
    return {
        "active_cameras": 1,
        "total_cameras": 1,
        "active_threats": critical,
        "people_tracked": len(TRACK_IDS),
        "events_today": today_events,
    }

@app.post("/api/live/start")
def start_live():
    global live_thread
    if live_thread and live_thread.is_alive():
        return {"status": "already running"}

    main.RUN_LIVE = True
    main.PAUSE_LIVE = False
    main.STOP_LIVE = False

    live_thread = threading.Thread(
        target=main.run_live,
        args=(0,), 
        daemon=True
    )
    live_thread.start()
    return {"status": "started"}

@app.post("/api/live/stop")
def stop_live():
    # 🔥 OFFLINE REPLAY STOP
    if main.CURRENT_MODE.startswith("OFFLINE"):
        main.STOP_LIVE = True
        main.RUN_LIVE = False
        main.PAUSE_LIVE = False
        return {"status": "offline replay stopped"}

    # 🔥 LIVE CAMERA STOP (existing behavior)
    main.RUN_LIVE = False
    main.STOP_LIVE = True
    time.sleep(0.3)
    main.STOP_LIVE = False

    with main.STREAM_LOCK:
        main.STREAM_FRAME = None

    main.CURRENT_MODE = "IDLE"
    return {"status": "live stopped"}


@app.post("/api/live/pause")
def pause_live():
    main.PAUSE_LIVE = True
    return {"status": "paused"}

@app.post("/api/live/restart")
def restart_live():
    main.RUN_LIVE = False
    main.PAUSE_LIVE = False
    time.sleep(0.3)
    main.RUN_LIVE = True
    # Default to 0 or 1 depending on your preference
    threading.Thread(target=lambda: main.run_live(0), daemon=True).start()
    return {"status": "restarted"}

@app.get("/api/live/status")
def live_status():
    return {"running": main.RUN_LIVE, "paused": main.PAUSE_LIVE}

@app.post("/api/live/resume")
def resume_live():
    main.PAUSE_LIVE = False
    return {"status": "running"}

# ===============================
# Event tap
# ===============================
_original_update = EventManager.update

def patched_update(self, *args, **kwargs):
    global EVENT_BUFFER, TRACK_IDS
    _original_update(self, *args, **kwargs)
    ev = self.current_event
    if not ev: return

    for tid in ev.get("persons", []):
        TRACK_IDS.add(tid)

    EVENT_BUFFER.append({
        "type": ev.get("type", "Unknown"),
        "severity": ev.get("severity", "LOW"),
        "time": datetime.now().isoformat(timespec="seconds"),
        "camera": ev.get("source", "LIVE"),
        "final": ev.get("final", "safe")
    })
    if len(EVENT_BUFFER) > 500: EVENT_BUFFER.pop(0)

EventManager.update = patched_update

@app.on_event("shutdown")
async def shutdown():
    for pc in list(PEER_CONNECTIONS):
        await pc.close()
    PEER_CONNECTIONS.clear()

