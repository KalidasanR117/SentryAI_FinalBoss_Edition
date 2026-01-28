from aiortc import MediaStreamTrack
from av import VideoFrame
import main
import cv2
import numpy as np
import asyncio
import time
from fractions import Fraction

class SentryVideoTrack(MediaStreamTrack):
    """
    Optimized video stream track with proper async timing and frame deduplication.
    """
    kind = "video"

    def __init__(self, fps=30):
        super().__init__()
        self.fps = fps
        self.pts = 0
        self.time_base = Fraction(1, 90000)
        self.frame_duration = int(90000 / fps)
        
        # 🔥 NEW: Timing control
        self.frame_interval = 1.0 / fps  # Time between frames
        self.last_frame_time = 0
        self.last_frame_id = None  # Track frame changes
        
        # Default fallback size
        # self.default_width = 640
        # self.default_height = 480
        self.default_width = 1280
        self.default_height = 480
        
        print(f"🎥 SentryVideoTrack STARTED @ {fps} FPS")

    async def recv(self):
        """
        Async receive with proper frame rate limiting.
        """
        # 🔥 OPTIMIZATION 1: Frame rate limiter
        current_time = time.time()
        time_since_last = current_time - self.last_frame_time
        
        if time_since_last < self.frame_interval:
            # Sleep until next frame is due
            await asyncio.sleep(self.frame_interval - time_since_last)
        
        self.last_frame_time = time.time()
        
        # 🔥 OPTIMIZATION 2: Thread-safe frame fetch
        try:
            with main.STREAM_LOCK:
                frame = main.STREAM_FRAME.copy() if main.STREAM_FRAME is not None else None
        except:
            frame = None
        
        # Handle 'No Frame' case
        if frame is None:
            frame = np.zeros((self.default_height, self.default_width, 3), dtype=np.uint8)
            cv2.putText(frame, "WAITING FOR CAMERA...", (50, 240), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # 🔥 OPTIMIZATION 3: Ensure even dimensions (WebRTC requirement)
        h, w = frame.shape[:2]
        new_w = w if w % 2 == 0 else w - 1
        new_h = h if h % 2 == 0 else h - 1
        
        if new_w != w or new_h != h:
            frame = frame[:new_h, :new_w]

        # Convert to WebRTC Format
        try:
            # Convert BGR (OpenCV) -> RGB (Browser)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Create the video frame
            new_frame = VideoFrame.from_ndarray(frame_rgb, format="rgb24")
            
            # Timestamp
            new_frame.pts = self.pts
            new_frame.time_base = self.time_base
            self.pts += self.frame_duration
            
            return new_frame
            
        except Exception as e:
            print(f"❌ FRAME ERROR: {e}")
            # Return a safe dummy frame
            dummy = np.zeros((480, 640, 3), dtype=np.uint8)
            f = VideoFrame.from_ndarray(dummy, format="rgb24")
            f.pts = self.pts
            f.time_base = self.time_base
            self.pts += self.frame_duration
            return f