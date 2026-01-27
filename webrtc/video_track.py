from aiortc import MediaStreamTrack
from av import VideoFrame
import main
import cv2
import numpy as np
import time
from fractions import Fraction

class SentryVideoTrack(MediaStreamTrack):
    """
    A video stream track that reads the latest frame from main.STREAM_FRAME
    and sends it to the WebRTC client.
    """
    kind = "video"

    def __init__(self, fps=30):
        super().__init__()
        self.fps = fps
        self.pts = 0
        self.time_base = Fraction(1, 90000)
        self.frame_duration = int(90000 / fps)
        
        # Default fallback size (only used if no camera frame exists yet)
        self.default_width = 640
        self.default_height = 480
        print(f"🎥 SentryVideoTrack STARTED")

    async def recv(self):
        # 1. Get frame (Non-blocking check)
        frame = main.STREAM_FRAME
        
        # 2. Handle 'No Frame' case
        if frame is None:
            # Create a black placeholder
            frame = np.zeros((self.default_height, self.default_width, 3), dtype=np.uint8)
            cv2.putText(frame, "WAITING FOR CAMERA...", (50, 240), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # 3. 🔥 DYNAMIC ASPECT RATIO FIX
        # We DO NOT force a resize here. We trust main.py to send the correct size.
        # However, WebRTC requires dimensions to be even numbers (divisible by 2).
        h, w = frame.shape[:2]
        new_w = w if w % 2 == 0 else w - 1
        new_h = h if h % 2 == 0 else h - 1
        
        if new_w != w or new_h != h:
            frame = frame[:new_h, :new_w]

        # 4. Convert to WebRTC Format & Add Timestamp
        try:
            # Convert BGR (OpenCV) -> RGB (Browser)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Create the video frame
            new_frame = VideoFrame.from_ndarray(frame, format="rgb24")
            
            # Manual Timestamp Logic
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