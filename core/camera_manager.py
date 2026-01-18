"""
Camera Manager Module for Multi-Camera Rotation
Save this as: core/camera_manager.py
"""

import cv2
import time
import threading
from enum import Enum
from typing import List, Optional, Tuple
from dataclasses import dataclass

cv2.setLogLevel(0)  # Suppress OpenCV internal errors


# ========================= SEVERITY =========================

class EventSeverity(Enum):
    NORMAL = "NORMAL"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


# ========================= CONFIG =========================

@dataclass
class CameraConfig:
    base_time_window: float = 30.0      # Seconds per camera
    max_scan_index: int = 10            # Max camera index to probe
    frame_test_count: int = 5           # Frames to validate camera
    camera_warmup_time: float = 1.0     # Seconds after camera switch


# ========================= DISCOVERY =========================

class CameraDiscovery:
    """Discovers physical cameras"""

    def __init__(self, max_index: int, test_frames: int):
        self.max_index = max_index
        self.test_frames = test_frames

    def scan_cameras(self) -> List[int]:
        valid = []
        print(f"[CAMERA DISCOVERY] Scanning indices 0-{self.max_index}...")

        for idx in range(self.max_index):
            if self._test_camera(idx):
                valid.append(idx)
                print(f"  ✓ Camera {idx} detected")

        print(f"[CAMERA DISCOVERY] Found {len(valid)} camera(s): {valid}")
        return valid

    def _test_camera(self, index: int) -> bool:
        cap = None
        try:
            cap = cv2.VideoCapture(index)
            if not cap.isOpened():
                return False

            ok = 0
            for _ in range(self.test_frames):
                ret, frame = cap.read()
                if ret and frame is not None and frame.size > 0:
                    ok += 1

            return ok >= int(self.test_frames * 0.8)

        except Exception:
            return False
        finally:
            if cap:
                cap.release()


# ========================= SCHEDULER =========================

class CameraScheduler:
    """Manages camera rotation with base-window reset on CRITICAL events"""

    def __init__(self, config: CameraConfig):
        self.config = config
        self.window_start_time = time.time()
        self.current_severity = EventSeverity.NORMAL
        self._lock = threading.Lock()

    def update_event(self, severity: EventSeverity):
        """
        Reset base window whenever a CRITICAL event is detected.
        """
        with self._lock:
            self.current_severity = severity

            if severity == EventSeverity.CRITICAL:
                # 🔥 HARD RESET to base window
                self.window_start_time = time.time()
                print("[SCHEDULER] CRITICAL detected → resetting base window")

    def should_rotate(self) -> bool:
        """
        Rotate only if base window fully expires with no CRITICAL reset.
        """
        with self._lock:
            elapsed = time.time() - self.window_start_time
            return elapsed >= self.config.base_time_window

    def reset_window(self):
        """Called when camera is switched"""
        with self._lock:
            self.window_start_time = time.time()
            self.current_severity = EventSeverity.NORMAL

    def get_remaining_time(self) -> float:
        with self._lock:
            elapsed = time.time() - self.window_start_time
            return max(0, self.config.base_time_window - elapsed)


# ========================= MANAGER =========================

class CameraManager:
    """Multi-camera orchestrator"""

    def __init__(self, config: CameraConfig):
        self.config = config
        self.discovery = CameraDiscovery(
            max_index=config.max_scan_index,
            test_frames=config.frame_test_count
        )
        self.scheduler = CameraScheduler(config)

        self.camera_indices: List[int] = []
        self.failed_cameras = set()
        self.current_index = 0
        self.active_camera: Optional[cv2.VideoCapture] = None

    def initialize(self) -> bool:
        self.camera_indices = self.discovery.scan_cameras()
        if not self.camera_indices:
            print("[CAMERA MANAGER] ERROR: No cameras detected")
            return False

        print(f"[CAMERA MANAGER] Initialized with {len(self.camera_indices)} camera(s)")
        return True

    def start_camera(self) -> bool:
        if not self.camera_indices:
            return False

        cam_id = self.camera_indices[self.current_index]
        if cam_id in self.failed_cameras:
            return False

        try:
            self.active_camera = cv2.VideoCapture(cam_id)
            if not self.active_camera.isOpened():
                raise RuntimeError("Open failed")

            time.sleep(self.config.camera_warmup_time)

            ret, _ = self.active_camera.read()
            if not ret:
                raise RuntimeError("Frame read failed")

            print(
                f"[CAMERA MANAGER] Started camera {cam_id} "
                f"({self.current_index + 1}/{len(self.camera_indices)})"
            )

            self.scheduler.reset_window()
            return True

        except Exception:
            print(f"[CAMERA MANAGER] Camera {cam_id} marked as failed")
            self.failed_cameras.add(cam_id)
            if self.active_camera:
                self.active_camera.release()
            return False

    def stop_camera(self):
        if self.active_camera:
            self.active_camera.release()
            self.active_camera = None

    def rotate_camera(self) -> bool:
        print("[CAMERA MANAGER] Rotating camera...")
        self.stop_camera()

        for _ in range(len(self.camera_indices)):
            self.current_index = (self.current_index + 1) % len(self.camera_indices)
            if self.camera_indices[self.current_index] in self.failed_cameras:
                continue
            if self.start_camera():
                return True

        print("[CAMERA MANAGER] ERROR: No usable cameras remaining")
        return False

    def read_frame(self) -> Tuple[bool, Optional[any]]:
        if not self.active_camera:
            return False, None
        return self.active_camera.read()

    def update_event_severity(self, severity: EventSeverity):
        self.scheduler.update_event(severity)

    def should_rotate(self) -> bool:
        return (
            len(self.camera_indices) > 1
            and self.scheduler.should_rotate()
        )

    def get_status(self) -> dict:
        return {
            "current_camera": (
                self.camera_indices[self.current_index]
                if self.camera_indices else None
            ),
            "remaining_time": int(self.scheduler.get_remaining_time()),
            "event_severity": self.scheduler.current_severity.value,
            "failed_cameras": list(self.failed_cameras),
        }

    def get_fps(self) -> float:
        if self.active_camera:
            return self.active_camera.get(cv2.CAP_PROP_FPS) or 30.0
        return 30.0


# ========================= UTILITY =========================

def map_severity_to_enum(severity_str: str) -> EventSeverity:
    return {
        "CRITICAL": EventSeverity.CRITICAL,
        "HIGH": EventSeverity.HIGH,
        "MEDIUM": EventSeverity.MEDIUM,
        "LOW": EventSeverity.LOW,
        "NORMAL": EventSeverity.NORMAL,
    }.get(severity_str, EventSeverity.NORMAL)
