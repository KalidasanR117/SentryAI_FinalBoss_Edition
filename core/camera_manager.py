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


class EventSeverity(Enum):
    """Event severity levels for adaptive scheduling"""
    NORMAL = "NORMAL"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


@dataclass
class CameraConfig:
    """Camera rotation configuration"""
    base_time_window: float = 30.0      # Base time per camera (seconds)
    extension_multiplier: float = 2.5   # Time extension for critical events
    max_scan_index: int = 10            # Maximum camera index to scan
    frame_test_count: int = 5           # Frames to test for validity
    camera_warmup_time: float = 1.0     # Warmup time after camera switch


class CameraDiscovery:
    """Discovers and validates physical cameras on the system"""
    
    def __init__(self, max_index: int = 10, test_frames: int = 5):
        self.max_index = max_index
        self.test_frames = test_frames
    
    def scan_cameras(self) -> List[int]:
        """Scan for available physical cameras"""
        valid_cameras = []
        
        print(f"[CAMERA DISCOVERY] Scanning indices 0-{self.max_index}...")
        
        for idx in range(self.max_index):
            if self._test_camera(idx):
                valid_cameras.append(idx)
                print(f"  ✓ Camera {idx} detected")
        
        print(f"[CAMERA DISCOVERY] Found {len(valid_cameras)} camera(s): {valid_cameras}")
        return valid_cameras
    
    def _test_camera(self, index: int) -> bool:
        """Test if a camera index is valid"""
        cap = None
        try:
            cap = cv2.VideoCapture(index)
            
            if not cap.isOpened():
                return False
            
            successful_reads = 0
            for _ in range(self.test_frames):
                ret, frame = cap.read()
                if ret and frame is not None and frame.size > 0:
                    successful_reads += 1
            
            return successful_reads >= (self.test_frames * 0.8)
            
        except Exception:
            return False
        finally:
            if cap is not None:
                cap.release()


class CameraScheduler:
    """Manages camera rotation with adaptive time windows"""
    
    def __init__(self, config: CameraConfig):
        self.config = config
        self.current_severity = EventSeverity.NORMAL
        self.window_start_time = time.time()
        self.extension_active = False
        self._lock = threading.Lock()
    
    def update_event(self, severity: EventSeverity):
        """Update current event severity and adjust time window"""
        with self._lock:
            self.current_severity = severity
            
            # Activate extension for HIGH/CRITICAL events
            if severity in [EventSeverity.HIGH, EventSeverity.CRITICAL]:
                if not self.extension_active:
                    self.extension_active = True
                    extended_time = int(self.config.base_time_window * self.config.extension_multiplier)
                    print(f"[SCHEDULER] {severity.value} event - extending window to {extended_time}s")
    
    def should_rotate(self) -> bool:
        """Determine if it's time to rotate to next camera"""
        with self._lock:
            elapsed = time.time() - self.window_start_time
            
            if self.extension_active:
                time_limit = self.config.base_time_window * self.config.extension_multiplier
            else:
                time_limit = self.config.base_time_window
            
            if elapsed >= time_limit:
                # Reset extension if event cleared
                if self.current_severity in [EventSeverity.NORMAL, EventSeverity.LOW, EventSeverity.MEDIUM]:
                    self.extension_active = False
                
                return True
            
            return False
    
    def reset_window(self):
        """Reset the time window (called after camera switch)"""
        with self._lock:
            self.window_start_time = time.time()
            self.extension_active = False
    
    def get_remaining_time(self) -> float:
        """Get remaining time in current window"""
        with self._lock:
            elapsed = time.time() - self.window_start_time
            if self.extension_active:
                time_limit = self.config.base_time_window * self.config.extension_multiplier
            else:
                time_limit = self.config.base_time_window
            return max(0, time_limit - elapsed)


class CameraManager:
    """Orchestrates multi-camera rotation and lifecycle management"""
    
    def __init__(self, config: CameraConfig):
        self.config = config
        self.discovery = CameraDiscovery(
            max_index=config.max_scan_index,
            test_frames=config.frame_test_count
        )
        self.scheduler = CameraScheduler(config)
        
        self.camera_indices: List[int] = []
        self.current_index: int = 0
        self.active_camera: Optional[cv2.VideoCapture] = None
    
    def initialize(self) -> bool:
        """Initialize camera system by discovering available cameras"""
        self.camera_indices = self.discovery.scan_cameras()
        
        if not self.camera_indices:
            print("[CAMERA MANAGER] ERROR: No cameras detected!")
            return False
        
        print(f"[CAMERA MANAGER] Initialized with {len(self.camera_indices)} camera(s)")
        return True
    
    def start_camera(self) -> bool:
        """Open the current camera in rotation"""
        if not self.camera_indices:
            return False
        
        camera_id = self.camera_indices[self.current_index]
        
        try:
            self.active_camera = cv2.VideoCapture(camera_id)
            
            if not self.active_camera.isOpened():
                print(f"[CAMERA MANAGER] Failed to open camera {camera_id}")
                return False
            
            # Warmup period
            time.sleep(self.config.camera_warmup_time)
            
            # Verify frame reading
            ret, frame = self.active_camera.read()
            if not ret:
                print(f"[CAMERA MANAGER] Camera {camera_id} cannot read frames")
                self.active_camera.release()
                return False
            
            print(f"[CAMERA MANAGER] Started camera {camera_id} ({self.current_index + 1}/{len(self.camera_indices)})")
            self.scheduler.reset_window()
            return True
            
        except Exception as e:
            print(f"[CAMERA MANAGER] Error starting camera {camera_id}: {e}")
            if self.active_camera:
                self.active_camera.release()
            return False
    
    def stop_camera(self):
        """Safely release the active camera"""
        if self.active_camera:
            self.active_camera.release()
            self.active_camera = None
    
    def rotate_camera(self) -> bool:
        """Rotate to the next camera in the list"""
        print(f"[CAMERA MANAGER] Rotating camera (time window expired)...")
        
        self.stop_camera()
        
        # Move to next camera (round-robin)
        self.current_index = (self.current_index + 1) % len(self.camera_indices)
        
        return self.start_camera()
    
    def read_frame(self) -> Tuple[bool, Optional[any]]:
        """Read a frame from the active camera"""
        if not self.active_camera:
            return False, None
        
        return self.active_camera.read()
    
    def update_event_severity(self, severity: EventSeverity):
        """Update event severity for adaptive scheduling"""
        self.scheduler.update_event(severity)
    
    def should_rotate(self) -> bool:
        """Check if camera rotation should occur"""
        if len(self.camera_indices) <= 1:
            return False
        
        return self.scheduler.should_rotate()
    
    def get_status(self) -> dict:
        """Get current status information"""
        return {
            "total_cameras": len(self.camera_indices),
            "current_camera": self.camera_indices[self.current_index] if self.camera_indices else None,
            "rotation_index": self.current_index,
            "remaining_time": self.scheduler.get_remaining_time(),
            "event_severity": self.scheduler.current_severity.value,
            "extension_active": self.scheduler.extension_active
        }
    
    def get_fps(self) -> float:
        """Get FPS from active camera"""
        if self.active_camera:
            return self.active_camera.get(cv2.CAP_PROP_FPS) or 30.0
        return 30.0


def map_severity_to_enum(severity_str: str) -> EventSeverity:
    """Map string severity to EventSeverity enum"""
    severity_map = {
        "CRITICAL": EventSeverity.CRITICAL,
        "HIGH": EventSeverity.HIGH,
        "MEDIUM": EventSeverity.MEDIUM,
        "LOW": EventSeverity.LOW,
        "NORMAL": EventSeverity.NORMAL
    }
    return severity_map.get(severity_str, EventSeverity.NORMAL)