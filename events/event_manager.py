# sentry/events/event_manager.py
import math

class EventManager:
    def __init__(self, fps, source):
        self.fps = fps
        self.source = source
        self.events = []
        self.current_event = None
        self.event_id = 0
    
    # 🔥 FIX: Added 'extend=False' to the signature to prevent the TypeError
    def update(self, frame_idx, label, severity,
           confidence=None, face_ids=None,
           override=None, cause=None,
           screenshot=None, extend=False):

        if face_ids is None:
            face_ids = []

        time_sec = frame_idx / self.fps

        # -------- Normal → close event --------
        if label in ["Normal", "Normal Motion"]:
            # 🔥 Do NOT close critical face-based events (Blacklist lock)
            if self.current_event and self.current_event.get("override") == "BLACKLIST":
                return
            self._close_event(time_sec)
            return

        # -------- Start new event --------
        if self.current_event is None:
            self.current_event = {
                "event_id": self.event_id,
                "type": label,
                "severity": severity,
                "start_time": round(time_sec, 2),
                "end_time": round(time_sec, 2),
                "duration": 0.0,
                "source": self.source,
                "confidence": confidence,
                "persons": set(face_ids),
                "override": override,
                "cause": cause,
                "screenshot": screenshot   # ✅ STORE ONCE
            }
            self.event_id += 1
            return

        # -------- Extend same event --------
        if self.current_event["type"] == label:
            self.current_event["end_time"] = round(time_sec, 2)
            self.current_event["duration"] = round(
                self.current_event["end_time"] -
                self.current_event["start_time"], 2
            )
            self.current_event["persons"].update(face_ids)
            return

        # -------- Different label → close + reopen --------
        self._close_event(time_sec)
        # Recursively call update for the new event
        self.update(frame_idx, label, severity,
                    confidence, face_ids, override, cause, screenshot, extend)


    def _close_event(self, time_sec):
        if self.current_event is None:
            return

        self.current_event["end_time"] = round(time_sec, 2)
        self.current_event["duration"] = round(
            self.current_event["end_time"] -
            self.current_event["start_time"], 2
        )
        # Convert set to list for JSON serialization
        self.current_event["persons"] = list(
            self.current_event["persons"]
        )
        self.events.append(self.current_event)
        self.current_event = None

    def end_current_event(self, frame_idx):
        """Force close the current event (used by main.py to release blacklist lock)"""
        time_sec = frame_idx / self.fps
        self._close_event(time_sec)

    def finalize(self):
        if self.current_event:
            self._close_event(
                self.current_event["end_time"] +
                (1.0 / self.fps)
            )
        if hasattr(self, "_active_keys"):
            self._active_keys.clear()

    def export(self):
        return self.events

    def is_new_event(self, track_id, label):
        """
        Returns True if this (track_id, label) pair
        is starting a new event.
        """
        key = (track_id, label)

        if not hasattr(self, "_active_keys"):
            self._active_keys = set()

        if key not in self._active_keys:
            self._active_keys.add(key)
            return True

        return False