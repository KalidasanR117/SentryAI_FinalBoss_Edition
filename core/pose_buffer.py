# sentry/core/pose_buffer.py
from collections import defaultdict, deque

class PoseBuffer:
    def __init__(self, max_len=30):
        self.buffer = defaultdict(lambda: deque(maxlen=max_len))

    def update(self, track_id, keypoints):
        self.buffer[track_id].append(keypoints)

    def get(self, track_id):
        return list(self.buffer[track_id])

    def ready(self, track_id):
        return len(self.buffer[track_id]) == self.buffer[track_id].maxlen
