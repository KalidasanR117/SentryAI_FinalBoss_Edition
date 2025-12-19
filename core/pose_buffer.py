# sentry/core/pose_buffer.py
from collections import defaultdict, deque

class PoseBuffer:
    def __init__(self, max_len=30):
        self.buffers = defaultdict(lambda: deque(maxlen=max_len))

    def update(self, track_id, keypoints):
        self.buffers[track_id].append(keypoints)

    def ready(self, track_id):
        return len(self.buffers[track_id]) == self.buffers[track_id].maxlen

    def get(self, track_id):
        if self.ready(track_id):
            return list(self.buffers[track_id])
        return None
