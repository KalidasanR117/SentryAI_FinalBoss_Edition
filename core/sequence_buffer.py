# sentry/core/sequence_buffer.py
import numpy as np
from collections import defaultdict, deque

class SequenceBuffer:
    def __init__(self, max_len=30):
        self.max_len = max_len
        self.storage = defaultdict(lambda: deque(maxlen=self.max_len))

    def add(self, track_id, keypoints):
        """
        Adds joint keypoints (17x2) for a person.
        Stores sequence of frames per ID.
        """
        self.storage[track_id].append(keypoints)

    def get(self, track_id):
        """
        Returns a list of frames or None if not enough.
        """
        seq = list(self.storage[track_id])
        return seq if len(seq) == self.max_len else None
