# sentry/core/sequence_buffer.py
from collections import defaultdict, deque

class SequenceBuffer:
    def __init__(self, max_len=64):
        self.buffer = defaultdict(lambda: deque(maxlen=max_len))
    def add(self, tid, features):
        self.buffer[tid].append(features)
    def get(self, tid):
        return list(self.buffer[tid])
