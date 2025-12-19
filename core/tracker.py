# sentry/core/tracker.py
import numpy as np
from collections import deque

class PoseTracker:
    def __init__(self, max_history=30, distance_threshold=50):
        self.next_id = 0
        self.tracks = {}
        self.max_history = max_history
        self.distance_threshold = distance_threshold

    def update(self, pose_list):
        """
        Updates tracks based on current pose centroids.
        pose_list: list of {"keypoints": np.ndarray((17,2)), ...}
        Returns updated tracks: {id: deque(history_of_centroids)}
        """
        new_tracks = {}
        centroids = [np.mean(p["keypoints"], axis=0) for p in pose_list]

        for idx, centroid in enumerate(centroids):
            best_id = None
            min_dist = float("inf")

            for track_id, hist in self.tracks.items():
                dist = np.linalg.norm(hist[-1] - centroid)
                if dist < min_dist:
                    min_dist = dist
                    best_id = track_id

            if best_id is None or min_dist > self.distance_threshold:
                best_id = self.next_id
                self.next_id += 1

            new_tracks[best_id] = self.tracks.get(best_id, deque(maxlen=self.max_history))
            new_tracks[best_id].append(centroid)

        self.tracks = new_tracks
        return self.tracks
