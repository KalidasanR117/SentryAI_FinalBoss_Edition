# sentry/core/features.py
import numpy as np

def compute_velocity(track_hist):
    """
    velocity between last two centroids
    """
    if len(track_hist) < 2: return 0.0
    return np.linalg.norm(np.array(track_hist[-1]) - np.array(track_hist[-2]))

def angle_between(a, b, c):
    """
    angle at b between segments ab and bc
    """
    a,b,c = map(np.array, (a,b,c))
    ba,bc = a-b, c-b
    cos = np.dot(ba, bc)/ (np.linalg.norm(ba)*np.linalg.norm(bc) + 1e-8)
    return np.degrees(np.arccos(np.clip(cos,-1,1)))
