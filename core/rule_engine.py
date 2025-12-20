import numpy as np
from collections import defaultdict, deque

SEVERITY_COLORS = {
    "CRITICAL": (0, 0, 255),
    "HIGH":     (0, 165, 255),
    "MEDIUM":   (0, 255, 255),
    "LOW":      (0, 255, 0)
}

# How many frames to consider in history
WINDOW = 10

class RuleEngine:
    def __init__(self, history=30):
        # pose history stores only keypoints sequences
        self.pose_hist = defaultdict(lambda: deque(maxlen=history))

    def safe_dist(self, a, b):
        return np.linalg.norm(a - b)

    def get_torso(self, kps, conf):
        # estimate torso center from available points
        pts = []
        for idx in [5,6,11,12]:
            if idx < len(kps) and conf[idx] > 0.3:
                pts.append(kps[idx])
        return np.mean(pts, axis=0) if pts else np.mean(kps, axis=0)

    def update(self, persons, objects):
        results = {}

        # update pose history
        for p in persons:
            tid = p["track_id"]
            self.pose_hist[tid].append((p["keypoints"], p["confidence"]))

        for p in persons:
            tid = p["track_id"]
            seq = self.pose_hist[tid]

            # require enough history
            if len(seq) < WINDOW:
                continue

            # compute motion features over last WINDOW frames
            motions = []
            torsos = []
            for i in range(1, WINDOW):
                prev_k, prev_c = seq[i-1]
                curr_k, curr_c = seq[i]
                torso_prev = self.get_torso(prev_k, prev_c)
                torso_curr = self.get_torso(curr_k, curr_c)
                torso_vel = self.safe_dist(torso_prev, torso_curr)

                # average hand speed (if available)
                hspeed = 0
                valid = 0
                if len(prev_k)>=11 and len(curr_k)>=11:
                    # left and right hands
                    for j in [9,10]:
                        if prev_c[j]>0.3 and curr_c[j]>0.3:
                            hspeed += self.safe_dist(prev_k[j], curr_k[j])
                            valid += 1
                if valid>0:
                    hand_speed = hspeed/valid
                else:
                    hand_speed = 0

                motions.append((hand_speed, torso_vel))
                torsos.append(torso_curr)

            # average over window
            avg_hand = np.mean([m[0] for m in motions])
            avg_torso = np.mean([m[1] for m in motions])

            # classify based on sequence features
            if avg_hand>15 and avg_torso>5:
                action="Physical Assault"
                sev="CRITICAL"
            elif avg_hand>10 and avg_torso>3:
                action="Aggressive Interaction"
                sev="HIGH"
            else:
                action="Normal Motion"
                sev="LOW"

            results[tid] = {
                "action": action,
                "severity": sev,
                "color": SEVERITY_COLORS[sev]
            }

        # simple theft detection
        for p in persons:
            pid = p["track_id"]
            torso = self.get_torso(p["keypoints"], p["confidence"])
            for obj in objects:
                bx1,by1,bx2,by2 = obj["bbox"]
                obj_center = np.array([(bx1+bx2)/2,(by1+by2)/2])
                d = self.safe_dist(torso, obj_center)
                if d < 60:
                    results[f"{pid}_theft"] = {
                        "action": "Possible Theft",
                        "severity": "HIGH",
                        "color": SEVERITY_COLORS["HIGH"]
                    }

        return results
