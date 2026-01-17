# sentry/core/rule_engine.py
import numpy as np
from collections import defaultdict, deque

# ===================== CONFIG =====================
WINDOW = 10  # frames used for motion analysis

SEVERITY_COLORS = {
    "CRITICAL": (0, 0, 255),
    "HIGH":     (0, 165, 255),
    "MEDIUM":   (0, 255, 255),
    "LOW":      (0, 255, 0)
}

JOINT_NAMES = {
    5: "left_shoulder",
    6: "right_shoulder",
    7: "left_elbow",
    8: "right_elbow",
    9: "left_wrist",
    10: "right_wrist",
    11: "left_hip",
    12: "right_hip"
}
# =================================================


class RuleEngine:
    def __init__(self, history=30):
        self.pose_hist = defaultdict(lambda: deque(maxlen=history))

    # ------------------ Geometry helpers ------------------
    def safe_dist(self, a, b):
        return float(np.linalg.norm(a - b))

    def get_torso(self, kps, conf):
        pts = []
        for idx in [5, 6, 11, 12]:
            if idx < len(kps) and conf[idx] > 0.3:
                pts.append(kps[idx])
        return np.mean(pts, axis=0) if pts else np.mean(kps, axis=0)

    # ------------------ Core update ------------------
    def update(self, persons, objects=None):
        """
        objects intentionally ignored
        Theft logic REMOVED
        """
        results = {}

        # ---- Update pose history ----
        for p in persons:
            tid = p["track_id"]
            self.pose_hist[tid].append((p["keypoints"], p["confidence"]))

        # ---- Evaluate motion ----
        for p in persons:
            tid = p["track_id"]
            seq = self.pose_hist[tid]

            if len(seq) < WINDOW:
                continue

            hand_speeds = []
            torso_speeds = []

            for i in range(1, WINDOW):
                prev_k, prev_c = seq[i - 1]
                curr_k, curr_c = seq[i]

                # torso velocity
                torso_prev = self.get_torso(prev_k, prev_c)
                torso_curr = self.get_torso(curr_k, curr_c)
                torso_speeds.append(self.safe_dist(torso_prev, torso_curr))

                # hand velocity
                hs = []
                for j in [9, 10]:  # wrists
                    if j < len(prev_k) and prev_c[j] > 0.3 and curr_c[j] > 0.3:
                        hs.append(self.safe_dist(prev_k[j], curr_k[j]))
                if hs:
                    hand_speeds.append(np.mean(hs))

            avg_hand = float(np.mean(hand_speeds)) if hand_speeds else 0.0
            avg_torso = float(np.mean(torso_speeds)) if torso_speeds else 0.0

            # ================= CLASSIFICATION =================
            if avg_hand > 20 and avg_torso > 7:
                action = "Physical Assault"
                severity = "CRITICAL"
                rule_name = "RAPID_ARM_SWING_HIGH_BODY_MOTION"

            elif avg_hand > 12 and avg_torso > 4:
                action = "Aggressive Interaction"
                severity = "HIGH"
                rule_name = "ELEVATED_ARM_AND_TORSO_MOTION"

            else:
                action = "Normal Motion"
                severity = "LOW"
                rule_name = None

            # ================= RESULT =================
            payload = {
                "action": action,
                "severity": severity,
                "color": SEVERITY_COLORS[severity]
            }

            # ---- Add explainability only if non-normal ----
            if action != "Normal Motion":
                payload["cause"] = {
                    "trigger": "POSE_RULE",
                    "rule_name": rule_name,
                    "description": (
                        "Rapid arm movement combined with torso displacement"
                        if severity == "CRITICAL"
                        else "Elevated arm and torso movement"
                    ),
                    "joints_involved": [
                        JOINT_NAMES[9],
                        JOINT_NAMES[10],
                        JOINT_NAMES[7],
                        JOINT_NAMES[8]
                    ],
                    "metrics": {
                        "avg_hand_speed": round(avg_hand, 2),
                        "avg_torso_speed": round(avg_torso, 2),
                        "window_frames": WINDOW
                    }
                }

            results[tid] = payload

        return results
