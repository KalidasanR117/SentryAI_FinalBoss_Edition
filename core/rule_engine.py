# sentry/core/rule_engine.py
import numpy as np
from collections import defaultdict, deque
from scipy.spatial.distance import euclidean

# ===================== CONFIG =====================
WINDOW = 12  # frames for motion analysis (reduced for faster response)
PROXIMITY_THRESHOLD = 180  # pixels for interaction detection (increased)
STRIKE_ZONE_THRESHOLD = 100  # pixels for strike detection (increased)
MIN_RAPID_FRAMES = 2  # frames within window (reduced from 3)

SEVERITY_COLORS = {
    "CRITICAL": (0, 0, 255),      # Red - Active violence
    "HIGH": (0, 100, 255),         # Orange - Aggressive behavior
    "MEDIUM": (0, 255, 255),       # Yellow - Suspicious activity
    "LOW": (0, 255, 0)             # Green - Normal
}

# COCO keypoint indices
KEYPOINTS = {
    "nose": 0,
    "left_eye": 1,
    "right_eye": 2,
    "left_ear": 3,
    "right_ear": 4,
    "left_shoulder": 5,
    "right_shoulder": 6,
    "left_elbow": 7,
    "right_elbow": 8,
    "left_wrist": 9,
    "right_wrist": 10,
    "left_hip": 11,
    "right_hip": 12,
    "left_knee": 13,
    "right_knee": 14,
    "left_ankle": 15,
    "right_ankle": 16
}
# =================================================


class RuleEngine:
    def __init__(self, history=30, fps=30):
        self.pose_hist = defaultdict(lambda: deque(maxlen=history))
        self.strike_hist = defaultdict(lambda: deque(maxlen=60))
        self.fps = fps
        self.frame_count = 0
        
    # ==================== GEOMETRIC HELPERS ====================
    
    def safe_point(self, kps, conf, idx, threshold=0.25):  # Lower confidence threshold
        """Get keypoint if confidence is sufficient"""
        if idx < len(kps) and conf[idx] > threshold:
            return kps[idx]
        return None
    
    def angle_between_vectors(self, v1, v2):
        """Calculate angle between two vectors in degrees"""
        v1_u = v1 / (np.linalg.norm(v1) + 1e-6)
        v2_u = v2 / (np.linalg.norm(v2) + 1e-6)
        angle = np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
        return np.degrees(angle)
    
    def get_joint_angle(self, p1, p2, p3):
        """Calculate angle at p2 formed by p1-p2-p3"""
        if p1 is None or p2 is None or p3 is None:
            return None
        v1 = p1 - p2
        v2 = p3 - p2
        return self.angle_between_vectors(v1, v2)
    
    def get_torso_center(self, kps, conf):
        """Get center of torso (shoulders + hips)"""
        pts = []
        for idx in [5, 6, 11, 12]:
            pt = self.safe_point(kps, conf, idx)
            if pt is not None:
                pts.append(pt)
        return np.mean(pts, axis=0) if len(pts) >= 2 else None
    
    def get_dynamic_scale(self, kps, conf):
        """Calculate scale factor based on torso height"""
        l_sh = self.safe_point(kps, conf, KEYPOINTS["left_shoulder"])
        r_sh = self.safe_point(kps, conf, KEYPOINTS["right_shoulder"])
        l_hip = self.safe_point(kps, conf, KEYPOINTS["left_hip"])
        r_hip = self.safe_point(kps, conf, KEYPOINTS["right_hip"])

        if all(x is not None for x in [l_sh, r_sh, l_hip, r_hip]):
            shoulder_mid = (l_sh + r_sh) / 2
            hip_mid = (l_hip + r_hip) / 2
            return euclidean(shoulder_mid, hip_mid)
        
        return 100.0  # Fallback
    
    def get_head_center(self, kps, conf):
        """Get center of head (nose, eyes, ears)"""
        pts = []
        for idx in range(5):
            pt = self.safe_point(kps, conf, idx)
            if pt is not None:
                pts.append(pt)
        return np.mean(pts, axis=0) if pts else None
    
    # ==================== BIOMECHANICAL ANALYSIS ====================
    
    def detect_punch_posture(self, kps, conf):
        """Detect punching posture - RELAXED thresholds"""
        features = []
        
        for side in ['left', 'right']:
            shoulder = self.safe_point(kps, conf, KEYPOINTS[f"{side}_shoulder"])
            elbow = self.safe_point(kps, conf, KEYPOINTS[f"{side}_elbow"])
            wrist = self.safe_point(kps, conf, KEYPOINTS[f"{side}_wrist"])
            
            if all(p is not None for p in [shoulder, elbow, wrist]):
                angle = self.get_joint_angle(shoulder, elbow, wrist)
                # RELAXED: 130° instead of 140°
                if angle and angle > 130:
                    arm_length = euclidean(shoulder, wrist)
                    shoulder_elbow = euclidean(shoulder, elbow)
                    extension = arm_length / (shoulder_elbow + 1e-6)
                    
                    features.append({
                        'side': side,
                        'angle': angle,
                        'extension': extension,
                        'wrist_pos': wrist,
                        'shoulder_pos': shoulder
                    })
        
        return features
    
    def detect_blocking_posture(self, kps, conf):
        """Detect defensive blocking"""
        head = self.get_head_center(kps, conf)
        if head is None:
            return False
        
        for side in ['left', 'right']:
            shoulder = self.safe_point(kps, conf, KEYPOINTS[f"{side}_shoulder"])
            elbow = self.safe_point(kps, conf, KEYPOINTS[f"{side}_elbow"])
            wrist = self.safe_point(kps, conf, KEYPOINTS[f"{side}_wrist"])
            
            if all(p is not None for p in [shoulder, elbow, wrist]):
                hand_head_dist = euclidean(wrist, head)
                angle = self.get_joint_angle(shoulder, elbow, wrist)
                
                # RELAXED: 120 -> 130 distance threshold
                if hand_head_dist < 130 and angle and 60 < angle < 120:
                    return True
        
        return False
    
    def detect_kicking_posture(self, kps, conf):
        """Detect kicking - RELAXED thresholds"""
        features = []
        
        for side in ['left', 'right']:
            hip = self.safe_point(kps, conf, KEYPOINTS[f"{side}_hip"])
            knee = self.safe_point(kps, conf, KEYPOINTS[f"{side}_knee"])
            ankle = self.safe_point(kps, conf, KEYPOINTS[f"{side}_ankle"])
            
            if all(p is not None for p in [hip, knee, ankle]):
                elevation = abs(hip[1] - knee[1])
                angle = self.get_joint_angle(hip, knee, ankle)
                
                # RELAXED: 50 -> 40 elevation, 150 -> 140 angle
                if elevation > 40 or (angle and angle > 140):
                    features.append({
                        'side': side,
                        'elevation': elevation,
                        'angle': angle
                    })
        
        return features
    
    def detect_aggressive_stance(self, kps, conf):
        """Detect aggressive fighting stance - RELAXED"""
        l_shoulder = self.safe_point(kps, conf, KEYPOINTS["left_shoulder"])
        r_shoulder = self.safe_point(kps, conf, KEYPOINTS["right_shoulder"])
        l_hip = self.safe_point(kps, conf, KEYPOINTS["left_hip"])
        r_hip = self.safe_point(kps, conf, KEYPOINTS["right_hip"])
        
        if not all(p is not None for p in [l_shoulder, r_shoulder, l_hip, r_hip]):
            return False
        
        shoulder_width = euclidean(l_shoulder, r_shoulder)
        torso_compression = euclidean(
            (l_shoulder + r_shoulder) / 2,
            (l_hip + r_hip) / 2
        )
        
        # RELAXED: 60 -> 50 shoulder width, 120 -> 140 torso compression
        return shoulder_width > 50 and torso_compression < 140
    
    # ==================== MOTION ANALYSIS ====================
    
    def analyze_hand_trajectory(self, track_id, torso_height):
        """Analyze hand movement - RELAXED thresholds"""
        seq = self.pose_hist[track_id]
        if len(seq) < 4:  # Reduced from 5
            return None

        # RELAXED: Lower velocity threshold
        STRIKE_VELOCITY_THRESHOLD = torso_height * 0.25  # Was 0.4
        STRIKE_ACCEL_THRESHOLD = 10  # Was 15

        metrics = {
            'max_velocity': 0.0,
            'acceleration': 0.0,
            'rapid_extension': False
        }

        velocities = []
        for i in range(1, min(len(seq), 8)):
            prev_k, prev_c = seq[i - 1]
            curr_k, curr_c = seq[i]
            
            for side in ['left', 'right']:
                idx = KEYPOINTS[f"{side}_wrist"]
                prev_w = self.safe_point(prev_k, prev_c, idx)
                curr_w = self.safe_point(curr_k, curr_c, idx)
                
                if prev_w is not None and curr_w is not None:
                    vel = euclidean(prev_w, curr_w)
                    velocities.append(vel)
                    metrics['max_velocity'] = max(metrics['max_velocity'], vel)
        
        if len(velocities) >= 2:
            accel = np.diff(velocities)
            metrics['acceleration'] = float(np.max(accel)) if len(accel) > 0 else 0.0
            
            # Check if meets strike criteria
            metrics['rapid_extension'] = (
                metrics['max_velocity'] > STRIKE_VELOCITY_THRESHOLD
                and metrics['acceleration'] > STRIKE_ACCEL_THRESHOLD
            )

        return metrics
    
    def analyze_torso_movement(self, track_id):
        """Analyze torso movement"""
        seq = self.pose_hist[track_id]
        if len(seq) < WINDOW:
            return None
        
        movements = []
        for i in range(1, WINDOW):
            prev_k, prev_c = seq[i - 1]
            curr_k, curr_c = seq[i]
            
            prev_torso = self.get_torso_center(prev_k, prev_c)
            curr_torso = self.get_torso_center(curr_k, curr_c)
            
            if prev_torso is not None and curr_torso is not None:
                movements.append(euclidean(prev_torso, curr_torso))
        
        if movements:
            return {
                'avg_speed': float(np.mean(movements)),
                'max_speed': float(np.max(movements)),
                'erratic': float(np.std(movements)) > 4  # Was 5
            }
        return None
    
    # ==================== INTERACTION DETECTION ====================
    
    def detect_proximity_interaction(self, persons):
        """Detect close interactions between people"""
        interactions = []
        
        for i, p1 in enumerate(persons):
            for p2 in persons[i + 1:]:
                t1 = self.get_torso_center(p1['keypoints'], p1['confidence'])
                t2 = self.get_torso_center(p2['keypoints'], p2['confidence'])
                
                if t1 is not None and t2 is not None:
                    dist = euclidean(t1, t2)
                    
                    if dist < PROXIMITY_THRESHOLD:
                        interactions.append({
                            'id1': p1['track_id'],
                            'id2': p2['track_id'],
                            'distance': dist,
                            'person1': p1,
                            'person2': p2
                        })
        
        return interactions
    
    def detect_strike_impact(self, p1, p2):
        """Detect if person 1's hand is in strike zone of person 2"""
        head2 = self.get_head_center(p2['keypoints'], p2['confidence'])
        torso2 = self.get_torso_center(p2['keypoints'], p2['confidence'])
        
        if head2 is None and torso2 is None:
            return False
        
        target_zones = [z for z in [head2, torso2] if z is not None]
        
        for side in ['left', 'right']:
            wrist1 = self.safe_point(
                p1['keypoints'],
                p1['confidence'],
                KEYPOINTS[f"{side}_wrist"]
            )
            
            if wrist1 is not None:
                for zone in target_zones:
                    if euclidean(wrist1, zone) < STRIKE_ZONE_THRESHOLD:
                        return True
        
        return False
    
    # ==================== MAIN UPDATE ====================
    
    def update(self, persons, objects=None):
        """Main evaluation logic"""
        results = {}
        self.frame_count += 1
        
        # Update pose history
        for p in persons:
            tid = p["track_id"]
            self.pose_hist[tid].append((p["keypoints"], p["confidence"]))
        
        # Detect interactions
        interactions = self.detect_proximity_interaction(persons)
        
        # Analyze each person
        for p in persons:
            tid = p["track_id"]
            kps = p["keypoints"]
            conf = p["confidence"]
            torso_height = self.get_dynamic_scale(kps, conf)
            
            # Skip if insufficient history
            if len(self.pose_hist[tid]) < 4:  # Reduced from 5
                results[tid] = {
                    "action": "Tracking",
                    "severity": "LOW",
                    "color": SEVERITY_COLORS["LOW"],
                    "cause": {
                        "trigger": "INITIALIZATION",
                        "rule_name": "INSUFFICIENT_HISTORY",
                        "description": "Person detected but insufficient pose history for analysis",
                        "joints_involved": [],
                        "metrics": {
                            "frames_tracked": f"{len(self.pose_hist[tid])} frames",
                            "frames_required": "4 frames"
                        }
                    }
                }
                continue
            
            # Analyze postures and movements
            punch_features = self.detect_punch_posture(kps, conf)
            kick_features = self.detect_kicking_posture(kps, conf)
            blocking = self.detect_blocking_posture(kps, conf)
            aggressive_stance = self.detect_aggressive_stance(kps, conf)
            
            hand_metrics = self.analyze_hand_trajectory(tid, torso_height)
            torso_metrics = self.analyze_torso_movement(tid)
            
            # Check interactions
            involved_interaction = None
            strike_detected = False
            
            for inter in interactions:
                if inter['id1'] == tid or inter['id2'] == tid:
                    involved_interaction = inter
                    
                    # Check for strike
                    if inter['id1'] == tid:
                        strike_detected = self.detect_strike_impact(p, inter['person2'])
                    else:
                        strike_detected = self.detect_strike_impact(inter['person1'], p)
            
            # Count recent strikes
            rapid_count = len(self.strike_hist[tid])
            
            # ==================== CLASSIFICATION ====================
            
            action = "Normal Motion"
            severity = "LOW"
            rule_name = None
            cause_desc = None
            joints = []
            metrics = {}
            
            # CRITICAL: Active physical violence
            if strike_detected and hand_metrics and hand_metrics['rapid_extension']:
                action = "Active Physical Assault"
                severity = "CRITICAL"
                rule_name = "STRIKE_IMPACT_DETECTED"
                cause_desc = "Direct strike detected with rapid arm extension in close proximity to another person"
                joints = ["wrist", "elbow", "shoulder"]
                metrics = {
                    "hand_velocity": f"{hand_metrics['max_velocity']:.1f} px/frame",
                    "interaction_distance": f"{involved_interaction['distance']:.1f} px" if involved_interaction else "N/A"
                }
                self.strike_hist[tid].append(self.frame_count)
            
            # RELAXED: Kicking with lower thresholds
            elif kick_features and involved_interaction and involved_interaction['distance'] < 150:
                action = "Kicking Attack"
                severity = "CRITICAL"
                rule_name = "KICK_IN_PROXIMITY"
                cause_desc = "Kicking motion detected in close proximity to another person"
                joints = ["hip", "knee", "ankle"]
                metrics = {
                    "leg_elevation": f"{kick_features[0]['elevation']:.1f} px",
                    "knee_angle": f"{kick_features[0]['angle']:.1f}°",
                    "distance_to_person": f"{involved_interaction['distance']:.1f} px"
                }
            
            # HIGH: Aggressive behavior patterns
            elif hand_metrics and hand_metrics['rapid_extension'] and involved_interaction:
                action = "Aggressive Striking Motion"
                severity = "HIGH"
                rule_name = "RAPID_STRIKE_NEAR_PERSON"
                cause_desc = "Rapid punching motion detected near another person"
                joints = ["wrist", "elbow", "shoulder"]
                metrics = {
                    "hand_velocity": f"{hand_metrics['max_velocity']:.1f} px/frame",
                    "acceleration": f"{hand_metrics['acceleration']:.1f} px/frame²",
                    "proximity": f"{involved_interaction['distance']:.1f} px"
                }
            
            elif blocking and involved_interaction:
                action = "Defensive Fighting Posture"
                severity = "HIGH"
                rule_name = "DEFENSIVE_STANCE_INTERACTION"
                cause_desc = "Defensive blocking posture near another person"
                joints = ["shoulder", "elbow", "wrist"]
                metrics = {"proximity": f"{involved_interaction['distance']:.1f} px"}
            
            elif aggressive_stance and involved_interaction and involved_interaction['distance'] < 120:
                action = "Confrontational Stance"
                severity = "HIGH"
                rule_name = "AGGRESSIVE_STANCE_CLOSE_PROXIMITY"
                cause_desc = "Aggressive fighting stance in very close proximity"
                joints = ["shoulder", "hip"]
                metrics = {"distance": f"{involved_interaction['distance']:.1f} px"}
            
            # MEDIUM: Suspicious behavior
            elif hand_metrics and hand_metrics['rapid_extension'] and rapid_count >= MIN_RAPID_FRAMES:
                action = "Shadow Boxing / Air Punching"
                severity = "MEDIUM"
                rule_name = "RAPID_PUNCH_NO_TARGET"
                cause_desc = "Rapid punching motion without a target (possible testing or intimidation)"
                joints = ["wrist", "elbow", "shoulder"]
                metrics = {
                    "hand_velocity": f"{hand_metrics['max_velocity']:.1f} px/frame",
                    "acceleration": f"{hand_metrics['acceleration']:.1f} px/frame²"
                }
            
            elif kick_features and not involved_interaction:
                action = "Kicking Motion (No Target)"
                severity = "MEDIUM"
                rule_name = "KICK_NO_PROXIMITY"
                cause_desc = "Kicking motion detected without nearby person"
                joints = ["hip", "knee", "ankle"]
                metrics = {"leg_elevation": f"{kick_features[0]['elevation']:.1f} px"}
            
            elif aggressive_stance and torso_metrics and torso_metrics['erratic']:
                action = "Agitated Movement"
                severity = "MEDIUM"
                rule_name = "ERRATIC_AGGRESSIVE_MOVEMENT"
                cause_desc = "Erratic movement with aggressive stance"
                joints = ["shoulder", "hip", "torso"]
                metrics = {
                    "torso_avg_speed": f"{torso_metrics['avg_speed']:.1f} px/frame",
                    "torso_max_speed": f"{torso_metrics['max_speed']:.1f} px/frame",
                    "erratic_movement": "Yes"
                }
            
            # Build result
            payload = {
                "action": action,
                "severity": severity,
                "color": SEVERITY_COLORS[severity]
            }
            
            # Add explainability for non-normal actions
            if action != "Normal Motion" and action != "Tracking":
                payload["cause"] = {
                    "trigger": "BIOMECHANICAL_RULE",
                    "rule_name": rule_name,
                    "description": cause_desc,
                    "joints_involved": joints,
                    "metrics": metrics,
                    "interaction": involved_interaction is not None,
                    "frame_window": f"{WINDOW} frames"
                }
                
                # Add recent strike count
                if rapid_count > 0:
                    payload["cause"]["recent_strikes"] = rapid_count
                    payload["cause"]["strike_window_frames"] = f"{len(self.strike_hist[tid])} frames"
            
            results[tid] = payload
        
        return results