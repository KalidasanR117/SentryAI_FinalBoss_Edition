# sentry/core/theft_detector.py

from collections import defaultdict, deque
import numpy as np

class TheftDetector:
    def __init__(self, static_thresh=8, corr_thresh=60):
        """
        static_thresh = number of frames to consider an object "static"
        corr_thresh   = px distance threshold for correlated motion
        """
        self.obj_history = defaultdict(lambda: deque(maxlen=120))
        self.person_history = defaultdict(lambda: deque(maxlen=120))
        self.static_thresh = static_thresh
        self.corr_thresh = corr_thresh
        self.stolen_log = set()

    def update_objects(self, objects):
        for o in objects:
            oid = o["track_id"]
            center = np.array([ (o["bbox"][0] + o["bbox"][2]) / 2,
                                (o["bbox"][1] + o["bbox"][3]) / 2 ])
            self.obj_history[oid].append(center)

    def update_persons(self, persons):
        for p in persons:
            pid = p["track_id"]
            kps = p["keypoints"]
            # basic torso center estimate
            torso = np.mean([kps[5], kps[6], kps[11], kps[12]], axis=0)
            self.person_history[pid].append(np.array(torso))

    def is_static(self, hist):
        if len(hist) < self.static_thresh:
            return False
        pts = np.array(hist)
        # low STD => roughly static
        return np.mean(np.std(pts, axis=0)) < 5

    def correlates_with_person(self, obj_hist, pers_hist):
        """
        Checks if object and person move similarly.
        We compare the last K frames from both histories.
        """
        if len(obj_hist) < self.static_thresh or len(pers_hist) < self.static_thresh:
            return False

        # take last static_thresh frames of both
        obj_pts = np.array(obj_hist)[-self.static_thresh:]
        pers_pts = np.array(pers_hist)[-self.static_thresh:]
        
        # now they have the same shape: (static_thresh, 2)
        dists = np.linalg.norm(obj_pts - pers_pts, axis=1)
        return np.all(dists < self.corr_thresh)


    def detect(self, persons, objects, frame_idx=None):
        events = []

        # update histories
        self.update_objects(objects)
        self.update_persons(persons)

        for oid, obj_hist in list(self.obj_history.items()):
            if self.is_static(obj_hist):
                static_center = obj_hist[-1]

                if len(obj_hist) >= obj_hist.maxlen:
                    for p in persons:
                        pid = p["track_id"]
                        pers_hist = self.person_history[pid]

                        if self.correlates_with_person(obj_hist, pers_hist):
                            curr_center = obj_hist[-1]
                            if np.linalg.norm(curr_center - static_center) > self.corr_thresh:
                                if (pid, oid) not in self.stolen_log:
                                    self.stolen_log.add((pid, oid))

                                    # Build event
                                    event = {
                                        "type": "theft_confirmed",
                                        "person_id": pid,
                                        "object_id": oid,
                                        "frame": frame_idx,
                                        "timestamp": datetime.datetime.now().strftime("%H:%M:%S")
                                    }
                                    events.append(event)

        return events
