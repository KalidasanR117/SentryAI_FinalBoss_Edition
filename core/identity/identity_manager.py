# sentry/core/identity/identity_manager.py

import time
import cv2
import numpy as np
from core.identity.types import Identity


class IdentityManager:
    def __init__(self, arcface_session, face_db):
        """
        arcface_session : ONNX Runtime session for ArcFace
        face_db         : {
                            "whitelist": {name: [embeddings]},
                            "blacklist": {name: [embeddings]}
                          }
        """
        self.arcface = arcface_session
        self.whitelist_db = face_db.get("whitelist", {})
        self.blacklist_db = face_db.get("blacklist", {})

        self.identity_cache = {}
        # track_id -> { identity, name, confidence, last_seen }

        self.WL_THRESHOLD = 0.50
        self.BL_THRESHOLD = 0.50
        self.TIMEOUT = 3.0  # seconds

        # cache ArcFace input details once
        meta = self.arcface.get_inputs()[0]
        self.arc_input_name = meta.name
        self.arc_input_shape = meta.shape

    # -----------------------------------------------------

    def _preprocess_face(self, face_img):
        shape = self.arc_input_shape

        if shape[1] == 3:  # NCHW
            H, W = shape[2], shape[3]
            face = cv2.resize(face_img, (W, H))
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB).astype(np.float32)
            face = (face - 127.5) / 128.0
            face = np.transpose(face, (2, 0, 1))
            face = np.expand_dims(face, 0)
        else:  # NHWC
            H, W = shape[1], shape[2]
            face = cv2.resize(face_img, (W, H))
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB).astype(np.float32)
            face = (face - 127.5) / 128.0
            face = np.expand_dims(face, 0)

        return face

    # -----------------------------------------------------

    def _compare_embedding(self, emb, db):
        best_name = None
        best_score = 0.0

        for name, refs in db.items():
            for ref in refs:
                score = float(np.dot(emb, ref))
                if score > best_score:
                    best_score = score
                    best_name = name

        return best_name, best_score

    # -----------------------------------------------------

    def update_identity(self, track_id, face_img):
        """
        Called only when a face crop is available
        """
        face_input = self._preprocess_face(face_img)

        emb = self.arcface.run(
            None, {self.arc_input_name: face_input}
        )[0].flatten()

        emb /= np.linalg.norm(emb + 1e-8)

        bl_name, bl_score = self._compare_embedding(emb, self.blacklist_db)
        wl_name, wl_score = self._compare_embedding(emb, self.whitelist_db)

        if bl_score >= self.BL_THRESHOLD:
            identity = Identity.BLACKLIST
            name = bl_name
            score = bl_score

        elif wl_score >= self.WL_THRESHOLD:
            identity = Identity.WHITELIST
            name = wl_name
            score = wl_score

        else:
            identity = Identity.UNKNOWN
            name = None
            score = max(bl_score, wl_score)

        self.identity_cache[track_id] = {
            "identity": identity,
            "name": name,
            "confidence": float(score),
            "last_seen": time.time()
        }

        return self.identity_cache[track_id]

    # -----------------------------------------------------

    def get_identity(self, track_id):
        data = self.identity_cache.get(track_id)

        if data is None:
            return {
                "identity": Identity.UNKNOWN,
                "name": None,
                "confidence": 0.0
            }

        if time.time() - data["last_seen"] > self.TIMEOUT:
            return {
                "identity": Identity.UNKNOWN,
                "name": None,
                "confidence": 0.0
            }

        data["last_seen"] = time.time()
        return data
