# sentry/scripts/run_pose.py
import cv2
import time
import numpy as np
import torch

from sentry.core.detector import PoseDetector
from sentry.core.pose_buffer import PoseBuffer
from sentry.utils.visualization import draw_pose

# (Future) Action model
from sentry.core.action_model import PoseLSTM
action_model = PoseLSTM(num_classes=3)
action_model.eval()

buffer = PoseBuffer(max_len=30)

def predict_action(seq):
    seq = np.array(seq, dtype=np.float32)
    x = torch.from_numpy(seq).unsqueeze(0)
    with torch.no_grad():
        out = action_model(x)
    return int(torch.argmax(out, dim=1))

def main(source=0):
    cap = cv2.VideoCapture(source)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    detector = PoseDetector()
    prev_time = time.time()
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detections = detector.infer(frame)

        # update pose buffers
        for det in detections:
            buffer.update(det["track_id"], det["keypoints"])

        # (optional) action inference every N frames
        action_labels = {}
        if frame_count % 5 == 0:
            for det in detections:
                tid = det["track_id"]
                seq = buffer.get(tid)
                if seq is not None:
                    action_labels[tid] = predict_action(seq)

        annotated = draw_pose(frame.copy(), detections)

        # FPS
        fps = 1 / (time.time() - prev_time)
        prev_time = time.time()
        cv2.putText(annotated, f"FPS:{fps:.1f}",
                    (10,30), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0,255,0), 2)

        cv2.imshow("Sentry – Pose + ByteTrack", annotated)
        frame_count += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
