import numpy as np

def coco17_to_flat(keypoints):
    return keypoints.flatten().astype("float32")
