# src/align.py
"""
Detector-agnostic ArcFace 5-point alignment: warps any face given its
5 keypoints (left eye, right eye, nose, mouth left, mouth right) onto the
canonical 112x112 ArcFace reference layout. Used by src/face_pipeline.py
and src/enroll.py regardless of which detector/landmarker produced the
points.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

# ArcFace reference 5pt for a 112x112 aligned crop
_ARCFACE_REF_112 = np.array(
    [
        [38.2946, 51.6963],  # left eye
        [73.5318, 51.5014],  # right eye
        [56.0252, 71.7366],  # nose tip
        [41.5493, 92.3655],  # mouth left
        [70.7299, 92.2041],  # mouth right
    ],
    dtype=np.float32,
)


@dataclass
class Face:
    x1: int
    y1: int
    x2: int
    y2: int
    kps: np.ndarray  # (5, 2) float32, order: left eye, right eye, nose, mouth left, mouth right


def align_face_5pt(
    img: np.ndarray, kps: np.ndarray, out_size: Tuple[int, int] = (112, 112)
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Warp img so the given 5 keypoints map onto the ArcFace reference layout."""
    out_w, out_h = int(out_size[0]), int(out_size[1])

    ref = _ARCFACE_REF_112.copy()
    ref[:, 0] *= out_w / 112.0
    ref[:, 1] *= out_h / 112.0

    M, _inliers = cv2.estimateAffinePartial2D(kps.astype(np.float32), ref, method=cv2.LMEDS)
    if M is None:
        return None, None

    aligned = cv2.warpAffine(img, M, (out_w, out_h), borderValue=0)
    return aligned, M
