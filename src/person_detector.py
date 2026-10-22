# src/person_detector.py
"""
Whole-body detection: OpenCV's built-in HOG + Linear SVM people detector.
Chosen over a DNN model so the whole stack stays CPU-only with no extra
download - it ships inside opencv-python and runs comfortably at webcam
frame rates on a laptop CPU.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np


@dataclass
class BodyDet:
    x1: int
    y1: int
    x2: int
    y2: int
    score: float


def iou(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = max(1, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(1, (bx2 - bx1) * (by2 - by1))
    return inter / float(area_a + area_b - inter)


def center(box) -> np.ndarray:
    x1, y1, x2, y2 = box
    return np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0], dtype=np.float32)


class PersonDetector:
    def __init__(
        self,
        hit_threshold: float = 0.0,
        win_stride: Tuple[int, int] = (8, 8),
        scale: float = 1.05,
        nms_thresh: float = 0.4,
        min_score: float = 0.3,
    ):
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        self.hit_threshold = hit_threshold
        self.win_stride = win_stride
        self.scale = scale
        self.nms_thresh = nms_thresh
        self.min_score = min_score

    def detect(self, frame_bgr: np.ndarray, max_people: int = 6) -> List[BodyDet]:
        boxes, weights = self.hog.detectMultiScale(
            frame_bgr,
            hitThreshold=self.hit_threshold,
            winStride=self.win_stride,
            padding=(8, 8),
            scale=self.scale,
        )
        if len(boxes) == 0:
            return []

        rects = [[int(x), int(y), int(w), int(h)] for (x, y, w, h) in boxes]
        scores = [float(w) for w in weights]
        keep = cv2.dnn.NMSBoxes(rects, scores, score_threshold=self.min_score, nms_threshold=self.nms_thresh)
        if len(keep) == 0:
            return []

        out = []
        for i in np.array(keep).reshape(-1):
            x, y, w, h = rects[i]
            out.append(BodyDet(x1=x, y1=y, x2=x + w, y2=y + h, score=scores[i]))
        out.sort(key=lambda b: b.score, reverse=True)
        return out[:max_people]
