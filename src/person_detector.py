# src/person_detector.py
"""
Whole-body detection: MediaPipe ObjectDetector (EfficientDet-Lite0, int8),
filtered to the "person" category. Replaces OpenCV's built-in HOG + Linear
SVM people detector, which is slow relative to its accuracy and fails
often under partial occlusion or non-upright poses - exactly the
path-crossing scenarios this tracker needs to survive. The Tasks API also
handles NMS internally, unlike the manual cv2.dnn.NMSBoxes call this
replaces. Runs on the same CPU XNNPACK delegate already proven fast
elsewhere in this project.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python import BaseOptions

_DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "efficientdet_lite0.tflite")


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


def _scale_for_detection(frame: np.ndarray, max_dim: int) -> Tuple[np.ndarray, float]:
    h, w = frame.shape[:2]
    scale = min(1.0, max_dim / float(max(h, w)))
    if scale >= 1.0:
        return frame, 1.0
    small = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LINEAR)
    return small, scale


class PersonDetector:
    def __init__(
        self,
        model_path: str = _DEFAULT_MODEL_PATH,
        min_confidence: float = 0.4,
        max_dim: int = 640,
        max_results: int = 8,
    ):
        self.max_dim = max_dim
        options = mp_vision.ObjectDetectorOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=mp_vision.RunningMode.VIDEO,
            max_results=max_results,
            score_threshold=min_confidence,
            category_allowlist=["person"],
        )
        self._detector = mp_vision.ObjectDetector.create_from_options(options)
        self._start_time = time.monotonic()

    def close(self) -> None:
        self._detector.close()

    def detect(self, frame_bgr: np.ndarray, max_people: int = 6) -> List[BodyDet]:
        small, scale = _scale_for_detection(frame_bgr, self.max_dim)
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms = int((time.monotonic() - self._start_time) * 1000)
        result = self._detector.detect_for_video(mp_image, timestamp_ms)
        if not result.detections:
            return []

        inv = 1.0 / scale
        out = []
        for d in result.detections:
            bb = d.bounding_box
            x1, y1 = int(bb.origin_x * inv), int(bb.origin_y * inv)
            x2, y2 = int((bb.origin_x + bb.width) * inv), int((bb.origin_y + bb.height) * inv)
            score = d.categories[0].score if d.categories else 1.0
            out.append(BodyDet(x1=x1, y1=y1, x2=x2, y2=y2, score=float(score)))

        out.sort(key=lambda b: b.score, reverse=True)
        return out[:max_people]
