# src/face_detector.py
"""
Whole-frame multi-face bounding-box detection: MediaPipe FaceDetector
(BlazeFace), replacing the Haar cascade this project started with. Haar
is frontal-only and stops detecting past roughly 15-20 deg of yaw; BlazeFace
tolerates much larger head turns, which is the whole point of swapping it
in - it keeps CONFIRMED alive through partial turns that used to drop
straight to TEMPORARY decay.

This module only finds face boxes. The precise 5-point landmarks ArcFace
alignment needs (left eye, right eye, nose, mouth-left, mouth-right) still
come from the per-ROI MediaPipe FaceLandmarker pass in face_pipeline.py -
BlazeFace's own 6 keypoints don't match that point set or order.
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

_DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "blaze_face_short_range.tflite")


@dataclass
class FaceBox:
    x1: int
    y1: int
    x2: int
    y2: int
    score: float


def _scale_for_detection(frame: np.ndarray, max_dim: int) -> Tuple[np.ndarray, float]:
    """Downscales so the longest side is <= max_dim before inference - the
    single biggest CPU-cost lever for a whole-frame detector pass. Returns
    the scale factor to map detected boxes back to the original frame."""
    h, w = frame.shape[:2]
    scale = min(1.0, max_dim / float(max(h, w)))
    if scale >= 1.0:
        return frame, 1.0
    small = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LINEAR)
    return small, scale


class FaceBoxDetector:
    def __init__(
        self,
        model_path: str = _DEFAULT_MODEL_PATH,
        min_confidence: float = 0.5,
        max_dim: int = 640,
        hold_frames: int = 2,
    ):
        self.max_dim = max_dim
        self.hold_frames = hold_frames

        options = mp_vision.FaceDetectorOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=mp_vision.RunningMode.VIDEO,
            min_detection_confidence=min_confidence,
        )
        self._detector = mp_vision.FaceDetector.create_from_options(options)
        self._start_time = time.monotonic()

        # debounce a single-frame miss instead of flickering empty, same idea
        # as the old Haar hold-last-boxes buffer
        self._last_boxes: List[FaceBox] = []
        self._missed = 0

    def close(self) -> None:
        self._detector.close()

    def detect(self, frame_bgr: np.ndarray, max_faces: int = 8) -> List[FaceBox]:
        small, scale = _scale_for_detection(frame_bgr, self.max_dim)
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms = int((time.monotonic() - self._start_time) * 1000)
        result = self._detector.detect_for_video(mp_image, timestamp_ms)

        if not result.detections:
            self._missed += 1
            return self._last_boxes if self._missed <= self.hold_frames else []

        self._missed = 0
        inv = 1.0 / scale
        boxes = []
        for d in result.detections:
            bb = d.bounding_box
            x1, y1 = int(bb.origin_x * inv), int(bb.origin_y * inv)
            x2, y2 = int((bb.origin_x + bb.width) * inv), int((bb.origin_y + bb.height) * inv)
            score = d.categories[0].score if d.categories else 1.0
            boxes.append(FaceBox(x1=x1, y1=y1, x2=x2, y2=y2, score=float(score)))

        boxes.sort(key=lambda b: (b.x2 - b.x1) * (b.y2 - b.y1), reverse=True)
        self._last_boxes = boxes[:max_faces]
        return self._last_boxes
