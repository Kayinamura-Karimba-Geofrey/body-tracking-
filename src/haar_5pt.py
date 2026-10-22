# src/haar_5pt.py
"""
Reusable face detection + 5-point alignment building blocks:
- Haar cascade for the face box (fast), smoothed/held across frames for stability
- MediaPipe FaceLandmarker (Tasks API) for 5 stable keypoints
- ArcFace-style 5pt -> square alignment warp

Used by src/align.py (and anything else that wants a stable face box + 5pt).
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python import BaseOptions

# 5-point indices (canonical face mesh topology, shared by FaceMesh and FaceLandmarker)
IDX_LEFT_EYE = 33
IDX_RIGHT_EYE = 263
IDX_NOSE_TIP = 1
IDX_MOUTH_LEFT = 61
IDX_MOUTH_RIGHT = 291

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

_DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "face_landmarker.task")
_DEFAULT_CASCADE_PATH = os.path.join(os.path.dirname(__file__), "haarcascade_frontalface_default.xml")


@dataclass
class Face:
    x1: int
    y1: int
    x2: int
    y2: int
    kps: np.ndarray  # (5, 2) float32, order: left eye, right eye, nose, mouth left, mouth right


class Haar5ptDetector:
    """Haar box (smoothed) + MediaPipe 5pt landmarks (smoothed), held across brief misses."""

    def __init__(
        self,
        cascade_path: str = _DEFAULT_CASCADE_PATH,
        model_path: str = _DEFAULT_MODEL_PATH,
        min_size: Tuple[int, int] = (60, 60),
        smooth_alpha: float = 0.4,
        max_missed_frames: int = 10,
        debug: bool = False,
    ):
        self.min_size = min_size
        self.smooth_alpha = smooth_alpha
        self.max_missed_frames = max_missed_frames
        self.debug = debug

        self._cascade = cv2.CascadeClassifier(cascade_path)
        if self._cascade.empty():
            raise RuntimeError(f"Failed to load cascade: {cascade_path}")

        if not os.path.exists(model_path):
            raise RuntimeError(f"Missing model file: {model_path}")

        options = mp_vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=mp_vision.RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._landmarker = mp_vision.FaceLandmarker.create_from_options(options)
        self._start_time = time.monotonic()

        self._smoothed_box: Optional[Tuple[float, float, float, float]] = None
        self._smoothed_kps: Optional[np.ndarray] = None
        self._missed_frames = 0

    def close(self):
        self._landmarker.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def detect(self, frame: np.ndarray, max_faces: int = 1) -> List[Face]:
        H, W = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # equalize contrast so uneven/low light doesn't starve Haar of edges
        gray = cv2.equalizeHist(gray)

        boxes = self._cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=self.min_size
        )

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms = int((time.monotonic() - self._start_time) * 1000)
        res = self._landmarker.detect_for_video(mp_image, timestamp_ms)

        have_box = len(boxes) > 0
        have_kps = bool(res.face_landmarks)

        # The landmarker tolerates head turns far better than the frontal-only Haar
        # cascade, so track off it alone once we have a face — Haar just tightens
        # the box when it also agrees.
        if have_kps:
            self._missed_frames = 0

            lm = res.face_landmarks[0]
            idxs = [IDX_LEFT_EYE, IDX_RIGHT_EYE, IDX_NOSE_TIP, IDX_MOUTH_LEFT, IDX_MOUTH_RIGHT]
            pts = np.array([[lm[i].x * W, lm[i].y * H] for i in idxs], dtype=np.float32)

            # enforce left/right ordering
            if pts[0, 0] > pts[1, 0]:
                pts[[0, 1]] = pts[[1, 0]]
            if pts[3, 0] > pts[4, 0]:
                pts[[3, 4]] = pts[[4, 3]]

            if have_box:
                x, y, w, h = max(boxes, key=lambda f: f[2] * f[3])
            else:
                x_min, y_min = pts[:, 0].min(), pts[:, 1].min()
                x_max, y_max = pts[:, 0].max(), pts[:, 1].max()
                pw, ph = x_max - x_min, y_max - y_min
                x, y = x_min - 0.4 * pw, y_min - 0.6 * ph
                w, h = pw * 1.8, ph * 2.0

            if self._smoothed_box is None:
                self._smoothed_box = (float(x), float(y), float(w), float(h))
            else:
                self._smoothed_box = tuple(
                    prev + self.smooth_alpha * (cur - prev)
                    for prev, cur in zip(self._smoothed_box, (x, y, w, h))
                )

            if self._smoothed_kps is None:
                self._smoothed_kps = pts
            else:
                self._smoothed_kps = self._smoothed_kps + self.smooth_alpha * (pts - self._smoothed_kps)
        else:
            self._missed_frames += 1
            if self._missed_frames > self.max_missed_frames:
                self._smoothed_box = None
                self._smoothed_kps = None
                if self.debug:
                    print("[haar_5pt] face lost")

        if self._smoothed_box is None or self._smoothed_kps is None:
            return []

        bx, by, bw, bh = (int(v) for v in self._smoothed_box)
        face = Face(x1=bx, y1=by, x2=bx + bw, y2=by + bh, kps=self._smoothed_kps.copy())
        return [face][:max_faces]


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
