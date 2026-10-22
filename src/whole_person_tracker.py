# src/whole_person_tracker.py
"""
Whole-person tracking with face association: once an enrolled target's face
is verified, the body box that face sits on is locked and carried forward
by spatial continuity (IoU + displacement) even while the face itself is
temporarily unreadable - a head turn, a hand over the face, someone briefly
crossing in front. Identity confidence for the lock is owned by
IdentityConfidence (src/identity_state.py); this module only ever reports
what that honest three-state model says, it never invents certainty.

Centering uses the BODY box center, not the face: the body stays trackable
through exactly the situations (turned head, occlusion) where the face
can't, so centering on it is what keeps a camera/servo pointed at the
person through those gaps instead of the output vanishing.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from .association import associate_faces_to_bodies
from .face_pipeline import ArcFaceEmbedderONNX, FaceDBMatcher, FaceDet, HaarFaceMesh5pt, identity_of
from .identity_state import IdentityConfidence, IdentitySnapshot, IdentityState
from .person_detector import BodyDet, PersonDetector, center, iou

logger = logging.getLogger(__name__)


@dataclass
class TrackingSignal:
    error_x: float
    error_y: float
    horizontal: str
    vertical: str


@dataclass
class TrackResult:
    body: BodyDet
    identity: IdentitySnapshot
    position: TrackingSignal
    face: Optional[FaceDet]  # the face currently seen in the locked body's head region, if any


class WholePersonTracker:
    def __init__(
        self,
        target_name: str,
        face_detector: HaarFaceMesh5pt,
        embedder: ArcFaceEmbedderONNX,
        matcher: FaceDBMatcher,
        body_detector: PersonDetector,
        verify_every: int = 10,
        ema_alpha: float = 0.30,
        dead_zone: float = 0.07,
    ):
        self.target_name = target_name
        self.face_detector = face_detector
        self.embedder = embedder
        self.matcher = matcher
        self.body_detector = body_detector
        self.verify_every = verify_every
        self.ema_alpha = ema_alpha
        self.dead_zone = dead_zone

        self.identity = IdentityConfidence()
        self.locked_body: Optional[BodyDet] = None
        self.smooth_center: Optional[np.ndarray] = None
        self.frame_index = 0

    @staticmethod
    def _box(b: BodyDet):
        return (b.x1, b.y1, b.x2, b.y2)

    def _acquire(self, frame, faces: List[FaceDet], bodies: List[BodyDet]) -> Optional[BodyDet]:
        face_bodies = associate_faces_to_bodies(faces, bodies)
        for face, body in zip(faces, face_bodies):
            if body is None:
                continue
            match = identity_of(self.embedder, self.matcher, frame, face)
            if match.accepted and match.name == self.target_name:
                logger.info("acquired target %r: body=%s", self.target_name, self._box(body))
                return body
        return None

    def _associate_body(self, bodies: List[BodyDet]) -> Optional[BodyDet]:
        """Carries the locked body box forward frame-to-frame by spatial continuity alone."""
        if self.locked_body is None or not bodies:
            return None
        if len(bodies) == 1:
            return bodies[0]

        last_box = self._box(self.locked_body)
        last_center = center(last_box)
        last_diag = max(np.linalg.norm([last_box[2] - last_box[0], last_box[3] - last_box[1]]), 1.0)

        best_score, best_body = -1e9, None
        for body in bodies:
            box = self._box(body)
            score = iou(last_box, box) - 0.20 * (np.linalg.norm(center(box) - last_center) / last_diag)
            if score > best_score:
                best_score, best_body = score, body
        return best_body if best_score > -0.45 else None

    def _face_on_locked_body(self, faces: List[FaceDet], bodies: List[BodyDet], candidate_body: BodyDet) -> Optional[FaceDet]:
        face_bodies = associate_faces_to_bodies(faces, bodies)
        for face, body in zip(faces, face_bodies):
            if body is candidate_body:
                return face
        return None

    def update(self, frame) -> Optional[TrackResult]:
        self.frame_index += 1
        faces = self.face_detector.detect(frame, max_faces=8)
        bodies = self.body_detector.detect(frame, max_people=6)

        if self.locked_body is None:
            candidate = self._acquire(frame, faces, bodies)
            if candidate is None:
                self.identity.tick_unknown()
                return None
            self.locked_body = candidate
            self.identity.confirm()
        else:
            candidate = self._associate_body(bodies)
            if candidate is None:
                snapshot = self.identity.hold_without_face(body_continuity_ok=False)
                self.identity.tick_unknown()
                if self.identity.state == IdentityState.SEARCHING:
                    self.locked_body = None
                    self.smooth_center = None
                    return None
                # briefly keep reporting the last known box while UNKNOWN, honestly
                # flagged as such, instead of the output disappearing outright
            else:
                self.locked_body = candidate
                face = self._face_on_locked_body(faces, bodies, candidate)
                due = self.identity.state != IdentityState.CONFIRMED or self.frame_index % self.verify_every == 0
                if face is not None and due:
                    match = identity_of(self.embedder, self.matcher, frame, face)
                    if match.accepted and match.name == self.target_name:
                        self.identity.confirm()
                    elif match.accepted:
                        # a *different* enrolled identity is now wearing this body box -
                        # a path crossing swapped who we're actually looking at, so the
                        # lock must drop rather than keep tracking the wrong person
                        logger.info("locked body now shows %r, not target %r -> dropping lock",
                                    match.name, self.target_name)
                        self.identity.reset()
                        self.locked_body = None
                        self.smooth_center = None
                        return None
                    else:
                        self.identity.hold_without_face(body_continuity_ok=True)
                elif face is None:
                    self.identity.hold_without_face(body_continuity_ok=True)

        snapshot = self.identity.snapshot()
        box = self._box(self.locked_body)
        raw_center = center(box)
        self.smooth_center = raw_center if self.smooth_center is None else (
            self.ema_alpha * raw_center + (1.0 - self.ema_alpha) * self.smooth_center
        )
        face = self._face_on_locked_body(faces, bodies, self.locked_body)
        return TrackResult(
            body=self.locked_body,
            identity=snapshot,
            position=self._position_signal(frame.shape),
            face=face,
        )

    def _position_signal(self, shape) -> TrackingSignal:
        height, width = shape[:2]
        ex = float((self.smooth_center[0] - width / 2.0) / (width / 2.0))
        ey = float((self.smooth_center[1] - height / 2.0) / (height / 2.0))
        horizontal = "LEFT" if ex < -self.dead_zone else "RIGHT" if ex > self.dead_zone else "CENTER"
        vertical = "UP" if ey < -self.dead_zone else "DOWN" if ey > self.dead_zone else "CENTER"
        return TrackingSignal(ex, ey, horizontal, vertical)
