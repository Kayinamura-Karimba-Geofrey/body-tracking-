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

Performance: on this CPU, EfficientDet-Lite0 (person detection) costs
roughly 10x a BlazeFace pass and dwarfs everything else per frame -
downscaling the input barely moves it, the cost is the model's own
compute. So once a body is locked, person detection only runs every
`body_detect_every` frames and relies on IoU+displacement continuity
(see _associate_body) to carry the box between detections. Likewise, the
expensive per-ROI FaceLandmarker + ArcFace embedding pass (needed for
*whose* face it is) only runs when actually due for re-verification;
every other locked frame uses the cheap BlazeFace-only box pass (no
landmarker) just to answer *is a face visible there at all*, which is
all the TEMPORARY/CONFIRMED transition needs most frames.

Appearance (src/appearance.py): a face can only ever confirm identity
when it's visible, which by definition never happens with someone's back
turned. An optional torso/clothing appearance signature (captured across
all four orientations at enrollment) lets the tracker re-acquire a lost
target - or disambiguate a crowded continuity match - using *where they
are and what they're wearing* instead, landing honestly in TEMPORARY
rather than CONFIRMED (see IdentityConfidence.confirm_by_appearance).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from .appearance import compare_signatures, torso_signature
from .association import associate_faces_to_bodies
from .face_pipeline import ArcFaceEmbedderONNX, FaceDBMatcher, FaceDet, MultiFaceDetector, identity_of
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
        face_detector: MultiFaceDetector,
        embedder: ArcFaceEmbedderONNX,
        matcher: FaceDBMatcher,
        body_detector: PersonDetector,
        verify_every: int = 10,
        body_detect_every: int = 2,
        ema_alpha: float = 0.30,
        dead_zone: float = 0.07,
        appearance_signature: Optional[np.ndarray] = None,
        appearance_threshold: float = 0.55,
    ):
        self.target_name = target_name
        self.face_detector = face_detector
        self.embedder = embedder
        self.matcher = matcher
        self.body_detector = body_detector
        self.verify_every = verify_every
        self.body_detect_every = max(1, body_detect_every)
        self.ema_alpha = ema_alpha
        self.dead_zone = dead_zone
        self.appearance_signature = appearance_signature
        self.appearance_threshold = appearance_threshold

        self.identity = IdentityConfidence()
        self.locked_body: Optional[BodyDet] = None
        self.smooth_center: Optional[np.ndarray] = None
        self.frame_index = 0
        self._cached_bodies: List[BodyDet] = []

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

    def _associate_body(self, frame, bodies: List[BodyDet]) -> Optional[BodyDet]:
        """Carries the locked body box forward frame-to-frame by spatial continuity,
        blended with appearance similarity when available - the latter is what keeps
        the lock on the right person rather than a nearby one in a path crossing."""
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
            if self.appearance_signature is not None:
                sig = torso_signature(frame, box)
                score += 0.30 * compare_signatures(sig, self.appearance_signature)
            if score > best_score:
                best_score, best_body = score, body
        return best_body if best_score > -0.45 else None

    def _reinforce_or_hold(self, frame, candidate: BodyDet) -> None:
        """Call when the locked body is still continuity-tracked but no face confirmed it
        this frame. Appearance is a weaker, ongoing signal (not just a one-time rescue on
        continuity break) - without this, a sustained back-turn free-decays through
        TEMPORARY into UNKNOWN and then just sits there for as long as the person's back
        stays turned, even though the same tracked body keeps matching their clothing the
        whole time."""
        if self.appearance_signature is not None:
            sig = torso_signature(frame, self._box(candidate))
            if compare_signatures(sig, self.appearance_signature) >= self.appearance_threshold:
                self.identity.confirm_by_appearance()
                return
        self.identity.hold_without_face(body_continuity_ok=True)

    def _acquire_by_appearance(self, frame, bodies: List[BodyDet]) -> Optional[BodyDet]:
        """Re-acquires by torso/clothing appearance alone - no face required, so this
        is what lets the tracker find the target again even with their back turned."""
        if self.appearance_signature is None or not bodies:
            return None
        best_score, best_body = -1.0, None
        for body in bodies:
            sig = torso_signature(frame, self._box(body))
            score = compare_signatures(sig, self.appearance_signature)
            if score > best_score:
                best_score, best_body = score, body
        if best_body is not None and best_score >= self.appearance_threshold:
            logger.info("re-acquired %r by appearance: body=%s score=%.3f",
                        self.target_name, self._box(best_body), best_score)
            return best_body
        return None

    def _face_on_locked_body(self, faces: List[FaceDet], bodies: List[BodyDet], candidate_body: BodyDet) -> Optional[FaceDet]:
        face_bodies = associate_faces_to_bodies(faces, bodies)
        for face, body in zip(faces, face_bodies):
            if body is candidate_body:
                return face
        return None

    def _detect_bodies(self, frame) -> List[BodyDet]:
        """Full EfficientDet-Lite0 pass only every body_detect_every frames - see module
        docstring. Other frames reuse the last detections; continuity (_associate_body)
        tolerates a frame or two of staleness fine."""
        if self.frame_index == 1 or self.frame_index % self.body_detect_every == 0:
            self._cached_bodies = self.body_detector.detect(frame, max_people=6)
        return self._cached_bodies

    def update(self, frame) -> Optional[TrackResult]:
        self.frame_index += 1
        bodies = self._detect_bodies(frame)

        # phase-offset by 1 so periodic re-verification frames don't land on the same
        # frame as periodic body re-detection (both are expensive; stacking them on one
        # frame is what caused the visible stutter every body_detect_every*verify_every
        # frames - e.g. every 10th frame when body_detect_every=2, verify_every=10,
        # both anchored at 0)
        due = self.identity.state != IdentityState.CONFIRMED or self.frame_index % self.verify_every == 1
        face: Optional[FaceDet] = None

        if self.locked_body is None:
            # need identity on every candidate face to acquire, so this is always the
            # full (expensive) detect - throttling here would only slow down finding
            # the target in the first place, not the steady-state tracking loop
            faces = self.face_detector.detect(frame, max_faces=8)
            candidate = self._acquire(frame, faces, bodies)
            if candidate is not None:
                self.locked_body = candidate
                self.identity.confirm()
                face = self._face_on_locked_body(faces, bodies, candidate)
            else:
                # no face match this frame (e.g. target's back is turned) - fall back
                # to appearance alone, so the tracker can still find them
                candidate = self._acquire_by_appearance(frame, bodies)
                if candidate is None:
                    self.identity.tick_unknown()
                    return None
                self.locked_body = candidate
                self.identity.confirm_by_appearance()
        else:
            candidate = self._associate_body(frame, bodies)
            if candidate is None:
                # continuity broke (occlusion / path crossing) - try an immediate
                # appearance-based re-lock rather than waiting out the UNKNOWN timeout
                candidate = self._acquire_by_appearance(frame, bodies)
                if candidate is not None:
                    self.locked_body = candidate
                    self.identity.confirm_by_appearance()
                else:
                    self.identity.hold_without_face(body_continuity_ok=False)
                    self.identity.tick_unknown()
                    if self.identity.state == IdentityState.SEARCHING:
                        self.locked_body = None
                        self.smooth_center = None
                        return None
                    # briefly keep reporting the last known box while UNKNOWN, honestly
                    # flagged as such, instead of the output disappearing outright
            else:
                self.locked_body = candidate
                if due:
                    # full pass: need the 5pt landmarks to actually check identity
                    faces = self.face_detector.detect(frame, max_faces=8)
                    face = self._face_on_locked_body(faces, bodies, candidate)
                    if face is not None:
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
                            self._reinforce_or_hold(frame, candidate)
                    else:
                        self._reinforce_or_hold(frame, candidate)
                else:
                    # cheap pass: only need to know a face is there, not whose -
                    # the landmarker + ArcFace embedding cost is skipped entirely
                    face_boxes = self.face_detector.box_detector.detect(frame, max_faces=8)
                    face_bodies = associate_faces_to_bodies(face_boxes, bodies)
                    if not any(b is candidate for b in face_bodies):
                        self._reinforce_or_hold(frame, candidate)

        snapshot = self.identity.snapshot()
        box = self._box(self.locked_body)
        raw_center = center(box)
        self.smooth_center = raw_center if self.smooth_center is None else (
            self.ema_alpha * raw_center + (1.0 - self.ema_alpha) * self.smooth_center
        )
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
