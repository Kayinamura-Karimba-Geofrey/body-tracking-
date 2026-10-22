# src/identity_state.py
"""
Honest three-state identity model for whole-person tracking.

CONFIRMED  - the locked body's head currently shows a face that ArcFace just
             matched to the target. confidence = 1.0.
TEMPORARY  - the face is gone (head turn, occlusion, looking away) but the
             body box is still being carried forward frame-to-frame by
             spatial continuity (IoU + displacement, see associate_body
             below). confidence decays exponentially from the moment the
             face was last confirmed - this module never pretends a
             TEMPORARY identity is as certain as a CONFIRMED one.
UNKNOWN    - confidence has decayed past the floor, or body continuity
             itself broke (occluded too long, or lost to a path crossing).
             The system is honest that it no longer knows who, if anyone,
             the locked box belongs to; only a fresh face match can
             re-establish CONFIRMED.

A body can also be re-acquired by clothing/torso appearance alone (see
src/appearance.py) when no face is visible at all - most importantly
someone whose back is turned, who by definition has no face to check.
That's weaker evidence than a face match, so confirm_by_appearance()
always lands in TEMPORARY, decaying like any other TEMPORARY state;
appearance alone can never produce CONFIRMED.

decay: confidence(t) = exp(-decay_rate * seconds_since_confirmed)
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


class IdentityState(Enum):
    SEARCHING = auto()  # no target locked at all yet / reacquiring after UNKNOWN
    CONFIRMED = auto()
    TEMPORARY = auto()
    UNKNOWN = auto()


@dataclass
class IdentitySnapshot:
    state: IdentityState
    confidence: float
    seconds_since_confirmed: float


class IdentityConfidence:
    """
    Tracks confidence for one locked target across confirmations and gaps.
    Pure state/math - carries no boxes or frames, so it is easy to unit test
    and to drive from recorded timestamps in the scenario evaluator.
    """

    def __init__(
        self,
        decay_rate: float = 0.35,     # per second; ~2s to decay to ~0.5
        temporary_floor: float = 0.35,  # below this, TEMPORARY -> UNKNOWN
        unknown_timeout_s: float = 8.0,  # UNKNOWN this long without reacquiring -> fully drop (SEARCHING)
    ):
        self.decay_rate = decay_rate
        self.temporary_floor = temporary_floor
        self.unknown_timeout_s = unknown_timeout_s

        self.state = IdentityState.SEARCHING
        self._confirmed_at: Optional[float] = None
        self._unknown_at: Optional[float] = None

    def confirm(self, now: Optional[float] = None) -> None:
        """Call when ArcFace just matched the target to the locked body's face."""
        now = now if now is not None else time.time()
        if self.state != IdentityState.CONFIRMED:
            logger.info("identity CONFIRMED (was %s)", self.state.name)
        self.state = IdentityState.CONFIRMED
        self._confirmed_at = now
        self._unknown_at = None

    def confirm_by_appearance(self, confidence: float = 0.6, now: Optional[float] = None) -> None:
        """
        Call when a body was re-acquired by torso/clothing appearance alone
        (no face available to check - e.g. back turned). Lands in TEMPORARY
        at the given starting confidence, which then decays normally; never
        CONFIRMED, since appearance alone is weaker evidence than a face match.
        """
        now = now if now is not None else time.time()
        confidence = min(max(confidence, 1e-6), 0.999)
        elapsed = -math.log(confidence) / self.decay_rate
        self._confirmed_at = now - elapsed
        self._unknown_at = None
        if self.state != IdentityState.TEMPORARY:
            logger.info("identity TEMPORARY (reacquired by appearance, confidence=%.2f)", confidence)
        self.state = IdentityState.TEMPORARY

    def hold_without_face(self, body_continuity_ok: bool, now: Optional[float] = None) -> IdentitySnapshot:
        """
        Call every frame the locked body is on-screen but its face couldn't
        be matched this frame (turned away, occluded, or just not due for
        re-verification). body_continuity_ok is False once spatial tracking
        itself lost the body box (occluded too long / crossed paths).
        """
        now = now if now is not None else time.time()

        if self._confirmed_at is None:
            # never actually confirmed anything - nothing to decay from
            return self.snapshot(now)

        elapsed = now - self._confirmed_at
        confidence = math.exp(-self.decay_rate * elapsed)

        if not body_continuity_ok:
            self._drop_to_unknown(now, reason="body continuity lost")
        elif confidence < self.temporary_floor:
            self._drop_to_unknown(now, reason=f"confidence decayed below floor ({confidence:.2f})")
        elif self.state == IdentityState.CONFIRMED:
            logger.info("identity TEMPORARY (face gone, confidence=%.2f)", confidence)
            self.state = IdentityState.TEMPORARY

        return self.snapshot(now)

    def _drop_to_unknown(self, now: float, reason: str) -> None:
        if self.state != IdentityState.UNKNOWN:
            logger.info("identity UNKNOWN (%s)", reason)
            self.state = IdentityState.UNKNOWN
            self._unknown_at = now

    def tick_unknown(self, now: Optional[float] = None) -> IdentityState:
        """Call every frame while UNKNOWN and no reacquisition has happened yet."""
        now = now if now is not None else time.time()
        if self.state == IdentityState.UNKNOWN and self._unknown_at is not None:
            if now - self._unknown_at > self.unknown_timeout_s:
                logger.info("identity dropped to SEARCHING (unknown for > %.1fs)", self.unknown_timeout_s)
                self.reset()
        return self.state

    def reset(self) -> None:
        self.state = IdentityState.SEARCHING
        self._confirmed_at = None
        self._unknown_at = None

    def snapshot(self, now: Optional[float] = None) -> IdentitySnapshot:
        now = now if now is not None else time.time()
        if self._confirmed_at is None:
            return IdentitySnapshot(self.state, 0.0, 0.0)
        elapsed = now - self._confirmed_at
        return IdentitySnapshot(self.state, math.exp(-self.decay_rate * elapsed), elapsed)
