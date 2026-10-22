# src/association.py
"""
Spatial face-to-body mapping: given faces from face_pipeline and bodies from
person_detector (two independent detectors, no shared IDs), decide which
body box each face belongs to. A face belongs to a body when it sits inside
the body box's top slice, roughly centered on the body's width - this holds
across head turns and is cheap, so it runs every frame on every pair.
"""

from __future__ import annotations

from typing import List, Optional

from .face_pipeline import FaceDet
from .person_detector import BodyDet


def face_in_body_head_region(face: FaceDet, body: BodyDet, top_frac: float = 0.35, x_margin_frac: float = 0.25) -> bool:
    bw, bh = body.x2 - body.x1, body.y2 - body.y1
    if bw <= 0 or bh <= 0:
        return False

    fcx = (face.x1 + face.x2) / 2.0
    fcy = (face.y1 + face.y2) / 2.0

    x_lo = body.x1 - x_margin_frac * bw
    x_hi = body.x2 + x_margin_frac * bw
    y_hi = body.y1 + top_frac * bh  # head must be in the upper slice of the body box

    return x_lo <= fcx <= x_hi and body.y1 - 0.15 * bh <= fcy <= y_hi


def best_body_for_face(face: FaceDet, bodies: List[BodyDet]) -> Optional[BodyDet]:
    candidates = [b for b in bodies if face_in_body_head_region(face, b)]
    if not candidates:
        return None
    # smallest body box among candidates: the tightest-fitting match, so a
    # face isn't stolen by a larger body box standing further behind it
    return min(candidates, key=lambda b: (b.x2 - b.x1) * (b.y2 - b.y1))


def associate_faces_to_bodies(faces: List[FaceDet], bodies: List[BodyDet]) -> List[Optional[BodyDet]]:
    """Returns, for each face (same order/index as `faces`), its matched body or None."""
    return [best_body_for_face(face, bodies) for face in faces]
