# src/enroll.py
"""
Enrollment: camera -> MultiFaceDetector (BlazeFace + FaceLandmarker 5pt) ->
align_face_5pt (112x112) -> ArcFace embedding. Stores the mean L2-normalized
embedding per identity in data/db/face_db.npz, read by WholePersonTracker's
FaceDBMatcher (src/face_pipeline.py).

Run (interactive):
    python -m src.enroll

Controls:
    SPACE : capture one sample
    a     : toggle auto-capture
    s     : save enrollment (needs a handful of samples)
    q     : quit

Run (non-interactive, e.g. scripted/headless):
    python -m src.enroll --name Alice --duration 10
Captures automatically for --duration seconds and saves at the end if
enough samples were found - no keyboard input needed.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np

from .align import align_face_5pt
from .camera import pick_camera_index
from .face_pipeline import ArcFaceEmbedderONNX, MultiFaceDetector, load_db_npz

DB_PATH = Path("data/db/face_db.npz")
SAMPLES_NEEDED = 15


def mean_embedding(embeddings: List[np.ndarray]) -> np.ndarray:
    m = np.stack(embeddings, axis=0).mean(axis=0)
    return (m / (np.linalg.norm(m) + 1e-12)).astype(np.float32)


def save_db(db: Dict[str, np.ndarray]) -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez(DB_PATH, **{k: v.astype(np.float32) for k, v in db.items()})


def _open_camera() -> cv2.VideoCapture:
    cap = cv2.VideoCapture(pick_camera_index())
    if not cap.isOpened():
        raise RuntimeError("Failed to open camera.")
    return cap


def run_interactive(name: str) -> None:
    det = MultiFaceDetector()
    embedder = ArcFaceEmbedderONNX()
    db = load_db_npz(DB_PATH)
    samples: List[np.ndarray] = []
    auto = False
    last_auto = 0.0

    cap = _open_camera()
    print("Controls: SPACE=capture, a=auto-capture, s=save, q=quit")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            vis = frame.copy()
            faces = det.detect(frame, max_faces=1)
            aligned = None
            if faces:
                f = faces[0]
                cv2.rectangle(vis, (f.x1, f.y1), (f.x2, f.y2), (0, 255, 0), 2)
                aligned, _ = align_face_5pt(frame, f.kps, out_size=(112, 112))

            now = time.time()
            if auto and aligned is not None and now - last_auto >= 0.25:
                samples.append(embedder.embed(aligned))
                last_auto = now

            cv2.putText(vis, f"{name}: {len(samples)}/{SAMPLES_NEEDED} samples "
                             f"(auto={'ON' if auto else 'OFF'})",
                        (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
            cv2.imshow("enroll", vis)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("a"):
                auto = not auto
            elif key == ord(" ") and aligned is not None:
                samples.append(embedder.embed(aligned))
                print(f"captured ({len(samples)})")
            elif key == ord("s"):
                if len(samples) < max(3, SAMPLES_NEEDED // 2):
                    print(f"not enough samples yet ({len(samples)})")
                    continue
                db[name] = mean_embedding(samples)
                save_db(db)
                print(f"saved '{name}' to {DB_PATH} ({len(db)} identities total)")
                samples.clear()
    finally:
        det.close()
        cap.release()
        cv2.destroyAllWindows()


def run_auto(name: str, duration: float, capture_every: float = 0.25) -> None:
    det = MultiFaceDetector()
    embedder = ArcFaceEmbedderONNX()
    db = load_db_npz(DB_PATH)
    samples: List[np.ndarray] = []
    last_capture = 0.0
    frames_seen = 0
    frames_with_face = 0

    cap = _open_camera()
    print(f"Auto-enrolling '{name}' for {duration:.0f}s - face the camera now.")
    try:
        deadline = time.time() + duration
        while time.time() < deadline:
            ok, frame = cap.read()
            if not ok:
                break
            frames_seen += 1

            faces = det.detect(frame, max_faces=1)
            if faces:
                frames_with_face += 1
                aligned, _ = align_face_5pt(frame, faces[0].kps, out_size=(112, 112))
                now = time.time()
                if aligned is not None and now - last_capture >= capture_every:
                    samples.append(embedder.embed(aligned))
                    last_capture = now
    finally:
        det.close()
        cap.release()

    print(f"frames_seen={frames_seen} frames_with_face={frames_with_face} samples={len(samples)}")
    if len(samples) < max(3, SAMPLES_NEEDED // 2):
        print(f"not enough samples captured ({len(samples)}) - no face was detected for most of "
              f"the window. Make sure you're facing the camera and well lit, then try again.")
        return

    db[name] = mean_embedding(samples)
    save_db(db)
    print(f"saved '{name}' to {DB_PATH} ({len(db)} identities total, {len(samples)} samples averaged)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default=None, help="identity to enroll; omit for interactive mode")
    parser.add_argument("--duration", type=float, default=10.0, help="auto-capture window in seconds (--name mode only)")
    args = parser.parse_args()

    if args.name:
        run_auto(args.name, args.duration)
    else:
        name = input("Enter person name to enroll (e.g., Alice): ").strip()
        if not name:
            print("No name provided. Exiting.")
            return
        run_interactive(name)


if __name__ == "__main__":
    main()
