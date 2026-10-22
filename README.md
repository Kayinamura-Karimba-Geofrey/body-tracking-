# tracking user wholly

Whole-person tracking and face-association module: keeps a lock on one
enrolled person's *body*, not just their face, so identity continuity
survives head turns, brief occlusions, and other people crossing the
target's path. CPU-only, no extra hardware - just a webcam.

Builds on the identity-lock pattern in `../face-recognition` (Haar ->
MediaPipe 5pt -> align -> ArcFace ONNX -> cosine match) and adds a second,
independent detector for the whole body (OpenCV's built-in HOG people
detector), a spatial face-to-body association step, and an honest
three-state identity model instead of a single locked/lost flag.

## Why whole-person, not just face

A face-only lock (see `../face-recognition/src/face_tracking.py`) reports
nothing the instant the face isn't legible - a head turn or a hand in front
of the mouth looks identical to the person having walked away. Tracking the
body box separately means the system can keep saying *where* the person
is during that gap, while being explicit about *how sure* it still is that
it's the right person.

## Three-state identity model (`src/identity_state.py`)

| State | Meaning |
|---|---|
| `CONFIRMED` | ArcFace matched the target's face this frame (or within the last `--verify-every` frames). confidence = 1.0. |
| `TEMPORARY` | Face isn't currently matchable, but the body box is still tracked frame-to-frame by spatial continuity (IoU + displacement). confidence decays as `exp(-decay_rate * seconds_since_confirmed)`. |
| `UNKNOWN` | Confidence decayed below `--temporary-floor`, or body continuity itself broke (occluded too long, or lost in a path crossing). Only a fresh face match can re-establish `CONFIRMED`. |

If a *different* enrolled identity's face appears on the locked body (a
path crossing that actually swapped who's standing there), the lock is
dropped immediately rather than kept on the wrong person.

## Setup

```bash
uv venv
uv pip install -r requirements.txt
```

`uv pip install` checks what's already installed and reuses it - if you
already have an identical `.venv` from a sibling project (e.g.
`../face-recognition/.venv`), you can `cp -a ../face-recognition/.venv .venv`
first and this step finishes in milliseconds instead of re-downloading
opencv/mediapipe/onnxruntime.

**Do not use `uv run` or `uv sync` in this directory.** Both insist on a
venv built entirely through `uv`'s own install chain and will delete and
fully re-download `.venv` even when it already has everything required -
confirmed by re-downloading ~90MB from scratch against an already-satisfied,
copied venv. Activate the venv or call its interpreter directly instead:

```bash
source .venv/bin/activate   # then just `python -m src....`
# or, without activating:
.venv/bin/python -m src....
```

## Scripts

All commands below assume the venv is active (`source .venv/bin/activate`);
otherwise prefix each with `.venv/bin/` (e.g. `.venv/bin/python -m src.enroll`).
Each was run end-to-end against a live webcam (`/dev/video0`) and a real
`cv2.imshow` window during development - see **Verified working** below.

### 1. Enroll a target

```bash
python -m src.enroll
```

Interactive: type a name, aim the camera at them, `SPACE` to capture a
sample (or `a` to auto-capture), `s` to save once you have a handful of
samples, `q` to quit. Writes `data/db/face_db.npz`.

### 2. Track

```bash
python -m src.track_wholly --target "SomeEnrolledName" --log runs/session.csv
```

Opens the camera, overlays the locked body box, state, and confidence.
`--log` records per-frame state/confidence/position to CSV for later
analysis. Useful flags: `--camera N` (force a camera index),
`--threshold` (ArcFace accept threshold), `--decay-rate` /
`--temporary-floor` (tune how fast TEMPORARY decays and when it drops to
UNKNOWN), `--debug` (verbose per-frame logs). Press `q` to quit.

### 3. Evaluate a recorded run

```bash
python -m src.evaluate runs/session.csv
```

Prints time spent in each state, state transition counts, and how many
`TEMPORARY` episodes recovered to `CONFIRMED` vs. dropped to `UNKNOWN` -
use this to score the head-turn / occlusion / path-crossing test scenarios.

## Verified working

Run from this directory with the `.venv` described above:

```bash
.venv/bin/python -m py_compile src/*.py                          # all modules compile
.venv/bin/python -m src.track_wholly --help                       # CLI parses
.venv/bin/python -m src.track_wholly --target X --log runs/t.csv  # opens camera, detects, logs CSV, q to quit
.venv/bin/python -m src.evaluate runs/t.csv                       # summarizes the CSV
.venv/bin/python -m src.enroll                                    # opens camera, enroll UI
```

All five ran clean (no tracebacks) against the real webcam at `/dev/video0`
on this machine; `track_wholly` correctly reported `SEARCHING` throughout
since no identity was enrolled in that pass.

## Test scenarios (no hardware needed)

Run `track_wholly` with `--log`, then act out each scenario in front of the
webcam and check the resulting state sequence:

- **Head turn**: face the camera until `CONFIRMED`, turn your head away for
  a few seconds, turn back. Expect `CONFIRMED -> TEMPORARY -> CONFIRMED`
  with no `UNKNOWN` if you turn back before confidence crosses
  `--temporary-floor`.
- **Occlusion**: cover your face with a hand/object for several seconds.
  Expect the same `TEMPORARY` decay; if held long enough, `UNKNOWN`, then
  `CONFIRMED` again once your face is matched.
- **Path crossing**: have a second (ideally enrolled) person walk between
  the camera and the target. Expect the body lock to either survive via
  continuity (`TEMPORARY`) or drop to `UNKNOWN`/`SEARCHING` if the body
  detector loses the box - and never silently relabel the lock onto the
  second person.

## Module layout

| File | Role |
|---|---|
| `src/person_detector.py` | HOG whole-body detection (CPU-only, built into OpenCV). |
| `src/face_pipeline.py` | Multi-face detect + ArcFace embed + DB match (trimmed from `face-recognition/src/recognize.py`). |
| `src/association.py` | Maps each detected face to the body box it sits on. |
| `src/identity_state.py` | The three-state confidence model, decoupled from boxes/frames. |
| `src/whole_person_tracker.py` | Integrates the above: acquire, body-continuity tracking, re-verification, centering signal. |
| `src/track_wholly.py` | CLI entry point: camera loop, overlay, CSV logging. |
| `src/evaluate.py` | Summarizes a logged run. |
| `src/enroll.py` | Enrolls a target identity into `data/db/face_db.npz`. |
