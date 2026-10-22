# tracking user wholly

Whole-person tracking and face-association module: keeps a lock on one
enrolled person's *body*, not just their face, so identity continuity
survives head turns, brief occlusions, and other people crossing the
target's path. CPU-only, no extra hardware - just a webcam.

Builds on the identity-lock pattern in `../face-recognition` (detect ->
MediaPipe 5pt -> align -> ArcFace ONNX -> cosine match) and adds a second,
independent detector for the whole body, a spatial face-to-body
association step, and an honest three-state identity model instead of a
single locked/lost flag.

## Why whole-person, not just face

A face-only lock (see `../face-recognition/src/face_tracking.py`) reports
nothing the instant the face isn't legible - a head turn or a hand in front
of the mouth looks identical to the person having walked away. Tracking the
body box separately means the system can keep saying *where* the person
is during that gap, while being explicit about *how sure* it still is that
it's the right person.

## Detectors

This started on a Haar cascade (face) and OpenCV's HOG people detector
(body) - both are decades-old and weak: Haar stops detecting a face past
roughly 15-20 deg of yaw, and HOG fails often under partial occlusion or
non-upright poses. Both are now replaced with CPU-efficient MediaPipe
Tasks models, which is most of what actually makes "even though the user
turns back" work:

| Stage | Model | Role |
|---|---|---|
| Face box (whole frame, every frame) | BlazeFace (`FaceDetector` task) | Tolerant of much larger head turns than Haar; cheap (~9ms/frame), so it also doubles as the "is a face visible at all" check on frames that skip full re-verification. |
| Face 5pt (per detected ROI) | MediaPipe `FaceLandmarker` | Unchanged from the original build - gives the exact `(left eye, right eye, nose, mouth-left, mouth-right)` points ArcFace alignment needs; BlazeFace's own keypoints don't match that set. |
| Identity | ArcFace ONNX + cosine match | Unchanged. |
| Body box (whole frame) | EfficientDet-Lite0 int8 (`ObjectDetector` task, filtered to `person`) | Multi-person, robust to occlusion/pose, NMS handled internally. This is the single most expensive op per frame (~85ms on a laptop CPU - downscaling the input barely moves it, the cost is the model's own compute), so see **Performance** below. |

## Performance

Once a target is locked, `src/whole_person_tracker.py` stops running the
expensive passes on every frame:

- **Body detection** (EfficientDet-Lite0) only runs every
  `--body-detect-every` frames (default 2); other frames reuse the last
  boxes and lean on IoU+displacement continuity to carry the lock.
- **Face identity** (FaceLandmarker + ArcFace embedding) only runs when
  actually due for re-verification (`--verify-every`, default every 10
  frames, or whenever state isn't already `CONFIRMED`). Every other
  locked frame uses the cheap BlazeFace-only box pass just to answer "is
  a face visible there at all" - which is all the `CONFIRMED`/`TEMPORARY`
  transition needs most frames.

Measured on this machine: ~16-17fps steady-state while locked (up from
~1.5-9fps before this throttling, and far above the single-digit fps the
old Haar+HOG combo could sustain at comparable accuracy).

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

### Model assets

`models/` is git-ignored (the ArcFace embedder alone is 167MB, over
GitHub's 100MB limit) and needs four files fetched once:

```bash
mkdir -p models
curl -sL -o models/blaze_face_short_range.tflite \
  "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite"
curl -sL -o models/efficientdet_lite0.tflite -C - --max-time 300 \
  "https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/int8/latest/efficientdet_lite0.tflite"
# face_landmarker.task and embedder_arcface.onnx: copy from ../face-recognition/models/
# if you already have that sibling project, otherwise see its README for sources.
```

Verify sizes: `blaze_face_short_range.tflite` ~230KB,
`efficientdet_lite0.tflite` ~4.6MB exactly (a truncated/wrong-version
download of this one is the most common failure - MediaPipe rejects it
with `ValueError: The model is not a valid Flatbuffer buffer`; re-run the
`curl` with `-C -` to resume if it does).

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
`--threshold` (ArcFace accept threshold), `--face-detect-conf` /
`--person-detect-conf` (detector confidence thresholds), `--body-detect-every`
(how often the expensive person detector reruns - see **Performance**),
`--decay-rate` / `--temporary-floor` (tune how fast TEMPORARY decays and
when it drops to UNKNOWN), `--debug` (verbose per-frame logs). Press `q`
to quit.

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
on this machine, both before and after the Haar/HOG -> BlazeFace/EfficientDet
swap; `track_wholly` correctly reported `SEARCHING` throughout since no
identity was enrolled in those passes. Steady-state throughput measured at
~16-17fps after the throttling in **Performance** (headless `update()`
benchmark: 30 frames in 1.79s).

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
| `src/person_detector.py` | EfficientDet-Lite0 whole-body detection (MediaPipe ObjectDetector, filtered to `person`). |
| `src/face_detector.py` | BlazeFace whole-frame multi-face box detection (MediaPipe FaceDetector). |
| `src/face_pipeline.py` | `MultiFaceDetector` (BlazeFace + per-ROI FaceLandmarker 5pt) + ArcFace embed + DB match. |
| `src/align.py` | Detector-agnostic ArcFace 5pt alignment warp. |
| `src/association.py` | Maps each detected face to the body box it sits on. |
| `src/identity_state.py` | The three-state confidence model, decoupled from boxes/frames. |
| `src/whole_person_tracker.py` | Integrates the above: acquire, body-continuity tracking, throttled re-verification, centering signal. |
| `src/track_wholly.py` | CLI entry point: camera loop, overlay, CSV logging. |
| `src/evaluate.py` | Summarizes a logged run. |
| `src/enroll.py` | Enrolls a target identity into `data/db/face_db.npz`. |
