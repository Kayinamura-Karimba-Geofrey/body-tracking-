# src/evaluate.py
"""
Summarizes a CSV log recorded by `python -m src.track_wholly --log <path>`
into state durations and transition counts - the basis for writing up the
head-turn / occlusion / path-crossing scenario results.

Run:
    python -m src.evaluate runs/session.csv
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path")
    args = parser.parse_args()

    rows = list(csv.DictReader(Path(args.csv_path).open()))
    if not rows:
        print("empty log")
        return

    state_seconds = Counter()
    transitions = Counter()
    prev_state = None
    prev_t = float(rows[0]["t"])
    recoveries = 0  # TEMPORARY -> CONFIRMED (face reacquired without ever hitting UNKNOWN)
    drops = 0       # -> UNKNOWN

    for row in rows:
        t = float(row["t"])
        state = row["state"]
        state_seconds[state] += t - prev_t
        if prev_state is not None and state != prev_state:
            transitions[f"{prev_state}->{state}"] += 1
            if prev_state == "TEMPORARY" and state == "CONFIRMED":
                recoveries += 1
            if state == "UNKNOWN":
                drops += 1
        prev_state, prev_t = state, t

    duration = float(rows[-1]["t"]) - float(rows[0]["t"])
    print(f"log: {args.csv_path}  frames={len(rows)}  duration={duration:.1f}s\n")
    print("time in each state:")
    for state, secs in state_seconds.most_common():
        print(f"  {state:10s} {secs:6.1f}s  ({100 * secs / duration:5.1f}%)")
    print("\nstate transitions:")
    for name, count in transitions.most_common():
        print(f"  {name:28s} x{count}")
    print(f"\nTEMPORARY recovered to CONFIRMED (no UNKNOWN gap): {recoveries}")
    print(f"dropped to UNKNOWN: {drops}")


if __name__ == "__main__":
    main()
