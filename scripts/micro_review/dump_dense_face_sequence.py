#!/usr/bin/env python
"""Dense per-frame face crops around peaks for borderline samples.

Renders a wide-but-dense sequence of full-face crops (every frame, not sampled)
around each sample's peak, so the reviewer can trace subtle frame-to-frame
changes without temporal downsampling hiding motion.

Usage:
    python scripts/micro_review/dump_dense_face_sequence.py [EAQ... IDs or ALL]
Output:
    .work/contact_sheets/dense/<ea_id>/<nnnn>_<tsec>s.png
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def load_cascade():
    candidates = [
        ROOT / "scripts" / "ea_features" / "data" / "haarcascade_frontalface_default.xml",
        Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml",
    ]
    for path in candidates:
        if path.is_file():
            cascade = cv2.CascadeClassifier(str(path))
            if not cascade.empty():
                return cascade
    raise RuntimeError("face cascade not found")


def main() -> None:
    ids = sys.argv[1:] if len(sys.argv) > 1 else ["ALL"]
    csv_path = ROOT / "source_index" / "m1_sample_20.csv"
    videos_dir = ROOT / ".work" / "m1_videos"
    micro_dir = ROOT / ".work" / "meta_micro"
    out_root = ROOT / ".work" / "contact_sheets" / "dense"

    cascade = load_cascade()
    rows = {}
    with csv_path.open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows[r["ea_id"]] = r
    micro_meta = {}
    for f in sorted(micro_dir.glob("*.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        micro_meta[d["ea_id"]] = d

    for ea_id in sorted(rows):
        if "ALL" not in ids and ea_id not in ids:
            continue
        row = rows[ea_id]
        meta = micro_meta.get(ea_id)
        if meta is None:
            continue
        peak_idx = meta.get("peak_frame_index", 0)

        if row["source_dataset"] == "CH-SIMS":
            vid = row["source_id"].split("/")[-1] + ".mp4"
            rel = videos_dir / "ch_sims" / vid
        else:
            parts = row["source_id"].split("/")
            base = f'{parts[-2]}_{parts[-1]}.mp4'
            rel = videos_dir / ("meld_test" if "test" in row["source_split"] else "meld_train") / base
        if not rel.is_file():
            print(f"{ea_id}: missing video")
            continue

        # Read ALL frames (no sampling) so the dense trace is exact.
        cap = cv2.VideoCapture(str(rel))
        if not cap.isOpened():
            print(f"{ea_id}: cannot open")
            continue
        fps = cap.get(cv2.CAP_PROP_FPS)
        all_frames = []
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            all_frames.append(fr)
        cap.release()

        # Map L2 peak idx -> true frame via face-present sampling (same as L2).
        max_frames = 48
        if len(all_frames) > max_frames:
            s_idx = np.linspace(0, len(all_frames) - 1, max_frames).astype(int)
            sampled = [all_frames[i] for i in s_idx]
        else:
            sampled = all_frames
        face_nums = []
        for i, fr in enumerate(sampled):
            gray = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
            faces = cascade.detectMultiScale(gray, 1.1, 4, minSize=(40, 40))
            if len(faces):
                face_nums.append(i)
        if not face_nums:
            print(f"{ea_id}: no face")
            continue
        if peak_idx >= len(face_nums):
            peak_idx = len(face_nums) - 1
        peak_true = face_nums[peak_idx]

        # Find peak face box
        peak_frame = all_frames[peak_true]
        gray = cv2.cvtColor(peak_frame, cv2.COLOR_BGR2GRAY)
        faces = cascade.detectMultiScale(gray, 1.1, 4, minSize=(40, 40))
        if len(faces) == 0:
            print(f"{ea_id}: no face at peak")
            continue
        fx, fy, fw, fh = max(faces, key=lambda b: b[2] * b[3])
        pad = 0.15
        x0 = max(0, int(fx - pad * fw)); y0 = max(0, int(fy - pad * fh))
        x1 = min(len(peak_frame[0]), int(fx + fw + pad * fw)); y1 = min(len(peak_frame), int(fy + fh + pad * fh))

        out_dir = out_root / ea_id
        out_dir.mkdir(parents=True, exist_ok=True)
        lo = max(0, peak_true - 4)
        hi = min(len(all_frames) - 1, peak_true + 4)
        for gi in range(lo, hi + 1):
            crop = all_frames[gi][y0:y1, x0:x1]
            crop = cv2.resize(crop, (400, 400), interpolation=cv2.INTER_CUBIC)
            cv2.putText(crop, f"{gi} {gi/fps:.3f}s", (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255) if gi == peak_true else (0, 255, 0), 2)
            cv2.imwrite(str(out_dir / f"{gi:04d}_{gi/fps:.3f}s.png"), crop)
        print(f"{ea_id}: peak_true={peak_true} t={peak_true/fps:.3f}s dense={hi-lo+1} frames")


if __name__ == "__main__":
    sys.exit(main())
