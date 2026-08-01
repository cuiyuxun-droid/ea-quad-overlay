#!/usr/bin/env python
"""Generate zoomed facial-region crops (eyes/brows, mouth) around micro peaks.

Helps a human reviewer make AU-level judgments: renders enlarged patches of the
eye+brow region and the mouth region for the peak frame +/- 2 frames, per sample.

Output: .work/contact_sheets/zoom/<ea_id>_zoom.png (rows: frames, cols: [eyes, mouth])
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


def sample_frames(video_path: Path, max_frames: int):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    collected = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        collected.append(frame)
    cap.release()
    if len(collected) <= max_frames:
        return collected, fps
    idx = np.linspace(0, len(collected) - 1, max_frames).astype(int)
    return [collected[i] for i in idx], fps


def main() -> None:
    csv_path = ROOT / "source_index" / "m1_sample_20.csv"
    videos_dir = ROOT / ".work" / "m1_videos"
    micro_dir = ROOT / ".work" / "meta_micro"
    out_dir = ROOT / ".work" / "contact_sheets" / "zoom"

    cascade = load_cascade()
    rows = {}
    with csv_path.open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows[r["ea_id"]] = r

    micro_meta = {}
    for f in sorted(micro_dir.glob("*.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        micro_meta[d["ea_id"]] = d

    out_dir.mkdir(parents=True, exist_ok=True)

    for ea_id in sorted(rows):
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
            continue

        frames, fps = sample_frames(rel, 48)
        # Recompute face-present sampled indices to map peak_idx -> true frame
        face_nums = []
        for i, fr in enumerate(frames):
            gray = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
            faces = cascade.detectMultiScale(gray, 1.1, 4, minSize=(40, 40))
            if len(faces):
                face_nums.append(i)
        if not face_nums:
            continue
        if peak_idx >= len(face_nums):
            peak_idx = len(face_nums) - 1
        peak_true = face_nums[peak_idx]

        rows_imgs = []
        for gi in range(max(0, peak_true - 2), min(len(frames), peak_true + 3)):
            frame = frames[gi]
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = cascade.detectMultiScale(gray, 1.1, 4, minSize=(40, 40))
            if len(faces) == 0:
                continue
            x, y, w, h = max(faces, key=lambda b: b[2] * b[3])
            # eyes/brows: upper third; mouth: lower third
            eye = frame[y: y + int(h * 0.42), x + int(w * 0.12): x + w - int(w * 0.12)]
            mouth = frame[y + int(h * 0.55): y + int(h * 0.95), x + int(w * 0.12): x + w - int(w * 0.12)]
            eye = cv2.resize(eye, (320, 200), interpolation=cv2.INTER_CUBIC)
            mouth = cv2.resize(mouth, (320, 200), interpolation=cv2.INTER_CUBIC)
            tag = "PEAK" if gi == peak_true else f"#{gi}"
            for img in (eye, mouth):
                cv2.putText(img, tag, (6, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255) if gi == peak_true else (0, 0, 0), 2)
            rows_imgs.append(np.hstack([eye, mouth]))

        if not rows_imgs:
            continue
        canvas = np.vstack(rows_imgs)
        cv2.imwrite(str(out_dir / f"{ea_id}_zoom.png"), canvas)
        print(f"{ea_id}: peak_true={peak_true} t={peak_true/fps:.3f}s frames={len(rows_imgs)}")


if __name__ == "__main__":
    sys.exit(main())
