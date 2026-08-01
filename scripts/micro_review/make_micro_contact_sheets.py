#!/usr/bin/env python
"""Generate micro-expression review contact sheets for M1 samples.

For each sample, locate the micro candidate peak frame (from the L2 micro
feature metadata) and render a window of frames around it so a human
reviewer can confirm whether a micro-expression is actually present.

Layout per sample:
  - top strip:  zoomed face ROI for each frame in the peak window (this is
                 where subtle, brief facial movements are visible)
  - bottom grid: full frames for the same window (scene/context)
  - peak frame is highlighted in red.

The peak_frame_index stored in the L2 micro JSON is an index into the
face-present sampled frames (the L2 extractor computes frame diffs only over
faces). This script re-runs the same sampling + face detection so it can map
that index back to the true video frame number and timestamps.

Usage:
    python scripts/micro_review/make_micro_contact_sheets.py \
        --index source_index/m1_sample_20.csv \
        --videos-dir .work/m1_videos \
        --micro-dir .work/meta_micro \
        --out-dir .work/contact_sheets
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np

try:
    from ea_features.face_utils import get_face_cascade
    _SYS_PATH_READY = True
except ModuleNotFoundError:
    _SYS_PATH_READY = False


def load_cascade() -> cv2.CascadeClassifier:
    if _SYS_PATH_READY:
        return get_face_cascade()
    candidates = [
        Path(__file__).resolve().parents[1]
        / "ea_features"
        / "data"
        / "haarcascade_frontalface_default.xml",
        Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml",
    ]
    for path in candidates:
        if path.is_file():
            cascade = cv2.CascadeClassifier(str(path))
            if not cascade.empty():
                return cascade
    raise RuntimeError("face cascade not found")


def crop_largest_face(cascade, frame_bgr):
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(40, 40))
    if len(faces) == 0:
        return None
    x, y, w, h = max(faces, key=lambda box: box[2] * box[3])
    return frame_bgr[y : y + h, x : x + w]


def sample_frames(video_path: Path, max_frames: int):
    """Read frames sequentially; sample down to max_frames like the L2 extractor."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    collected = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        collected.append(frame)
    cap.release()
    if not collected:
        raise RuntimeError(f"no frames in video: {video_path}")
    if len(collected) <= max_frames:
        return collected, fps, list(range(len(collected)))
    idx = np.linspace(0, len(collected) - 1, max_frames).astype(int)
    return [collected[i] for i in idx], fps, [int(i) for i in idx]


def find_peak_face_frame(cascade, frames, frame_nums, peak_idx):
    """Map L2 peak index -> sampled index and true video frame number."""
    face_frames = []
    for i, frame in enumerate(frames):
        if crop_largest_face(cascade, frame) is not None:
            face_frames.append((i, frame_nums[i]))
    if not face_frames:
        return None, None, []
    if peak_idx >= len(face_frames):
        peak_idx = len(face_frames) - 1
    peak_sample_idx, peak_true_frame = face_frames[peak_idx]
    return peak_sample_idx, peak_true_frame, face_frames


def draw_strip(cascade, frames, frame_nums, win, peak_sample_idx, out_path: Path) -> None:
    """Top strip: zoomed face ROIs across the window."""
    cell_w, cell_h = 200, 200
    n = len(win)
    strip = np.full((cell_h + 30, cell_w * n, 3), 245, dtype=np.uint8)
    for j, gi in enumerate(win):
        frame = frames[gi]
        crop = crop_largest_face(cascade, frame)
        if crop is None:
            tile = np.full((cell_h, cell_w, 3), 120, dtype=np.uint8)
            cv2.putText(
                tile,
                "no face",
                (30, cell_h // 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                1,
            )
        else:
            tile = cv2.resize(crop, (cell_w, cell_h), interpolation=cv2.INTER_LINEAR)
        if gi == peak_sample_idx:
            cv2.rectangle(tile, (0, 0), (cell_w - 1, cell_h - 1), (0, 0, 255), 6)
        x = j * cell_w
        strip[0:cell_h, x : x + cell_w] = tile
        cv2.putText(
            strip,
            f"#{frame_nums[gi]}",
            (x + 4, cell_h + 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (40, 40, 40),
            1,
        )
    cv2.imwrite(str(out_path), strip)


def draw_grid(cascade, frames, frame_nums, win, peak_sample_idx, out_path: Path) -> None:
    """Bottom grid: full frames across the window for context."""
    thumb_w, thumb_h = 160, 90
    n = len(win)
    cols = min(6, n)
    rows = int(np.ceil(n / cols))
    grid = np.full((rows * (thumb_h + 26), cols * (thumb_w + 6), 3), 255, dtype=np.uint8)
    for j, gi in enumerate(win):
        frame = frames[gi]
        tile = cv2.resize(frame, (thumb_w, thumb_h), interpolation=cv2.INTER_AREA)
        r, c = divmod(j, cols)
        y = r * (thumb_h + 26)
        x = c * (thumb_w + 6)
        if gi == peak_sample_idx:
            border = (0, 0, 255)
            cv2.putText(tile, "PEAK", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        else:
            border = (120, 120, 120)
        grid[y + 3 : y + 3 + thumb_h, x + 3 : x + 3 + thumb_w] = tile
        cv2.rectangle(
            grid,
            (x + 3, y + 3),
            (x + 3 + thumb_w - 1, y + 3 + thumb_h - 1),
            border,
            1,
        )
        cv2.putText(
            grid,
            f"#{frame_nums[gi]}",
            (x + 8, y + thumb_h + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (60, 60, 60),
            1,
        )
    cv2.imwrite(str(out_path), grid)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", required=True, type=Path)
    parser.add_argument("--videos-dir", required=True, type=Path)
    parser.add_argument("--micro-dir", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--radius", type=int, default=5)
    args = parser.parse_args()

    cascade = load_cascade()

    rows = {}
    with args.index.open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows[r["ea_id"]] = r

    micro_meta = {}
    for f in sorted(args.micro_dir.glob("*.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        micro_meta[d["ea_id"]] = d

    args.out_dir.mkdir(parents=True, exist_ok=True)

    summary = {}
    for ea_id in sorted(rows):
        row = rows[ea_id]
        meta = micro_meta.get(ea_id)
        if meta is None:
            print(f"{ea_id}: no micro meta, skipped")
            continue
        peak_idx = meta.get("peak_frame_index", 0)

        # Rebuild video path the same way as .work/sample_video_map.json
        if row["source_dataset"] == "CH-SIMS":
            vid = row["source_id"].split("/")[-1] + ".mp4"
            rel = args.videos_dir / "ch_sims" / vid
        else:
            parts = row["source_id"].split("/")
            base = f'{parts[-2]}_{parts[-1]}.mp4'
            split_dir = "meld_test" if "test" in row["source_split"] else "meld_train"
            rel = args.videos_dir / split_dir / base

        if not rel.is_file():
            print(f"{ea_id}: missing video {rel}")
            continue

        frames, fps, frame_nums = sample_frames(rel, 48)
        peak_sample_idx, peak_true_frame, face_nums = find_peak_face_frame(
            cascade, frames, frame_nums, peak_idx
        )

        if peak_sample_idx is None:
            print(f"{ea_id}: no face detected, skipped")
            continue

        lo = max(0, peak_sample_idx - args.radius)
        hi = min(len(frames) - 1, peak_sample_idx + args.radius)
        win = list(range(lo, hi + 1))

        strip_path = args.out_dir / f"{ea_id}_face_strip.png"
        grid_path = args.out_dir / f"{ea_id}_frame_grid.png"
        draw_strip(cascade, frames, frame_nums, win, peak_sample_idx, strip_path)
        draw_grid(cascade, frames, frame_nums, win, peak_sample_idx, grid_path)

        peak_sec = peak_true_frame / fps if fps else float("nan")
        summary[ea_id] = {
            "peak_frame_index": peak_idx,
            "peak_sampled_frame_index": peak_sample_idx,
            "peak_true_frame": peak_true_frame,
            "peak_sec": round(peak_sec, 3),
            "fps": round(fps, 3),
            "window_frames": [f"{frame_nums[i]}:{frame_nums[i]/fps:.2f}s" for i in win],
            "candidate_score": meta.get("candidate_score"),
            "frames_with_face": meta.get("frames_with_face"),
            "peak_true_frame_index_in_video": peak_true_frame,
        }
        print(
            f"{ea_id}: peak L2 idx={peak_idx} sampled_frame={peak_sample_idx} "
            f"true_frame={peak_true_frame} t={peak_sec:.2f}s score={meta.get('candidate_score')}"
        )

    out_json = args.out_dir / "_peak_map.json"
    out_json.write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(f"\nwritten {len(summary)} contact sheets to {args.out_dir}")


if __name__ == "__main__":
    sys.exit(main())
