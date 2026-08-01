"""Shared face ROI helpers using OpenCV cascades (no MediaPipe dependency)."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import cv2
import numpy as np


_CASCADE: cv2.CascadeClassifier | None = None


def _ascii_cascade_path(src: Path) -> Path:
    """OpenCV on Windows often fails on non-ASCII paths; copy to temp."""
    dst = Path(tempfile.gettempdir()) / "ea_haarcascade_frontalface_default.xml"
    if (not dst.is_file()) or dst.stat().st_size != src.stat().st_size:
        shutil.copyfile(src, dst)
    return dst


def get_face_cascade() -> cv2.CascadeClassifier:
    global _CASCADE
    if _CASCADE is None:
        candidates = [
            Path(__file__).resolve().parent / "data" / "haarcascade_frontalface_default.xml",
            Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml",
        ]
        cascade = None
        for path in candidates:
            if not path.is_file():
                continue
            load_path = _ascii_cascade_path(path)
            cascade = cv2.CascadeClassifier(str(load_path))
            if not cascade.empty():
                break
        if cascade is None or cascade.empty():
            raise RuntimeError(f"failed to load face cascade from {candidates}")
        _CASCADE = cascade
    return _CASCADE


def crop_largest_face(frame_bgr: np.ndarray) -> np.ndarray | None:
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    faces = get_face_cascade().detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=4, minSize=(40, 40)
    )
    if len(faces) == 0:
        return None
    x, y, w, h = max(faces, key=lambda box: box[2] * box[3])
    return frame_bgr[y : y + h, x : x + w]


def read_frames_sequential(video_path: Path, max_frames: int) -> list[np.ndarray]:
    """Read frames sequentially. Avoid CAP_PROP_POS_FRAMES seeks (can hang)."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []
    collected: list[np.ndarray] = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        collected.append(frame)
    cap.release()
    if not collected:
        return []
    if len(collected) <= max_frames:
        return collected
    idx = np.linspace(0, len(collected) - 1, max_frames).astype(int)
    return [collected[i] for i in idx]


sample_frames = read_frames_sequential
read_all_frames = read_frames_sequential
