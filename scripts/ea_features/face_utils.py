"""Shared face ROI helpers using OpenCV cascades (no MediaPipe dependency)."""

from __future__ import annotations

import shutil
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

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


def detect_largest_face(frame_bgr: np.ndarray) -> tuple[int, int, int, int] | None:
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    faces = get_face_cascade().detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=4, minSize=(40, 40)
    )
    if len(faces) == 0:
        return None
    x, y, w, h = max(faces, key=lambda box: int(box[2]) * int(box[3]))
    return int(x), int(y), int(w), int(h)


def crop_largest_face(frame_bgr: np.ndarray) -> np.ndarray | None:
    face = detect_largest_face(frame_bgr)
    if face is None:
        return None
    x, y, w, h = face
    return frame_bgr[y : y + h, x : x + w]


@dataclass(frozen=True)
class FaceQuality:
    frames_sampled: int
    frames_with_face: int
    face_detect_rate: float
    mean_face_ratio: float
    usable_for_micro: bool
    filter_reason: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def assess_face_quality(
    video_path: Path | None,
    *,
    max_frames: int = 12,
    min_detect_rate: float = 0.5,
    min_face_ratio: float = 0.01,
) -> FaceQuality:
    """Measure face visibility before expensive micro extraction.

    ``mean_face_ratio`` is the detected face bounding-box area divided by the
    frame area. A sample must satisfy both configured thresholds.
    """
    if video_path is None or not Path(video_path).is_file():
        return FaceQuality(0, 0, 0.0, 0.0, False, "missing_video")

    frames = read_frames_sequential(Path(video_path), max_frames=max_frames)
    if not frames:
        return FaceQuality(0, 0, 0.0, 0.0, False, "no_frames")

    ratios: list[float] = []
    for frame in frames:
        face = detect_largest_face(frame)
        if face is None:
            continue
        _, _, width, height = face
        frame_height, frame_width = frame.shape[:2]
        ratios.append((width * height) / max(frame_width * frame_height, 1))

    detect_rate = len(ratios) / len(frames)
    mean_ratio = float(np.mean(ratios)) if ratios else 0.0
    reasons = []
    if detect_rate < min_detect_rate:
        reasons.append("low_face_detect_rate")
    if mean_ratio < min_face_ratio:
        reasons.append("small_face")
    usable = not reasons
    return FaceQuality(
        frames_sampled=len(frames),
        frames_with_face=len(ratios),
        face_detect_rate=round(detect_rate, 6),
        mean_face_ratio=round(mean_ratio, 6),
        usable_for_micro=usable,
        filter_reason="ok" if usable else ";".join(reasons),
    )


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
