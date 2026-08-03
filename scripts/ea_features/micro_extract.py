"""Micro-expression candidate features from face ROI motion peaks."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ea_features.face_utils import crop_largest_face, read_all_frames


FEATURE_DIM = 16
MODEL_NAME = "face_roi_framediff_opticalflow_v1"


class MicroExtractor:
    def __init__(self, max_frames: int = 48) -> None:
        self.max_frames = max_frames
        self.model_name = MODEL_NAME

    def extract(self, video_path: Path | None) -> tuple[np.ndarray, dict[str, Any]]:
        started = time.perf_counter()
        zero = np.zeros((FEATURE_DIM,), dtype=np.float32)
        if video_path is None or not Path(video_path).is_file():
            return zero, {
                "model": self.model_name,
                "status": "missing_video",
                "skipped": True,
                "candidate_score": 0.0,
                "elapsed_sec": round(time.perf_counter() - started, 4),
            }

        frames = read_all_frames(Path(video_path), self.max_frames)
        if len(frames) < 2:
            return zero, {
                "model": self.model_name,
                "status": "too_few_frames",
                "skipped": True,
                "candidate_score": 0.0,
                "elapsed_sec": round(time.perf_counter() - started, 4),
            }

        gray_faces: list[np.ndarray] = []
        for frame in frames:
            crop = crop_largest_face(frame)
            if crop is None:
                continue
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (64, 64))
            gray_faces.append(gray)

        if len(gray_faces) < 2:
            return zero, {
                "model": self.model_name,
                "status": "no_face",
                "skipped": True,
                "candidate_score": 0.0,
                "frames_sampled": len(frames),
                "elapsed_sec": round(time.perf_counter() - started, 4),
            }

        diffs = []
        flows = []
        for prev, cur in zip(gray_faces[:-1], gray_faces[1:]):
            diff = np.mean(np.abs(cur.astype(np.float32) - prev.astype(np.float32)))
            flow = cv2.calcOpticalFlowFarneback(
                prev, cur, None, 0.5, 3, 15, 3, 5, 1.2, 0
            )
            mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
            diffs.append(float(diff))
            flows.append(float(np.mean(mag)))

        diff_arr = np.asarray(diffs, dtype=np.float32)
        flow_arr = np.asarray(flows, dtype=np.float32)
        score_arr = 0.5 * _zscore(diff_arr) + 0.5 * _zscore(flow_arr)
        peak_idx = int(np.argmax(score_arr))
        candidate_score = float(score_arr[peak_idx])

        window = _window_slice(score_arr, peak_idx, radius=3)
        feat = np.zeros((FEATURE_DIM,), dtype=np.float32)
        feat[0] = float(np.mean(diff_arr))
        feat[1] = float(np.std(diff_arr))
        feat[2] = float(np.max(diff_arr))
        feat[3] = float(np.mean(flow_arr))
        feat[4] = float(np.std(flow_arr))
        feat[5] = float(np.max(flow_arr))
        feat[6] = candidate_score
        feat[7] = float(peak_idx) / max(len(score_arr) - 1, 1)
        feat[8 : 8 + min(len(window), 8)] = window[:8]

        return feat, {
            "model": self.model_name,
            "status": "ok",
            "skipped": False,
            "candidate_score": round(candidate_score, 6),
            "peak_frame_index": peak_idx,
            "frames_with_face": len(gray_faces),
            "frames_sampled": len(frames),
            "elapsed_sec": round(time.perf_counter() - started, 4),
        }

    def close(self) -> None:
        return None


def _zscore(arr: np.ndarray) -> np.ndarray:
    mean = float(np.mean(arr))
    std = float(np.std(arr))
    if std < 1e-6:
        return np.zeros_like(arr)
    return (arr - mean) / std


def _window_slice(arr: np.ndarray, center: int, radius: int) -> np.ndarray:
    left = max(center - radius, 0)
    right = min(center + radius + 1, len(arr))
    chunk = arr[left:right]
    out = np.zeros((2 * radius + 1,), dtype=np.float32)
    out[: chunk.size] = chunk
    return out
