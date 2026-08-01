"""Feature IO helpers for npy + sidecar JSON meta."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
FEATURE_ROOT = ROOT / "features"
MODALITIES = ("text", "speech", "macro", "micro")


def segment_id(ea_id: str, seg: int = 1) -> str:
    return f"{ea_id}_seg{seg:03d}"


def feature_stem(ea_id: str, modality: str, seg: int = 1) -> str:
    return f"{segment_id(ea_id, seg)}_{modality}"


def feature_paths(ea_id: str, modality: str, seg: int = 1) -> tuple[Path, Path]:
    stem = feature_stem(ea_id, modality, seg)
    base = FEATURE_ROOT / modality
    return base / f"{stem}.npy", base / f"{stem}.json"


def save_feature(
    ea_id: str,
    modality: str,
    vector: np.ndarray,
    meta: dict[str, Any],
    seg: int = 1,
) -> tuple[Path, Path]:
    npy_path, meta_path = feature_paths(ea_id, modality, seg)
    npy_path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(vector, dtype=np.float32)
    np.save(npy_path, arr)
    payload = {
        "ea_id": ea_id,
        "segment_id": segment_id(ea_id, seg),
        "modality": modality,
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "feature_path": str(npy_path.relative_to(ROOT)).replace("\\", "/"),
        **meta,
    }
    meta_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return npy_path, meta_path


def load_feature(npy_path: Path) -> np.ndarray:
    return np.load(npy_path)


def feature_exists(ea_id: str, modality: str, seg: int = 1) -> bool:
    npy_path, _ = feature_paths(ea_id, modality, seg)
    return npy_path.is_file() and npy_path.stat().st_size > 0
