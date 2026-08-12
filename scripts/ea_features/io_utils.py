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


def feature_paths(
    ea_id: str,
    modality: str,
    seg: int = 1,
    *,
    feature_root: Path | None = None,
) -> tuple[Path, Path]:
    stem = feature_stem(ea_id, modality, seg)
    base = (feature_root or FEATURE_ROOT) / modality
    return base / f"{stem}.npy", base / f"{stem}.json"


def save_feature(
    ea_id: str,
    modality: str,
    vector: np.ndarray,
    meta: dict[str, Any],
    seg: int = 1,
    *,
    feature_root: Path | None = None,
    path_root: Path | None = None,
) -> tuple[Path, Path]:
    npy_path, meta_path = feature_paths(
        ea_id,
        modality,
        seg,
        feature_root=feature_root,
    )
    npy_path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(vector, dtype=np.float32)
    npy_tmp = npy_path.with_suffix(".npy.tmp")
    with npy_tmp.open("wb") as handle:
        np.save(handle, arr)
    npy_tmp.replace(npy_path)
    relative_root = path_root or ROOT
    try:
        feature_path = npy_path.relative_to(relative_root).as_posix()
    except ValueError:
        feature_path = npy_path.as_posix()
    payload = {
        "ea_id": ea_id,
        "segment_id": segment_id(ea_id, seg),
        "modality": modality,
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "feature_path": feature_path,
        **meta,
    }
    meta_tmp = meta_path.with_suffix(".json.tmp")
    meta_tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    meta_tmp.replace(meta_path)
    return npy_path, meta_path


def load_feature(npy_path: Path) -> np.ndarray:
    return np.load(npy_path)


def feature_exists(
    ea_id: str,
    modality: str,
    seg: int = 1,
    *,
    feature_root: Path | None = None,
) -> bool:
    npy_path, meta_path = feature_paths(
        ea_id,
        modality,
        seg,
        feature_root=feature_root,
    )
    return (
        npy_path.is_file()
        and npy_path.stat().st_size > 0
        and meta_path.is_file()
        and meta_path.stat().st_size > 0
    )
