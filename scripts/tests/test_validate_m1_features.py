"""Tests for validate_m1_features against synthetic feature files."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[1]
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import validate_m1_features as validator


def _write_index(path: Path, n: int = 20) -> None:
    fields = [
        "ea_id",
        "source_dataset",
        "source_split",
        "source_id",
        "video_path",
        "audio_path",
        "text_path",
        "start",
        "end",
        "language",
        "face_quality",
        "audio_quality",
        "text_quality",
        "usable_for_micro",
        "usable_for_l4",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for i in range(1, n + 1):
            writer.writerow(
                {
                    "ea_id": f"EAQ{i:06d}",
                    "source_dataset": "CH-SIMS" if i <= 11 else "MELD",
                    "source_split": "train",
                    "source_id": f"dummy/{i}",
                    "video_path": "x.mp4",
                    "audio_path": "x.mp4",
                    "text_path": "x.csv#a",
                    "start": "0.0",
                    "end": "1.0",
                    "language": "zh",
                    "face_quality": "high",
                    "audio_quality": "high",
                    "text_quality": "high",
                    "usable_for_micro": "true",
                    "usable_for_l4": "true",
                }
            )


def test_validate_pass(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    index = tmp_path / "index.csv"
    report = tmp_path / "report.md"
    _write_index(index, 20)
    feature_root = tmp_path / "features"

    def fake_paths(ea_id: str, modality: str, seg: int = 1):
        stem = f"{ea_id}_seg{seg:03d}_{modality}"
        base = feature_root / modality
        return base / f"{stem}.npy", base / f"{stem}.json"

    monkeypatch.setattr(validator, "feature_paths", fake_paths)

    for i in range(1, 21):
        ea_id = f"EAQ{i:06d}"
        for modality in ("text", "speech", "macro", "micro"):
            npy, meta = fake_paths(ea_id, modality)
            npy.parent.mkdir(parents=True, exist_ok=True)
            np.save(npy, np.ones((4,), dtype=np.float32))
            meta.write_text(json.dumps({"status": "ok"}), encoding="utf-8")

    monkeypatch.setattr(
        sys,
        "argv",
        ["validate_m1_features.py", "--index", str(index), "--report", str(report)],
    )
    validator.main()
    assert report.is_file()
    text = report.read_text(encoding="utf-8")
    assert "none" in text
    assert "`text`" in text


def test_validate_missing_feature_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    index = tmp_path / "index.csv"
    report = tmp_path / "report.md"
    _write_index(index, 20)
    feature_root = tmp_path / "features"

    def fake_paths(ea_id: str, modality: str, seg: int = 1):
        stem = f"{ea_id}_seg{seg:03d}_{modality}"
        base = feature_root / modality
        return base / f"{stem}.npy", base / f"{stem}.json"

    monkeypatch.setattr(validator, "feature_paths", fake_paths)
    npy, meta = fake_paths("EAQ000001", "text")
    npy.parent.mkdir(parents=True, exist_ok=True)
    np.save(npy, np.ones((2,), dtype=np.float32))
    meta.write_text("{}", encoding="utf-8")

    monkeypatch.setattr(
        sys,
        "argv",
        ["validate_m1_features.py", "--index", str(index), "--report", str(report)],
    )
    with pytest.raises(SystemExit) as exc:
        validator.main()
    assert exc.value.code == 1
