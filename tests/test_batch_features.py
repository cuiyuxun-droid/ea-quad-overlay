"""Tests for the unified FeatureBank extraction runner."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from ea_features import face_utils  # noqa: E402
from ea_features.batch import (  # noqa: E402
    BatchConfig,
    BatchExtractionError,
    BatchFeatureRunner,
    parse_modalities,
)

FIELDS = [
    "ea_id",
    "source_dataset",
    "source_split",
    "source_id",
    "video_path",
    "audio_path",
    "text_path",
    "start",
    "end",
    "usable_for_micro",
]


def _write_index(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


class DummyResolver:
    def __init__(self, cache_root: Path) -> None:
        self.cache_root = cache_root

    def resolve_text(self, text_spec: str, source_dataset: str = "") -> str:
        return text_spec

    def resolve_video(
        self,
        video_spec: str,
        sample_cache: Path,
        start: float = 0.0,
        end: float = 0.0,
    ) -> Path:
        return Path(video_spec)

    def resolve_audio(
        self,
        audio_spec: str,
        video_path: Path | None,
        sample_cache: Path,
        start: float = 0.0,
        end: float = 0.0,
    ) -> Path:
        return Path(audio_spec)


class DummyExtractor:
    def __init__(self, modality: str, calls: list[str], *, fail: bool = False) -> None:
        self.modality = modality
        self.calls = calls
        self.fail = fail

    def extract(self, payload: Any) -> tuple[np.ndarray, dict[str, Any]]:
        self.calls.append(self.modality)
        if self.fail:
            raise RuntimeError("synthetic extraction failure")
        return np.ones((3,), dtype=np.float32), {"model": "dummy", "status": "ok"}


def _factories(calls: list[str], *, failing: str | None = None):
    return {
        modality: (
            lambda modality=modality: DummyExtractor(modality, calls, fail=modality == failing)
        )
        for modality in ("text", "speech", "macro", "micro")
    }


def _config(tmp_path: Path, index: Path, modalities: str) -> BatchConfig:
    return BatchConfig(
        root=tmp_path,
        output_root=tmp_path / "out",
        index_paths=(index,),
        modalities=parse_modalities(modalities),
    )


def test_batch_writes_features_quality_and_manifest(tmp_path: Path) -> None:
    index = tmp_path / "dataset.csv"
    _write_index(
        index,
        [
            {
                "ea_id": "EAQ000101",
                "source_dataset": "demo",
                "source_split": "train",
                "source_id": "demo/1",
                "video_path": str(tmp_path / "one.mp4"),
                "audio_path": str(tmp_path / "one.wav"),
                "text_path": "hello",
                "start": "0",
                "end": "1",
                "usable_for_micro": "true",
            },
            {
                "ea_id": "EAQ000102",
                "source_dataset": "demo",
                "source_split": "train",
                "source_id": "demo/2",
                "video_path": str(tmp_path / "two.mp4"),
                "audio_path": str(tmp_path / "two.wav"),
                "text_path": "world",
                "start": "0",
                "end": "1",
                "usable_for_micro": "false",
            },
        ],
    )
    calls: list[str] = []
    runner = BatchFeatureRunner(
        _config(tmp_path, index, "text,speech,macro,micro"),
        resolver=DummyResolver(tmp_path / "cache"),
        extractor_factories=_factories(calls),
        face_quality_checker=lambda *_args, **_kwargs: {
            "frames_sampled": 10,
            "frames_with_face": 9,
            "face_detect_rate": 0.9,
            "mean_face_ratio": 0.12,
            "usable_for_micro": True,
            "filter_reason": "ok",
        },
    )

    summary = runner.run()

    assert summary["samples"] == 2
    assert summary["failed_samples"] == 0
    assert summary["filtered_samples"] == 1
    assert calls.count("text") == 2
    assert calls.count("micro") == 1
    assert (tmp_path / "out/features/text/EAQ000101_seg001_text.npy").is_file()
    assert not (tmp_path / "out/features/micro/EAQ000102_seg001_micro.npy").exists()

    with (tmp_path / "out/reports/feature_quality.csv").open(
        newline="", encoding="utf-8-sig"
    ) as handle:
        quality = list(csv.DictReader(handle))
    assert quality[0]["face_detect_rate"] == "0.9"
    assert quality[1]["filtered_modalities"] == "micro"
    assert quality[1]["micro_filter_reason"] == "source_index_usable_for_micro=false"

    manifest = [
        json.loads(line)
        for line in (tmp_path / "out/manifests/feature_bank.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert manifest[0]["feature_paths"]["text"].startswith("features/text/")
    assert "micro" in manifest[0]["feature_paths"]
    assert "micro" not in manifest[1]["feature_paths"]


def test_skip_existing_does_not_load_extractor(tmp_path: Path) -> None:
    index = tmp_path / "dataset.csv"
    _write_index(
        index,
        [
            {
                "ea_id": "EAQ000201",
                "source_dataset": "demo",
                "source_split": "test",
                "source_id": "demo/201",
                "video_path": "",
                "audio_path": "",
                "text_path": "already done",
                "start": "0",
                "end": "0",
                "usable_for_micro": "false",
            }
        ],
    )
    calls: list[str] = []
    config = _config(tmp_path, index, "text")
    BatchFeatureRunner(
        config,
        resolver=DummyResolver(tmp_path / "cache"),
        extractor_factories=_factories(calls),
    ).run()
    BatchFeatureRunner(
        config,
        resolver=DummyResolver(tmp_path / "cache"),
        extractor_factories={
            "text": lambda: pytest.fail("extractor should not load for an existing feature")
        },
    ).run()

    assert calls == ["text"]


def test_failure_is_recorded_without_blocking_other_modalities(tmp_path: Path) -> None:
    index = tmp_path / "dataset.csv"
    _write_index(
        index,
        [
            {
                "ea_id": "EAQ000301",
                "source_dataset": "demo",
                "source_split": "dev",
                "source_id": "demo/301",
                "video_path": "clip.mp4",
                "audio_path": "clip.wav",
                "text_path": "keep going",
                "start": "0",
                "end": "1",
                "usable_for_micro": "false",
            }
        ],
    )
    calls: list[str] = []
    runner = BatchFeatureRunner(
        _config(tmp_path, index, "text,speech"),
        resolver=DummyResolver(tmp_path / "cache"),
        extractor_factories=_factories(calls, failing="speech"),
    )

    summary = runner.run()

    assert summary["failed_samples"] == 1
    assert (tmp_path / "out/features/text/EAQ000301_seg001_text.npy").is_file()
    assert not (tmp_path / "out/features/speech/EAQ000301_seg001_speech.npy").exists()
    report = (tmp_path / "out/reports/feature_quality.csv").read_text(encoding="utf-8-sig")
    assert "synthetic extraction failure" in report


def test_parse_modalities_rejects_unknown_values() -> None:
    assert parse_modalities("text,text,macro") == ("text", "macro")
    with pytest.raises(BatchExtractionError, match="unsupported modality"):
        parse_modalities("text,thermal")


def test_config_rejects_invalid_face_threshold(tmp_path: Path) -> None:
    with pytest.raises(BatchExtractionError, match="min_face_detect_rate"):
        BatchConfig(
            root=tmp_path,
            output_root=tmp_path,
            index_paths=(tmp_path / "index.csv",),
            modalities=("micro",),
            min_face_detect_rate=1.1,
        )


def test_face_quality_uses_detection_rate_and_face_area(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"video-placeholder")
    frames = [np.zeros((100, 200, 3), dtype=np.uint8) for _ in range(4)]
    detections = iter([(0, 0, 40, 50), (0, 0, 20, 20), None, (0, 0, 20, 20)])
    monkeypatch.setattr(face_utils, "read_frames_sequential", lambda *_args, **_kwargs: frames)
    monkeypatch.setattr(face_utils, "detect_largest_face", lambda _frame: next(detections))

    quality = face_utils.assess_face_quality(
        video,
        min_detect_rate=0.8,
        min_face_ratio=0.02,
    )

    assert quality.frames_sampled == 4
    assert quality.frames_with_face == 3
    assert quality.face_detect_rate == 0.75
    assert quality.mean_face_ratio == pytest.approx(0.046667, abs=1e-6)
    assert not quality.usable_for_micro
    assert quality.filter_reason == "low_face_detect_rate"
