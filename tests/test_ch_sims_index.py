from __future__ import annotations

from pathlib import Path

import pytest

from ea_quad_overlay.ch_sims_index import (
    ChSimsIndexError,
    ChSimsRecord,
    MediaProbeResult,
    assign_ea_ids,
    build_index_and_label_rows,
    generate_ch_sims_index,
    read_index_csv,
    read_labels_csv,
    validate_index_rows,
    validate_labels_rows,
)


def _write_label_csv(path: Path, rows: list[list[str]]) -> None:
    import csv

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerows(rows)


def _write_m1_index(path: Path, rows: list[dict[str, str]]) -> None:
    import csv

    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_assign_ea_ids_preserves_m1_and_starts_new_at_021() -> None:
    records = [
        ChSimsRecord("video_0001", "0001", "a", "0", "0", "0", "0", "Neutral", "train"),
        ChSimsRecord("video_0001", "0002", "b", "0", "0", "0", "0", "Neutral", "train"),
        ChSimsRecord("video_0001", "0003", "c", "0", "0", "0", "0", "Neutral", "test"),
    ]
    reserved = {"video_0001/0002": "EAQ000005"}
    assigned = assign_ea_ids(records, reserved)
    assert assigned["video_0001/0002"] == "EAQ000005"
    assert assigned["video_0001/0001"] == "EAQ000021"
    assert assigned["video_0001/0003"] == "EAQ000022"
    assert "EAQ000012" not in assigned.values()


def test_unprobed_rows_are_not_marked_usable(tmp_path: Path) -> None:
    records = [
        ChSimsRecord("video_0001", "0001", "你好世界", "1.0", "1.0", "1.0", "1.0", "Positive", "train"),
    ]
    index_rows, label_rows, provenance = build_index_and_label_rows(
        records,
        dataset_root="/data/ch_sims",
    )
    assert index_rows[0]["face_quality"] == "missing"
    assert index_rows[0]["audio_quality"] == "missing"
    assert index_rows[0]["usable_for_micro"] == "false"
    assert index_rows[0]["usable_for_l4"] == "false"
    assert index_rows[0]["start"] == ""
    assert index_rows[0]["end"] == ""
    assert provenance["atomic_empty"] == 1
    assert label_rows[0]["label"] == "1.0"
    assert label_rows[0]["annotation"] == "Positive"


def test_probe_ok_fills_duration_and_usability() -> None:
    records = [
        ChSimsRecord("video_0001", "0001", "你好世界", "1.0", "1.0", "1.0", "1.0", "Positive", "train"),
    ]
    probes = {
        "video_0001/0001": MediaProbeResult(
            source_key="video_0001/0001",
            video_resolved_path="/data/Raw/video_0001/0001.mp4",
            duration_sec=1.323,
            has_video_stream=True,
            has_audio_stream=True,
            probe_status="ok",
        )
    }
    index_rows, _labels, provenance = build_index_and_label_rows(
        records,
        dataset_root="/data/ch_sims",
        probes=probes,
    )
    assert index_rows[0]["start"] == "0.00"
    assert index_rows[0]["end"] == "1.32"
    assert index_rows[0]["usable_for_micro"] == "true"
    assert index_rows[0]["usable_for_l4"] == "true"
    assert provenance["ffprobe"] == 1


def test_generate_and_validate_roundtrip(tmp_path: Path) -> None:
    label_csv = tmp_path / "label.csv"
    _write_label_csv(
        label_csv,
        [
            ["video_0001", "0001", "样本一", "1.0", "1.0", "1.0", "1.0", "Positive", "train"],
            ["video_0001", "0002", "样本二", "-1.0", "-1.0", "-1.0", "-1.0", "Negative", "test"],
        ],
    )
    m1_index = tmp_path / "m1_sample_20.csv"
    _write_m1_index(
        m1_index,
        [
            {
                "ea_id": "EAQ000001",
                "source_dataset": "CH-SIMS",
                "source_split": "train",
                "source_id": "CH-SIMS/video_0001/0001",
                "video_path": "v",
                "audio_path": "a",
                "text_path": "t",
                "start": "0.00",
                "end": "1.32",
                "language": "zh",
                "face_quality": "high",
                "audio_quality": "high",
                "text_quality": "high",
                "usable_for_micro": "true",
                "usable_for_l4": "true",
            }
        ],
    )
    output_csv = tmp_path / "ch_sims_index.csv"
    labels_csv = tmp_path / "ch_sims_labels.csv"
    generate_ch_sims_index(
        label_csv=label_csv,
        output_csv=output_csv,
        labels_csv=labels_csv,
        dataset_root="/data/ch_sims",
        m1_index_path=m1_index,
    )
    rows = read_index_csv(output_csv)
    labels = read_labels_csv(labels_csv)
    summary = validate_index_rows(
        rows,
        reserved_by_source_key={"video_0001/0001": "EAQ000001"},
        min_rows=2,
    )
    labels_summary = validate_labels_rows(labels, rows)
    assert summary["total"] == 2
    assert rows[0]["ea_id"] == "EAQ000001"
    assert rows[0]["end"] == "1.32"
    assert rows[0]["face_quality"] == "high"
    assert rows[0]["usable_for_micro"] == "true"
    assert labels_summary["missing_label_fields"] == 0
    assert labels[0]["annotation"] == "Positive"


def test_usable_micro_with_missing_face_is_rejected() -> None:
    with pytest.raises(ChSimsIndexError, match="usable_for_micro requires face quality"):
        validate_index_rows(
            [
                {
                    "ea_id": "EAQ000021",
                    "source_dataset": "CH-SIMS",
                    "source_split": "train",
                    "source_id": "CH-SIMS/video_0001/0001",
                    "video_path": "v",
                    "audio_path": "a",
                    "text_path": "t",
                    "start": "",
                    "end": "",
                    "language": "zh",
                    "face_quality": "missing",
                    "audio_quality": "missing",
                    "text_quality": "high",
                    "usable_for_micro": "true",
                    "usable_for_l4": "false",
                }
            ],
            min_rows=1,
        )
