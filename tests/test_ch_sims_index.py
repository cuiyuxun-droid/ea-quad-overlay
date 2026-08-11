from __future__ import annotations

from pathlib import Path

import pytest

from ea_quad_overlay.ch_sims_index import (
    ChSimsIndexError,
    ChSimsRecord,
    assign_ea_ids,
    build_index_rows,
    generate_ch_sims_index,
    read_index_csv,
    validate_index_rows,
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


def test_assign_ea_ids_preserves_m1_and_skips_meld_slots() -> None:
    records = [
        ChSimsRecord("video_0001", "0001", "a", "0", "0", "0", "0", "Neutral", "train"),
        ChSimsRecord("video_0001", "0002", "b", "0", "0", "0", "0", "Neutral", "train"),
        ChSimsRecord("video_0001", "0003", "c", "0", "0", "0", "0", "Neutral", "test"),
    ]
    reserved = {"video_0001/0002": "EAQ000005"}
    assigned = assign_ea_ids(records, reserved)
    assert assigned["video_0001/0001"] == "EAQ000001"
    assert assigned["video_0001/0002"] == "EAQ000005"
    assert assigned["video_0001/0003"] == "EAQ000002"
    assert "EAQ000012" not in assigned.values()


def test_build_index_rows_have_required_fields(tmp_path: Path) -> None:
    label_csv = tmp_path / "label.csv"
    _write_label_csv(
        label_csv,
        [
            ["video_0001", "0001", "你好", "1.0", "1.0", "1.0", "1.0", "Positive", "train"],
            ["video_0001", "0002", "测试", "-1.0", "-1.0", "-1.0", "-1.0", "Negative", "valid"],
        ],
    )
    records = [
        ChSimsRecord("video_0001", "0001", "你好", "1.0", "1.0", "1.0", "1.0", "Positive", "train"),
        ChSimsRecord("video_0001", "0002", "测试", "-1.0", "-1.0", "-1.0", "-1.0", "Negative", "valid"),
    ]
    rows = build_index_rows(records, dataset_root="/data/ch_sims", label_csv=label_csv)
    assert rows[0]["ea_id"] == "EAQ000001"
    assert rows[0]["source_split"] == "train"
    assert rows[1]["source_split"] == "validation"
    assert rows[0]["video_path"].endswith("Raw/video_0001/0001.mp4")
    assert rows[0]["text_path"].endswith("#video_0001/0001")


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
                "end": "1.00",
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
    generate_ch_sims_index(
        label_csv=label_csv,
        output_csv=output_csv,
        dataset_root="/data/ch_sims",
        m1_index_path=m1_index,
    )
    rows = read_index_csv(output_csv)
    summary = validate_index_rows(rows, reserved_by_source_key={"video_0001/0001": "EAQ000001"}, min_rows=2)
    assert summary["total"] == 2
    assert rows[0]["ea_id"] == "EAQ000001"


def test_invalid_start_end_rejected() -> None:
    with pytest.raises(ChSimsIndexError, match="invalid start/end"):
        validate_index_rows(
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
                    "end": "0.00",
                    "language": "zh",
                    "face_quality": "high",
                    "audio_quality": "high",
                    "text_quality": "high",
                    "usable_for_micro": "true",
                    "usable_for_l4": "true",
                }
            ],
            min_rows=1,
        )
