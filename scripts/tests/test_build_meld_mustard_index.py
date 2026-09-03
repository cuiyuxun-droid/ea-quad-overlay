"""Tests for MELD / MUStARD source index builder."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ea_quad_overlay.meld_mustard_index import (  # noqa: E402
    INDEX_COLUMNS,
    MELD_EA_ID_START,
    MUSTARD_EA_ID_START,
    MELD_VIDEO_DIRS,
    assign_ea_ids,
    build_meld_row,
    build_mustard_row,
    duration_from_timestamps,
    emotion_sentiment_mismatch,
    generate_dialogue_indexes,
    load_meld_utterances,
    load_mustard_clips,
    quality_flags,
    resolve_mustard_video_path,
)


def _write_meld_ann(root: Path) -> Path:
    ann = root / "annotations"
    ann.mkdir(parents=True)
    header = (
        "Sr No.,Utterance,Speaker,Emotion,Sentiment,Dialogue_ID,Utterance_ID,"
        "Season,Episode,StartTime,EndTime\n"
    )
    (ann / "train_sent_emo.csv").write_text(
        header
        + '1,Hello,A,joy,negative,0,0,1,1,"00:00:01,000","00:00:03,500"\n'
        + '2,World,B,neutral,neutral,0,1,1,1,"00:00:03,500","00:00:04,000"\n'
        + '3,Seed line,C,anger,negative,0,4,1,1,"00:00:04,000","00:00:10,000"\n',
        encoding="utf-8",
    )
    (ann / "dev_sent_emo.csv").write_text(
        header
        + '1,Dev line,C,anger,positive,1,0,1,2,"00:00:10,000","00:00:12,000"\n',
        encoding="utf-8",
    )
    (ann / "test_sent_emo.csv").write_text(
        header
        + '1,Test line,D,sadness,negative,2,0,1,3,"00:00:00,000","00:00:02,000"\n',
        encoding="utf-8",
    )
    return root


def _write_mustard(json_path: Path) -> Path:
    payload = {
        "1_10": {
            "utterance": "Nice weather.",
            "speaker": "A",
            "context": [],
            "context_speakers": [],
            "show": "BBT",
            "sarcasm": True,
        },
        "1_20": {
            "utterance": "Plain statement.",
            "speaker": "B",
            "context": [],
            "context_speakers": [],
            "show": "FRIENDS",
            "sarcasm": False,
        },
    }
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload), encoding="utf-8")
    return json_path


def _write_m1(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
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
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "ea_id": "EAQ000012",
                "source_dataset": "MELD",
                "source_split": "train",
                "source_id": "MELD/train/dia0/utt4",
                "video_path": "/root/x.mp4",
                "audio_path": "/root/x.mp4",
                "text_path": "/root/a.csv#Dialogue_ID=0&Utterance_ID=4",
                "start": "0.00",
                "end": "6.49",
                "language": "en",
                "face_quality": "high",
                "audio_quality": "high",
                "text_quality": "high",
                "usable_for_micro": "true",
                "usable_for_l4": "true",
            }
        )
    return path


def test_emotion_sentiment_mismatch() -> None:
    assert emotion_sentiment_mismatch("joy", "negative") is True
    assert emotion_sentiment_mismatch("anger", "positive") is True
    assert emotion_sentiment_mismatch("joy", "positive") is False
    assert emotion_sentiment_mismatch("neutral", "negative") is False


def test_duration_from_timestamps() -> None:
    assert duration_from_timestamps("00:00:01,000", "00:00:03,500") == 2.5
    assert duration_from_timestamps("", "00:00:01,000") is None


def test_server_path_layouts() -> None:
    assert MELD_VIDEO_DIRS["dev"] == "extracted/MELD.Raw/dev_splits_complete"
    assert resolve_mustard_video_path("1_60", "/root/autodl-tmp/data/datasets/mustard").endswith(
        "/raw/clips/utterances_final/1_60.mp4"
    )


def test_quality_flags_unchecked_media() -> None:
    face, audio, text, micro, l4 = quality_flags(
        text="hello",
        video_present=None,
        duration=None,
        sarcasm_candidate=True,
    )
    assert face == "missing"
    assert audio == "missing"
    assert text == "high"
    assert micro == "false"
    assert l4 == "true"


def test_generate_indexes_preserves_m1_and_paths(tmp_path: Path) -> None:
    meld_root = _write_meld_ann(tmp_path / "meld")
    mustard_json = _write_mustard(tmp_path / "data" / "sarcasm_data.json")
    m1 = _write_m1(tmp_path / "m1_sample_20.csv")
    meld_out = tmp_path / "meld_index.csv"
    mustard_out = tmp_path / "mustard_index.csv"
    alloc = tmp_path / "alloc.csv"

    meld_rows, mustard_rows, meld_sum, mustard_sum = generate_dialogue_indexes(
        meld_ann_root=meld_root,
        mustard_json=mustard_json,
        meld_path_root="/root/autodl-tmp/data/datasets/meld",
        mustard_path_root="/root/autodl-tmp/data/datasets/mustard",
        meld_output=meld_out,
        mustard_output=mustard_out,
        m1_index_path=m1,
        allocation_map_path=alloc,
        check_media=False,
    )

    assert meld_sum["total"] == 5
    assert mustard_sum["total"] == 2
    assert meld_sum["seed_rows_inherited"] == 1
    assert meld_sum["usable_for_micro"] == 1  # only M1 seed
    assert mustard_sum["usable_for_micro"] == 0

    by_sid = {row["source_id"]: row for row in meld_rows}
    assert by_sid["MELD/train/dia0/utt4"]["ea_id"] == "EAQ000012"
    assert by_sid["MELD/train/dia0/utt4"]["usable_for_micro"] == "true"
    assert by_sid["MELD/train/dia0/utt4"]["end"] == "6.49"
    assert by_sid["MELD/train/dia0/utt0"]["ea_id"].startswith("EAQ1")
    assert by_sid["MELD/train/dia0/utt0"]["usable_for_micro"] == "false"
    assert by_sid["MELD/train/dia0/utt0"]["face_quality"] == "missing"
    assert "dev_splits_complete" in by_sid["MELD/dev/dia1/utt0"]["video_path"]

    mustard_by_id = {row["source_id"]: row for row in mustard_rows}
    assert mustard_by_id["1_10"]["ea_id"].startswith("EAQ2")
    assert "/raw/clips/utterances_final/1_10.mp4" in mustard_by_id["1_10"]["video_path"]
    assert mustard_by_id["1_10"]["text_path"].endswith(
        "/data/sarcasm_data.json#utterance_id=1_10"
    )
    assert mustard_by_id["1_10"]["is_sarcasm_candidate"] == "true"
    assert mustard_by_id["1_10"]["candidate_reason"] == "mustard_label"

    with meld_out.open(encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == list(INDEX_COLUMNS)

    # Stability: regenerating with an inserted earlier source keeps prior IDs.
    prior = {row["source_id"]: row["ea_id"] for row in meld_rows}
    ann = meld_root / "annotations" / "train_sent_emo.csv"
    text = ann.read_text(encoding="utf-8")
    # Insert a new earliest train row (dia0/utt-1 not valid); use dia0/utt2 new.
    header, *rest = text.splitlines()
    new_line = '4,Inserted,Z,joy,positive,0,2,1,1,"00:00:00,000","00:00:01,000"'
    # Put inserted row after header but source_id sorts after utt1? utt2 is new.
    ann.write_text("\n".join([header, rest[0], rest[1], new_line, rest[2]]) + "\n", encoding="utf-8")

    meld_rows2, _, _, _ = generate_dialogue_indexes(
        meld_ann_root=meld_root,
        mustard_json=mustard_json,
        meld_path_root="/root/autodl-tmp/data/datasets/meld",
        mustard_path_root="/root/autodl-tmp/data/datasets/mustard",
        meld_output=meld_out,
        mustard_output=mustard_out,
        m1_index_path=m1,
        allocation_map_path=alloc,
        check_media=False,
    )
    by2 = {row["source_id"]: row["ea_id"] for row in meld_rows2}
    for sid, ea_id in prior.items():
        assert by2[sid] == ea_id
    assert "MELD/train/dia0/utt2" in by2
    assert by2["MELD/train/dia0/utt2"] not in prior.values()


def test_assign_ea_ids_skips_seed_range() -> None:
    assigned = assign_ea_ids(
        dataset="MELD",
        source_ids=["MELD/train/dia0/utt0", "MELD/train/dia0/utt4"],
        range_start=MELD_EA_ID_START,
        range_end_exclusive=MUSTARD_EA_ID_START,
        seed_reservations={"MELD/train/dia0/utt4": "EAQ000012"},
        existing_map={},
    )
    assert assigned["MELD/train/dia0/utt4"] == "EAQ000012"
    assert assigned["MELD/train/dia0/utt0"] == f"EAQ{MELD_EA_ID_START:06d}"


def test_load_helpers(tmp_path: Path) -> None:
    meld_root = _write_meld_ann(tmp_path / "meld")
    mustard_json = _write_mustard(tmp_path / "sarcasm_data.json")
    assert len(load_meld_utterances(meld_root)) == 5
    assert len(load_mustard_clips(mustard_json)) == 2
