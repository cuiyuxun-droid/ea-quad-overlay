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
    build_meld_row,
    build_mustard_row,
    duration_from_timestamps,
    emotion_sentiment_mismatch,
    generate_dialogue_indexes,
    load_meld_utterances,
    load_mustard_clips,
)


def _write_meld_ann(root: Path) -> Path:
    ann = root / "annotations"
    ann.mkdir(parents=True)
    header = (
        "Sr No.,Utterance,Speaker,Emotion,Sentiment,Dialogue_ID,Utterance_ID,"
        "Season,Episode,StartTime,EndTime\n"
    )
    # train: mismatch joy + negative
    (ann / "train_sent_emo.csv").write_text(
        header
        + '1,Hello,A,joy,negative,0,0,1,1,"00:00:01,000","00:00:03,500"\n'
        + '2,World,B,neutral,neutral,0,1,1,1,"00:00:03,500","00:00:04,000"\n',
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


def test_emotion_sentiment_mismatch() -> None:
    assert emotion_sentiment_mismatch("joy", "negative") is True
    assert emotion_sentiment_mismatch("anger", "positive") is True
    assert emotion_sentiment_mismatch("joy", "positive") is False
    assert emotion_sentiment_mismatch("neutral", "negative") is False
    assert emotion_sentiment_mismatch("sadness", "neutral") is False


def test_duration_from_timestamps() -> None:
    assert duration_from_timestamps("00:00:01,000", "00:00:03,500") == 2.5
    assert duration_from_timestamps("", "00:00:01,000") is None


def test_generate_indexes(tmp_path: Path) -> None:
    meld_root = _write_meld_ann(tmp_path / "meld")
    mustard_json = _write_mustard(tmp_path / "sarcasm_data.json")
    meld_out = tmp_path / "meld_index.csv"
    mustard_out = tmp_path / "mustard_index.csv"

    meld_rows, mustard_rows, meld_sum, mustard_sum = generate_dialogue_indexes(
        meld_ann_root=meld_root,
        mustard_json=mustard_json,
        meld_path_root="/root/autodl-tmp/data/datasets/meld",
        mustard_path_root="/root/autodl-tmp/data/datasets/mustard",
        meld_output=meld_out,
        mustard_output=mustard_out,
        check_media=False,
    )

    assert meld_sum["total"] == 4
    assert mustard_sum["total"] == 2
    assert meld_rows[0]["ea_id"] == f"EAQ{MELD_EA_ID_START:06d}"
    assert mustard_rows[0]["ea_id"] == f"EAQ{MUSTARD_EA_ID_START:06d}"

    # source_id format
    assert meld_rows[0]["source_id"] == "MELD/train/dia0/utt0"
    assert "Dialogue_ID=0&Utterance_ID=0" in meld_rows[0]["text_path"]
    assert meld_rows[0]["video_path"].endswith("train_splits/dia0_utt0.mp4")

    # sarcasm mismatch on first train + dev
    by_sid = {row["source_id"]: row for row in meld_rows}
    assert by_sid["MELD/train/dia0/utt0"]["is_sarcasm_candidate"] == "true"
    assert by_sid["MELD/train/dia0/utt0"]["candidate_reason"] == (
        "meld_emotion_sentiment_mismatch"
    )
    assert by_sid["MELD/train/dia0/utt1"]["is_sarcasm_candidate"] == "false"
    assert by_sid["MELD/dev/dia1/utt0"]["is_sarcasm_candidate"] == "true"
    assert by_sid["MELD/test/dia2/utt0"]["is_sarcasm_candidate"] == "false"

    # MUStARD label passthrough
    mustard_by_id = {row["source_id"]: row for row in mustard_rows}
    assert mustard_by_id["1_10"]["is_sarcasm_candidate"] == "true"
    assert mustard_by_id["1_10"]["candidate_reason"] == "mustard_label"
    assert mustard_by_id["1_20"]["is_sarcasm_candidate"] == "false"
    assert mustard_by_id["1_20"]["candidate_reason"] == ""
    assert mustard_by_id["1_10"]["text_path"].endswith("#utterance_id=1_10")

    with meld_out.open(encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == list(INDEX_COLUMNS)
        assert len(list(reader)) == 4


def test_load_helpers(tmp_path: Path) -> None:
    meld_root = _write_meld_ann(tmp_path / "meld")
    mustard_json = _write_mustard(tmp_path / "sarcasm_data.json")
    assert len(load_meld_utterances(meld_root)) == 4
    assert len(load_mustard_clips(mustard_json)) == 2


def test_build_row_helpers() -> None:
    from ea_quad_overlay.meld_mustard_index import MeldUtterance, MustardClip

    meld = MeldUtterance(
        split="train",
        dialogue_id=0,
        utterance_id=4,
        utterance="hi",
        emotion="joy",
        sentiment="negative",
        start_time="00:00:00,000",
        end_time="00:00:02,000",
    )
    row = build_meld_row(
        meld,
        ea_id="EAQ100000",
        meld_root="/root/autodl-tmp/data/datasets/meld",
        check_media=False,
    )
    assert row["end"] == "2.00"
    assert row["is_sarcasm_candidate"] == "true"

    clip = MustardClip(
        utterance_id="9_9",
        utterance="x",
        sarcasm=True,
        show="BBT",
        speaker="A",
    )
    mrow = build_mustard_row(
        clip,
        ea_id="EAQ200000",
        mustard_root="/root/autodl-tmp/data/datasets/mustard",
        json_path="/root/autodl-tmp/data/datasets/mustard/sarcasm_data.json",
        check_media=False,
    )
    assert mrow["video_path"].endswith("utterances_final/9_9.mp4")
