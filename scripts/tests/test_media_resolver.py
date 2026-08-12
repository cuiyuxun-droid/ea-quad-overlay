"""Unit tests for MediaResolver without server media."""

from __future__ import annotations

import csv
import sys
import wave
import zipfile
from pathlib import Path

import numpy as np

SCRIPTS = Path(__file__).resolve().parents[1]
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from ea_features.media import (  # noqa: E402
    MediaResolver,
    read_ch_sims_text,
    read_keyed_transcript,
    read_meld_utterance,
)


def _write_wav(path: Path, frames: int = 1600, sr: int = 16000) -> None:
    with wave.open(str(path), "w") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sr)
        handle.writeframes((np.zeros(frames, dtype=np.int16)).tobytes())


def test_zip_member_materialize(tmp_path: Path) -> None:
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"fake-mp4-bytes")
    zip_path = tmp_path / "Raw.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.write(video, arcname="Raw/video_0001/0001.mp4")

    resolver = MediaResolver(cache_root=tmp_path / "cache")
    out = resolver.materialize_path(
        f"{zip_path}::Raw/video_0001/0001.mp4",
        sample_cache=tmp_path / "cache" / "EAQ000001",
        preferred_name="video",
    )
    assert out is not None
    assert out.is_file()
    assert out.read_bytes() == b"fake-mp4-bytes"


def test_read_ch_sims_text(tmp_path: Path) -> None:
    label = tmp_path / "label.csv"
    with label.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "text"])
        writer.writeheader()
        writer.writerow({"id": "video_0001/0001", "text": "你好世界"})
        writer.writerow({"id": "video_0001/0002", "text": "第二句"})
    assert read_ch_sims_text(label, "video_0001/0001") == "你好世界"


def test_read_ch_sims_text_headerless(tmp_path: Path) -> None:
    label = tmp_path / "label.csv"
    label.write_text(
        "video_0001,0001,我不想嫁给李茶,-1.0,-1.0,-1.0,-1.0,Negative,train\n"
        "video_0001,0002,你这是嫁入豪门啊！,1.0,1.0,0.8,1.0,Positive,train\n",
        encoding="utf-8",
    )
    assert read_ch_sims_text(label, "video_0001/0001") == "我不想嫁给李茶"


def test_read_ch_sims_text_split_video_clip_columns(tmp_path: Path) -> None:
    label = tmp_path / "label.csv"
    with label.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["video_id", "clip_id", "text"])
        writer.writeheader()
        writer.writerow({"video_id": "video_0001", "clip_id": "0001", "text": "first"})
        writer.writerow({"video_id": "video_0002", "clip_id": "0001", "text": "second"})
    assert read_ch_sims_text(label, "video_0001/0001") == "first"


def test_read_meld_utterance(tmp_path: Path) -> None:
    csv_path = tmp_path / "train_sent_emo.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["Dialogue_ID", "Utterance_ID", "Utterance"],
        )
        writer.writeheader()
        writer.writerow({"Dialogue_ID": "0", "Utterance_ID": "4", "Utterance": "Hello there"})
    assert read_meld_utterance(csv_path, 0, 4) == "Hello there"


def test_read_iemocap_keyed_transcript(tmp_path: Path) -> None:
    transcript = tmp_path / "session.txt"
    transcript.write_text(
        "Ses01F_impro01_F000 [006.2901-008.2357]: Excuse me.\n"
        "Ses01F_impro01_M000 [007.5712-010.4750]: Do you have your forms?\n",
        encoding="utf-8",
    )
    assert read_keyed_transcript(transcript, "Ses01F_impro01_F000") == "Excuse me."
    resolver = MediaResolver(cache_root=tmp_path / "cache")
    assert (
        resolver.resolve_text(f"{transcript}#Ses01F_impro01_M000", "IEMOCAP")
        == "Do you have your forms?"
    )


def test_resolve_text_pointers(tmp_path: Path) -> None:
    label = tmp_path / "label.csv"
    with label.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "text"])
        writer.writeheader()
        writer.writerow({"id": "video_0001/0001", "text": "中文文本"})

    meld = tmp_path / "train_sent_emo.csv"
    with meld.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["Dialogue_ID", "Utterance_ID", "Utterance"],
        )
        writer.writeheader()
        writer.writerow({"Dialogue_ID": "1", "Utterance_ID": "2", "Utterance": "English text"})

    resolver = MediaResolver(cache_root=tmp_path / "cache")
    assert resolver.resolve_text(f"{label}#video_0001/0001") == "中文文本"
    assert resolver.resolve_text(f"{meld}#Dialogue_ID=1&Utterance_ID=2") == "English text"
