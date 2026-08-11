#!/usr/bin/env python
"""Build source indexes for IEMOCAP and CMU-MOSEI/MOSI.

The script is intentionally conservative: it only emits rows whose source
annotation record can be parsed. Media/text paths are marked usable only when
the relevant source paths are present in the discovered dataset tree.
"""

from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
SOURCE_INDEX = ROOT / "source_index"

BASE_FIELDS = [
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
LABEL_FIELDS = [
    "raw_emotion",
    "raw_sentiment",
    "raw_valence",
    "raw_arousal",
    "weak_label_hint",
    "label_source",
]
FIELDNAMES = BASE_FIELDS + LABEL_FIELDS

IEMOCAP_EVAL_RE = re.compile(
    r"^\[(?P<start>[-\d.]+)\s*-\s*(?P<end>[-\d.]+)\]\s+"
    r"(?P<utt>\S+)\s+(?P<emotion>\S+)\s+\["
    r"(?P<vad>[^\]]+)\]"
)
IEMOCAP_TRANSCRIPT_RE = re.compile(r"^(?P<utt>\S+)\s+\[[^\]]+\]:\s*(?P<text>.*)$")
MEDIA_EXTENSIONS = {".avi", ".mp4", ".mov", ".mkv", ".wav", ".mp3", ".flac", ".m4a"}
TEXT_EXTENSIONS = {".txt", ".csv", ".tsv"}

EMOTION_WEAK_LABELS = {
    "ang": "negative",
    "anger": "negative",
    "dis": "negative",
    "disgust": "negative",
    "exc": "positive",
    "excited": "positive",
    "fea": "negative",
    "fear": "negative",
    "fru": "negative",
    "frustration": "negative",
    "hap": "positive",
    "happy": "positive",
    "neu": "neutral",
    "neutral": "neutral",
    "sad": "negative",
    "sadness": "negative",
    "sur": "positive",
    "surprise": "positive",
}


@dataclass(frozen=True)
class SourceRow:
    source_dataset: str
    source_split: str
    source_id: str
    video_path: str
    audio_path: str
    text_path: str
    start: str
    end: str
    language: str
    face_quality: str
    audio_quality: str
    text_quality: str
    usable_for_micro: str
    usable_for_l4: str
    raw_emotion: str = ""
    raw_sentiment: str = ""
    raw_valence: str = ""
    raw_arousal: str = ""
    weak_label_hint: str = ""
    label_source: str = ""

    def with_ea_id(self, ea_id: str) -> dict[str, str]:
        row = {field: getattr(self, field) for field in FIELDNAMES if field != "ea_id"}
        return {"ea_id": ea_id, **row}


def truth(value: bool) -> str:
    return "true" if value else "false"


def quality(value: bool) -> str:
    return "high" if value else "low"


def weak_from_sentiment(value: str) -> str:
    if not value:
        return ""
    try:
        numeric = float(value)
    except ValueError:
        normalized = value.strip().lower()
        if normalized in {"positive", "pos", "1"}:
            return "positive"
        if normalized in {"negative", "neg", "-1"}:
            return "negative"
        if normalized in {"neutral", "neu", "0"}:
            return "neutral"
        return ""
    if numeric > 0:
        return "positive"
    if numeric < 0:
        return "negative"
    return "neutral"


def session_split(session_name: str) -> str:
    if session_name in {"Session1", "Session2", "Session3"}:
        return "train"
    if session_name == "Session4":
        return "dev"
    return "test"


def read_iemocap_transcripts(session_dir: Path) -> dict[str, Path]:
    transcript_dir = session_dir / "dialog" / "transcriptions"
    paths: dict[str, Path] = {}
    if not transcript_dir.is_dir():
        return paths
    for path in transcript_dir.glob("*.txt"):
        with path.open(encoding="utf-8", errors="replace") as handle:
            for line in handle:
                match = IEMOCAP_TRANSCRIPT_RE.match(line.strip())
                if match:
                    paths[match.group("utt")] = path
    return paths


def find_iemocap_dialog_video(session_dir: Path, dialog_id: str) -> Path | None:
    avi_root = session_dir / "dialog" / "avi"
    if not avi_root.is_dir():
        return None
    candidates = [
        avi_root / "DivX" / f"{dialog_id}.avi",
        avi_root / f"{dialog_id}.avi",
        avi_root / f"{dialog_id}.mp4",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    for path in avi_root.rglob(f"{dialog_id}.*"):
        if path.suffix.lower() in MEDIA_EXTENSIONS and path.is_file():
            return path
    return None


def build_iemocap_rows(root: Path) -> list[SourceRow]:
    if not root.is_dir():
        return []

    rows: list[SourceRow] = []
    for session_dir in sorted(root.glob("Session*")):
        eval_dir = session_dir / "dialog" / "EmoEvaluation"
        if not eval_dir.is_dir():
            continue
        transcripts = read_iemocap_transcripts(session_dir)
        split = session_split(session_dir.name)
        for eval_path in sorted(eval_dir.glob("*.txt")):
            dialog_id = eval_path.stem
            video_path = find_iemocap_dialog_video(session_dir, dialog_id)
            with eval_path.open(encoding="utf-8", errors="replace") as handle:
                for line in handle:
                    match = IEMOCAP_EVAL_RE.match(line.strip())
                    if not match:
                        continue
                    utt_id = match.group("utt")
                    emotion = match.group("emotion")
                    vad = [part.strip() for part in match.group("vad").split(",")]
                    valence = vad[0] if len(vad) >= 1 else ""
                    arousal = vad[1] if len(vad) >= 2 else ""
                    audio_path = (
                        session_dir / "sentences" / "wav" / dialog_id / f"{utt_id}.wav"
                    )
                    transcript_path = transcripts.get(utt_id)
                    has_video = video_path is not None and video_path.is_file()
                    has_audio = audio_path.is_file()
                    has_text = transcript_path is not None and transcript_path.is_file()
                    weak_label = EMOTION_WEAK_LABELS.get(emotion.lower(), "")
                    usable_l4 = has_video and has_audio and has_text and bool(weak_label)
                    rows.append(
                        SourceRow(
                            source_dataset="IEMOCAP",
                            source_split=split,
                            source_id=f"IEMOCAP/{session_dir.name}/{utt_id}",
                            video_path=str(video_path) if video_path else "",
                            audio_path=str(audio_path) if has_audio else "",
                            text_path=f"{transcript_path}#{utt_id}" if transcript_path else "",
                            start=f"{float(match.group('start')):.2f}",
                            end=f"{float(match.group('end')):.2f}",
                            language="en",
                            face_quality=quality(has_video),
                            audio_quality=quality(has_audio),
                            text_quality=quality(has_text),
                            usable_for_micro=truth(has_video and has_audio and has_text),
                            usable_for_l4=truth(usable_l4),
                            raw_emotion=emotion,
                            raw_valence=valence,
                            raw_arousal=arousal,
                            weak_label_hint=weak_label,
                            label_source=f"{eval_path}#{utt_id}",
                        )
                    )
    return rows


def normalize_header(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.strip().lower()).strip("_")


def first_existing(row: dict[str, str], names: Iterable[str]) -> str:
    for name in names:
        value = row.get(name, "").strip()
        if value:
            return value
    return ""


def sniff_csvs(root: Path) -> list[Path]:
    wanted = ("label", "sentiment", "annotation", "metadata", "segments")
    return [
        path
        for path in sorted(root.rglob("*.csv"))
        if any(token in path.name.lower() for token in wanted)
    ]


def index_media_files(root: Path) -> dict[str, list[Path]]:
    media: dict[str, list[Path]] = {}
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in MEDIA_EXTENSIONS:
            continue
        stem = path.stem.lower()
        media.setdefault(stem, []).append(path)
    return media


def find_media(media: dict[str, list[Path]], keys: Iterable[str], prefer_audio: bool) -> Path | None:
    for key in keys:
        normalized = Path(key).stem.lower()
        for path in media.get(normalized, []):
            suffix = path.suffix.lower()
            if prefer_audio and suffix in {".wav", ".mp3", ".flac", ".m4a"}:
                return path
            if not prefer_audio and suffix in {".mp4", ".avi", ".mov", ".mkv"}:
                return path
        if normalized in media and media[normalized]:
            return media[normalized][0]
    return None


def build_mosei_like_rows(root: Path, dataset_name: str) -> list[SourceRow]:
    if not root.is_dir():
        return []
    media = index_media_files(root)
    rows: list[SourceRow] = []
    for csv_path in sniff_csvs(root):
        with csv_path.open(newline="", encoding="utf-8", errors="replace") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames:
                continue
            normalized_names = {name: normalize_header(name) for name in reader.fieldnames}
            for raw_row in reader:
                row = {normalized_names[name]: (value or "") for name, value in raw_row.items()}
                clip_id = first_existing(
                    row,
                    (
                        "input_clip",
                        "clip_id",
                        "segment_id",
                        "video_id",
                        "id",
                        "file",
                        "filename",
                        "utterance_id",
                    ),
                )
                video_id = first_existing(
                    row,
                    ("input_video_id", "video_id", "movie", "show", "file", "filename"),
                )
                if not clip_id and not video_id:
                    continue
                sentiment = first_existing(
                    row,
                    (
                        "sentiment",
                        "answer_sentiment",
                        "sentiment_label",
                        "label",
                        "annotation",
                        "mosi_sentiment",
                        "mosei_sentiment",
                    ),
                )
                emotion = first_existing(
                    row,
                    (
                        "emotion",
                        "emotion_label",
                        "raw_emotion",
                        "answer_anger",
                        "answer_disgust",
                        "answer_fear",
                        "answer_happiness",
                        "answer_sadness",
                        "answer_surprise",
                    ),
                )
                if not sentiment and not emotion:
                    continue
                start = first_existing(row, ("start", "start_time", "begin", "timestamp_start"))
                end = first_existing(row, ("end", "end_time", "finish", "timestamp_end"))
                source_key = (
                    f"{video_id}/{clip_id}" if video_id and clip_id else clip_id or video_id
                )
                media_keys = [
                    source_key,
                    clip_id,
                    video_id,
                    f"{video_id}_{clip_id}",
                    f"{clip_id}_{video_id}",
                    f"{video_id}[{clip_id}]",
                ]
                video_path = find_media(media, media_keys, prefer_audio=False)
                audio_path = find_media(media, media_keys, prefer_audio=True)
                has_video = video_path is not None and video_path.is_file()
                has_audio = audio_path is not None and audio_path.is_file()
                text_value = first_existing(row, ("text", "transcript", "utterance", "sentence"))
                text_path = f"{csv_path}#{clip_id or video_id}" if text_value or csv_path.is_file() else ""
                weak_label = weak_from_sentiment(sentiment) or EMOTION_WEAK_LABELS.get(
                    emotion.lower(), ""
                )
                usable_l4 = has_video and has_audio and bool(text_path) and bool(weak_label)
                rows.append(
                    SourceRow(
                        source_dataset=dataset_name,
                        source_split=first_existing(row, ("split", "partition", "mode")) or "unknown",
                        source_id=f"{dataset_name}/{source_key}",
                        video_path=str(video_path) if video_path else "",
                        audio_path=str(audio_path) if audio_path else "",
                        text_path=text_path,
                        start=f"{float(start):.2f}" if start else "0.00",
                        end=f"{float(end):.2f}" if end else "0.00",
                        language="en",
                        face_quality=quality(has_video),
                        audio_quality=quality(has_audio),
                        text_quality=quality(bool(text_path)),
                        usable_for_micro=truth(has_video and has_audio and bool(text_path)),
                        usable_for_l4=truth(usable_l4),
                        raw_emotion=emotion,
                        raw_sentiment=sentiment,
                        weak_label_hint=weak_label,
                        label_source=f"{csv_path}#{source_key}",
                    )
                )
    return dedupe_rows(rows)


def dedupe_rows(rows: list[SourceRow]) -> list[SourceRow]:
    seen: set[str] = set()
    unique: list[SourceRow] = []
    for row in rows:
        key = f"{row.source_dataset}:{row.source_id}:{row.label_source}"
        if key in seen:
            continue
        seen.add(key)
        unique.append(row)
    return unique


def assign_ids(rows: list[SourceRow], start_id: int) -> list[dict[str, str]]:
    return [row.with_ea_id(f"EAQ{index:06d}") for index, row in enumerate(rows, start=start_id)]


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iemocap-root", type=Path, default=Path("/root/autodl-tmp/data/datasets/iemocap"))
    parser.add_argument("--mosei-root", type=Path, default=Path("/root/autodl-tmp/data/datasets/mosei"))
    parser.add_argument("--mosi-root", type=Path, default=Path("/root/autodl-tmp/data/datasets/mosi"))
    parser.add_argument("--output-dir", type=Path, default=SOURCE_INDEX)
    parser.add_argument("--start-id", type=int, default=21)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    iemocap_rows = build_iemocap_rows(args.iemocap_root)
    mosei_rows = build_mosei_like_rows(args.mosei_root, "MOSEI")
    mosi_rows = build_mosei_like_rows(args.mosi_root, "MOSI")

    iemocap_index = assign_ids(iemocap_rows, args.start_id)
    mosei_index = assign_ids(mosei_rows + mosi_rows, args.start_id + len(iemocap_rows))

    write_csv(args.output_dir / "iemocap_index.csv", iemocap_index)
    write_csv(args.output_dir / "mosei_index.csv", mosei_index)

    print(f"Wrote {len(iemocap_index)} rows to {args.output_dir / 'iemocap_index.csv'}")
    print(f"Wrote {len(mosei_index)} rows to {args.output_dir / 'mosei_index.csv'}")


if __name__ == "__main__":
    main()
