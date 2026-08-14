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
from collections import defaultdict
from dataclasses import dataclass, field
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
DATASET_ID_RANGES = {
    "IEMOCAP": (300000, 399999),
    "MOSEI": (400000, 499999),
    "MOSI": (500000, 599999),
}

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
    wanted = ("label", "sentiment", "annotation", "metadata", "segments", "batch")
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


@dataclass
class MoseiAnnotation:
    dataset_name: str
    video_id: str
    clip_id: str
    source_key: str
    sentiments: list[float] = field(default_factory=list)
    emotions: dict[str, list[float]] = field(default_factory=lambda: defaultdict(list))
    label_sources: set[str] = field(default_factory=set)


def parse_number(value: str) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def format_number(value: float) -> str:
    return f"{value:.3f}".rstrip("0").rstrip(".")


def read_mosei_segments(root: Path) -> dict[str, dict[str, tuple[Path, str, str]]]:
    transcript_root = root / "extracted" / "Raw" / "Transcript" / "Segmented" / "Combined"
    segments: dict[str, dict[str, tuple[Path, str, str]]] = {}
    if not transcript_root.is_dir():
        return segments
    for path in sorted(transcript_root.glob("*.txt")):
        video_id = path.stem
        clips: dict[str, tuple[Path, str, str]] = {}
        with path.open(encoding="utf-8", errors="replace") as handle:
            for line in handle:
                parts = line.rstrip("\n").split("___", 4)
                if len(parts) < 5:
                    continue
                _, clip_id, start, end, text = parts
                if text.strip():
                    clips[clip_id] = (path, start, end)
        if clips:
            segments[video_id] = clips
    return segments


def mosei_segment_video(root: Path, video_id: str, clip_id: str) -> Path:
    return root / "extracted" / "Raw" / "Videos" / "Segmented" / "Combined" / f"{video_id}_{clip_id}.mp4"


def mosei_audio(root: Path, video_id: str) -> Path:
    return root / "extracted" / "Raw" / "Audio" / "Full" / "WAV_16000" / f"{video_id}.wav"


def collect_mosei_annotations(root: Path, dataset_name: str) -> dict[str, MoseiAnnotation]:
    annotations: dict[str, MoseiAnnotation] = {}
    for csv_path in sniff_csvs(root):
        with csv_path.open(newline="", encoding="utf-8", errors="replace") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames:
                continue
            normalized_names = {name: normalize_header(name) for name in reader.fieldnames}
            for raw_row in reader:
                row = {normalized_names[name]: (value or "") for name, value in raw_row.items()}
                clip_id = first_existing(row, ("input_clip", "clip_id", "segment_id"))
                video_id = first_existing(row, ("input_video_id", "video_id"))
                if not video_id or not clip_id:
                    continue
                sentiment = first_existing(
                    row,
                    (
                        "answer_sentiment",
                        "sentiment",
                        "sentiment_label",
                        "mosi_sentiment",
                        "mosei_sentiment",
                    ),
                )
                emotion_values = {
                    "anger": first_existing(row, ("answer_anger", "anger")),
                    "disgust": first_existing(row, ("answer_disgust", "disgust")),
                    "fear": first_existing(row, ("answer_fear", "fear")),
                    "happiness": first_existing(row, ("answer_happiness", "happiness")),
                    "sadness": first_existing(row, ("answer_sadness", "sadness")),
                    "surprise": first_existing(row, ("answer_surprise", "surprise")),
                }
                sentiment_number = parse_number(sentiment)
                if sentiment_number is None and not any(emotion_values.values()):
                    continue
                source_key = f"{video_id}/{clip_id}"
                annotation = annotations.setdefault(
                    source_key,
                    MoseiAnnotation(
                        dataset_name=dataset_name,
                        video_id=video_id,
                        clip_id=clip_id,
                        source_key=source_key,
                    ),
                )
                if sentiment_number is not None:
                    annotation.sentiments.append(sentiment_number)
                for emotion, value in emotion_values.items():
                    number = parse_number(value)
                    if number is not None:
                        annotation.emotions[emotion].append(number)
                annotation.label_sources.add(
                    f"{csv_path}#Input.VIDEO_ID={video_id}&Input.CLIP={clip_id}"
                )
    return annotations


def summarize_emotions(annotation: MoseiAnnotation) -> str:
    parts = []
    for emotion in sorted(annotation.emotions):
        values = annotation.emotions[emotion]
        if values:
            parts.append(f"{emotion}={format_number(sum(values) / len(values))}")
    return ";".join(parts)


def build_mosei_like_rows(root: Path, dataset_name: str) -> list[SourceRow]:
    if not root.is_dir():
        return []
    rows: list[SourceRow] = []
    segments = read_mosei_segments(root)
    annotations = collect_mosei_annotations(root, dataset_name)
    for annotation in annotations.values():
        segment = segments.get(annotation.video_id, {}).get(annotation.clip_id)
        transcript_path, start, end = segment if segment else (None, "", "")
        video_path = mosei_segment_video(root, annotation.video_id, annotation.clip_id)
        audio_path = mosei_audio(root, annotation.video_id)
        has_video = video_path.is_file()
        has_audio = audio_path.is_file()
        has_text = transcript_path is not None and transcript_path.is_file()
        sentiment = ""
        if annotation.sentiments:
            sentiment = format_number(sum(annotation.sentiments) / len(annotation.sentiments))
        weak_label = weak_from_sentiment(sentiment)
        usable_l4 = has_video and has_audio and has_text and bool(weak_label)
        rows.append(
            SourceRow(
                source_dataset=dataset_name,
                source_split="unknown",
                source_id=f"{dataset_name}/{annotation.source_key}",
                video_path=str(video_path) if has_video else "",
                audio_path=str(audio_path) if has_audio else "",
                text_path=(
                    f"{transcript_path}#clip={annotation.clip_id}" if transcript_path else ""
                ),
                start=f"{float(start):.2f}" if start else "",
                end=f"{float(end):.2f}" if end else "",
                language="en",
                face_quality=quality(has_video),
                audio_quality=quality(has_audio),
                text_quality=quality(has_text),
                usable_for_micro=truth(has_video and has_audio and has_text),
                usable_for_l4=truth(usable_l4),
                raw_emotion=summarize_emotions(annotation),
                raw_sentiment=sentiment,
                weak_label_hint=weak_label,
                label_source=";".join(sorted(annotation.label_sources)),
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


def load_existing_ids(paths: Iterable[Path]) -> dict[tuple[str, str], str]:
    assignments: dict[tuple[str, str], str] = {}
    for path in paths:
        if not path.is_file():
            continue
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                dataset = row.get("source_dataset", "")
                source_id = row.get("source_id", "")
                ea_id = row.get("ea_id", "")
                if dataset and source_id and ea_id:
                    assignments[(dataset, source_id)] = ea_id
    return assignments


def id_number(ea_id: str) -> int | None:
    match = re.fullmatch(r"EAQ(\d{6})", ea_id)
    return int(match.group(1)) if match else None


def assign_dataset_ids(
    rows: list[SourceRow],
    dataset_name: str,
    existing_ids: dict[tuple[str, str], str],
) -> list[dict[str, str]]:
    start_id, end_id = DATASET_ID_RANGES[dataset_name]
    next_id = start_id
    output: list[dict[str, str]] = []
    used: set[str] = set()
    for row in rows:
        existing = existing_ids.get((row.source_dataset, row.source_id), "")
        number = id_number(existing)
        if number is not None and start_id <= number <= end_id and existing not in used:
            ea_id = existing
        else:
            while f"EAQ{next_id:06d}" in used:
                next_id += 1
            if next_id > end_id:
                raise SystemExit(f"{dataset_name} ID range exhausted")
            ea_id = f"EAQ{next_id:06d}"
            next_id += 1
        used.add(ea_id)
        output.append(row.with_ea_id(ea_id))
    return output


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
    parser.add_argument(
        "--id-registry",
        type=Path,
        nargs="*",
        default=[],
        help="Existing source index CSV files used to preserve prior assignments.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    iemocap_rows = build_iemocap_rows(args.iemocap_root)
    mosei_rows = build_mosei_like_rows(args.mosei_root, "MOSEI")
    mosi_rows = build_mosei_like_rows(args.mosi_root, "MOSI")

    registry_paths = args.id_registry or [
        args.output_dir / "iemocap_index.csv",
        args.output_dir / "mosei_index.csv",
    ]
    existing_ids = load_existing_ids(registry_paths)
    iemocap_index = assign_dataset_ids(iemocap_rows, "IEMOCAP", existing_ids)
    mosei_index = (
        assign_dataset_ids(mosei_rows, "MOSEI", existing_ids)
        + assign_dataset_ids(mosi_rows, "MOSI", existing_ids)
    )

    write_csv(args.output_dir / "iemocap_index.csv", iemocap_index)
    write_csv(args.output_dir / "mosei_index.csv", mosei_index)

    print(f"Wrote {len(iemocap_index)} rows to {args.output_dir / 'iemocap_index.csv'}")
    print(f"Wrote {len(mosei_index)} rows to {args.output_dir / 'mosei_index.csv'}")


if __name__ == "__main__":
    main()
