"""MELD / MUStARD source index generation and reporting (Issue #9)."""

from __future__ import annotations

import csv
import json
import re
import urllib.request
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

MUSTARD_JSON_URL = (
    "https://raw.githubusercontent.com/soujanyaporia/MUStARD/master/data/sarcasm_data.json"
)

TEMPLATE_COLUMNS = (
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
)
EXTRA_COLUMNS = ("is_sarcasm_candidate", "candidate_reason")
INDEX_COLUMNS = TEMPLATE_COLUMNS + EXTRA_COLUMNS
ALLOCATION_COLUMNS = ("source_dataset", "source_id", "ea_id")

MELD_SPLITS = ("train", "dev", "test")
# Server layout under /root/autodl-tmp/data/datasets/meld/extracted/MELD.Raw/
MELD_VIDEO_DIRS = {
    "train": "extracted/MELD.Raw/train_splits",
    "dev": "extracted/MELD.Raw/dev_splits_complete",
    "test": "extracted/MELD.Raw/output_repeated_splits_test",
}

MELD_EA_ID_START = 100_000
MUSTARD_EA_ID_START = 200_000
# Historical M1 seed IDs must never be minted for new dataset rows.
RESERVED_SEED_EA_IDS = {f"EAQ{i:06d}" for i in range(1, 21)}

EMOTION_POLARITY = {
    "joy": "pos",
    "surprise": "pos",
    "anger": "neg",
    "disgust": "neg",
    "fear": "neg",
    "sadness": "neg",
    "neutral": "neu",
}
SENTIMENT_POLARITY = {
    "positive": "pos",
    "negative": "neg",
    "neutral": "neu",
}

TIMESTAMP_RE = re.compile(
    r"^(?P<h>\d+):(?P<m>\d{2}):(?P<s>\d{2})[,.](?P<ms>\d+)$"
)


class DialogueIndexError(ValueError):
    """Raised when dialogue index generation fails."""


@dataclass(frozen=True)
class MeldUtterance:
    split: str
    dialogue_id: int
    utterance_id: int
    utterance: str
    emotion: str
    sentiment: str
    start_time: str
    end_time: str

    @property
    def sort_key(self) -> tuple[int, int, int]:
        split_rank = {name: idx for idx, name in enumerate(MELD_SPLITS)}
        return (
            split_rank.get(self.split, 99),
            self.dialogue_id,
            self.utterance_id,
        )

    @property
    def source_id(self) -> str:
        return f"MELD/{self.split}/dia{self.dialogue_id}/utt{self.utterance_id}"


@dataclass(frozen=True)
class MustardClip:
    utterance_id: str
    utterance: str
    sarcasm: bool
    show: str
    speaker: str

    @property
    def source_id(self) -> str:
        return self.utterance_id


def fetch_mustard_json(dest: Path) -> Path:
    """Download official MUStARD sarcasm_data.json."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(MUSTARD_JSON_URL, headers={"User-Agent": "ea-quad-overlay/1.0"})
    with urllib.request.urlopen(request, timeout=120) as response:
        dest.write_bytes(response.read())
    return dest


def parse_timestamp_to_sec(value: str) -> float | None:
    text = (value or "").strip().strip('"')
    if not text:
        return None
    match = TIMESTAMP_RE.match(text)
    if not match:
        return None
    hours = int(match.group("h"))
    minutes = int(match.group("m"))
    seconds = int(match.group("s"))
    millis = int(match.group("ms").ljust(3, "0")[:3])
    return hours * 3600 + minutes * 60 + seconds + millis / 1000.0


def duration_from_timestamps(start: str, end: str) -> float | None:
    start_sec = parse_timestamp_to_sec(start)
    end_sec = parse_timestamp_to_sec(end)
    if start_sec is None or end_sec is None:
        return None
    duration = end_sec - start_sec
    if duration <= 0:
        return None
    return round(duration, 4)


def emotion_sentiment_mismatch(emotion: str, sentiment: str) -> bool:
    emo = EMOTION_POLARITY.get((emotion or "").strip().lower())
    sent = SENTIMENT_POLARITY.get((sentiment or "").strip().lower())
    if emo is None or sent is None:
        return False
    if emo == "neu" or sent == "neu":
        return False
    return emo != sent


def bool_str(value: bool) -> str:
    return "true" if value else "false"


def format_end(duration: float | None) -> str:
    if duration is None:
        return ""
    return f"{duration:.2f}"


def media_exists(path: str, check_media: bool) -> bool | None:
    """Return True/False when checking, else None (unknown)."""
    if not check_media:
        return None
    if not path:
        return False
    candidate = Path(path)
    if candidate.is_file():
        return True
    if path.startswith("/root/") or path.startswith("/data/"):
        return None
    return False


def resolve_mustard_video_path(utterance_id: str, mustard_root: str) -> str:
    root = mustard_root.rstrip("/\\").replace("\\", "/")
    # AutoDL layout confirmed by Issue #9 review.
    return f"{root}/raw/clips/utterances_final/{utterance_id}.mp4"


def probe_mustard_video_path(utterance_id: str, mustard_root: Path) -> Path | None:
    relatives = (
        mustard_root / "raw" / "clips" / "utterances_final" / f"{utterance_id}.mp4",
        mustard_root / "raw" / "clips" / "utterances_final" / f"{utterance_id}_c.mp4",
        mustard_root / "utterances_final" / f"{utterance_id}.mp4",
        mustard_root / "videos" / f"{utterance_id}.mp4",
        mustard_root / f"{utterance_id}.mp4",
    )
    for path in relatives:
        if path.is_file():
            return path
    return None


def mustard_json_server_path(mustard_root: str) -> str:
    root = mustard_root.rstrip("/\\").replace("\\", "/")
    return f"{root}/data/sarcasm_data.json"


def quality_flags(
    *,
    text: str,
    video_present: bool | None,
    duration: float | None,
    sarcasm_candidate: bool,
) -> tuple[str, str, str, str, str]:
    """Return face/audio/text quality and usable_for_micro / usable_for_l4.

    When media was not verified (``video_present is None``), face/audio are
    ``missing`` and ``usable_for_micro`` is always false per
    ``docs/source_index_contract.md``.
    """
    if text.strip():
        text_quality = "high"
    else:
        text_quality = "missing"

    if video_present is True:
        face_quality = "medium"
        audio_quality = "medium"
    else:
        # False (confirmed missing) or None (unchecked): not resolvable yet.
        face_quality = "missing"
        audio_quality = "missing"

    if video_present is not True:
        usable_micro = False
    elif duration is None:
        # Atomic clip with confirmed media; duration optional.
        usable_micro = True
    else:
        usable_micro = 0.8 <= duration <= 30.0

    usable_l4 = text_quality != "missing" and (
        sarcasm_candidate or face_quality != "missing"
    )
    return (
        face_quality,
        audio_quality,
        text_quality,
        bool_str(usable_micro),
        bool_str(usable_l4),
    )


def read_meld_split(ann_csv: Path, split: str) -> list[MeldUtterance]:
    if not ann_csv.is_file():
        raise DialogueIndexError(f"MELD annotation missing: {ann_csv}")
    rows: list[MeldUtterance] = []
    with ann_csv.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {"Utterance", "Emotion", "Sentiment", "Dialogue_ID", "Utterance_ID"}
        if reader.fieldnames is None or not required.issubset(set(reader.fieldnames)):
            raise DialogueIndexError(f"{ann_csv}: missing required MELD columns")
        for row_number, row in enumerate(reader, start=2):
            try:
                dialogue_id = int(str(row["Dialogue_ID"]).strip())
                utterance_id = int(str(row["Utterance_ID"]).strip())
            except (KeyError, ValueError) as exc:
                raise DialogueIndexError(
                    f"{ann_csv}:{row_number}: invalid Dialogue_ID/Utterance_ID"
                ) from exc
            rows.append(
                MeldUtterance(
                    split=split,
                    dialogue_id=dialogue_id,
                    utterance_id=utterance_id,
                    utterance=(row.get("Utterance") or "").strip(),
                    emotion=(row.get("Emotion") or "").strip(),
                    sentiment=(row.get("Sentiment") or "").strip(),
                    start_time=(row.get("StartTime") or "").strip(),
                    end_time=(row.get("EndTime") or "").strip(),
                )
            )
    return rows


def load_meld_utterances(meld_root: Path) -> list[MeldUtterance]:
    ann_dir = meld_root / "annotations"
    all_rows: list[MeldUtterance] = []
    for split in MELD_SPLITS:
        all_rows.extend(read_meld_split(ann_dir / f"{split}_sent_emo.csv", split))
    all_rows.sort(key=lambda item: item.sort_key)
    if not all_rows:
        raise DialogueIndexError(f"no MELD utterances under {ann_dir}")
    return all_rows


def load_mustard_clips(json_path: Path) -> list[MustardClip]:
    if not json_path.is_file():
        raise DialogueIndexError(f"MUStARD json missing: {json_path}")
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not payload:
        raise DialogueIndexError(f"{json_path}: expected non-empty object")
    clips: list[MustardClip] = []
    for key in sorted(payload.keys()):
        item = payload[key]
        if not isinstance(item, dict):
            raise DialogueIndexError(f"{json_path}: entry {key} is not an object")
        sarcasm = item.get("sarcasm")
        if not isinstance(sarcasm, bool):
            raise DialogueIndexError(f"{json_path}: entry {key} missing bool sarcasm")
        clips.append(
            MustardClip(
                utterance_id=str(key),
                utterance=str(item.get("utterance") or "").strip(),
                sarcasm=sarcasm,
                show=str(item.get("show") or "").strip(),
                speaker=str(item.get("speaker") or "").strip(),
            )
        )
    return clips


def meld_video_path(meld_root: str, split: str, dialogue_id: int, utterance_id: int) -> str:
    root = meld_root.rstrip("/\\").replace("\\", "/")
    rel = MELD_VIDEO_DIRS[split]
    return f"{root}/{rel}/dia{dialogue_id}_utt{utterance_id}.mp4"


def meld_text_path(meld_root: str, split: str, dialogue_id: int, utterance_id: int) -> str:
    root = meld_root.rstrip("/\\").replace("\\", "/")
    return (
        f"{root}/annotations/{split}_sent_emo.csv"
        f"#Dialogue_ID={dialogue_id}&Utterance_ID={utterance_id}"
    )


def mustard_text_path(json_path: str, utterance_id: str) -> str:
    return f"{json_path.replace(chr(92), '/')}#utterance_id={utterance_id}"


def read_m1_meld_reservations(m1_index_path: Path) -> dict[str, str]:
    """Map MELD source_id -> historical M1 ea_id."""
    if not m1_index_path.is_file():
        return {}
    reserved: dict[str, str] = {}
    with m1_index_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("source_dataset") != "MELD":
                continue
            source_id = (row.get("source_id") or "").strip()
            ea_id = (row.get("ea_id") or "").strip()
            if source_id and ea_id:
                reserved[source_id] = ea_id
    return reserved


def read_m1_meld_seed_metadata(m1_index_path: Path) -> dict[str, dict[str, str]]:
    """Reuse measured M1 durations and accepted quality flags for seed rows."""
    if not m1_index_path.is_file():
        return {}
    metadata: dict[str, dict[str, str]] = {}
    with m1_index_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("source_dataset") != "MELD":
                continue
            source_id = (row.get("source_id") or "").strip()
            if not source_id:
                continue
            metadata[source_id] = {
                "start": row.get("start") or "0.00",
                "end": row.get("end") or "",
                "face_quality": row.get("face_quality") or "high",
                "audio_quality": row.get("audio_quality") or "high",
                "text_quality": row.get("text_quality") or "high",
                "usable_for_micro": row.get("usable_for_micro") or "true",
                "usable_for_l4": row.get("usable_for_l4") or "true",
            }
    return metadata


def read_allocation_map(path: Path | None) -> dict[tuple[str, str], str]:
    """Load persisted (source_dataset, source_id) -> ea_id assignments."""
    if path is None or not path.is_file():
        return {}
    mapping: dict[tuple[str, str], str] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            dataset = (row.get("source_dataset") or "").strip()
            source_id = (row.get("source_id") or "").strip()
            ea_id = (row.get("ea_id") or "").strip()
            if dataset and source_id and ea_id:
                mapping[(dataset, source_id)] = ea_id
    return mapping


def read_index_allocation(path: Path | None, dataset: str) -> dict[tuple[str, str], str]:
    """Recover allocation from an existing dataset index CSV."""
    if path is None or not path.is_file():
        return {}
    mapping: dict[tuple[str, str], str] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("source_dataset") != dataset:
                continue
            source_id = (row.get("source_id") or "").strip()
            ea_id = (row.get("ea_id") or "").strip()
            if source_id and ea_id:
                mapping[(dataset, source_id)] = ea_id
    return mapping


def write_allocation_map(path: Path, mapping: Mapping[tuple[str, str], str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {"source_dataset": dataset, "source_id": source_id, "ea_id": ea_id}
        for (dataset, source_id), ea_id in sorted(
            mapping.items(), key=lambda item: (item[0][0], item[1])
        )
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(ALLOCATION_COLUMNS))
        writer.writeheader()
        writer.writerows(rows)


def _next_free_id(start: int, used: set[str], end_exclusive: int) -> int:
    current = start
    while current < end_exclusive:
        candidate = f"EAQ{current:06d}"
        if candidate not in used and candidate not in RESERVED_SEED_EA_IDS:
            return current
        current += 1
    raise DialogueIndexError(f"exhausted EA ID range starting at {start}")


def assign_ea_ids(
    *,
    dataset: str,
    source_ids: Sequence[str],
    range_start: int,
    range_end_exclusive: int,
    seed_reservations: Mapping[str, str],
    existing_map: Mapping[tuple[str, str], str],
) -> dict[str, str]:
    """Assign stable ea_id values for one dataset.

    Priority: M1 seed reservation > persisted allocation map > new IDs in range.
    """
    assignments: dict[str, str] = {}
    used: set[str] = set(RESERVED_SEED_EA_IDS)
    used.update(existing_map.values())
    used.update(seed_reservations.values())

    for source_id in source_ids:
        if source_id in seed_reservations:
            ea_id = seed_reservations[source_id]
            assignments[source_id] = ea_id
            used.add(ea_id)
            continue
        key = (dataset, source_id)
        if key in existing_map:
            ea_id = existing_map[key]
            assignments[source_id] = ea_id
            used.add(ea_id)
            continue

    next_id = range_start
    for source_id in source_ids:
        if source_id in assignments:
            continue
        next_id = _next_free_id(next_id, used, range_end_exclusive)
        ea_id = f"EAQ{next_id:06d}"
        assignments[source_id] = ea_id
        used.add(ea_id)
        next_id += 1
    return assignments


def build_meld_row(
    record: MeldUtterance,
    *,
    ea_id: str,
    meld_root: str,
    check_media: bool,
    local_meld_root: Path | None = None,
    seed_meta: Mapping[str, str] | None = None,
) -> dict[str, str]:
    video = meld_video_path(meld_root, record.split, record.dialogue_id, record.utterance_id)
    present = media_exists(video, check_media)
    if present is None and check_media and local_meld_root is not None:
        local = (
            local_meld_root
            / MELD_VIDEO_DIRS[record.split]
            / f"dia{record.dialogue_id}_utt{record.utterance_id}.mp4"
        )
        if local.is_file():
            present = True
        elif (local_meld_root / "extracted").exists():
            present = any(
                local_meld_root.rglob(f"dia{record.dialogue_id}_utt{record.utterance_id}.mp4")
            )

    sarcasm = emotion_sentiment_mismatch(record.emotion, record.sentiment)
    subtitle_duration = duration_from_timestamps(record.start_time, record.end_time)

    if seed_meta:
        # Preserve already-accepted M1 seed evidence.
        face_q = seed_meta["face_quality"]
        audio_q = seed_meta["audio_quality"]
        text_q = seed_meta["text_quality"]
        usable_micro = seed_meta["usable_for_micro"]
        usable_l4 = seed_meta["usable_for_l4"]
        start = seed_meta.get("start") or "0.00"
        end = seed_meta.get("end") or ""
    else:
        face_q, audio_q, text_q, usable_micro, usable_l4 = quality_flags(
            text=record.utterance,
            video_present=present,
            duration=subtitle_duration if present is True else None,
            sarcasm_candidate=sarcasm,
        )
        # Atomic utterance mp4: leave end empty unless media was probed with a duration.
        start = "0.00"
        end = format_end(subtitle_duration) if present is True and subtitle_duration else ""

    return {
        "ea_id": ea_id,
        "source_dataset": "MELD",
        "source_split": record.split,
        "source_id": record.source_id,
        "video_path": video,
        "audio_path": video,
        "text_path": meld_text_path(meld_root, record.split, record.dialogue_id, record.utterance_id),
        "start": start,
        "end": end,
        "language": "en",
        "face_quality": face_q,
        "audio_quality": audio_q,
        "text_quality": text_q,
        "usable_for_micro": usable_micro,
        "usable_for_l4": usable_l4,
        "is_sarcasm_candidate": bool_str(sarcasm),
        "candidate_reason": "meld_emotion_sentiment_mismatch" if sarcasm else "",
    }


def build_mustard_row(
    clip: MustardClip,
    *,
    ea_id: str,
    mustard_root: str,
    json_path: str,
    check_media: bool,
    local_mustard_root: Path | None = None,
) -> dict[str, str]:
    video = resolve_mustard_video_path(clip.utterance_id, mustard_root)
    present = media_exists(video, check_media)
    if present is None and check_media and local_mustard_root is not None:
        local = probe_mustard_video_path(clip.utterance_id, local_mustard_root)
        if local is not None:
            present = True
        elif local_mustard_root.exists():
            present = False

    sarcasm = clip.sarcasm
    face_q, audio_q, text_q, usable_micro, usable_l4 = quality_flags(
        text=clip.utterance,
        video_present=present,
        duration=None,
        sarcasm_candidate=sarcasm,
    )
    return {
        "ea_id": ea_id,
        "source_dataset": "MUStARD",
        "source_split": "all",
        "source_id": clip.source_id,
        "video_path": video,
        "audio_path": video,
        "text_path": mustard_text_path(json_path, clip.utterance_id),
        "start": "",
        "end": "",
        "language": "en",
        "face_quality": face_q,
        "audio_quality": audio_q,
        "text_quality": text_q,
        "usable_for_micro": usable_micro,
        "usable_for_l4": usable_l4,
        "is_sarcasm_candidate": bool_str(sarcasm),
        "candidate_reason": "mustard_label" if sarcasm else "",
    }


def write_index_csv(path: Path, rows: Sequence[Mapping[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(INDEX_COLUMNS))
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col, "") for col in INDEX_COLUMNS})


def summarize_rows(
    rows: Sequence[Mapping[str, str]],
    *,
    seed_inherited: int = 0,
) -> dict[str, Any]:
    split_counts = Counter(row["source_split"] for row in rows)
    face_counts = Counter(row["face_quality"] for row in rows)
    sarcasm_n = sum(1 for row in rows if row.get("is_sarcasm_candidate") == "true")
    usable_l4 = sum(1 for row in rows if row.get("usable_for_l4") == "true")
    usable_micro = sum(1 for row in rows if row.get("usable_for_micro") == "true")
    media_unknown = sum(1 for row in rows if row.get("face_quality") == "missing")
    sorted_ids = sorted(row["ea_id"] for row in rows)
    new_ids = [row["ea_id"] for row in rows if row["ea_id"] not in RESERVED_SEED_EA_IDS]
    return {
        "total": len(rows),
        "splits": dict(sorted(split_counts.items())),
        "face_quality": dict(sorted(face_counts.items())),
        "sarcasm_candidates": sarcasm_n,
        "usable_for_l4": usable_l4,
        "usable_for_micro": usable_micro,
        "media_unverified_or_missing": media_unknown,
        "ea_id_first": sorted_ids[0] if sorted_ids else "",
        "ea_id_last": sorted_ids[-1] if sorted_ids else "",
        "seed_rows_inherited": seed_inherited,
        "new_rows_allocated": len(rows) - seed_inherited,
        "new_ea_id_first": min(new_ids) if new_ids else "",
        "new_ea_id_last": max(new_ids) if new_ids else "",
    }


def read_m1_meld_source_ids(m1_index_path: Path) -> list[str]:
    if not m1_index_path.is_file():
        return []
    ids: list[str] = []
    with m1_index_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("source_dataset") == "MELD":
                ids.append(row["source_id"])
    return ids


def generate_dialogue_indexes(
    *,
    meld_ann_root: Path,
    mustard_json: Path,
    meld_path_root: str,
    mustard_path_root: str,
    meld_output: Path,
    mustard_output: Path,
    m1_index_path: Path | None = None,
    allocation_map_path: Path | None = None,
    check_media: bool = False,
) -> tuple[list[dict[str, str]], list[dict[str, str]], dict[str, Any], dict[str, Any]]:
    meld_records = load_meld_utterances(meld_ann_root)
    mustard_clips = load_mustard_clips(mustard_json)

    local_meld = meld_ann_root if check_media else None
    local_mustard = (
        Path(mustard_path_root) if check_media and not mustard_path_root.startswith("/root/") else None
    )
    if check_media and local_mustard is None and mustard_json.parent.exists():
        # Prefer probing under mustard dataset root siblings when available.
        candidate = mustard_json.parent.parent if mustard_json.parent.name == "data" else mustard_json.parent
        local_mustard = candidate

    if mustard_path_root.startswith("/root/") or mustard_path_root.startswith("/data/"):
        mustard_json_for_text = mustard_json_server_path(mustard_path_root)
    else:
        mustard_json_for_text = str(mustard_json).replace("\\", "/")

    m1_path = m1_index_path or Path()
    seed_reservations = read_m1_meld_reservations(m1_path)
    seed_meta = read_m1_meld_seed_metadata(m1_path)

    existing = read_allocation_map(allocation_map_path)
    existing.update(read_index_allocation(meld_output, "MELD"))
    existing.update(read_index_allocation(mustard_output, "MUStARD"))

    meld_ids = assign_ea_ids(
        dataset="MELD",
        source_ids=[record.source_id for record in meld_records],
        range_start=MELD_EA_ID_START,
        range_end_exclusive=MUSTARD_EA_ID_START,
        seed_reservations=seed_reservations,
        existing_map=existing,
    )
    mustard_ids = assign_ea_ids(
        dataset="MUStARD",
        source_ids=[clip.source_id for clip in mustard_clips],
        range_start=MUSTARD_EA_ID_START,
        range_end_exclusive=300_000,
        seed_reservations={},
        existing_map=existing,
    )

    meld_rows: list[dict[str, str]] = []
    for record in meld_records:
        meld_rows.append(
            build_meld_row(
                record,
                ea_id=meld_ids[record.source_id],
                meld_root=meld_path_root,
                check_media=check_media,
                local_meld_root=local_meld,
                seed_meta=seed_meta.get(record.source_id),
            )
        )

    mustard_rows: list[dict[str, str]] = []
    for clip in mustard_clips:
        mustard_rows.append(
            build_mustard_row(
                clip,
                ea_id=mustard_ids[clip.source_id],
                mustard_root=mustard_path_root,
                json_path=mustard_json_for_text,
                check_media=check_media,
                local_mustard_root=local_mustard,
            )
        )

    write_index_csv(meld_output, meld_rows)
    write_index_csv(mustard_output, mustard_rows)

    combined_map: dict[tuple[str, str], str] = dict(existing)
    for source_id, ea_id in meld_ids.items():
        combined_map[("MELD", source_id)] = ea_id
    for source_id, ea_id in mustard_ids.items():
        combined_map[("MUStARD", source_id)] = ea_id
    if allocation_map_path is not None:
        write_allocation_map(allocation_map_path, combined_map)

    meld_summary = summarize_rows(meld_rows, seed_inherited=len(seed_reservations))
    mustard_summary = summarize_rows(mustard_rows, seed_inherited=0)
    return meld_rows, mustard_rows, meld_summary, mustard_summary


def render_dialogue_report(
    *,
    meld_summary: Mapping[str, Any],
    mustard_summary: Mapping[str, Any],
    meld_path_root: str,
    mustard_path_root: str,
    meld_ann_source: str,
    mustard_json_source: str,
    meld_output: str,
    mustard_output: str,
    allocation_map_source: str,
    m1_meld_ids: Sequence[str],
    meld_rows: Sequence[Mapping[str, str]],
    check_media: bool,
    generation_command: str,
) -> str:
    by_source = {row["source_id"]: row for row in meld_rows}
    m1_hits = [sid for sid in m1_meld_ids if sid in by_source]
    m1_misses = [sid for sid in m1_meld_ids if sid not in by_source]
    m1_id_mismatches = [
        f"{sid} expected seed id missing or remapped to {by_source[sid]['ea_id']}"
        for sid in m1_hits
        if sid in by_source and by_source[sid]["ea_id"] not in RESERVED_SEED_EA_IDS
    ]
    # Stronger check: compare against reservations when available via ea_id in 000012-000020
    seed_ok = sum(
        1
        for sid in m1_hits
        if by_source[sid]["ea_id"] in RESERVED_SEED_EA_IDS
    )

    lines = [
        "# Dialogue Dataset Index Report (MELD / MUStARD)",
        "",
        "GitHub issue: <https://github.com/cuiyuxun-droid/ea-quad-overlay/issues/9>",
        "",
        "## Scope",
        "",
        "- Build utterance-level `source_index` for MELD and MUStARD.",
        "- Mark face/audio/text usability for micro and L4 workflows.",
        "- Explicitly flag sarcasm candidates via extension columns.",
        "- Preserve M1 MELD seed `ea_id` values per `docs/source_index_contract.md`.",
        "",
        "## Outputs",
        "",
        f"- `{meld_output}`",
        f"- `{mustard_output}`",
        f"- allocation map: `{allocation_map_source}`",
        "",
        "## Path Roots",
        "",
        f"- MELD path root (written into CSV): `{meld_path_root}`",
        f"- MUStARD path root (written into CSV): `{mustard_path_root}`",
        f"- MELD annotations used for generation: `{meld_ann_source}`",
        f"- MUStARD JSON used for generation: `{mustard_json_source}`",
        f"- `--check-media`: `{bool_str(check_media)}`",
        f"- generation command: `{generation_command}`",
        "",
        "## Allocation",
        "",
        "```text",
        "dataset: MELD",
        f"first_ea_id: {meld_summary.get('ea_id_first')}",
        f"last_ea_id: {meld_summary.get('ea_id_last')}",
        f"seed_rows_inherited: {meld_summary.get('seed_rows_inherited')}",
        f"new_rows_allocated: {meld_summary.get('new_rows_allocated')}",
        f"new_ea_id_first: {meld_summary.get('new_ea_id_first')}",
        f"new_ea_id_last: {meld_summary.get('new_ea_id_last')}",
        "allocation_map_source: source_index/m1_sample_20.csv + "
        f"{allocation_map_source} + docs/source_index_contract.md",
        "```",
        "",
        "```text",
        "dataset: MUStARD",
        f"first_ea_id: {mustard_summary.get('ea_id_first')}",
        f"last_ea_id: {mustard_summary.get('ea_id_last')}",
        f"seed_rows_inherited: {mustard_summary.get('seed_rows_inherited')}",
        f"new_rows_allocated: {mustard_summary.get('new_rows_allocated')}",
        "allocation_map_source: "
        f"{allocation_map_source} + docs/source_index_contract.md",
        "```",
        "",
        "## Schema Extension",
        "",
        "Canonical template columns plus:",
        "",
        "- `is_sarcasm_candidate`: `true` / `false`",
        "- `candidate_reason`: `meld_emotion_sentiment_mismatch` or `mustard_label`",
        "",
        "## MELD Summary",
        "",
        f"- Total utterances: **{meld_summary['total']}**",
        f"- Split counts: `{meld_summary['splits']}`",
        f"- Face quality: `{meld_summary['face_quality']}`",
        f"- Media unverified or missing (`face_quality=missing`): "
        f"**{meld_summary['media_unverified_or_missing']}**",
        f"- Sarcasm candidates (emotion/sentiment polarity mismatch): "
        f"**{meld_summary['sarcasm_candidates']}**",
        f"- `usable_for_l4=true`: **{meld_summary['usable_for_l4']}**",
        f"- `usable_for_micro=true`: **{meld_summary['usable_for_micro']}**",
        "",
        "### Traceability example",
        "",
        "- `source_id`: `MELD/train/dia0/utt0`",
        "- `text_path`: `{annotations}/train_sent_emo.csv#Dialogue_ID=0&Utterance_ID=0`",
        "- `video_path`: `{extracted}/MELD.Raw/train_splits/dia0_utt0.mp4`",
        "- `video_path` (dev): `{extracted}/MELD.Raw/dev_splits_complete/dia*_utt*.mp4`",
        "",
        "## MUStARD Summary",
        "",
        f"- Total clips: **{mustard_summary['total']}**",
        f"- Split counts: `{mustard_summary['splits']}`",
        f"- Face quality: `{mustard_summary['face_quality']}`",
        f"- Media unverified or missing (`face_quality=missing`): "
        f"**{mustard_summary['media_unverified_or_missing']}**",
        f"- Sarcasm candidates (official label): **{mustard_summary['sarcasm_candidates']}**",
        f"- `usable_for_l4=true`: **{mustard_summary['usable_for_l4']}**",
        f"- `usable_for_micro=true`: **{mustard_summary['usable_for_micro']}**",
        "",
        "### Traceability example",
        "",
        "- `source_id`: original MUStARD utterance key (e.g. `1_60`)",
        "- `text_path`: `{mustard_root}/data/sarcasm_data.json#utterance_id=1_60`",
        "- `video_path`: `{mustard_root}/raw/clips/utterances_final/{id}.mp4`",
        "",
        "## Cross-check with M1 seed index",
        "",
        f"- M1 MELD source_ids: **{len(m1_meld_ids)}**",
        f"- Found in `meld_index.csv`: **{len(m1_hits)}**",
        f"- Seed ea_id preserved: **{seed_ok}**",
        f"- Missing: **{len(m1_misses)}**"
        + (f" (`{', '.join(m1_misses)}`)" if m1_misses else ""),
        f"- Remapped away from seed range: **{len(m1_id_mismatches)}**"
        + (f" (`{'; '.join(m1_id_mismatches)}`)" if m1_id_mismatches else ""),
        "",
        "## Selection / quality rules",
        "",
        "1. Keep all annotated rows; do not drop low-quality samples.",
        "2. MELD sarcasm candidate when Emotion polarity conflicts with Sentiment polarity "
        "(neutral either side is not a conflict).",
        "3. MUStARD sarcasm candidate when official `sarcasm=true`.",
        "4. Unchecked media (`--check-media` off) => `face_quality=missing`, "
        "`usable_for_micro=false` (contract).",
        "5. M1 MELD seed rows inherit measured duration and accepted quality from "
        "`m1_sample_20.csv`.",
        "6. `usable_for_l4` requires usable text and (sarcasm candidate or non-missing face).",
        "7. EA IDs persist via allocation map; inserting new source rows does not renumber "
        "existing assignments.",
        "",
        "## Known limitations",
        "",
        "- Full-corpus face detection was not run.",
        "- Default generation does not verify server media existence; re-run on the dataset "
        "host with `--check-media` after mounting the AutoDL paths to flip verified rows.",
        "- MUStARD `start/end` are empty (atomic clip files).",
        "",
    ]
    return "\n".join(lines)
