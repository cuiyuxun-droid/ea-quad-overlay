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

MELD_SPLITS = ("train", "dev", "test")
MELD_VIDEO_DIRS = {
    "train": "extracted/MELD.Raw/train_splits",
    "dev": "extracted/MELD.Raw/dev_splits",
    "test": "extracted/MELD.Raw/output_repeated_splits_test",
}

MELD_EA_ID_START = 100_000
MUSTARD_EA_ID_START = 200_000

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
    # Only local/Windows/absolute paths that exist on this machine can be verified.
    candidate = Path(path)
    if candidate.is_file():
        return True
    # POSIX server paths are not checkable on Windows unless mounted.
    if path.startswith("/root/") or path.startswith("/data/"):
        return None
    return False


def resolve_mustard_video_path(utterance_id: str, mustard_root: str) -> str:
    root = mustard_root.rstrip("/\\").replace("\\", "/")
    # Canonical guess used in indexes; local probes may resolve alternate layouts.
    return f"{root}/utterances_final/{utterance_id}.mp4"


def probe_mustard_video_path(utterance_id: str, mustard_root: Path) -> Path | None:
    relatives = (
        mustard_root / "utterances_final" / f"{utterance_id}.mp4",
        mustard_root / "utterances_final" / f"{utterance_id}_c.mp4",
        mustard_root / "videos" / f"{utterance_id}.mp4",
        mustard_root / "extracted" / "utterances_final" / f"{utterance_id}.mp4",
        mustard_root / f"{utterance_id}.mp4",
    )
    for path in relatives:
        if path.is_file():
            return path
    return None


def quality_flags(
    *,
    text: str,
    video_present: bool | None,
    duration: float | None,
    sarcasm_candidate: bool,
) -> tuple[str, str, str, str, str]:
    if text.strip():
        text_quality = "high"
    else:
        text_quality = "missing"

    if video_present is True:
        face_quality = "medium"
        audio_quality = "medium"
    elif video_present is False:
        face_quality = "missing"
        audio_quality = "missing"
    else:
        # Media not checked (typical for server-path indexes generated locally).
        face_quality = "medium" if text_quality != "missing" else "low"
        audio_quality = "medium" if text_quality != "missing" else "low"

    if video_present is False:
        usable_micro = False
    elif duration is None:
        usable_micro = video_present is not False and face_quality != "missing"
    else:
        usable_micro = face_quality != "missing" and 0.8 <= duration <= 30.0

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


def build_meld_row(
    record: MeldUtterance,
    *,
    ea_id: str,
    meld_root: str,
    check_media: bool,
    local_meld_root: Path | None = None,
) -> dict[str, str]:
    duration = duration_from_timestamps(record.start_time, record.end_time)
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
        elif (local_meld_root / "extracted").exists() or (local_meld_root / "annotations").exists():
            # Local tree present but this clip missing.
            if any(local_meld_root.rglob(f"dia{record.dialogue_id}_utt{record.utterance_id}.mp4")):
                present = True
            else:
                # Only treat as missing when extracted media tree exists.
                extracted = local_meld_root / "extracted"
                present = False if extracted.exists() else None

    sarcasm = emotion_sentiment_mismatch(record.emotion, record.sentiment)
    face_q, audio_q, text_q, usable_micro, usable_l4 = quality_flags(
        text=record.utterance,
        video_present=present,
        duration=duration,
        sarcasm_candidate=sarcasm,
    )
    return {
        "ea_id": ea_id,
        "source_dataset": "MELD",
        "source_split": record.split,
        "source_id": record.source_id,
        "video_path": video,
        "audio_path": video,
        "text_path": meld_text_path(meld_root, record.split, record.dialogue_id, record.utterance_id),
        "start": "0.00",
        "end": format_end(duration),
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
        present = True if local is not None else (
            False if local_mustard_root.exists() else None
        )

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
        "start": "0.00",
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


def summarize_rows(rows: Sequence[Mapping[str, str]]) -> dict[str, Any]:
    split_counts = Counter(row["source_split"] for row in rows)
    face_counts = Counter(row["face_quality"] for row in rows)
    sarcasm_n = sum(1 for row in rows if row.get("is_sarcasm_candidate") == "true")
    usable_l4 = sum(1 for row in rows if row.get("usable_for_l4") == "true")
    usable_micro = sum(1 for row in rows if row.get("usable_for_micro") == "true")
    return {
        "total": len(rows),
        "splits": dict(sorted(split_counts.items())),
        "face_quality": dict(sorted(face_counts.items())),
        "sarcasm_candidates": sarcasm_n,
        "usable_for_l4": usable_l4,
        "usable_for_micro": usable_micro,
        "ea_id_first": rows[0]["ea_id"] if rows else "",
        "ea_id_last": rows[-1]["ea_id"] if rows else "",
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
    check_media: bool = False,
) -> tuple[list[dict[str, str]], list[dict[str, str]], dict[str, Any], dict[str, Any]]:
    meld_records = load_meld_utterances(meld_ann_root)
    mustard_clips = load_mustard_clips(mustard_json)

    local_meld = meld_ann_root if check_media else None
    local_mustard = mustard_json.parent if check_media else None
    mustard_json_str = str(mustard_json if mustard_json.is_absolute() else mustard_json).replace(
        "\\", "/"
    )
    # Prefer server-style text pointer when path root is the canonical server tree.
    if mustard_path_root.startswith("/root/"):
        mustard_json_for_text = f"{mustard_path_root.rstrip('/')}/sarcasm_data.json"
    else:
        mustard_json_for_text = mustard_json_str

    meld_rows: list[dict[str, str]] = []
    for offset, record in enumerate(meld_records):
        ea_id = f"EAQ{MELD_EA_ID_START + offset:06d}"
        meld_rows.append(
            build_meld_row(
                record,
                ea_id=ea_id,
                meld_root=meld_path_root,
                check_media=check_media,
                local_meld_root=local_meld,
            )
        )

    mustard_rows: list[dict[str, str]] = []
    for offset, clip in enumerate(mustard_clips):
        ea_id = f"EAQ{MUSTARD_EA_ID_START + offset:06d}"
        mustard_rows.append(
            build_mustard_row(
                clip,
                ea_id=ea_id,
                mustard_root=mustard_path_root,
                json_path=mustard_json_for_text,
                check_media=check_media,
                local_mustard_root=local_mustard,
            )
        )

    write_index_csv(meld_output, meld_rows)
    write_index_csv(mustard_output, mustard_rows)
    return meld_rows, mustard_rows, summarize_rows(meld_rows), summarize_rows(mustard_rows)


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
    m1_meld_ids: Sequence[str],
    meld_source_ids: Iterable[str],
    check_media: bool,
) -> str:
    meld_id_set = set(meld_source_ids)
    m1_hits = [sid for sid in m1_meld_ids if sid in meld_id_set]
    m1_misses = [sid for sid in m1_meld_ids if sid not in meld_id_set]

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
        "",
        "## Outputs",
        "",
        f"- `{meld_output}`",
        f"- `{mustard_output}`",
        "",
        "## Path Roots",
        "",
        f"- MELD path root (written into CSV): `{meld_path_root}`",
        f"- MUStARD path root (written into CSV): `{mustard_path_root}`",
        f"- MELD annotations used for generation: `{meld_ann_source}`",
        f"- MUStARD JSON used for generation: `{mustard_json_source}`",
        f"- `--check-media`: `{bool_str(check_media)}`",
        "",
        "## EA ID Ranges",
        "",
        f"- MELD: `{meld_summary.get('ea_id_first')}` … `{meld_summary.get('ea_id_last')}` "
        f"(start `EAQ{MELD_EA_ID_START:06d}`)",
        f"- MUStARD: `{mustard_summary.get('ea_id_first')}` … `{mustard_summary.get('ea_id_last')}` "
        f"(start `EAQ{MUSTARD_EA_ID_START:06d}`)",
        "- Reserved separately from M1 seed IDs `EAQ000001`–`EAQ000020`.",
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
        "",
        "## MUStARD Summary",
        "",
        f"- Total clips: **{mustard_summary['total']}**",
        f"- Split counts: `{mustard_summary['splits']}`",
        f"- Face quality: `{mustard_summary['face_quality']}`",
        f"- Sarcasm candidates (official label): **{mustard_summary['sarcasm_candidates']}**",
        f"- `usable_for_l4=true`: **{mustard_summary['usable_for_l4']}**",
        f"- `usable_for_micro=true`: **{mustard_summary['usable_for_micro']}**",
        "",
        "### Traceability example",
        "",
        "- `source_id`: original MUStARD utterance key (e.g. `1_60`)",
        "- `text_path`: `sarcasm_data.json#utterance_id=1_60`",
        "- `video_path`: `{mustard_root}/utterances_final/{id}.mp4` (canonical guess)",
        "",
        "## Cross-check with M1 seed index",
        "",
        f"- M1 MELD source_ids: **{len(m1_meld_ids)}**",
        f"- Found in `meld_index.csv`: **{len(m1_hits)}**",
        f"- Missing: **{len(m1_misses)}**"
        + (f" (`{', '.join(m1_misses)}`)" if m1_misses else ""),
        "",
        "## Selection / quality rules",
        "",
        "1. Keep all annotated rows; do not drop low-quality samples.",
        "2. MELD sarcasm candidate when Emotion polarity conflicts with Sentiment polarity "
        "(neutral either side is not a conflict).",
        "3. MUStARD sarcasm candidate when official `sarcasm=true`.",
        "4. Without local face detection, existing media is marked `face_quality=medium` "
        "(not `high`).",
        "5. `usable_for_l4` requires usable text and (sarcasm candidate or non-missing face).",
        "",
        "## Known limitations",
        "",
        "- Full-corpus face detection was not run; face quality is heuristic.",
        "- MUStARD video filenames vary by unpack layout; path is a best-effort canonical guess.",
        "- MELD `end` uses subtitle StartTime/EndTime delta when available; not ffprobe duration.",
        "- Server media existence was not verified unless `--check-media` was enabled on a machine "
        "that can see those files.",
        "",
    ]
    return "\n".join(lines)
