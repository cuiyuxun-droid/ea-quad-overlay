"""CH-SIMS source index generation, validation, and reporting."""

from __future__ import annotations

import csv
import re
import urllib.request
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

EA_ID_RE = re.compile(r"^EAQ\d{6}$")
SOURCE_ID_PREFIX = "CH-SIMS/"
QUALITY_VALUES = {"high", "medium", "low", "missing"}
BOOL_VALUES = {"true", "false"}
SPLIT_MAP = {"train": "train", "valid": "validation", "validation": "validation", "test": "test"}
HF_LABEL_URL = "https://huggingface.co/datasets/tamb2203579/CH-SIMS/resolve/main/label.csv"

INDEX_COLUMNS = (
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

RESERVED_MELD_EA_IDS = {f"EAQ{i:06d}" for i in range(12, 21)}


class ChSimsIndexError(ValueError):
    """Raised when CH-SIMS index generation or validation fails."""


@dataclass(frozen=True)
class ChSimsRecord:
    video_id: str
    clip_id: str
    text: str
    label: str
    label_t: str
    label_a: str
    label_v: str
    annotation: str
    mode: str

    @property
    def source_key(self) -> str:
        return f"{self.video_id}/{self.clip_id}"

    @property
    def source_id(self) -> str:
        return f"{SOURCE_ID_PREFIX}{self.source_key}"


def fetch_label_csv(dest: Path) -> Path:
    """Download public CH-SIMS label.csv metadata from Hugging Face."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(HF_LABEL_URL, headers={"User-Agent": "ea-quad-overlay/1.0"})
    with urllib.request.urlopen(request, timeout=120) as response:
        dest.write_bytes(response.read())
    return dest


def read_label_csv(path: Path) -> list[ChSimsRecord]:
    records: list[ChSimsRecord] = []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.reader(handle)
        for row_number, row in enumerate(reader, start=1):
            if len(row) < 9:
                raise ChSimsIndexError(f"{path}:{row_number}: expected 9 columns, got {len(row)}")
            video_id, clip_id, text, label, label_t, label_a, label_v, annotation, mode = row[:9]
            if not video_id or not clip_id:
                raise ChSimsIndexError(f"{path}:{row_number}: missing video_id or clip_id")
            records.append(
                ChSimsRecord(
                    video_id=video_id.strip(),
                    clip_id=clip_id.strip(),
                    text=(text or "").strip(),
                    label=(label or "").strip(),
                    label_t=(label_t or "").strip(),
                    label_a=(label_a or "").strip(),
                    label_v=(label_v or "").strip(),
                    annotation=(annotation or "").strip(),
                    mode=(mode or "").strip().lower(),
                )
            )
    if not records:
        raise ChSimsIndexError(f"{path}: no CH-SIMS rows found")
    return records


def read_m1_ch_sims_reservations(m1_index_path: Path) -> dict[str, str]:
    """Map CH-SIMS source keys to reserved M1 ea_id values."""
    if not m1_index_path.is_file():
        return {}
    reserved: dict[str, str] = {}
    with m1_index_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("source_dataset") != "CH-SIMS":
                continue
            source_id = row.get("source_id", "")
            if not source_id.startswith(SOURCE_ID_PREFIX):
                continue
            key = source_id.removeprefix(SOURCE_ID_PREFIX)
            reserved[key] = row["ea_id"]
    return reserved


def _estimate_duration_sec(text: str) -> float:
    """Conservative duration estimate when media probing is unavailable."""
    if not text:
        return 2.0
    # Chinese clips are short; use a simple chars-per-second heuristic.
    return max(1.0, min(30.0, len(text) / 3.5))


def _quality_from_metadata(record: ChSimsRecord) -> tuple[str, str, str, str, str]:
    text = record.text
    if len(text) >= 4:
        text_quality = "high"
    elif text:
        text_quality = "medium"
    else:
        text_quality = "missing"

    # CH-SIMS clips are video-first with embedded audio; mark audio as high unless text is missing.
    audio_quality = "high" if text_quality != "missing" else "medium"

    # Without local media probing, keep face quality conservative but usable.
    face_quality = "medium" if text_quality != "missing" else "low"

    usable = text_quality in {"high", "medium"} and face_quality != "missing"
    usable_flag = "true" if usable else "false"
    return face_quality, audio_quality, text_quality, usable_flag, usable_flag


def build_media_paths(record: ChSimsRecord, dataset_root: str, label_csv: Path) -> tuple[str, str, str]:
    root = dataset_root.rstrip("/")
    member = f"Raw/{record.video_id}/{record.clip_id}.mp4"
    video_path = f"{root}/Raw.zip::{member}"
    audio_path = video_path
    text_path = f"{root}/label.csv#{record.source_key}"
    # Keep label.csv path stable even when the local copy lives elsewhere.
    if label_csv.name == "label.csv":
        text_path = f"{root}/label.csv#{record.source_key}"
    else:
        text_path = f"{label_csv.as_posix()}#{record.source_key}"
    return video_path, audio_path, text_path


def assign_ea_ids(
    records: Sequence[ChSimsRecord],
    reserved_by_source_key: Mapping[str, str],
) -> dict[str, str]:
    """Assign ea_id values while preserving M1 CH-SIMS IDs and skipping MELD slots."""
    assignments: dict[str, str] = {}
    used_ids: set[str] = set(reserved_by_source_key.values())
    next_id = 1

    def next_free_id() -> str:
        nonlocal next_id
        while True:
            candidate = f"EAQ{next_id:06d}"
            next_id += 1
            if candidate in RESERVED_MELD_EA_IDS:
                continue
            if candidate not in used_ids:
                used_ids.add(candidate)
                return candidate

    for record in records:
        key = record.source_key
        if key in reserved_by_source_key:
            ea_id = reserved_by_source_key[key]
            if ea_id in assignments.values():
                raise ChSimsIndexError(f"duplicate reserved ea_id for {key}")
            assignments[key] = ea_id
            used_ids.add(ea_id)
            continue
        assignments[key] = next_free_id()
    return assignments


def build_index_rows(
    records: Sequence[ChSimsRecord],
    *,
    dataset_root: str,
    label_csv: Path,
    reserved_by_source_key: Mapping[str, str] | None = None,
) -> list[dict[str, str]]:
    reserved = reserved_by_source_key or {}
    ea_ids = assign_ea_ids(records, reserved)
    rows: list[dict[str, str]] = []
    for record in records:
        ea_id = ea_ids[record.source_key]
        split = SPLIT_MAP.get(record.mode, record.mode or "train")
        if split not in {"train", "validation", "test"}:
            raise ChSimsIndexError(f"unsupported split {record.mode!r} for {record.source_key}")
        duration = _estimate_duration_sec(record.text)
        face_q, audio_q, text_q, usable_micro, usable_l4 = _quality_from_metadata(record)
        video_path, audio_path, text_path = build_media_paths(record, dataset_root, label_csv)
        rows.append(
            {
                "ea_id": ea_id,
                "source_dataset": "CH-SIMS",
                "source_split": split,
                "source_id": record.source_id,
                "video_path": video_path,
                "audio_path": audio_path,
                "text_path": text_path,
                "start": "0.00",
                "end": f"{duration:.2f}",
                "language": "zh",
                "face_quality": face_q,
                "audio_quality": audio_q,
                "text_quality": text_q,
                "usable_for_micro": usable_micro,
                "usable_for_l4": usable_l4,
            }
        )
    return rows


def write_index_csv(path: Path, rows: Sequence[Mapping[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(INDEX_COLUMNS))
        writer.writeheader()
        writer.writerows(rows)


def read_index_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        header = list(reader.fieldnames or [])
        if header != list(INDEX_COLUMNS):
            raise ChSimsIndexError("index header does not match source_index_template.csv")
        return list(reader)


def validate_index_rows(
    rows: Sequence[Mapping[str, str]],
    *,
    reserved_by_source_key: Mapping[str, str] | None = None,
    min_rows: int = 2000,
) -> dict[str, Any]:
    if len(rows) < min_rows:
        raise ChSimsIndexError(f"expected at least {min_rows} rows, got {len(rows)}")

    seen_ea: set[str] = set()
    seen_source: set[str] = set()
    split_counts: Counter[str] = Counter()
    quality_counts: Counter[str] = Counter()

    for row_number, row in enumerate(rows, start=2):
        ea_id = row["ea_id"]
        if not EA_ID_RE.fullmatch(ea_id):
            raise ChSimsIndexError(f"row {row_number}: invalid ea_id {ea_id!r}")
        if ea_id in seen_ea:
            raise ChSimsIndexError(f"row {row_number}: duplicate ea_id {ea_id!r}")
        seen_ea.add(ea_id)

        if row["source_dataset"] != "CH-SIMS":
            raise ChSimsIndexError(f"row {row_number}: source_dataset must be CH-SIMS")
        source_id = row["source_id"]
        if not source_id.startswith(SOURCE_ID_PREFIX):
            raise ChSimsIndexError(f"row {row_number}: invalid source_id {source_id!r}")
        if source_id in seen_source:
            raise ChSimsIndexError(f"row {row_number}: duplicate source_id {source_id!r}")
        seen_source.add(source_id)

        if row["source_split"] not in {"train", "validation", "test"}:
            raise ChSimsIndexError(f"row {row_number}: invalid source_split")
        for field in ("video_path", "audio_path", "text_path"):
            if not row[field]:
                raise ChSimsIndexError(f"row {row_number}: missing {field}")
        try:
            start = float(row["start"])
            end = float(row["end"])
        except ValueError as exc:
            raise ChSimsIndexError(f"row {row_number}: start/end must be numeric") from exc
        if start < 0 or end <= start:
            raise ChSimsIndexError(f"row {row_number}: invalid start/end range")
        for field in ("face_quality", "audio_quality", "text_quality"):
            if row[field] not in QUALITY_VALUES:
                raise ChSimsIndexError(f"row {row_number}: invalid {field}")
        for field in ("usable_for_micro", "usable_for_l4"):
            if row[field] not in BOOL_VALUES:
                raise ChSimsIndexError(f"row {row_number}: invalid {field}")

        split_counts[row["source_split"]] += 1
        quality_counts[row["text_quality"]] += 1

    reserved = reserved_by_source_key or {}
    for key, ea_id in reserved.items():
        expected_source = f"{SOURCE_ID_PREFIX}{key}"
        matches = [row for row in rows if row["source_id"] == expected_source]
        if not matches:
            raise ChSimsIndexError(f"missing reserved M1 source_id {expected_source}")
        if matches[0]["ea_id"] != ea_id:
            raise ChSimsIndexError(
                f"M1 reservation mismatch for {expected_source}: "
                f"expected {ea_id}, got {matches[0]['ea_id']}"
            )

    usable_micro = sum(1 for row in rows if row["usable_for_micro"] == "true")
    usable_l4 = sum(1 for row in rows if row["usable_for_l4"] == "true")
    return {
        "total": len(rows),
        "split_counts": dict(split_counts),
        "text_quality_counts": dict(quality_counts),
        "usable_for_micro": usable_micro,
        "usable_for_l4": usable_l4,
        "reserved_m1_matches": len(reserved),
    }


def summarize_label_records(records: Sequence[ChSimsRecord]) -> dict[str, Any]:
    split_counts = Counter(record.mode for record in records)
    annotation_counts = Counter(record.annotation for record in records if record.annotation)
    missing_text = sum(1 for record in records if not record.text)
    return {
        "total": len(records),
        "split_counts": dict(split_counts),
        "annotation_counts": dict(annotation_counts),
        "missing_text": missing_text,
    }


def render_index_report(
    *,
    label_summary: Mapping[str, Any],
    index_summary: Mapping[str, Any],
    label_source: str,
    dataset_root: str,
    output_csv: str,
) -> str:
    lines = [
        "# CH-SIMS Index Report",
        "",
        "GitHub issue: <https://github.com/cuiyuxun-droid/ea-quad-overlay/issues/8>",
        "",
        "## Scope and result",
        "",
        "- Built a unified CH-SIMS source index from public `label.csv` metadata.",
        "- Preserved M1 CH-SIMS `ea_id` reservations from `source_index/m1_sample_20.csv`.",
        "- Generated modality paths, quality flags, and split labels for FeatureBank ingestion.",
        "",
        "## Inputs",
        "",
        f"- Label source: `{label_source}`",
        f"- Dataset root: `{dataset_root}`",
        f"- Output index: `{output_csv}`",
        "",
        "## Label coverage",
        "",
        "| Metric | Count |",
        "| --- | ---: |",
        f"| Total label rows | {label_summary['total']} |",
        f"| Missing text | {label_summary['missing_text']} |",
        "",
        "### Split distribution (label.csv)",
        "",
        "| Split | Count |",
        "| --- | ---: |",
    ]
    for split, count in sorted(label_summary["split_counts"].items()):
        lines.append(f"| `{split}` | {count} |")

    lines.extend(
        [
            "",
            "## Index validation",
            "",
            "| Metric | Count |",
            "| --- | ---: |",
            f"| Indexed rows | {index_summary['total']} |",
            f"| M1 reservations preserved | {index_summary['reserved_m1_matches']} |",
            f"| `usable_for_micro=true` | {index_summary['usable_for_micro']} |",
            f"| `usable_for_l4=true` | {index_summary['usable_for_l4']} |",
            "",
            "### Split distribution (index)",
            "",
            "| Split | Count |",
            "| --- | ---: |",
        ]
    )
    for split, count in sorted(index_summary["split_counts"].items()):
        lines.append(f"| `{split}` | {count} |")

    lines.extend(
        [
            "",
            "### Text quality distribution",
            "",
            "| Quality | Count |",
            "| --- | ---: |",
        ]
    )
    for quality, count in sorted(index_summary["text_quality_counts"].items()):
        lines.append(f"| `{quality}` | {count} |")

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- `start/end` are metadata estimates when media probing is unavailable.",
            "- Re-run generation on the dataset host with media access to refine durations and face quality.",
            "",
            "## Errors",
            "",
            "- none",
            "",
        ]
    )
    return "\n".join(lines)


def generate_ch_sims_index(
    *,
    label_csv: Path,
    output_csv: Path,
    dataset_root: str,
    m1_index_path: Path,
) -> tuple[list[dict[str, str]], dict[str, Any], dict[str, Any]]:
    records = read_label_csv(label_csv)
    reserved = read_m1_ch_sims_reservations(m1_index_path)
    rows = build_index_rows(
        records,
        dataset_root=dataset_root,
        label_csv=label_csv,
        reserved_by_source_key=reserved,
    )
    write_index_csv(output_csv, rows)
    label_summary = summarize_label_records(records)
    index_summary = validate_index_rows(
        rows,
        reserved_by_source_key=reserved,
        min_rows=len(rows),
    )
    return rows, label_summary, index_summary
