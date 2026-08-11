#!/usr/bin/env python
"""Validate IEMOCAP and MOSEI/MOSI source index files."""

from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE_INDEX = ROOT / "source_index"
IEMOCAP_INDEX = SOURCE_INDEX / "iemocap_index.csv"
MOSEI_INDEX = SOURCE_INDEX / "mosei_index.csv"

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
EA_ID_RE = re.compile(r"^EAQ\d{6}$")
BOOL_VALUES = {"true", "false"}
QUALITY_VALUES = {"high", "medium", "low"}
WEAK_LABELS = {"positive", "neutral", "negative"}


def fail(message: str) -> None:
    print(f"ERROR: {message}", file=sys.stderr)
    raise SystemExit(1)


def read_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    if not path.exists():
        fail(f"missing {path.relative_to(ROOT)}")
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def validate_row(path: Path, row_number: int, row: dict[str, str], check_paths: bool) -> None:
    prefix = f"{path.relative_to(ROOT)} row {row_number}"
    if not EA_ID_RE.match(row["ea_id"]):
        fail(f"{prefix}: invalid ea_id {row['ea_id']!r}")
    if row["source_dataset"] not in {"IEMOCAP", "MOSEI", "MOSI"}:
        fail(f"{prefix}: unexpected source_dataset {row['source_dataset']!r}")
    if not row["source_id"]:
        fail(f"{prefix}: missing source_id")
    try:
        start = float(row["start"])
        end = float(row["end"])
    except ValueError:
        fail(f"{prefix}: start/end must be numeric")
    if start < 0 or (end != 0 and end <= start):
        fail(f"{prefix}: invalid start/end range")
    for field in ("face_quality", "audio_quality", "text_quality"):
        if row[field] not in QUALITY_VALUES:
            fail(f"{prefix}: invalid {field} {row[field]!r}")
    for field in ("usable_for_micro", "usable_for_l4"):
        if row[field] not in BOOL_VALUES:
            fail(f"{prefix}: invalid {field} {row[field]!r}")
    if row["usable_for_micro"] == "true":
        for field in ("video_path", "audio_path", "text_path"):
            if not row[field]:
                fail(f"{prefix}: usable_for_micro row has empty {field}")
    if row["usable_for_l4"] == "true":
        for field in ("video_path", "audio_path", "text_path"):
            if not row[field]:
                fail(f"{prefix}: usable_for_l4 row has empty {field}")
        if row["weak_label_hint"] not in WEAK_LABELS:
            fail(f"{prefix}: usable_for_l4 row has unmapped weak_label_hint")
        if not row["label_source"]:
            fail(f"{prefix}: usable_for_l4 row has empty label_source")
    if row["source_dataset"] == "IEMOCAP":
        if row["usable_for_l4"] == "true" and not row["raw_emotion"]:
            fail(f"{prefix}: IEMOCAP L4 row missing raw_emotion")
        if row["raw_valence"] and row["raw_arousal"]:
            float(row["raw_valence"])
            float(row["raw_arousal"])
    if row["source_dataset"] in {"MOSEI", "MOSI"}:
        if row["usable_for_l4"] == "true" and not row["raw_sentiment"]:
            fail(f"{prefix}: MOSEI/MOSI L4 row missing raw_sentiment")
    if check_paths:
        for field in ("video_path", "audio_path"):
            value = row[field]
            if value and not Path(value).exists():
                fail(f"{prefix}: {field} does not exist locally: {value}")
        text_file = row["text_path"].split("#", 1)[0]
        if text_file and not Path(text_file).exists():
            fail(f"{prefix}: text_path does not exist locally: {text_file}")


def validate_file(path: Path, allow_empty: bool, check_paths: bool) -> Counter[str]:
    header, rows = read_rows(path)
    if header != FIELDNAMES:
        fail(f"{path.relative_to(ROOT)} header does not match expected schema")
    if not rows and not allow_empty:
        fail(f"{path.relative_to(ROOT)} has no rows")
    seen: set[str] = set()
    counts: Counter[str] = Counter()
    for row_number, row in enumerate(rows, start=2):
        if row["ea_id"] in seen:
            fail(f"{path.relative_to(ROOT)} row {row_number}: duplicate ea_id {row['ea_id']!r}")
        seen.add(row["ea_id"])
        validate_row(path, row_number, row, check_paths)
        counts[row["source_dataset"]] += 1
    return counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-empty", action="store_true")
    parser.add_argument("--check-paths", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    iemocap_counts = validate_file(IEMOCAP_INDEX, args.allow_empty, args.check_paths)
    mosei_counts = validate_file(MOSEI_INDEX, args.allow_empty, args.check_paths)
    total = iemocap_counts + mosei_counts
    print(
        "OK: validated "
        f"{sum(total.values())} rows "
        f"(IEMOCAP={total['IEMOCAP']}, MOSEI={total['MOSEI']}, MOSI={total['MOSI']})"
    )


if __name__ == "__main__":
    main()
