#!/usr/bin/env python
"""Validate IEMOCAP and MOSEI/MOSI source index files."""

from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable


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
DATASET_ID_RANGES = {
    "CH-SIMS": (21, 99999),
    "MELD": (100000, 199999),
    "MUStARD": (200000, 299999),
    "IEMOCAP": (300000, 399999),
    "MOSEI": (400000, 499999),
    "MOSI": (500000, 599999),
}
SEED_EXCEPTIONS = {
    "CH-SIMS": (1, 11),
    "MELD": (12, 20),
}


def fail(message: str) -> None:
    print(f"ERROR: {message}", file=sys.stderr)
    raise SystemExit(1)


def read_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    if not path.exists():
        fail(f"missing {path.relative_to(ROOT)}")
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = []
        for row in reader:
            rows.append({field: row.get(field, "") for field in FIELDNAMES})
        return list(reader.fieldnames or []), rows


def ea_id_number(ea_id: str) -> int:
    match = EA_ID_RE.match(ea_id)
    return int(ea_id[3:]) if match else -1


def resolve_text_pointer(row: dict[str, str]) -> bool:
    text_path = row["text_path"]
    if not text_path:
        return False
    file_text, _, pointer = text_path.partition("#")
    path = Path(file_text)
    if not path.exists():
        return False
    dataset = row["source_dataset"]
    if dataset == "IEMOCAP":
        if not pointer:
            return False
        with path.open(encoding="utf-8", errors="replace") as handle:
            return any(line.startswith(f"{pointer} ") for line in handle)
    if dataset in {"MOSEI", "MOSI"}:
        match = re.fullmatch(r"clip=([^&]+)", pointer)
        if not match:
            return False
        clip_id = match.group(1)
        with path.open(encoding="utf-8", errors="replace") as handle:
            return any(line.split("___", 2)[:2] == [path.stem, clip_id] for line in handle)
    return path.exists()


def validate_row(
    path: Path,
    row_number: int,
    row: dict[str, str],
    check_paths: bool,
    require_label_fields: bool,
) -> None:
    prefix = f"{path.relative_to(ROOT)} row {row_number}"
    if not EA_ID_RE.match(row["ea_id"]):
        fail(f"{prefix}: invalid ea_id {row['ea_id']!r}")
    if row["source_dataset"] not in DATASET_ID_RANGES:
        fail(f"{prefix}: unexpected source_dataset {row['source_dataset']!r}")
    start_id, end_id = DATASET_ID_RANGES[row["source_dataset"]]
    number = ea_id_number(row["ea_id"])
    seed_start, seed_end = SEED_EXCEPTIONS.get(row["source_dataset"], (-1, -1))
    in_dataset_range = start_id <= number <= end_id
    in_seed_exception = seed_start <= number <= seed_end
    if not in_dataset_range and not in_seed_exception:
        fail(
            f"{prefix}: {row['source_dataset']} ea_id {row['ea_id']} outside "
            f"EAQ{start_id:06d}-EAQ{end_id:06d}"
        )
    if not row["source_id"]:
        fail(f"{prefix}: missing source_id")
    if row["start"] or row["end"]:
        if not row["start"] or not row["end"]:
            fail(f"{prefix}: start/end must both be present or both be empty")
        try:
            start = float(row["start"])
            end = float(row["end"])
        except ValueError:
            fail(f"{prefix}: start/end must be numeric")
        if start < 0 or end <= start:
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
        if require_label_fields:
            if row["weak_label_hint"] not in WEAK_LABELS:
                fail(f"{prefix}: usable_for_l4 row has unmapped weak_label_hint")
            if not row["label_source"]:
                fail(f"{prefix}: usable_for_l4 row has empty label_source")
    if row["source_dataset"] == "IEMOCAP":
        if require_label_fields and row["usable_for_l4"] == "true" and not row["raw_emotion"]:
            fail(f"{prefix}: IEMOCAP L4 row missing raw_emotion")
        if row["raw_valence"] and row["raw_arousal"]:
            float(row["raw_valence"])
            float(row["raw_arousal"])
    if row["source_dataset"] in {"MOSEI", "MOSI"}:
        if require_label_fields and row["usable_for_l4"] == "true" and not row["raw_sentiment"]:
            fail(f"{prefix}: MOSEI/MOSI L4 row missing raw_sentiment")
    if check_paths:
        for field in ("video_path", "audio_path"):
            value = row[field]
            if value and not Path(value).exists():
                fail(f"{prefix}: {field} does not exist locally: {value}")
        text_file = row["text_path"].split("#", 1)[0]
        if text_file and not Path(text_file).exists():
            fail(f"{prefix}: text_path does not exist locally: {text_file}")
        if row["text_path"] and not resolve_text_pointer(row):
            fail(f"{prefix}: text_path pointer does not resolve: {row['text_path']}")


def validate_file(path: Path, allow_empty: bool, check_paths: bool) -> tuple[Counter[str], list[dict[str, str]]]:
    header, rows = read_rows(path)
    missing = [field for field in BASE_FIELDS if field not in header]
    if missing:
        fail(f"{path.relative_to(ROOT)} missing required columns: {', '.join(missing)}")
    if path.name in {"iemocap_index.csv", "mosei_index.csv"}:
        missing_label_fields = [field for field in LABEL_FIELDS if field not in header]
        if missing_label_fields:
            fail(
                f"{path.relative_to(ROOT)} missing label columns: "
                f"{', '.join(missing_label_fields)}"
            )
    require_label_fields = all(field in header for field in LABEL_FIELDS)
    if not rows and not allow_empty:
        fail(f"{path.relative_to(ROOT)} has no rows")
    seen: set[str] = set()
    counts: Counter[str] = Counter()
    for row_number, row in enumerate(rows, start=2):
        if row["ea_id"] in seen:
            fail(f"{path.relative_to(ROOT)} row {row_number}: duplicate ea_id {row['ea_id']!r}")
        seen.add(row["ea_id"])
        validate_row(path, row_number, row, check_paths, require_label_fields)
        counts[row["source_dataset"]] += 1
    return counts, rows


def default_index_paths() -> list[Path]:
    return [IEMOCAP_INDEX, MOSEI_INDEX]


def all_source_index_paths() -> list[Path]:
    return sorted(
        path
        for path in SOURCE_INDEX.glob("*.csv")
        if path.name != "source_index_template.csv"
    )


def validate_repository(paths: Iterable[Path], allow_empty: bool, check_paths: bool) -> Counter[str]:
    global_ea_ids: dict[str, str] = {}
    global_sources: dict[tuple[str, str], str] = {}
    total: Counter[str] = Counter()
    for path in paths:
        counts, rows = validate_file(path, allow_empty, check_paths)
        total += counts
        for row in rows:
            location = str(path.relative_to(ROOT))
            previous = global_ea_ids.get(row["ea_id"])
            if previous:
                fail(f"{location}: duplicate global ea_id {row['ea_id']} also in {previous}")
            global_ea_ids[row["ea_id"]] = location
            source_key = (row["source_dataset"], row["source_id"])
            previous_ea_id = global_sources.get(source_key)
            if previous_ea_id and previous_ea_id != row["ea_id"]:
                fail(
                    f"{location}: source identity {source_key} maps to both "
                    f"{previous_ea_id} and {row['ea_id']}"
                )
            global_sources[source_key] = row["ea_id"]
    return total


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-empty", action="store_true")
    parser.add_argument("--check-paths", action="store_true")
    parser.add_argument(
        "--indexes",
        type=Path,
        nargs="*",
        default=None,
        help="Index CSV files to validate together; defaults to IEMOCAP and MOSEI.",
    )
    parser.add_argument(
        "--all-source-indexes",
        action="store_true",
        help="Validate every CSV in source_index/ together.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = all_source_index_paths() if args.all_source_indexes else args.indexes or default_index_paths()
    total = validate_repository(paths, args.allow_empty, args.check_paths)
    print(
        "OK: validated "
        f"{sum(total.values())} rows "
        f"(IEMOCAP={total['IEMOCAP']}, MOSEI={total['MOSEI']}, MOSI={total['MOSI']})"
    )


if __name__ == "__main__":
    main()
