#!/usr/bin/env python
"""Validate the M1 seed source index."""

from __future__ import annotations

import csv
import re
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
INDEX_PATH = ROOT / "source_index" / "m1_sample_20.csv"
TEMPLATE_PATH = ROOT / "source_index" / "source_index_template.csv"

EA_ID_RE = re.compile(r"^EAQ\d{6}$")
QUALITY_VALUES = {"high", "medium", "low"}
BOOL_VALUES = {"true", "false"}


def fail(message: str) -> None:
    print(f"ERROR: {message}", file=sys.stderr)
    raise SystemExit(1)


def read_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def main() -> None:
    if not INDEX_PATH.exists():
        fail(f"missing {INDEX_PATH.relative_to(ROOT)}")

    template_header, _ = read_rows(TEMPLATE_PATH)
    header, rows = read_rows(INDEX_PATH)

    if header != template_header:
        fail("m1_sample_20.csv header does not match source_index_template.csv")

    if len(rows) < 20:
        fail("expected at least 20 rows")

    dataset_counts = Counter(row["source_dataset"] for row in rows)
    if not 10 <= dataset_counts["CH-SIMS"] <= 12:
        fail("expected 10-12 CH-SIMS rows")
    if not 8 <= dataset_counts["MELD"] <= 10:
        fail("expected 8-10 MELD rows")

    seen_ids: set[str] = set()
    for row_number, row in enumerate(rows, start=2):
        ea_id = row["ea_id"]
        if not EA_ID_RE.match(ea_id):
            fail(f"row {row_number}: invalid ea_id {ea_id!r}")
        if ea_id in seen_ids:
            fail(f"row {row_number}: duplicate ea_id {ea_id!r}")
        seen_ids.add(ea_id)

        if row["source_dataset"] not in {"CH-SIMS", "MELD"}:
            fail(f"row {row_number}: unexpected source_dataset {row['source_dataset']!r}")
        if not row["source_id"]:
            fail(f"row {row_number}: missing source_id")
        if not row["video_path"] or not row["audio_path"] or not row["text_path"]:
            fail(f"row {row_number}: missing modality path")
        try:
            start = float(row["start"])
            end = float(row["end"])
        except ValueError:
            fail(f"row {row_number}: start/end must be numeric")
        if start < 0 or end <= start:
            fail(f"row {row_number}: invalid start/end range")
        for field in ("face_quality", "audio_quality", "text_quality"):
            if row[field] not in QUALITY_VALUES:
                fail(f"row {row_number}: invalid {field} {row[field]!r}")
        for field in ("usable_for_micro", "usable_for_l4"):
            if row[field] not in BOOL_VALUES:
                fail(f"row {row_number}: invalid {field} {row[field]!r}")
            if row[field] != "true":
                fail(f"row {row_number}: {field} must be true for the M1 seed set")

    print(
        "OK: validated "
        f"{len(rows)} rows "
        f"({dataset_counts['CH-SIMS']} CH-SIMS, {dataset_counts['MELD']} MELD)"
    )


if __name__ == "__main__":
    main()
