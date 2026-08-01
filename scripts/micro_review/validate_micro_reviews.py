#!/usr/bin/env python
"""Validate M1 micro-review annotations (Issue 05).

Checks:
  - one review file per EA ID in the M1 source index (20 expected)
  - JSON parses and required fields exist
  - verdict is one of positive / negative / uncertain
  - positive reviews carry a non-null event (onset/apex/offset, AU, intensity)
  - negative / uncertain reviews do not claim an event
  - review files match the canonical EA ID / segment ID naming

Usage:
    python scripts/micro_review/validate_micro_reviews.py
"""

from __future__ import annotations

import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
INDEX_PATH = ROOT / "source_index" / "m1_sample_20.csv"
REVIEW_DIR = ROOT / "annotations" / "micro_review"

EA_ID_RE = re.compile(r"^EAQ\d{6}$")
SEG_RE = re.compile(r"^(EAQ\d{6})_seg\d{3}_micro_review\.json$")
VERDICTS = {"positive", "negative", "uncertain"}


def fail(message: str) -> None:
    print(f"ERROR: {message}", file=sys.stderr)
    raise SystemExit(1)


def main() -> None:
    rows = []
    with INDEX_PATH.open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append(r)

    expected = [r["ea_id"] for r in rows]
    if len(expected) < 20:
        fail(f"expected at least 20 samples in index, got {len(expected)}")

    review_files = {p.name: p for p in REVIEW_DIR.glob("*_micro_review.json")}
    if len(review_files) < 20:
        fail(f"expected at least 20 review files, got {len(review_files)}")

    # 1. every expected EA ID has a review file
    parsed: dict[str, dict] = {}
    for ea_id in expected:
        match = [name for name in review_files if name.startswith(ea_id + "_")]
        if not match:
            fail(f"missing review file for {ea_id}")
        path = review_files[match[0]]
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            fail(f"{path.name}: invalid JSON ({exc})")
        parsed[ea_id] = data

    # 2. naming + required fields
    for ea_id, data in parsed.items():
        if data.get("ea_id") != ea_id:
            fail(f"{ea_id}: ea_id field mismatch")
        if data.get("segment_id") != f"{ea_id}_seg001":
            fail(f"{ea_id}: unexpected segment_id {data.get('segment_id')!r}")
        for field in ("review_status", "has_micro_expression", "reviewer", "review_date", "evidence"):
            if field not in data:
                fail(f"{ea_id}: missing field {field!r}")

        verdict = data["review_status"]
        if verdict not in VERDICTS:
            fail(f"{ea_id}: invalid review_status {verdict!r}")
        expected_bool = {"positive": True, "negative": False, "uncertain": None}
        if data["has_micro_expression"] != expected_bool[verdict]:
            fail(f"{ea_id}: has_micro_expression inconsistent with {verdict}")

        if verdict == "positive":
            event = data.get("event")
            if event is None:
                fail(f"{ea_id}: positive review must carry an event")
            for field in ("onset_sec", "apex_sec", "offset_sec", "aus", "intensity", "confidence"):
                if field not in event:
                    fail(f"{ea_id}: positive event missing {field!r}")
        elif data.get("event") is not None:
            fail(f"{ea_id}: {verdict} review must have null event")

    counts = Counter(parsed[ea_id]["review_status"] for ea_id in expected)
    print(
        f"OK: validated {len(expected)} reviews "
        f"({counts['positive']} positive, {counts['negative']} negative, "
        f"{counts['uncertain']} uncertain)"
    )


if __name__ == "__main__":
    sys.exit(main())
