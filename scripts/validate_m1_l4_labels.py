#!/usr/bin/env python
"""Validate the 20 M1 L4 annotation files."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ea_quad_overlay.l4_labels import (  # noqa: E402
    L4ValidationError,
    summarize_annotations,
    validate_dataset,
)


DEFAULT_INDEX = ROOT / "source_index" / "m1_sample_20.csv"
DEFAULT_ANNOTATIONS = ROOT / "annotations" / "l4_gold"


def read_index(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or [])
        required = {"ea_id", "source_dataset"}
        if not required <= fields:
            missing = ", ".join(sorted(required - fields))
            raise L4ValidationError(f"source index missing columns: {missing}")
        return list(reader)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--annotations", type=Path, default=DEFAULT_ANNOTATIONS)
    args = parser.parse_args(argv)

    try:
        rows = read_index(args.index)
        labels = validate_dataset(rows, args.annotations)
    except (OSError, csv.Error, json.JSONDecodeError, L4ValidationError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    summary = summarize_annotations(labels)
    print(f"OK: validated {summary['total']} L4 labels")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
