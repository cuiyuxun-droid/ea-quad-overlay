#!/usr/bin/env python
"""Validate global ea_id and source-record identity consistency."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ea_quad_overlay.ch_sims_index import (  # noqa: E402
    ChSimsIndexError,
    validate_global_index_paths,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("indexes", type=Path, nargs="+", help="Dataset index CSV files.")
    args = parser.parse_args(argv)
    try:
        summary = validate_global_index_paths(args.indexes)
    except (OSError, ChSimsIndexError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(
        "OK: validated "
        f"{summary['rows']} rows across {summary['files']} indexes "
        f"(unique_ea_ids={summary['unique_ea_ids']}, "
        f"unique_source_records={summary['unique_source_records']})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
