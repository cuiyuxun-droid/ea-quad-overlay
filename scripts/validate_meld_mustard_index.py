#!/usr/bin/env python
"""Validate MELD / MUStARD source indexes and global identity constraints."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ea_quad_overlay.ch_sims_index import validate_global_index_paths  # noqa: E402
from ea_quad_overlay.meld_mustard_index import (  # noqa: E402
    read_index_csv,
    read_m1_meld_reservations,
    validate_index_rows,
)

DEFAULT_MELD = ROOT / "source_index" / "meld_index.csv"
DEFAULT_MUSTARD = ROOT / "source_index" / "mustard_index.csv"
DEFAULT_M1 = ROOT / "source_index" / "m1_sample_20.csv"
DEFAULT_CH_SIMS = ROOT / "source_index" / "ch_sims_index.csv"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--meld", type=Path, default=DEFAULT_MELD)
    parser.add_argument("--mustard", type=Path, default=DEFAULT_MUSTARD)
    parser.add_argument("--m1-index", type=Path, default=DEFAULT_M1)
    parser.add_argument(
        "--related-index",
        type=Path,
        action="append",
        default=[DEFAULT_CH_SIMS],
    )
    args = parser.parse_args(argv)
    try:
        meld_rows = read_index_csv(args.meld)
        mustard_rows = read_index_csv(args.mustard)
        meld_summary = validate_index_rows(
            meld_rows,
            dataset="MELD",
            seed_reservations=read_m1_meld_reservations(args.m1_index),
        )
        mustard_summary = validate_index_rows(mustard_rows, dataset="MUStARD")
        related = [path for path in args.related_index if path.is_file()]
        global_summary = validate_global_index_paths(
            [args.meld, args.mustard, args.m1_index, *related]
        )
    except (OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(
        "OK: validated "
        f"MELD={meld_summary['total']} MUStARD={mustard_summary['total']} rows "
        f"(unique_ea_ids={global_summary['unique_ea_ids']}, "
        f"unique_source_records={global_summary['unique_source_records']})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
