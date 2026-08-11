#!/usr/bin/env python
"""Validate CH-SIMS source index."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ea_quad_overlay.ch_sims_index import (  # noqa: E402
    ChSimsIndexError,
    read_index_csv,
    read_m1_ch_sims_reservations,
    validate_index_rows,
)


DEFAULT_INDEX = ROOT / "source_index" / "ch_sims_index.csv"
DEFAULT_M1 = ROOT / "source_index" / "m1_sample_20.csv"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--m1-index", type=Path, default=DEFAULT_M1)
    args = parser.parse_args(argv)

    try:
        rows = read_index_csv(args.index)
        reserved = read_m1_ch_sims_reservations(args.m1_index)
        summary = validate_index_rows(rows, reserved_by_source_key=reserved)
    except (OSError, ChSimsIndexError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(
        "OK: validated "
        f"{summary['total']} CH-SIMS rows "
        f"(micro={summary['usable_for_micro']}, l4={summary['usable_for_l4']})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
