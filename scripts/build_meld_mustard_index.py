#!/usr/bin/env python
"""Build MELD / MUStARD source indexes for Issue #9."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ea_quad_overlay.meld_mustard_index import (  # noqa: E402
    DialogueIndexError,
    fetch_mustard_json,
    generate_dialogue_indexes,
    read_m1_meld_source_ids,
    render_dialogue_report,
)


DEFAULT_MELD_ANN = ROOT / "data" / "m1" / "meld"
DEFAULT_MUSTARD_JSON = ROOT / ".cache" / "mustard" / "sarcasm_data.json"
DEFAULT_MELD_OUT = ROOT / "source_index" / "meld_index.csv"
DEFAULT_MUSTARD_OUT = ROOT / "source_index" / "mustard_index.csv"
DEFAULT_REPORT = ROOT / "reports" / "dialogue_dataset_index_report.md"
DEFAULT_M1 = ROOT / "source_index" / "m1_sample_20.csv"
DEFAULT_MELD_PATH_ROOT = "/root/autodl-tmp/data/datasets/meld"
DEFAULT_MUSTARD_PATH_ROOT = "/root/autodl-tmp/data/datasets/mustard"


def _rel_or_abs(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--meld-ann-root",
        type=Path,
        default=DEFAULT_MELD_ANN,
        help="Local root containing annotations/{train,dev,test}_sent_emo.csv",
    )
    parser.add_argument(
        "--mustard-json",
        type=Path,
        default=DEFAULT_MUSTARD_JSON,
        help="Path to sarcasm_data.json",
    )
    parser.add_argument(
        "--meld-path-root",
        default=DEFAULT_MELD_PATH_ROOT,
        help="Path prefix written into meld_index.csv (server-style by default)",
    )
    parser.add_argument(
        "--mustard-path-root",
        default=DEFAULT_MUSTARD_PATH_ROOT,
        help="Path prefix written into mustard_index.csv",
    )
    parser.add_argument("--meld-output", type=Path, default=DEFAULT_MELD_OUT)
    parser.add_argument("--mustard-output", type=Path, default=DEFAULT_MUSTARD_OUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--m1-index", type=Path, default=DEFAULT_M1)
    parser.add_argument(
        "--fetch-mustard",
        action="store_true",
        help="Download official MUStARD sarcasm_data.json before generation",
    )
    parser.add_argument(
        "--check-media",
        action="store_true",
        help="Probe local media existence when possible",
    )
    args = parser.parse_args(argv)

    try:
        if args.fetch_mustard or not args.mustard_json.is_file():
            fetch_mustard_json(args.mustard_json)

        meld_rows, _mustard_rows, meld_summary, mustard_summary = generate_dialogue_indexes(
            meld_ann_root=args.meld_ann_root,
            mustard_json=args.mustard_json,
            meld_path_root=args.meld_path_root,
            mustard_path_root=args.mustard_path_root,
            meld_output=args.meld_output,
            mustard_output=args.mustard_output,
            check_media=args.check_media,
        )

        m1_ids = read_m1_meld_source_ids(args.m1_index)
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            render_dialogue_report(
                meld_summary=meld_summary,
                mustard_summary=mustard_summary,
                meld_path_root=args.meld_path_root,
                mustard_path_root=args.mustard_path_root,
                meld_ann_source=_rel_or_abs(args.meld_ann_root),
                mustard_json_source=_rel_or_abs(args.mustard_json),
                meld_output=_rel_or_abs(args.meld_output),
                mustard_output=_rel_or_abs(args.mustard_output),
                m1_meld_ids=m1_ids,
                meld_source_ids=(row["source_id"] for row in meld_rows),
                check_media=args.check_media,
            ),
            encoding="utf-8",
        )
    except (OSError, DialogueIndexError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(
        "OK: wrote "
        f"{meld_summary['total']} MELD rows -> {args.meld_output} ; "
        f"{mustard_summary['total']} MUStARD rows -> {args.mustard_output}"
    )
    print(f"OK: report -> {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
