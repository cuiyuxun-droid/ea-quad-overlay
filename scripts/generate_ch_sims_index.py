#!/usr/bin/env python
"""Generate CH-SIMS source index and label companion CSV."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ea_quad_overlay.ch_sims_index import (  # noqa: E402
    ChSimsIndexError,
    fetch_label_csv,
    generate_ch_sims_index,
    probe_dataset_media,
    read_label_csv,
    render_index_report,
    write_probe_csv,
)


DEFAULT_LABEL = ROOT / ".cache" / "ch_sims" / "label.csv"
DEFAULT_OUTPUT = ROOT / "source_index" / "ch_sims_index.csv"
DEFAULT_LABELS = ROOT / "source_index" / "ch_sims_labels.csv"
DEFAULT_PROBE = ROOT / "source_index" / "ch_sims_media_probe.csv"
DEFAULT_M1 = ROOT / "source_index" / "m1_sample_20.csv"
DEFAULT_REPORT = ROOT / "reports" / "ch_sims_index_report.md"
DEFAULT_DATASET_ROOT = "/root/autodl-tmp/data/datasets/ch_sims"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label-csv", type=Path, default=DEFAULT_LABEL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--labels-output", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--probe-output", type=Path, default=DEFAULT_PROBE)
    parser.add_argument("--m1-index", type=Path, default=DEFAULT_M1)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--dataset-root", default=DEFAULT_DATASET_ROOT)
    parser.add_argument(
        "--fetch-label",
        action="store_true",
        help="Download public CH-SIMS label.csv from Hugging Face before generation.",
    )
    parser.add_argument(
        "--probe-media",
        action="store_true",
        help="Probe extracted Raw/ videos under --dataset-root with ffprobe.",
    )
    parser.add_argument(
        "--probe-limit",
        type=int,
        default=None,
        help="Optional limit for media probing (debug).",
    )
    parser.add_argument(
        "--load-probe-csv",
        type=Path,
        default=None,
        help="Reuse a previously written probe CSV instead of probing again.",
    )
    args = parser.parse_args(argv)

    try:
        if args.fetch_label or not args.label_csv.is_file():
            fetch_label_csv(args.label_csv)

        probes = None
        probe_csv = args.load_probe_csv
        if args.probe_media:
            records = read_label_csv(args.label_csv)
            probes = probe_dataset_media(
                records,
                Path(args.dataset_root),
                limit=args.probe_limit,
            )
            write_probe_csv(args.probe_output, probes)
            probe_csv = args.probe_output

        (
            _index_rows,
            _label_rows,
            label_summary,
            index_summary,
            duration_provenance,
            extras,
        ) = generate_ch_sims_index(
            label_csv=args.label_csv,
            output_csv=args.output,
            labels_csv=args.labels_output,
            dataset_root=args.dataset_root,
            m1_index_path=args.m1_index,
            probe_csv=probe_csv,
            probes=probes,
        )
        labels_summary = {
            "total": extras["total"],
            "missing_label_fields": extras["missing_label_fields"],
        }
        probe_summary = extras["probe"]
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            render_index_report(
                label_summary=label_summary,
                index_summary=index_summary,
                labels_summary=labels_summary,
                duration_provenance=duration_provenance,
                probe_summary=probe_summary,
                label_source=str(args.label_csv.relative_to(ROOT))
                if args.label_csv.is_relative_to(ROOT)
                else str(args.label_csv),
                dataset_root=args.dataset_root,
                output_csv=str(args.output.relative_to(ROOT))
                if args.output.is_relative_to(ROOT)
                else str(args.output),
                labels_csv=str(args.labels_output.relative_to(ROOT))
                if args.labels_output.is_relative_to(ROOT)
                else str(args.labels_output),
            ),
            encoding="utf-8",
        )
    except (OSError, ChSimsIndexError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(
        "OK: wrote "
        f"{index_summary['total']} CH-SIMS index rows and "
        f"{labels_summary['total']} label rows "
        f"(reserved M1={index_summary['reserved_m1_matches']}, "
        f"micro={index_summary['usable_for_micro']}, "
        f"l4={index_summary['usable_for_l4']})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
