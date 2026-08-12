#!/usr/bin/env python
"""Batch-extract FeatureBank modalities from one or more source-index CSVs."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from ea_features.batch import (  # noqa: E402
    BatchConfig,
    BatchExtractionError,
    BatchFeatureRunner,
    discover_index_paths,
    parse_modalities,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--index",
        action="append",
        type=Path,
        dest="indexes",
        help="Source-index CSV; repeat for multiple datasets. Defaults to source_index/*.csv.",
    )
    parser.add_argument("--index-dir", type=Path, default=ROOT / "source_index")
    parser.add_argument("--output-root", type=Path, default=ROOT)
    parser.add_argument(
        "--modalities",
        default="text,speech,macro,micro",
        help="Comma-separated subset of text,speech,macro,micro",
    )
    parser.add_argument("--skip-existing", action="store_true", default=True)
    parser.add_argument(
        "--overwrite",
        action="store_false",
        dest="skip_existing",
        help="Recompute requested features even when npy + meta already exist.",
    )
    parser.add_argument("--limit", type=int, default=0, help="Limit total rows for smoke tests")
    parser.add_argument("--device", choices=("cpu", "cuda"), default=None)
    parser.add_argument("--no-face-filter", action="store_true")
    parser.add_argument("--face-sample-frames", type=int, default=12)
    parser.add_argument("--min-face-detect-rate", type=float, default=0.5)
    parser.add_argument("--min-face-ratio", type=float, default=0.01)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument("--fail-on-error", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def pick_device(preferred: str | None) -> str:
    if preferred:
        return preferred
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

    try:
        modalities = parse_modalities(args.modalities)
        indexes = tuple(args.indexes or discover_index_paths(args.index_dir))
        config = BatchConfig(
            root=ROOT,
            output_root=args.output_root.resolve(),
            index_paths=tuple(path.resolve() for path in indexes),
            modalities=modalities,
            skip_existing=args.skip_existing,
            limit=args.limit,
            device=pick_device(args.device),
            face_filter=not args.no_face_filter,
            face_sample_frames=args.face_sample_frames,
            min_face_detect_rate=args.min_face_detect_rate,
            min_face_ratio=args.min_face_ratio,
            report_path=args.report.resolve() if args.report else None,
            manifest_path=args.manifest.resolve() if args.manifest else None,
            cache_root=args.cache_root.resolve() if args.cache_root else None,
        )
        summary = BatchFeatureRunner(config).run()
    except BatchExtractionError as exc:
        logging.error("%s", exc)
        raise SystemExit(2) from exc

    logging.info(
        "Done: samples=%d failed=%d micro_filtered=%d report=%s manifest=%s",
        summary["samples"],
        summary["failed_samples"],
        summary["filtered_samples"],
        summary["report_path"],
        summary["manifest_path"],
    )
    if args.fail_on_error and summary["failed_samples"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
