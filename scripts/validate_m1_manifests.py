#!/usr/bin/env python
"""Validate M1 manifests with repository read logic."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ea_quad_overlay.manifests import (  # noqa: E402
    ManifestError,
    validate_m1_manifests,
)


DEFAULT_INDEX = ROOT / "source_index" / "m1_sample_20.csv"
DEFAULT_MANIFESTS = ROOT / "manifests"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--manifest-dir", type=Path, default=DEFAULT_MANIFESTS)
    args = parser.parse_args(argv)

    try:
        summary = validate_m1_manifests(
            root=args.root,
            manifest_dir=args.manifest_dir,
            index_path=args.index,
        )
    except (OSError, ManifestError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    l2 = summary["l2_counts"]
    print(
        "OK: validated manifests "
        f"(text={l2['text']}, speech={l2['speech']}, "
        f"macro={l2['macro']}, micro={l2['micro']}, "
        f"fusion={summary['fusion_count']})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
