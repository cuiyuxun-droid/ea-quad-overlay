#!/usr/bin/env python
"""Generate M1 L2 + fusion manifests from Issue #4/#5/#6 artifacts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ea_quad_overlay.manifests import (  # noqa: E402
    ManifestError,
    generate_m1_manifests,
    render_manifest_report,
    validate_m1_manifests,
    write_m1_manifests,
)


DEFAULT_INDEX = ROOT / "source_index" / "m1_sample_20.csv"
DEFAULT_OUT = ROOT / "manifests"
DEFAULT_REPORT = ROOT / "reports" / "m1_manifest_check.md"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument(
        "--skip-validate",
        action="store_true",
        help="Write manifests without running the repository read validator.",
    )
    args = parser.parse_args(argv)

    try:
        manifests = generate_m1_manifests(root=args.root, index_path=args.index)
        written = write_m1_manifests(manifests, args.out_dir)
        if not args.skip_validate:
            summary = validate_m1_manifests(
                root=args.root,
                manifest_dir=args.out_dir,
                index_path=args.index,
            )
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(render_manifest_report(summary), encoding="utf-8")
        else:
            summary = {"total": len(manifests["fusion"])}
    except (OSError, ManifestError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(
        "OK: wrote "
        f"{len(written)} manifests for {summary['total']} samples "
        f"under {args.out_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
