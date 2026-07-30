#!/usr/bin/env python
"""Validate M1 feature outputs and write m1_feature_check.md."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from ea_features.io_utils import feature_paths
from ea_features.media import read_source_index


DEFAULT_INDEX = ROOT / "source_index" / "m1_sample_20.csv"
REPORT_PATH = ROOT / "reports" / "m1_feature_check.md"
REQUIRED = ("text", "speech", "macro")
OPTIONAL_MICRO = "micro"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--report", type=Path, default=REPORT_PATH)
    parser.add_argument("--strict-micro", action="store_true", help="Require micro for usable rows")
    return parser.parse_args()


def check_feature(ea_id: str, modality: str) -> dict:
    npy_path, meta_path = feature_paths(ea_id, modality)
    try:
        rel_path = str(npy_path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        rel_path = str(npy_path).replace("\\", "/")
    result = {
        "ea_id": ea_id,
        "modality": modality,
        "ok": False,
        "shape": "",
        "status": "",
        "path": rel_path,
        "error": "",
    }
    if not npy_path.is_file():
        result["error"] = "missing npy"
        return result
    if npy_path.stat().st_size <= 0:
        result["error"] = "empty npy"
        return result
    try:
        arr = np.load(npy_path)
    except Exception as exc:  # noqa: BLE001
        result["error"] = f"load failed: {exc}"
        return result
    if arr.size == 0:
        result["error"] = "zero-size array"
        return result
    result["shape"] = str(tuple(arr.shape))
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        result["status"] = str(meta.get("status", ""))
    result["ok"] = True
    return result


def main() -> None:
    args = parse_args()
    rows = read_source_index(args.index)
    if len(rows) < 20:
        print(f"ERROR: expected at least 20 rows, got {len(rows)}", file=sys.stderr)
        raise SystemExit(1)

    checks: list[dict] = []
    errors: list[str] = []

    for row in rows:
        ea_id = row["ea_id"]
        for modality in REQUIRED:
            item = check_feature(ea_id, modality)
            checks.append(item)
            if not item["ok"]:
                errors.append(f"{ea_id}/{modality}: {item['error']}")

        micro_item = check_feature(ea_id, OPTIONAL_MICRO)
        checks.append(micro_item)
        usable = row.get("usable_for_micro", "true").lower() == "true"
        if usable and args.strict_micro and not micro_item["ok"]:
            errors.append(f"{ea_id}/micro: {micro_item['error']}")
        elif usable and not micro_item["ok"]:
            # Soft requirement: still report but default to hard fail for M1 seed
            # because all seed rows are usable_for_micro=true.
            errors.append(f"{ea_id}/micro: {micro_item['error']}")

    by_mod = Counter()
    ok_mod = Counter()
    for item in checks:
        by_mod[item["modality"]] += 1
        if item["ok"]:
            ok_mod[item["modality"]] += 1

    try:
        index_display = args.index.relative_to(ROOT).as_posix()
    except ValueError:
        index_display = args.index.as_posix()

    lines = [
        "# M1 Feature Check",
        "",
        f"Index: `{index_display}`",
        f"Samples: {len(rows)}",
        "",
        "## Summary",
        "",
        "| Modality | OK | Total |",
        "| --- | ---: | ---: |",
    ]
    for modality in (*REQUIRED, OPTIONAL_MICRO):
        lines.append(f"| `{modality}` | {ok_mod[modality]} | {by_mod[modality]} |")

    lines.extend(
        [
            "",
            "## Per-sample",
            "",
            "| EA ID | text | speech | macro | micro |",
            "| --- | --- | --- | --- | --- |",
        ]
    )

    by_ea: dict[str, dict[str, dict]] = {}
    for item in checks:
        by_ea.setdefault(item["ea_id"], {})[item["modality"]] = item

    for row in rows:
        ea_id = row["ea_id"]
        cells = []
        for modality in (*REQUIRED, OPTIONAL_MICRO):
            item = by_ea[ea_id][modality]
            if item["ok"]:
                cells.append(f"ok {item['shape']}")
            else:
                cells.append(f"FAIL ({item['error']})")
        lines.append(f"| `{ea_id}` | " + " | ".join(cells) + " |")

    lines.extend(["", "## Errors", ""])
    if errors:
        for err in errors:
            lines.append(f"- {err}")
    else:
        lines.append("- none")

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {args.report}")

    if errors:
        print(f"ERROR: {len(errors)} validation issues", file=sys.stderr)
        raise SystemExit(1)
    print(
        "OK: validated features "
        f"(text={ok_mod['text']}, speech={ok_mod['speech']}, "
        f"macro={ok_mod['macro']}, micro={ok_mod['micro']})"
    )


if __name__ == "__main__":
    main()
